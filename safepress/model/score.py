from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import pandas as pd
import torch
from tqdm import tqdm

from safepress.model.blocks import chunk_indices, iter_linear_modules


@dataclass
class ScoreRow:
    module: str
    block_idx: int
    out_start: int
    out_end: int
    in_features: int
    num_params: int
    score: float


@torch.no_grad()
def _quant_dequant_symmetric_groupwise(
    w: torch.Tensor, *, bits: int, group_size: int
) -> torch.Tensor:
    """
    Groupwise symmetric quant-dequant along the last dimension (in_features).
    This is a *simulation* used only to estimate quantization error Δw.

    Processes in row chunks to limit peak memory for large weight matrices.
    """
    if w.ndim != 2:
        raise ValueError(f"Expected 2D weight matrix, got shape={tuple(w.shape)}")
    if bits < 2 or bits > 8:
        raise ValueError(f"bits must be in [2,8], got {bits}")

    out_features, in_features = w.shape
    if group_size <= 0 or group_size > in_features:
        group_size = in_features

    qmax = (2 ** (bits - 1)) - 1  # signed

    # Process in row chunks to limit peak memory.
    # Use smaller chunks (64 rows) to avoid OOM on multi-GPU splits.
    chunk_rows = min(64, out_features)
    w_hat = torch.empty_like(w)

    for start in range(0, out_features, chunk_rows):
        end = min(start + chunk_rows, out_features)
        # Compute on CPU if GPU memory is tight, then move back
        orig_device = w.device
        try:
            w_chunk = w[start:end, :].float()
        except torch.cuda.OutOfMemoryError:
            w_chunk = w[start:end, :].cpu().float()

        # Pad in_features to multiple of group_size
        pad = (group_size - (in_features % group_size)) % group_size
        if pad:
            w_chunk = torch.nn.functional.pad(w_chunk, (0, pad))
        in_padded = w_chunk.shape[1]
        n_groups = in_padded // group_size

        w_g = w_chunk.view(end - start, n_groups, group_size)
        max_abs = w_g.abs().amax(dim=2, keepdim=True)
        scale = max_abs / float(qmax)
        scale = torch.clamp(scale, min=1e-8)
        q = torch.round(w_g / scale).clamp(-qmax, qmax)
        result = (q * scale).view(end - start, in_padded)
        if pad:
            result = result[:, :in_features]
        w_hat[start:end, :] = result.to(device=orig_device, dtype=w.dtype)

    return w_hat


def _apply_chat_template_for_scoring(tokenizer, prompt: str) -> str:
    """Wrap a raw prompt in the tokenizer's chat template, matching the form
    used at eval time by :func:`safepress.eval.basic.generate_completions`.

    Critical for scoring alignment: if scoring sees raw prompts but eval sees
    chat-formatted prompts (with ``<|im_start|>user`` / ``<|im_end|>`` tokens
    on Qwen3, ``<|begin_of_text|>`` on Llama, etc.), the gradients identify a
    different set of safety-critical weights than what the model actually
    uses at deployment. Pre-2026 versions of this scorer had that mismatch
    and were a real review-killer; we now always apply the chat template
    when one is available.
    """
    if not hasattr(tokenizer, "apply_chat_template") or tokenizer.chat_template is None:
        return prompt
    messages = [{"role": "user", "content": prompt}]
    try:
        # Disable Qwen3-style thinking mode for deterministic scoring.
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )


def build_refusal_supervision(
    tokenizer,
    prompts: List[str],
    refusal_template: str,
    max_length: int = 2048,
    use_chat_template: bool = True,
):
    """Build a batch for causal LM loss that supervises only the refusal
    template tokens.

    labels: -100 for prompt tokens, token ids for refusal tokens.

    By default we apply the tokenizer's chat template to each prompt so that
    the gradient computation sees the *same* token distribution that the model
    sees at eval / generation time. This is essential for instruction-tuned
    models whose safety alignment lives behind the chat tokens (e.g. Qwen3,
    Llama-3.x, Gemma-2). Pass ``use_chat_template=False`` to recover the legacy
    raw-prompt behaviour (only useful for base/non-chat models).
    """
    # Apply chat template to each prompt when available (default).
    if use_chat_template:
        formatted_prompts = [
            _apply_chat_template_for_scoring(tokenizer, p) for p in prompts
        ]
    else:
        formatted_prompts = list(prompts)

    # Tokenize prompt and refusal separately to create labels.
    # add_special_tokens=False because the chat template already includes them.
    prompt_enc = tokenizer(formatted_prompts, add_special_tokens=False)
    refusal_enc = tokenizer([refusal_template] * len(prompts), add_special_tokens=False)

    input_ids_list = []
    labels_list = []
    for p_ids, r_ids in zip(prompt_enc["input_ids"], refusal_enc["input_ids"]):
        ids = p_ids + r_ids
        ids = ids[:max_length]
        # labels only on refusal part
        cutoff = min(len(p_ids), max_length)
        lab = [-100] * cutoff
        # refusal labels
        r_labs = r_ids[: max(0, max_length - cutoff)]
        lab.extend(r_labs)
        # If prompt got truncated, refusal labels might be empty; that's OK (loss will be 0)
        input_ids_list.append(ids)
        labels_list.append(lab)

    # Pad input_ids using tokenizer, then manually pad labels with -100
    batch = tokenizer.pad(
        {"input_ids": input_ids_list},
        padding=True,
        return_tensors="pt",
    )
    # Manually pad labels to same length as input_ids
    max_len = batch["input_ids"].shape[1]
    padded_labels = []
    for lab in labels_list:
        pad_len = max_len - len(lab)
        padded_labels.append(lab + [-100] * pad_len)
    batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)
    return batch


def build_lm_supervision(tokenizer, texts: List[str], max_length: int = 2048):
    """Build a standard causal-LM batch (labels=input_ids)."""
    enc = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    enc["labels"] = enc["input_ids"].clone()
    return enc


SUPPORTED_METRICS = {
    "taylor_abs",  # sum |g * (Q(w)-w)|
    "grad_abs",    # sum |g|
    "grad_sq",     # sum g^2
    "snip",        # sum |g*w|
}


def compute_block_scores(
    model: torch.nn.Module,
    tokenizer,
    prompts: List[str],
    *,
    refusal_template: str = "I'm sorry, but I can't help with that.",
    metric: str = "taylor_abs",
    prompt_mode: str = "refusal",  # 'refusal' or 'lm'
    bits: int = 4,
    group_size: int = 128,
    block_size: int = 64,
    max_length: int = 2048,
    batch_size: int = 1,
    device: torch.device | str | None = None,
    show_progress: bool = True,
) -> pd.DataFrame:
    """
    Compute per-(Linear, out_block) safety drift scores using a Taylor proxy.

    Output: pandas DataFrame with columns:
      module, block_idx, out_start, out_end, in_features, num_params, score
    """
    if metric not in SUPPORTED_METRICS:
        raise ValueError(f"Unsupported metric '{metric}'. Supported: {sorted(SUPPORTED_METRICS)}")
    if prompt_mode not in {"refusal", "lm"}:
        raise ValueError("prompt_mode must be 'refusal' or 'lm'")

    if device is None:
        try:
            device = next(p.device for p in model.parameters() if p.is_floating_point())
        except StopIteration:
            device = torch.device("cpu")
    device = torch.device(device)

    model.eval()
    # Make sure grads are enabled
    for p in model.parameters():
        p.requires_grad_(True)

    # Create an index so we can map module names -> weight param
    linear_modules: List[Tuple[str, torch.nn.Linear]] = list(iter_linear_modules(model))
    if len(linear_modules) == 0:
        raise RuntimeError("No torch.nn.Linear modules found. Unsupported architecture?")

    # We'll accumulate scores as float64 in Python.
    # Key: (module_name, block_idx) -> score
    score_accum: Dict[Tuple[str, int], float] = {}
    meta: Dict[Tuple[str, int], ScoreRow] = {}

    # Pre-initialize meta rows
    for mod_name, mod in linear_modules:
        out_features, in_features = mod.weight.shape
        blocks = chunk_indices(out_features, block_size)
        for b_idx, (s, e) in enumerate(blocks):
            num_params = (e - s) * in_features
            meta[(mod_name, b_idx)] = ScoreRow(
                module=mod_name,
                block_idx=b_idx,
                out_start=s,
                out_end=e,
                in_features=in_features,
                num_params=num_params,
                score=0.0,
            )
            score_accum[(mod_name, b_idx)] = 0.0

    # Mini-batching prompts
    it = range(0, len(prompts), batch_size)
    if show_progress:
        it = tqdm(list(it), desc="Scoring batches", leave=False)

    for start in it:
        batch_prompts = prompts[start : start + batch_size]
        if prompt_mode == "refusal":
            batch_inputs = build_refusal_supervision(
                tokenizer, batch_prompts,
                refusal_template=refusal_template,
                max_length=max_length,
            )
        else:
            batch_inputs = build_lm_supervision(tokenizer, batch_prompts, max_length=max_length)
        batch_inputs = {k: v.to(device) for k, v in batch_inputs.items()}

        # Forward + backward
        model.zero_grad(set_to_none=True)
        out = model(**batch_inputs)
        loss = out.loss
        if loss is None:
            raise RuntimeError("Model did not return a loss. Does it support labels?")
        loss.backward()

        # Compute per-linear per-block contributions
        with torch.no_grad():
            for mod_name, mod in linear_modules:
                w = mod.weight
                g = mod.weight.grad
                if g is None:
                    continue
                out_features, in_features = w.shape
                blocks = chunk_indices(out_features, block_size)
                # process each block independently to limit temporary memory
                for b_idx, (s, e) in enumerate(blocks):
                    w_blk = w[s:e, :]
                    g_blk = g[s:e, :]
                    # simulate quantization error for this block
                    w_hat = _quant_dequant_symmetric_groupwise(
                        w_blk, bits=bits, group_size=group_size
                    )
                    delta = (w_hat - w_blk).float()
                    if metric == "taylor_abs":
                        contrib_t = (g_blk.float() * delta).abs().sum()
                    elif metric == "grad_abs":
                        contrib_t = g_blk.float().abs().sum()
                    elif metric == "grad_sq":
                        contrib_t = (g_blk.float() ** 2).sum()
                    elif metric == "snip":
                        contrib_t = (g_blk.float() * w_blk.float()).abs().sum()
                    else:
                        raise AssertionError("unreachable")
                    contrib = float(contrib_t.item())
                    score_accum[(mod_name, b_idx)] += float(contrib)

    # Build dataframe
    rows: List[Dict[str, object]] = []
    for key, score in score_accum.items():
        sr = meta[key]
        rows.append(
            dict(
                module=sr.module,
                block_idx=sr.block_idx,
                out_start=sr.out_start,
                out_end=sr.out_end,
                in_features=sr.in_features,
                num_params=sr.num_params,
                score=float(score),
            )
        )

    df = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    return df
