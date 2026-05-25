"""
Real Q-resafe baseline reproduction.

The published Q-resafe (arXiv 2506.20251, Hong et al. 2025) re-aligns a
quantized LLM by:

1.  Scoring weights with SNIP (|w * grad|) on a safety calibration set and
    keeping the top-K *safety-critical* sub-modules.
2.  Constructing a safety-patching dataset of (prompt, chosen, rejected)
    triples where ``chosen`` is the pre-quantization (FP16) refusal-style
    response and ``rejected`` is the quantized model's (likely harmful)
    response.
3.  DPO-training **only** the safety-critical sub-modules to pull the
    quantized model back toward its FP16 refusal behaviour.

The existing ``qresafe_noft`` entry in ``safepress.methods.registry`` does
step 1 only and is therefore strictly weaker than the published method --
adequate as a *scoring ablation* control but not as a fair baseline against
SSMP. This module provides the proper version.

Compute footprint
-----------------
* Adds LoRA adapters (rank=16 by default) on the safety-critical linear
  modules of a quantized backbone (the rest is frozen and stays 4-bit).
* Requires a frozen FP16 reference model on the same prompts to compute the
  reference log-likelihoods for the DPO loss. A separate ``reference_model``
  argument is required; the caller is responsible for loading and freeing it.

External deps
-------------
``peft`` is required for the LoRA adapters; ``transformers`` for tokenisation
and generation. Both are lazily imported so importing this module on a CPU-only
node (e.g. during ``safepress --help``) does not pull torch into memory.

The implementation is deliberately self-contained -- no TRL ``DPOTrainer`` --
to keep the training loop auditable for the paper. The DPO loss is the
standard sigmoid form (Rafailov et al. 2023):

    L_DPO = - E[log sigmoid( beta * ( (log p_th(chosen) - log p_ref(chosen))
                                     - (log p_th(rejected) - log p_ref(rejected)) ) )]

where p_th and p_ref are the trainable (quantized + LoRA) and frozen FP16
reference model respectively. Only LoRA parameters receive gradients.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class QResafeConfig:
    """Hyperparameters for the Q-resafe DPO patch.

    Defaults follow the published paper where stated; otherwise standard DPO
    settings (Rafailov et al. 2023). Override per-experiment as needed.
    """

    # --- Safety dataset construction ---
    refusal_template: str = "I'm sorry, but I can't help with that."
    max_chosen_tokens: int = 64        # short refusal target
    max_rejected_tokens: int = 256     # quantized model can ramble
    max_input_length: int = 512        # truncate the prompt portion

    # --- Critical-weight selection ---
    snip_budget_ratio: float = 0.10    # 10% of params are "safety-critical"
    block_size: int = 64

    # --- LoRA ---
    lora_rank: int = 16
    lora_alpha: float = 32.0
    lora_dropout: float = 0.0
    lora_target_module_substrings: Sequence[str] = field(
        default_factory=lambda: (
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        )
    )

    # --- DPO training ---
    n_steps: int = 200
    batch_size: int = 1
    grad_accum: int = 4
    learning_rate: float = 1e-5
    beta: float = 0.1
    weight_decay: float = 0.0
    warmup_steps: int = 10
    seed: int = 0
    log_every: int = 20


@dataclass
class QResafeReport:
    """Bookkeeping for one Q-resafe patch."""

    n_adapted_modules: int
    n_trainable_params: int
    n_critical_modules: int
    n_pairs: int
    steps_completed: int
    final_loss: float
    mean_chosen_logp: float
    mean_rejected_logp: float


# ---------------------------------------------------------------------------
# Step 1: pick the safety-critical modules to adapt
# ---------------------------------------------------------------------------

def _select_critical_modules_from_scores(
    scores_df,
    *,
    target_substrings: Sequence[str],
    budget_ratio: float,
) -> List[str]:
    """Select the top-K linear modules by aggregated SNIP score.

    We aggregate per-module (sum of block-level scores), filter to the
    LoRA-targetable submodule names (``q_proj``, etc.), then take the top
    ``budget_ratio`` fraction by aggregate score.
    """
    df = scores_df.copy()
    if "module" not in df.columns or "score" not in df.columns:
        raise ValueError("scores dataframe must have 'module' and 'score' columns")
    agg = df.groupby("module", as_index=False)["score"].sum()
    keep_mask = agg["module"].apply(
        lambda name: any(s in name for s in target_substrings)
    )
    agg = agg[keep_mask].sort_values("score", ascending=False).reset_index(drop=True)
    if agg.empty:
        return []
    n_keep = max(1, int(round(len(agg) * float(budget_ratio))))
    return agg["module"].iloc[:n_keep].tolist()


# ---------------------------------------------------------------------------
# Step 2: construct chosen / rejected pairs
# ---------------------------------------------------------------------------

@torch.no_grad()
def _construct_pairs(
    quantized_model,
    reference_model,
    tokenizer,
    prompts: Sequence[str],
    *,
    config: QResafeConfig,
    device,
) -> List[Tuple[str, str, str]]:
    """Build a list of (prompt, chosen, rejected) triples.

    * ``chosen``   -- short refusal-style continuation drawn from the FP16
                       reference. We force the canonical refusal template; this
                       removes one degree of freedom from the comparison and is
                       what the Q-resafe paper does for the "patching" dataset.
    * ``rejected`` -- the quantized model's own (possibly harmful) generation.
    """
    from safepress.eval.basic import GenConfig, generate_completions

    rejected = generate_completions(
        quantized_model, tokenizer, list(prompts),
        gen=GenConfig(max_new_tokens=config.max_rejected_tokens),
        max_length=config.max_input_length + config.max_rejected_tokens,
        device=device,
        show_progress=True,
    )
    chosen = [config.refusal_template] * len(prompts)
    return list(zip(prompts, chosen, rejected))


# ---------------------------------------------------------------------------
# Step 3: DPO loss
# ---------------------------------------------------------------------------

def _tokenise_pair(
    tokenizer,
    prompt: str,
    chosen: str,
    rejected: str,
    *,
    max_input_length: int,
    max_chosen_tokens: int,
    max_rejected_tokens: int,
    device,
) -> Dict[str, torch.Tensor]:
    """Tokenise a single (prompt, chosen, rejected) triple.

    Returns input_ids and a per-token mask that is 1 on the response tokens
    only (we condition on the prompt and only score the continuation).
    """

    def _encode(prompt_str: str, response_str: str, max_resp_tokens: int) -> Dict[str, torch.Tensor]:
        # Apply chat template to the prompt
        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
            try:
                p_text = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt_str}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            except TypeError:
                p_text = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt_str}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
        else:
            p_text = prompt_str

        p_ids = tokenizer(
            p_text, return_tensors="pt", truncation=True,
            max_length=max_input_length,
            add_special_tokens=False,
        )["input_ids"][0]
        r_ids = tokenizer(
            response_str, return_tensors="pt", truncation=True,
            max_length=max_resp_tokens,
            add_special_tokens=False,
        )["input_ids"][0]

        input_ids = torch.cat([p_ids, r_ids], dim=0).to(device)
        # Response-only mask
        response_mask = torch.zeros_like(input_ids)
        response_mask[len(p_ids):] = 1
        return {"input_ids": input_ids, "response_mask": response_mask}

    c = _encode(prompt, chosen, max_chosen_tokens)
    r = _encode(prompt, rejected, max_rejected_tokens)
    return {"chosen": c, "rejected": r}


def _logprob_of_response(
    model,
    input_ids: torch.Tensor,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    """Sum of token log-probs under *model*, restricted to the response tokens.

    Returns a scalar tensor (one prompt at a time -- batch size 1 is fine for
    a defensive baseline).

    To support the memory layout where the reference model is parked on CPU
    (policy + LoRA grads need full VRAM during the DPO loop), we move the
    input tensors to whichever device *model* lives on, then return the
    scalar result without copying back -- callers are responsible for
    ``.to(policy_device)`` before arithmetic across the two models.
    """
    # Find the model's actual device (works for HF AutoModelForCausalLM,
    # peft-wrapped, and CPU-parked models alike).
    try:
        model_device = next(model.parameters()).device
    except StopIteration:
        model_device = input_ids.device
    # Move inputs to the model's device only if they're not already there.
    if input_ids.device != model_device:
        input_ids = input_ids.to(model_device)
    if response_mask.device != model_device:
        response_mask = response_mask.to(model_device)
    ids = input_ids.unsqueeze(0)
    logits = model(ids).logits[:, :-1, :].float()  # predict next token
    targets = ids[:, 1:]
    logp_all = F.log_softmax(logits, dim=-1).gather(2, targets.unsqueeze(-1)).squeeze(-1)
    # Mask: a target token at position t corresponds to response_mask at t (we
    # shifted by 1).
    mask = response_mask[1:].unsqueeze(0).to(logp_all.dtype)
    return (logp_all * mask).sum(dim=-1).squeeze(0)


def _dpo_loss(
    chosen_diff: torch.Tensor,
    rejected_diff: torch.Tensor,
    *,
    beta: float,
) -> torch.Tensor:
    """Standard DPO sigmoid loss given the policy-vs-reference log-prob
    differences for the chosen and rejected responses.
    """
    return -F.logsigmoid(beta * (chosen_diff - rejected_diff))


# ---------------------------------------------------------------------------
# Step 4: attach LoRA adapters to the chosen safety-critical modules
# ---------------------------------------------------------------------------

def _attach_lora(
    model,
    target_modules: Sequence[str],
    *,
    rank: int,
    alpha: float,
    dropout: float,
):
    """Inject LoRA adapters into *target_modules*. Returns the wrapped peft model.

    Lazy-imports ``peft`` so the module is importable on CPU-only nodes.

    For a quantized (e.g. bitsandbytes-4bit) base model we call
    ``prepare_model_for_kbit_training`` first; this re-enables gradients on the
    relevant LayerNorm + lm_head layers and casts outputs back to FP32 for
    numerical stability of the DPO loss.
    """
    try:
        from peft import LoraConfig, get_peft_model

        try:
            from peft import prepare_model_for_kbit_training
        except ImportError:
            prepare_model_for_kbit_training = None  # type: ignore[assignment]
    except ImportError as exc:
        raise ImportError(
            "peft is required for Q-resafe DPO patching. "
            "Install with `pip install peft`."
        ) from exc

    if not target_modules:
        raise ValueError("No target modules selected for Q-resafe LoRA.")

    # IMPORTANT: peft.LoraConfig treats every string in ``target_modules`` as a
    # *match pattern* against module names. Passing bare suffixes like
    # ``"q_proj"`` would adapt **every** q_proj in the model -- the wrong
    # behaviour for a baseline that's supposed to only re-train the
    # SNIP-identified safety-critical subset.  Pass the *full dotted paths*
    # instead so each pattern matches exactly one module.
    #
    # We dedup and sort for determinism (different runs with the same scores
    # should produce identical adapter placement).
    full_paths: List[str] = sorted(set(target_modules))

    # Detect a bitsandbytes-quantized base (presence of any int8/uint8 Linear
    # weight is the canonical signal). If so, prepare it for k-bit training.
    is_kbit = False
    try:
        import torch as _torch
        for p in model.parameters():
            if p.dtype in (_torch.int8, _torch.uint8):
                is_kbit = True
                break
    except Exception:  # noqa: BLE001
        is_kbit = False

    if is_kbit and prepare_model_for_kbit_training is not None:
        model = prepare_model_for_kbit_training(model)

    cfg = LoraConfig(
        r=int(rank),
        lora_alpha=float(alpha),
        lora_dropout=float(dropout),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=full_paths,
    )
    wrapped = get_peft_model(model, cfg)
    if hasattr(wrapped, "print_trainable_parameters"):
        wrapped.print_trainable_parameters()
    # Sanity: count how many LoRA adapters actually got attached and warn if
    # zero. (If the full-path matching misses every name -- e.g. when the model
    # gets re-wrapped by accelerate -- LoRA silently attaches nothing and the
    # DPO loop becomes a no-op.)
    n_attached = 0
    for name, _m in wrapped.named_modules():
        if name.endswith(".lora_A.default") or name.endswith(".lora_B.default"):
            n_attached += 1
    if n_attached == 0:
        logger.error(
            "Q-resafe: LoRA attached 0 adapters. Target paths may not match "
            "the model's module names. First few targets: %s",
            full_paths[:3],
        )
        raise RuntimeError(
            "Q-resafe LoRA attached 0 adapters; aborting before DPO loop."
        )
    return wrapped


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def qresafe_patch(
    quantized_model,
    tokenizer,
    reference_model,
    *,
    safety_prompts: Sequence[str],
    scores_df,
    config: Optional[QResafeConfig] = None,
    device: torch.device | str | None = None,
) -> Tuple[Any, QResafeReport]:
    """Apply the Q-resafe DPO patch to *quantized_model* in place.

    Parameters
    ----------
    quantized_model:
        A 4-bit (or lower) quantized causal LM whose safety alignment has
        degraded.  Must accept ``model(input_ids).logits`` and ``model.generate``.
    tokenizer:
        Tokenizer shared between ``quantized_model`` and ``reference_model``.
    reference_model:
        Frozen FP16 (or BF16) reference for the DPO loss. Typically the
        pre-quantization model. The caller is responsible for not training it
        (we explicitly disable its gradients here as well).
    safety_prompts:
        Harmful prompts used to build the patching dataset (AdvBench is the
        Q-resafe default).
    scores_df:
        SNIP scores from ``compute_block_scores(metric='snip', prompt_mode='refusal')``.
        Used to pick the safety-critical modules to attach LoRA adapters to.
    config:
        Hyperparameters; see :class:`QResafeConfig`. ``None`` uses defaults.
    device:
        Target device. If ``None`` we infer from ``quantized_model``.

    Returns
    -------
    (patched_model, report)
        ``patched_model`` is the peft-wrapped quantized model with LoRA adapters
        merged into the forward pass. ``report`` summarises what was adapted
        and the final training stats. The base quantized weights are not
        modified; the adapters store the safety patch.
    """
    cfg = config or QResafeConfig()
    if device is None:
        device = next(
            (p.device for p in quantized_model.parameters() if p.is_floating_point()),
            torch.device("cpu"),
        )
    device = torch.device(device)

    # 1. Pick safety-critical modules
    critical = _select_critical_modules_from_scores(
        scores_df,
        target_substrings=cfg.lora_target_module_substrings,
        budget_ratio=cfg.snip_budget_ratio,
    )
    logger.info("Q-resafe: %d safety-critical modules selected", len(critical))

    # 2. Construct (prompt, chosen, rejected) pairs
    # The reference model generates nothing here -- chosen is the canonical
    # refusal template -- so freeze it now.
    for p in reference_model.parameters():
        p.requires_grad_(False)
    reference_model.eval()

    pairs = _construct_pairs(
        quantized_model, reference_model, tokenizer, list(safety_prompts),
        config=cfg, device=device,
    )
    logger.info("Q-resafe: built %d (prompt, chosen, rejected) triples", len(pairs))

    # 3. Attach LoRA adapters
    patched = _attach_lora(
        quantized_model,
        target_modules=critical,
        rank=cfg.lora_rank,
        alpha=cfg.lora_alpha,
        dropout=cfg.lora_dropout,
    )
    patched.train()

    # 4. Trainable param count
    trainable_params = [p for p in patched.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable_params)
    logger.info("Q-resafe: %d trainable LoRA params", n_trainable)

    optim = torch.optim.AdamW(
        trainable_params,
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    torch.manual_seed(cfg.seed)
    n_pairs = len(pairs)
    step = 0
    final_loss = 0.0
    mean_chosen = 0.0
    mean_rejected = 0.0
    accum_loss = 0.0
    accum_count = 0

    optim.zero_grad(set_to_none=True)
    while step < cfg.n_steps:
        # Cycle through pairs
        for prompt, chosen, rejected in pairs:
            if step >= cfg.n_steps:
                break

            enc = _tokenise_pair(
                tokenizer, prompt, chosen, rejected,
                max_input_length=cfg.max_input_length,
                max_chosen_tokens=cfg.max_chosen_tokens,
                max_rejected_tokens=cfg.max_rejected_tokens,
                device=device,
            )

            # Policy log-probs (trainable)
            policy_chosen = _logprob_of_response(
                patched, enc["chosen"]["input_ids"], enc["chosen"]["response_mask"],
            )
            policy_rejected = _logprob_of_response(
                patched, enc["rejected"]["input_ids"], enc["rejected"]["response_mask"],
            )

            # Reference log-probs (no grad). Move to the policy device before
            # subtracting -- ``reference_model`` may live on a different cuda
            # device when both are loaded with ``device_map='auto'``.
            with torch.no_grad():
                ref_chosen = _logprob_of_response(
                    reference_model, enc["chosen"]["input_ids"], enc["chosen"]["response_mask"],
                ).to(policy_chosen.device)
                ref_rejected = _logprob_of_response(
                    reference_model, enc["rejected"]["input_ids"], enc["rejected"]["response_mask"],
                ).to(policy_rejected.device)

            chosen_diff = policy_chosen - ref_chosen
            rejected_diff = policy_rejected - ref_rejected
            loss = _dpo_loss(chosen_diff, rejected_diff, beta=cfg.beta)
            loss = loss / float(max(1, cfg.grad_accum))
            loss.backward()

            accum_loss += float(loss.item()) * float(max(1, cfg.grad_accum))
            accum_count += 1

            if accum_count % cfg.grad_accum == 0:
                optim.step()
                optim.zero_grad(set_to_none=True)
                if step % cfg.log_every == 0:
                    logger.info(
                        "[Q-resafe DPO] step=%d  loss=%.4f  chosen_lp=%.3f  rejected_lp=%.3f",
                        step,
                        accum_loss / float(max(1, accum_count)),
                        float(policy_chosen.detach().item()),
                        float(policy_rejected.detach().item()),
                    )
                step += 1
                final_loss = accum_loss / float(max(1, accum_count))
                mean_chosen = float(policy_chosen.detach().item())
                mean_rejected = float(policy_rejected.detach().item())
                accum_loss = 0.0
                accum_count = 0

    # Any leftover accumulation
    if accum_count > 0:
        optim.step()
        optim.zero_grad(set_to_none=True)

    report = QResafeReport(
        n_adapted_modules=len(critical),
        n_trainable_params=int(n_trainable),
        n_critical_modules=len(critical),
        n_pairs=n_pairs,
        steps_completed=step,
        final_loss=float(final_loss),
        mean_chosen_logp=float(mean_chosen),
        mean_rejected_logp=float(mean_rejected),
    )

    patched.eval()
    return patched, report
