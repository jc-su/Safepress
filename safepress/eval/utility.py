"""
Utility benchmarks for measuring general capability retention.

Measures whether quantization preserves general task performance
while (potentially) degrading safety alignment.
"""
from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional

import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resolve_device(model: torch.nn.Module, device: Optional[str | torch.device]) -> torch.device:
    """Return an explicit device, falling back to the model's first parameter device."""
    if device is not None:
        return torch.device(device)
    try:
        return next(p.device for p in model.parameters() if p.is_floating_point())
    except StopIteration:
        return torch.device("cpu")


# ---------------------------------------------------------------------------
# Perplexity
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_perplexity(
    model: torch.nn.Module,
    tokenizer,
    *,
    dataset_name: str = "wikitext",
    dataset_config: str = "wikitext-2-raw-v1",
    split: str = "test",
    n_samples: int = 100,
    max_length: int = 2048,
    stride: int = 512,
    device: Optional[str | torch.device] = None,
    show_progress: bool = True,
) -> Dict[str, Any]:
    """
    Compute perplexity on a language modelling dataset (default: WikiText-2 test).

    Uses a sliding-window approach so that long documents are handled correctly
    even when ``max_length`` is shorter than the full text.

    Args:
        model: A causal language model.
        tokenizer: The corresponding tokenizer.
        dataset_name: HuggingFace dataset name.
        dataset_config: Dataset configuration / subset.
        split: Which split to evaluate on.
        n_samples: Maximum number of text samples to use.
        max_length: Context window used per forward pass.
        stride: Stride for the sliding window.
        device: Torch device.  Auto-detected from model if ``None``.
        show_progress: Show tqdm progress bar.

    Returns:
        Dict with ``perplexity`` and ``n_samples``.
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "The `datasets` library is required for perplexity evaluation. "
            "Install with `pip install datasets`."
        ) from exc

    device = _resolve_device(model, device)
    model.eval()

    logger.info(
        "Loading dataset %s/%s split=%s for perplexity evaluation.",
        dataset_name,
        dataset_config,
        split,
    )
    ds = load_dataset(dataset_name, dataset_config, split=split)

    # Concatenate all texts and tokenise once (standard approach for WikiText PPL).
    texts: List[str] = []
    for row in ds:
        text = row.get("text", "")
        if text.strip():
            texts.append(text)
        if len(texts) >= n_samples:
            break

    full_text = "\n\n".join(texts)
    encodings = tokenizer(full_text, return_tensors="pt")
    input_ids = encodings["input_ids"]  # (1, seq_len)
    seq_len = input_ids.size(1)

    nlls: List[float] = []
    n_tokens_scored: int = 0

    positions = list(range(0, seq_len, stride))
    if show_progress:
        positions = tqdm(positions, desc="Perplexity", leave=False)

    for begin_loc in positions:
        end_loc = min(begin_loc + max_length, seq_len)
        # Tokens that form the input window
        trg_len = end_loc - begin_loc
        input_chunk = input_ids[:, begin_loc:end_loc].to(device)

        # Labels: mask out the first `overlap` tokens so we don't double-count
        target_chunk = input_chunk.clone()
        # For sliding window, only score the last (stride) tokens except for
        # the very first window which scores everything.
        if begin_loc != 0:
            target_chunk[:, :-stride] = -100

        outputs = model(input_ids=input_chunk, labels=target_chunk)
        # outputs.loss is the mean NLL over non-masked tokens in the chunk
        neg_log_likelihood = outputs.loss

        # Count how many tokens were actually scored
        n_scored = (target_chunk != -100).sum().item()
        nlls.append(neg_log_likelihood.float().item() * n_scored)
        n_tokens_scored += n_scored

        if end_loc >= seq_len:
            break

    if n_tokens_scored == 0:
        logger.warning("No tokens scored -- perplexity is undefined.")
        return {"perplexity": float("inf"), "n_samples": len(texts), "n_tokens_scored": 0}

    avg_nll = sum(nlls) / n_tokens_scored
    ppl = math.exp(avg_nll)

    return {
        "perplexity": ppl,
        "n_samples": len(texts),
        "n_tokens_scored": n_tokens_scored,
    }


# ---------------------------------------------------------------------------
# MMLU (lite)
# ---------------------------------------------------------------------------

_MMLU_CHOICES = ["A", "B", "C", "D"]


def _format_mmlu_prompt(question: str, choices: List[str]) -> str:
    """Build a simple multiple-choice prompt for MMLU."""
    lines = [question]
    for letter, choice in zip(_MMLU_CHOICES, choices):
        lines.append(f"{letter}. {choice}")
    lines.append("Answer:")
    return "\n".join(lines)


@torch.no_grad()
def eval_mmlu_lite(
    model: torch.nn.Module,
    tokenizer,
    *,
    n_questions: int = 200,
    subjects: Optional[List[str]] = None,
    dataset_id: str = "cais/mmlu",
    dataset_fallback_id: str = "hails/mmlu_no_train",
    split: str = "test",
    device: Optional[str | torch.device] = None,
    show_progress: bool = True,
) -> Dict[str, Any]:
    """
    Simple MMLU evaluation using log-probability scoring of answer choices.

    For each question we compute the log-probability of each choice token
    (A / B / C / D) given the formatted prompt and pick the highest.

    Args:
        model: A causal language model.
        tokenizer: Corresponding tokenizer.
        n_questions: Maximum number of questions to evaluate.
        subjects: If provided, restrict to these MMLU subjects.
        dataset_id: Primary HuggingFace dataset identifier.
        dataset_fallback_id: Fallback dataset id if primary fails.
        split: Dataset split.
        device: Torch device (auto-detected if None).
        show_progress: Show progress bar.

    Returns:
        Dict with ``accuracy``, ``n_questions``, and ``per_subject`` breakdown.
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "The `datasets` library is required for MMLU evaluation."
        ) from exc

    device = _resolve_device(model, device)
    model.eval()

    # Try loading dataset -- fall back if primary id is unavailable.
    ds = None
    for ds_id in [dataset_id, dataset_fallback_id]:
        try:
            logger.info("Attempting to load MMLU dataset: %s (split=%s)", ds_id, split)
            ds = load_dataset(ds_id, "all", split=split)
            break
        except Exception:
            logger.info("Failed to load %s, trying fallback...", ds_id)
            continue

    if ds is None:
        logger.warning("Could not load any MMLU dataset. Returning empty results.")
        return {"accuracy": 0.0, "n_questions": 0, "per_subject": {}}

    # Optionally filter by subject
    if subjects is not None:
        subject_set = {s.lower() for s in subjects}
        ds = ds.filter(lambda row: row.get("subject", "").lower() in subject_set)

    # Limit the number of questions
    if len(ds) > n_questions:
        ds = ds.select(range(n_questions))

    # Pre-compute token ids for A, B, C, D
    choice_token_ids: List[int] = []
    for letter in _MMLU_CHOICES:
        ids = tokenizer.encode(letter, add_special_tokens=False)
        # Take the last token id (handles tokenizers that split differently)
        choice_token_ids.append(ids[-1])

    correct = 0
    total = 0
    per_subject: Dict[str, Dict[str, int]] = {}  # subject -> {correct, total}

    it = range(len(ds))
    if show_progress:
        it = tqdm(it, desc="MMLU-lite", leave=False)

    for idx in it:
        row = ds[idx]
        question = row.get("question", "")
        choices = row.get("choices", [])
        answer_idx = row.get("answer", None)
        subject = row.get("subject", "unknown")

        if len(choices) < 4 or answer_idx is None:
            continue

        # Ensure answer_idx is int (some datasets store it as string letter)
        if isinstance(answer_idx, str):
            answer_idx = _MMLU_CHOICES.index(answer_idx.upper()) if answer_idx.upper() in _MMLU_CHOICES else int(answer_idx)

        prompt_text = _format_mmlu_prompt(question, choices)
        enc = tokenizer(
            prompt_text,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        ).to(device)

        outputs = model(**enc)
        # Get logits at the last position (next-token prediction)
        logits = outputs.logits[0, -1, :]  # (vocab_size,)
        choice_logits = torch.tensor(
            [logits[tid].item() for tid in choice_token_ids],
            dtype=torch.float32,
        )
        predicted = int(choice_logits.argmax().item())

        is_correct = predicted == answer_idx
        if is_correct:
            correct += 1
        total += 1

        # Track per-subject
        if subject not in per_subject:
            per_subject[subject] = {"correct": 0, "total": 0}
        per_subject[subject]["total"] += 1
        if is_correct:
            per_subject[subject]["correct"] += 1

    accuracy = correct / total if total > 0 else 0.0
    per_subject_acc: Dict[str, float] = {}
    for subj, counts in per_subject.items():
        per_subject_acc[subj] = (
            counts["correct"] / counts["total"] if counts["total"] > 0 else 0.0
        )

    return {
        "accuracy": accuracy,
        "n_questions": total,
        "per_subject": per_subject_acc,
    }


# ---------------------------------------------------------------------------
# TruthfulQA (lite)
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_truthfulqa_lite(
    model: torch.nn.Module,
    tokenizer,
    *,
    n_questions: int = 100,
    dataset_id: str = "truthfulqa/truthful_qa",
    dataset_config: str = "multiple_choice",
    split: str = "validation",
    device: Optional[str | torch.device] = None,
    show_progress: bool = True,
) -> Dict[str, Any]:
    """
    Simple TruthfulQA multiple-choice evaluation.

    For each question, we score each candidate answer by its average
    log-probability under the model and pick the best one.

    Args:
        model: A causal language model.
        tokenizer: Corresponding tokenizer.
        n_questions: Maximum questions to evaluate.
        dataset_id: HuggingFace dataset identifier.
        dataset_config: Dataset configuration.
        split: Dataset split (TruthfulQA only has ``validation``).
        device: Torch device (auto-detected if None).
        show_progress: Show progress bar.

    Returns:
        Dict with ``accuracy`` and ``n_questions``.
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "The `datasets` library is required for TruthfulQA evaluation."
        ) from exc

    device = _resolve_device(model, device)
    model.eval()

    logger.info("Loading TruthfulQA dataset: %s/%s split=%s", dataset_id, dataset_config, split)
    try:
        ds = load_dataset(dataset_id, dataset_config, split=split)
    except Exception as exc:
        logger.warning("Could not load TruthfulQA dataset: %s", exc)
        return {"accuracy": 0.0, "n_questions": 0}

    if len(ds) > n_questions:
        ds = ds.select(range(n_questions))

    correct = 0
    total = 0

    it = range(len(ds))
    if show_progress:
        it = tqdm(it, desc="TruthfulQA-lite", leave=False)

    for idx in it:
        row = ds[idx]
        question = row.get("question", "")
        mc1_targets = row.get("mc1_targets", None)

        if mc1_targets is None:
            continue

        choices = mc1_targets.get("choices", [])
        labels = mc1_targets.get("labels", [])

        if not choices or not labels:
            continue

        # Find the correct answer index (label == 1)
        correct_idx: Optional[int] = None
        for i, lab in enumerate(labels):
            if lab == 1:
                correct_idx = i
                break
        if correct_idx is None:
            continue

        # Score each choice by average log-prob
        best_score = float("-inf")
        best_idx = 0

        prompt_prefix = f"Q: {question}\nA:"

        for c_idx, choice in enumerate(choices):
            full_text = f"{prompt_prefix} {choice}"
            enc = tokenizer(
                full_text,
                return_tensors="pt",
                truncation=True,
                max_length=2048,
            ).to(device)

            input_ids = enc["input_ids"]
            outputs = model(**enc)
            logits = outputs.logits  # (1, seq_len, vocab_size)

            # Compute log-prob over the answer tokens only
            prefix_enc = tokenizer(
                prompt_prefix,
                return_tensors="pt",
                truncation=True,
                max_length=2048,
            )
            prefix_len = prefix_enc["input_ids"].shape[1]

            # Shift logits and labels for causal LM scoring
            shift_logits = logits[0, prefix_len - 1 : -1, :]  # predict tokens at positions [prefix_len, ...)
            shift_labels = input_ids[0, prefix_len:]

            if shift_labels.numel() == 0:
                score = float("-inf")
            else:
                log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
                token_log_probs = log_probs.gather(
                    1, shift_labels.unsqueeze(-1)
                ).squeeze(-1)
                score = token_log_probs.mean().item()

            if score > best_score:
                best_score = score
                best_idx = c_idx

        if best_idx == correct_idx:
            correct += 1
        total += 1

    accuracy = correct / total if total > 0 else 0.0
    return {
        "accuracy": accuracy,
        "n_questions": total,
    }


# ---------------------------------------------------------------------------
# Combined utility evaluation
# ---------------------------------------------------------------------------

def eval_all_utility(
    model: torch.nn.Module,
    tokenizer,
    *,
    device: Optional[str | torch.device] = None,
    perplexity_kwargs: Optional[Dict[str, Any]] = None,
    mmlu_kwargs: Optional[Dict[str, Any]] = None,
    truthfulqa_kwargs: Optional[Dict[str, Any]] = None,
    run_perplexity: bool = True,
    run_mmlu: bool = True,
    run_truthfulqa: bool = True,
) -> Dict[str, Any]:
    """
    Run all utility benchmarks and return combined results.

    Each benchmark is wrapped in a try/except so that a single failure does not
    block the others.

    Args:
        model: A causal language model.
        tokenizer: Corresponding tokenizer.
        device: Torch device (auto-detected if None).
        perplexity_kwargs: Extra kwargs forwarded to :func:`eval_perplexity`.
        mmlu_kwargs: Extra kwargs forwarded to :func:`eval_mmlu_lite`.
        truthfulqa_kwargs: Extra kwargs forwarded to :func:`eval_truthfulqa_lite`.
        run_perplexity: Whether to run the perplexity benchmark.
        run_mmlu: Whether to run the MMLU benchmark.
        run_truthfulqa: Whether to run the TruthfulQA benchmark.

    Returns:
        Dict with keys ``perplexity``, ``mmlu``, and ``truthfulqa``, each either
        a results dict or an error dict ``{"ok": False, "error": ...}``.
    """
    device = _resolve_device(model, device)
    results: Dict[str, Any] = {}

    # Perplexity
    if run_perplexity:
        try:
            ppl_kw = dict(device=device)
            if perplexity_kwargs:
                ppl_kw.update(perplexity_kwargs)
            results["perplexity"] = eval_perplexity(model, tokenizer, **ppl_kw)
        except Exception as exc:
            logger.warning("Perplexity evaluation failed: %s", exc)
            results["perplexity"] = {"ok": False, "error": str(exc)}

    # MMLU
    if run_mmlu:
        try:
            mmlu_kw = dict(device=device)
            if mmlu_kwargs:
                mmlu_kw.update(mmlu_kwargs)
            results["mmlu"] = eval_mmlu_lite(model, tokenizer, **mmlu_kw)
        except Exception as exc:
            logger.warning("MMLU evaluation failed: %s", exc)
            results["mmlu"] = {"ok": False, "error": str(exc)}

    # TruthfulQA
    if run_truthfulqa:
        try:
            tqa_kw = dict(device=device)
            if truthfulqa_kwargs:
                tqa_kw.update(truthfulqa_kwargs)
            results["truthfulqa"] = eval_truthfulqa_lite(model, tokenizer, **tqa_kw)
        except Exception as exc:
            logger.warning("TruthfulQA evaluation failed: %s", exc)
            results["truthfulqa"] = {"ok": False, "error": str(exc)}

    return results
