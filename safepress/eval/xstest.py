"""
XSTest over-refusal evaluation.

XSTest pairs 250 "safe" prompts (that *look* unsafe) with 200 "unsafe"
contrast prompts. A well-aligned model should refuse the unsafe set and comply
with the safe set; a poorly-aligned or over-cautious model will refuse the
safe set too.

This module exposes a single entry point :func:`run_xstest` that loads a JSONL
file produced by :func:`safepress.data.prepare.prepare_safety_prompts` with
``source="xstest"`` and reports:

* ``safe_compliance_rate``  -- 1 - refusal rate on the safe subset
  (higher is better; near FP16 is the goal).
* ``unsafe_refusal_rate``   -- refusal rate on the unsafe subset
  (higher is better; near FP16 is the goal).
* ``safety_score``          -- harmonic mean of the two (Pareto-friendly).

The refusal detector is the same heuristic as :mod:`safepress.eval.basic`; we
optionally accept a callable to plug in a stronger classifier (Llama-Guard-3
or HarmBench) for higher-fidelity reporting.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import torch

from safepress.eval.basic import GenConfig, generate_completions, is_refusal
from safepress.utils.io import read_jsonl


@dataclass
class XSTestConfig:
    prompts_path: str = "data/xstest.jsonl"
    label_key: str = "label"      # "safe" or "unsafe"
    prompt_key: str = "prompt"
    max_new_tokens: int = 256
    temperature: float = 0.0
    top_p: float = 1.0
    do_sample: bool = False
    max_length: int = 2048
    max_prompts_per_label: Optional[int] = None


def _harmonic_mean(a: float, b: float) -> float:
    if a <= 0.0 or b <= 0.0:
        return 0.0
    return 2.0 * a * b / (a + b)


def run_xstest(
    model: torch.nn.Module,
    tokenizer,
    config: XSTestConfig,
    *,
    refusal_detector: Optional[Callable[[str], bool]] = None,
    device: torch.device | str | None = None,
    show_progress: bool = True,
) -> Dict[str, Any]:
    """Run XSTest and return safe-compliance + unsafe-refusal rates.

    The returned dict has top-level keys:

    * ``safe_compliance_rate`` (float in [0, 1])
    * ``unsafe_refusal_rate`` (float in [0, 1])
    * ``safety_score`` (harmonic mean of the two)
    * ``per_prompt`` (list of dicts with prompt / label / refused / response)
    * ``n_safe``, ``n_unsafe``

    The list is order-preserving: rows for the safe split come first, then
    rows for the unsafe split (each in the order they appear in the JSONL).

    Parameters
    ----------
    refusal_detector:
        Callable mapping a response string to ``True`` (refused) or ``False``.
        Defaults to :func:`safepress.eval.basic.is_refusal`. Pass a Llama-Guard
        or HarmBench-classifier wrapper for higher-fidelity reporting.
    """
    detector = refusal_detector or is_refusal

    rows = read_jsonl(config.prompts_path)
    safe_rows = [r for r in rows if str(r.get(config.label_key, "")).lower() == "safe"]
    unsafe_rows = [r for r in rows if str(r.get(config.label_key, "")).lower() == "unsafe"]

    if config.max_prompts_per_label is not None:
        safe_rows = safe_rows[: config.max_prompts_per_label]
        unsafe_rows = unsafe_rows[: config.max_prompts_per_label]

    safe_prompts = [str(r[config.prompt_key]) for r in safe_rows]
    unsafe_prompts = [str(r[config.prompt_key]) for r in unsafe_rows]

    gen_cfg = GenConfig(
        max_new_tokens=config.max_new_tokens,
        temperature=config.temperature,
        top_p=config.top_p,
        do_sample=config.do_sample,
    )

    safe_responses = generate_completions(
        model, tokenizer, safe_prompts,
        gen=gen_cfg, max_length=config.max_length, device=device,
        show_progress=show_progress,
    ) if safe_prompts else []

    unsafe_responses = generate_completions(
        model, tokenizer, unsafe_prompts,
        gen=gen_cfg, max_length=config.max_length, device=device,
        show_progress=show_progress,
    ) if unsafe_prompts else []

    safe_refused = [bool(detector(r)) for r in safe_responses]
    unsafe_refused = [bool(detector(r)) for r in unsafe_responses]

    n_safe = len(safe_prompts)
    n_unsafe = len(unsafe_prompts)
    safe_refusal_rate = (sum(safe_refused) / n_safe) if n_safe else 0.0
    unsafe_refusal_rate = (sum(unsafe_refused) / n_unsafe) if n_unsafe else 0.0
    safe_compliance_rate = 1.0 - safe_refusal_rate

    per_prompt: List[Dict[str, Any]] = []
    for p, r, refused in zip(safe_prompts, safe_responses, safe_refused):
        per_prompt.append({"prompt": p, "label": "safe", "refused": refused, "response": r})
    for p, r, refused in zip(unsafe_prompts, unsafe_responses, unsafe_refused):
        per_prompt.append({"prompt": p, "label": "unsafe", "refused": refused, "response": r})

    return {
        "safe_compliance_rate": safe_compliance_rate,
        "unsafe_refusal_rate": unsafe_refusal_rate,
        "safety_score": _harmonic_mean(safe_compliance_rate, unsafe_refusal_rate),
        "safe_refusal_rate": safe_refusal_rate,
        "n_safe": n_safe,
        "n_unsafe": n_unsafe,
        "per_prompt": per_prompt,
        "gen_config": gen_cfg.__dict__,
    }
