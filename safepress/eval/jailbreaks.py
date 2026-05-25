"""
Adversarial-attack evaluation for safety-protected models.

Loads pre-generated HarmBench attack prompts (GCG / AutoDAN / PAIR) produced
by :func:`safepress.data.prepare.prepare_harmbench_attacks` and joins them
against the canonical HarmBench behaviors (``data/harmbench.jsonl``) to feed
the HarmBench classifier. The HarmBench classifier requires the *behavior*
description -- not the adversarial prompt -- to judge whether a generation
satisfies the underlying harmful intent.

Workflow per (model, condition, attack):

    1. Read attack prompts (each carries a ``behavior_id``).
    2. Look up the canonical behavior description for each ``behavior_id``.
    3. Generate model completions for the attack prompts (the model never sees
       the canonical behavior text).
    4. Classify (behavior, generation) pairs with the HarmBench classifier.
       Optionally also run the heuristic detector for a cheap fallback.
    5. Report ASR plus the per-prompt outcome vector that the paired
       bootstrap / McNemar tests in ``safepress.eval.stats`` need.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch

from safepress.eval.basic import GenConfig, generate_completions, is_refusal
from safepress.utils.io import read_jsonl


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class JailbreakConfig:
    """Configuration for a single (attack, model) jailbreak run."""

    attack_prompts_path: str           # JSONL with prompt+behavior+attack fields
    behaviors_path: str = "data/harmbench.jsonl"  # canonical behavior descriptions
    behavior_id_key: str = "behavior_id"
    behavior_text_key: str = "prompt"
    attack_prompt_key: str = "prompt"
    attack_behavior_key: str = "behavior"
    attack_name_key: str = "attack"

    max_new_tokens: int = 256
    temperature: float = 0.0
    top_p: float = 1.0
    do_sample: bool = False
    max_length: int = 2048
    max_prompts: Optional[int] = None

    # Classifier choices
    use_harmbench_classifier: bool = True
    harmbench_classifier_id: str = "cais/HarmBench-Llama-2-13b-cls"
    harmbench_batch_size: int = 4
    # Fallback if classifier unavailable: heuristic non-refusal -> "harmful"
    heuristic_fallback: bool = True

    # IO
    save_generations_to: Optional[str] = None  # JSONL: prompt+behavior+response+labels


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_behavior_map(
    behaviors_path: str | Path,
    *,
    id_key: str,
    text_key: str,
) -> Dict[str, str]:
    """Build {behavior_id: canonical behavior text}."""
    behavior_map: Dict[str, str] = {}
    for row in read_jsonl(behaviors_path):
        bid = str(row.get(id_key, "")).strip()
        if not bid:
            continue
        behavior_map[bid] = str(row.get(text_key, "")).strip()
    return behavior_map


def _load_attack_rows(
    attack_prompts_path: str | Path,
    *,
    prompt_key: str,
    behavior_key: str,
    attack_key: str,
    max_prompts: Optional[int],
) -> List[Dict[str, str]]:
    rows = read_jsonl(attack_prompts_path)
    out: List[Dict[str, str]] = []
    for r in rows:
        prompt = str(r.get(prompt_key, "")).strip()
        if not prompt:
            continue
        out.append(
            {
                "prompt": prompt,
                "behavior_id": str(r.get(behavior_key, "")),
                # JBB-style attack JSONLs carry the canonical goal string
                # in ``behavior_text``; preserve it so we can feed the
                # classifier the right behavior description without having
                # to join against a separate behaviors file.
                "behavior_text_inline": str(r.get("behavior_text", "")),
                "attack": str(r.get(attack_key, "")),
            }
        )
        if max_prompts is not None and len(out) >= max_prompts:
            break
    return out


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_jailbreak_eval(
    model: torch.nn.Module,
    tokenizer,
    config: JailbreakConfig,
    *,
    device: torch.device | str | None = None,
    show_progress: bool = True,
    classifier_model: Optional[torch.nn.Module] = None,
    classifier_tokenizer: Optional[Any] = None,
) -> Dict[str, Any]:
    """Run a single jailbreak evaluation pass.

    Returns a dict with:

    * ``asr`` -- attack success rate (fraction of generations judged harmful)
    * ``n`` -- number of (behavior, generation) pairs scored
    * ``per_prompt_harmful`` -- 0/1 outcome vector aligned to attack prompts
      (suitable for paired bootstrap / McNemar)
    * ``per_prompt_refused`` -- heuristic refusal 0/1 vector
    * ``classifier`` -- ``"harmbench"``, ``"heuristic"``, or ``"unavailable"``
    * ``attack_name`` -- attack tag (gcg / autodan / pair) inferred from rows
    * ``records`` -- per-row details (behavior, prompt, response, judgements)
    """
    behavior_map = _load_behavior_map(
        config.behaviors_path,
        id_key=config.behavior_id_key,
        text_key=config.behavior_text_key,
    )
    attack_rows = _load_attack_rows(
        config.attack_prompts_path,
        prompt_key=config.attack_prompt_key,
        behavior_key=config.attack_behavior_key,
        attack_key=config.attack_name_key,
        max_prompts=config.max_prompts,
    )
    if not attack_rows:
        return {
            "asr": 0.0,
            "n": 0,
            "per_prompt_harmful": [],
            "per_prompt_refused": [],
            "classifier": "unavailable",
            "attack_name": "",
            "records": [],
        }

    attack_name = attack_rows[0]["attack"] or Path(config.attack_prompts_path).stem
    attack_prompts = [r["prompt"] for r in attack_rows]
    # Prefer the inline ``behavior_text`` (JBB-style artifact JSONLs carry the
    # canonical goal string directly); fall back to the behaviors-file join
    # when the inline field is empty; final fallback is the adversarial
    # prompt itself, which still works with the HarmBench classifier but is
    # not the canonical setup.
    behavior_texts = []
    for r in attack_rows:
        t = r.get("behavior_text_inline", "").strip()
        if not t:
            t = behavior_map.get(r["behavior_id"], "")
        behavior_texts.append(t)

    gen_cfg = GenConfig(
        max_new_tokens=config.max_new_tokens,
        temperature=config.temperature,
        top_p=config.top_p,
        do_sample=config.do_sample,
    )
    responses = generate_completions(
        model, tokenizer, attack_prompts,
        gen=gen_cfg, max_length=config.max_length, device=device,
        show_progress=show_progress,
    )

    refused = [bool(is_refusal(r)) for r in responses]

    classifier_used = "unavailable"
    harmful_flags: List[bool] = []
    if config.use_harmbench_classifier:
        try:
            from safepress.eval.harmbench import (
                classify_harmful,
                load_harmbench_classifier,
            )

            if classifier_model is None or classifier_tokenizer is None:
                classifier_model, classifier_tokenizer = load_harmbench_classifier(
                    model_id=config.harmbench_classifier_id,
                )
            # Use the behavior text where available; fall back to attack prompt.
            paired_behaviors = [
                b if b else p for b, p in zip(behavior_texts, attack_prompts)
            ]
            harmful_flags = classify_harmful(
                classifier_model, classifier_tokenizer,
                paired_behaviors, responses,
                batch_size=config.harmbench_batch_size,
                show_progress=show_progress,
            )
            classifier_used = "harmbench"
        except Exception as exc:  # noqa: BLE001
            if not config.heuristic_fallback:
                raise
            # Heuristic fallback: non-refusal -> harmful
            harmful_flags = [not r for r in refused]
            classifier_used = f"heuristic ({exc.__class__.__name__})"

    if not harmful_flags:
        # No classifier requested and we still need an outcome vector for stats.
        harmful_flags = [not r for r in refused]
        if classifier_used == "unavailable":
            classifier_used = "heuristic"

    n = len(attack_rows)
    asr = sum(harmful_flags) / n if n > 0 else 0.0

    records: List[Dict[str, Any]] = []
    for row, resp, harmful, ref in zip(attack_rows, responses, harmful_flags, refused):
        records.append(
            {
                "behavior_id": row["behavior_id"],
                "behavior": behavior_map.get(row["behavior_id"], ""),
                "prompt": row["prompt"],
                "response": resp,
                "harmful": bool(harmful),
                "refused": bool(ref),
                "attack": attack_name,
            }
        )

    if config.save_generations_to:
        from safepress.utils.io import write_jsonl

        write_jsonl(config.save_generations_to, records)

    return {
        "asr": float(asr),
        "n": n,
        "per_prompt_harmful": [int(x) for x in harmful_flags],
        "per_prompt_refused": [int(x) for x in refused],
        "classifier": classifier_used,
        "attack_name": attack_name,
        "records": records,
        "gen_config": gen_cfg.__dict__,
    }


# ---------------------------------------------------------------------------
# Batch convenience: run multiple attacks under one classifier load
# ---------------------------------------------------------------------------

def run_multi_attack_eval(
    model: torch.nn.Module,
    tokenizer,
    attack_configs: Sequence[JailbreakConfig],
    *,
    device: torch.device | str | None = None,
    show_progress: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """Run several attacks, sharing one HarmBench classifier load across all
    of them to amortize VRAM cost.

    Returns ``{attack_name: result_dict}``.
    """
    classifier_model = None
    classifier_tokenizer = None
    out: Dict[str, Dict[str, Any]] = {}
    for cfg in attack_configs:
        # Lazy-load the classifier on the first attack that wants it.
        if cfg.use_harmbench_classifier and classifier_model is None:
            try:
                from safepress.eval.harmbench import load_harmbench_classifier

                classifier_model, classifier_tokenizer = load_harmbench_classifier(
                    model_id=cfg.harmbench_classifier_id,
                )
            except Exception:  # noqa: BLE001
                classifier_model, classifier_tokenizer = None, None

        result = run_jailbreak_eval(
            model, tokenizer, cfg,
            device=device,
            show_progress=show_progress,
            classifier_model=classifier_model,
            classifier_tokenizer=classifier_tokenizer,
        )
        out[result["attack_name"] or Path(cfg.attack_prompts_path).stem] = result

    return out
