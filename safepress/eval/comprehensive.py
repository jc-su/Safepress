"""
Comprehensive evaluation runner that combines all safety and utility metrics.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ComprehensiveEvalConfig:
    """Configuration for a comprehensive safety + utility evaluation run."""

    # --- Safety: prompt-based refusal evaluation ---
    eval_prompts_path: Optional[str] = None
    prompt_key: str = "prompt"
    max_prompts: Optional[int] = None
    max_new_tokens: int = 256
    temperature: float = 0.0
    top_p: float = 1.0
    do_sample: bool = False
    max_length: int = 2048

    # --- Safety: HarmBench classifier ---
    use_harmbench_cls: bool = False
    harmbench_behaviors_path: Optional[str] = None
    harmbench_generations_jsonl: Optional[str] = None
    harmbench_classifier_id: str = "cais/HarmBench-Llama-2-13b-cls"
    harmbench_batch_size: int = 4

    # --- Safety: StrongREJECT ---
    use_strongreject: bool = False
    strongreject_evaluator: str = "strongreject_rubric"

    # --- Utility: perplexity ---
    use_perplexity: bool = True
    perplexity_dataset: str = "wikitext"
    perplexity_config: str = "wikitext-2-raw-v1"
    perplexity_split: str = "test"
    perplexity_n_samples: int = 100
    perplexity_max_length: int = 2048
    perplexity_stride: int = 512

    # --- Utility: MMLU ---
    use_mmlu: bool = False
    mmlu_n_questions: int = 200
    mmlu_subjects: Optional[List[str]] = None

    # --- Utility: TruthfulQA ---
    use_truthfulqa: bool = False
    truthfulqa_n_questions: int = 100

    # --- Output ---
    out_path: Optional[str] = None


# ---------------------------------------------------------------------------
# Comprehensive runner
# ---------------------------------------------------------------------------

def run_comprehensive_eval(
    model: torch.nn.Module,
    tokenizer,
    config: ComprehensiveEvalConfig,
) -> Dict[str, Any]:
    """
    Run all configured evaluations and return a unified results dictionary.

    The result has top-level keys:

    - ``"safety"`` -- dict of safety evaluation results.
    - ``"utility"`` -- dict of utility evaluation results.
    - ``"metadata"`` -- timing and configuration info.

    Each sub-evaluation is wrapped in a try/except so that a single failure
    does not prevent other evaluations from running.

    Args:
        model: A causal language model.
        tokenizer: Corresponding tokenizer.
        config: :class:`ComprehensiveEvalConfig` controlling which evals to run.

    Returns:
        Nested results dict.
    """
    start_time = time.time()

    try:
        device = next(p.device for p in model.parameters() if p.is_floating_point())
    except StopIteration:
        device = torch.device("cpu")

    safety_results: Dict[str, Any] = {}
    utility_results: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Safety evaluations
    # ------------------------------------------------------------------

    # 1. Heuristic refusal rate + optional StrongREJECT
    if config.eval_prompts_path is not None:
        try:
            from safepress.utils.io import read_prompts_jsonl
            from safepress.eval.basic import (
                GenConfig,
                generate_completions,
                refusal_rate,
                try_strongreject_eval,
            )

            prompts = read_prompts_jsonl(config.eval_prompts_path, key=config.prompt_key)
            if config.max_prompts is not None:
                prompts = prompts[: config.max_prompts]

            gen_cfg = GenConfig(
                max_new_tokens=config.max_new_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                do_sample=config.do_sample,
            )
            responses = generate_completions(
                model,
                tokenizer,
                prompts,
                gen=gen_cfg,
                max_length=config.max_length,
                device=device,
                show_progress=True,
            )

            rr = refusal_rate(responses)
            safety_results["refusal"] = {
                "refusal_rate": rr,
                "n": len(prompts),
                "gen_config": gen_cfg.__dict__,
            }

            # StrongREJECT
            if config.use_strongreject:
                sr = try_strongreject_eval(
                    prompts,
                    responses,
                    evaluator=config.strongreject_evaluator,
                )
                safety_results["strongreject"] = sr

            # If HarmBench classifier is requested and we already have generations,
            # run HarmBench on the same (prompt, response) pairs.
            if config.use_harmbench_cls and config.harmbench_behaviors_path is None:
                try:
                    from safepress.eval.harmbench import harmbench_attack_success_rate

                    hb = harmbench_attack_success_rate(
                        prompts,
                        responses,
                        classifier_id=config.harmbench_classifier_id,
                        batch_size=config.harmbench_batch_size,
                    )
                    safety_results["harmbench"] = hb
                except Exception as exc:
                    logger.warning("HarmBench classifier evaluation failed: %s", exc)
                    safety_results["harmbench"] = {"ok": False, "error": str(exc)}

        except Exception as exc:
            logger.warning("Refusal / basic safety evaluation failed: %s", exc)
            safety_results["refusal"] = {"ok": False, "error": str(exc)}

    # 2. HarmBench from file (separate behaviors CSV + generations JSONL)
    if (
        config.use_harmbench_cls
        and config.harmbench_behaviors_path is not None
        and config.harmbench_generations_jsonl is not None
    ):
        try:
            from safepress.eval.harmbench import harmbench_eval_from_file

            hb_file = harmbench_eval_from_file(
                config.harmbench_behaviors_path,
                config.harmbench_generations_jsonl,
                classifier_id=config.harmbench_classifier_id,
                batch_size=config.harmbench_batch_size,
            )
            safety_results["harmbench_file"] = hb_file
        except Exception as exc:
            logger.warning("HarmBench file-based evaluation failed: %s", exc)
            safety_results["harmbench_file"] = {"ok": False, "error": str(exc)}

    # ------------------------------------------------------------------
    # Utility evaluations
    # ------------------------------------------------------------------

    if config.use_perplexity:
        try:
            from safepress.eval.utility import eval_perplexity

            ppl_result = eval_perplexity(
                model,
                tokenizer,
                dataset_name=config.perplexity_dataset,
                dataset_config=config.perplexity_config,
                split=config.perplexity_split,
                n_samples=config.perplexity_n_samples,
                max_length=config.perplexity_max_length,
                stride=config.perplexity_stride,
                device=device,
            )
            utility_results["perplexity"] = ppl_result
        except Exception as exc:
            logger.warning("Perplexity evaluation failed: %s", exc)
            utility_results["perplexity"] = {"ok": False, "error": str(exc)}

    if config.use_mmlu:
        try:
            from safepress.eval.utility import eval_mmlu_lite

            mmlu_result = eval_mmlu_lite(
                model,
                tokenizer,
                n_questions=config.mmlu_n_questions,
                subjects=config.mmlu_subjects,
                device=device,
            )
            utility_results["mmlu"] = mmlu_result
        except Exception as exc:
            logger.warning("MMLU evaluation failed: %s", exc)
            utility_results["mmlu"] = {"ok": False, "error": str(exc)}

    if config.use_truthfulqa:
        try:
            from safepress.eval.utility import eval_truthfulqa_lite

            tqa_result = eval_truthfulqa_lite(
                model,
                tokenizer,
                n_questions=config.truthfulqa_n_questions,
                device=device,
            )
            utility_results["truthfulqa"] = tqa_result
        except Exception as exc:
            logger.warning("TruthfulQA evaluation failed: %s", exc)
            utility_results["truthfulqa"] = {"ok": False, "error": str(exc)}

    # ------------------------------------------------------------------
    # Assemble result
    # ------------------------------------------------------------------

    elapsed = time.time() - start_time
    result: Dict[str, Any] = {
        "safety": safety_results,
        "utility": utility_results,
        "metadata": {
            "elapsed_seconds": round(elapsed, 2),
            "device": str(device),
            "config": {
                k: v
                for k, v in config.__dict__.items()
                if not k.startswith("_")
            },
        },
    }

    # Optionally persist
    if config.out_path is not None:
        try:
            from safepress.utils.logging import save_json

            save_json(config.out_path, result)
            logger.info("Comprehensive evaluation saved to %s", config.out_path)
        except Exception as exc:
            logger.warning("Failed to save results to %s: %s", config.out_path, exc)

    return result


# ---------------------------------------------------------------------------
# Multi-model comparison
# ---------------------------------------------------------------------------

def compare_models(results: Dict[str, Dict[str, Any]]) -> "pd.DataFrame":
    """
    Build a comparison table from multiple model evaluation results.

    Args:
        results: Mapping from model name to the dict returned by
            :func:`run_comprehensive_eval`.

    Returns:
        A :class:`pandas.DataFrame` with one row per model and columns for each
        metric.  Suitable for inclusion in paper tables.

    Example::

        res = {
            "llama2-7b-fp16": run_comprehensive_eval(m1, t1, cfg),
            "llama2-7b-nf4": run_comprehensive_eval(m2, t2, cfg),
            "llama2-7b-safepress": run_comprehensive_eval(m3, t3, cfg),
        }
        df = compare_models(res)
        print(df.to_latex(index=True))
    """
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError(
            "pandas is required for compare_models. Install with `pip install pandas`."
        ) from exc

    rows: List[Dict[str, Any]] = []

    for model_name, res in results.items():
        row: Dict[str, Any] = {"model": model_name}

        safety = res.get("safety", {})
        utility = res.get("utility", {})

        # --- Safety metrics ---
        refusal = safety.get("refusal", {})
        if "refusal_rate" in refusal:
            row["refusal_rate"] = refusal["refusal_rate"]

        strongreject = safety.get("strongreject", {})
        if strongreject.get("ok") and "avg_score" in strongreject:
            row["strongreject_score"] = strongreject["avg_score"]

        harmbench = safety.get("harmbench", {})
        if "asr" in harmbench:
            row["harmbench_asr"] = harmbench["asr"]

        harmbench_file = safety.get("harmbench_file", {})
        if "asr" in harmbench_file:
            row["harmbench_file_asr"] = harmbench_file["asr"]

        # --- Utility metrics ---
        ppl = utility.get("perplexity", {})
        if "perplexity" in ppl:
            row["perplexity"] = ppl["perplexity"]

        mmlu = utility.get("mmlu", {})
        if "accuracy" in mmlu:
            row["mmlu_accuracy"] = mmlu["accuracy"]

        tqa = utility.get("truthfulqa", {})
        if "accuracy" in tqa:
            row["truthfulqa_accuracy"] = tqa["accuracy"]

        # --- Metadata ---
        meta = res.get("metadata", {})
        if "elapsed_seconds" in meta:
            row["elapsed_seconds"] = meta["elapsed_seconds"]

        rows.append(row)

    df = pd.DataFrame(rows)
    if "model" in df.columns:
        df = df.set_index("model")

    return df
