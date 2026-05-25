"""Budget and cross-model sweep experiments."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import torch
from tqdm import tqdm

from safepress.eval.basic import (
    GenConfig,
    generate_completions,
    refusal_rate,
    try_strongreject_eval,
)
from safepress.model.load import load_fp_model
from safepress.model.protect import select_top_blocks
from safepress.model.quantize import quantize_bnb4
from safepress.model.score import compute_block_scores
from safepress.model.split_linear import apply_block_splitting
from safepress.utils.io import read_prompts_jsonl
from safepress.utils.logging import save_json


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class SweepResult:
    """Container for a parameter sweep result."""
    experiment: str
    model_id: str
    sweep_param: str
    results: Dict[str, Dict[str, Any]]
    summary: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _evaluate_model(
    model: torch.nn.Module,
    tokenizer,
    prompts: List[str],
    *,
    device: torch.device | str | None = None,
) -> Dict[str, Any]:
    """Generate completions and return refusal statistics."""
    gen = GenConfig()
    responses = generate_completions(
        model, tokenizer, prompts, gen=gen, device=device, show_progress=True,
    )
    rr = refusal_rate(responses)
    sr = try_strongreject_eval(prompts, responses)
    return {
        "refusal_rate": rr,
        "n": len(prompts),
        "strongreject": sr,
    }


def _apply_ssmp_and_quantize(
    model: torch.nn.Module,
    plan,
    block_size: int,
) -> None:
    """Apply block splitting + bnb4 quantization in-place."""
    split_report = apply_block_splitting(model, plan.protect_map, block_size=block_size)
    quantize_bnb4(
        model,
        modules_to_not_convert=split_report.protected_modules_to_skip,
    )


def _quantize_full(model: torch.nn.Module) -> None:
    """Quantize every Linear layer (no protection) in-place."""
    quantize_bnb4(model, modules_to_not_convert=[])


# ---------------------------------------------------------------------------
# 1. Budget Sweep
# ---------------------------------------------------------------------------

def budget_sweep(
    model_id: str,
    scores_csv: str | Path,
    eval_prompts: List[str],
    *,
    budgets: Optional[List[float]] = None,
    block_size: int = 64,
    out_dir: str | Path,
    device_map: str = "auto",
    dtype: str = "float16",
) -> SweepResult:
    """
    Sweep the FP16 budget ratio and measure safety at each level.

    For each budget in *budgets*, build an SSMP model and evaluate its
    refusal rate.  Also evaluates FP16 baseline and full-quant baseline
    as anchors.

    Parameters
    ----------
    model_id : str
        HuggingFace model ID or local path.
    scores_csv : str | Path
        Path to the block-scores CSV produced by ``safepress score``.
    eval_prompts : list[str]
        Harmful prompts for refusal-rate evaluation.
    budgets : list[float], optional
        Budget ratios to sweep. Defaults to ``[0.005, 0.01, 0.02, 0.04, 0.08]``.
    block_size : int
        Block granularity for splitting.
    out_dir : str | Path
        Directory to write results.
    device_map, dtype
        Forwarded to :func:`load_fp_model`.

    Returns
    -------
    SweepResult
    """
    if budgets is None:
        budgets = [0.005, 0.01, 0.02, 0.04, 0.08]

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scores = pd.read_csv(scores_csv)

    results: Dict[str, Dict[str, Any]] = {}

    # -- FP16 baseline -----------------------------------------------------
    print("[budget_sweep] Evaluating FP16 baseline ...")
    loaded = load_fp_model(model_id, dtype=dtype, device_map=device_map)
    results["fp16_baseline"] = _evaluate_model(
        loaded.model, loaded.tokenizer, eval_prompts, device=loaded.device,
    )
    del loaded
    torch.cuda.empty_cache()

    # -- Full quant baseline -----------------------------------------------
    print("[budget_sweep] Evaluating full 4-bit quantization ...")
    loaded = load_fp_model(model_id, dtype=dtype, device_map=device_map)
    _quantize_full(loaded.model)
    results["full_quant"] = _evaluate_model(
        loaded.model, loaded.tokenizer, eval_prompts, device=loaded.device,
    )
    del loaded
    torch.cuda.empty_cache()

    # -- Each budget level -------------------------------------------------
    for b in tqdm(budgets, desc="Budget sweep"):
        label = f"budget_{b:.4f}"
        print(f"[budget_sweep] Evaluating budget={b:.4f} ...")
        loaded = load_fp_model(model_id, dtype=dtype, device_map=device_map)
        plan = select_top_blocks(scores, budget_ratio=b, block_size=block_size)
        _apply_ssmp_and_quantize(loaded.model, plan, block_size)
        eval_out = _evaluate_model(
            loaded.model, loaded.tokenizer, eval_prompts, device=loaded.device,
        )
        eval_out["budget"] = b
        eval_out["protected_params"] = plan.protected_params
        eval_out["total_params"] = plan.total_params
        results[label] = eval_out
        del loaded
        torch.cuda.empty_cache()

    # -- Save results ------------------------------------------------------
    summary_parts = [f"FP16={results['fp16_baseline']['refusal_rate']:.3f}"]
    summary_parts.append(f"FullQ={results['full_quant']['refusal_rate']:.3f}")
    for b in budgets:
        label = f"budget_{b:.4f}"
        summary_parts.append(f"b={b:.3f}:{results[label]['refusal_rate']:.3f}")

    result = SweepResult(
        experiment="budget_sweep",
        model_id=model_id,
        sweep_param="budget",
        results=results,
        summary=", ".join(summary_parts),
    )
    save_json(out_dir / "budget_sweep_results.json", result.to_dict())
    print(f"[budget_sweep] Done. {result.summary}")
    return result


# ---------------------------------------------------------------------------
# 2. Cross-Model Sweep
# ---------------------------------------------------------------------------

def cross_model_sweep(
    model_ids: List[str],
    eval_prompts_path: str | Path,
    calib_prompts_path: str | Path,
    *,
    budget: float = 0.02,
    block_size: int = 64,
    out_dir: str | Path,
    device_map: str = "auto",
    dtype: str = "float16",
) -> SweepResult:
    """
    Run the full SSMP pipeline on multiple models and compare.

    For each model: compute scores -> build SSMP -> evaluate.
    Also evaluates FP16 baseline and full-quant baseline per model.

    Parameters
    ----------
    model_ids : list[str]
        HuggingFace model IDs or local paths.
    eval_prompts_path : str | Path
        JSONL file with evaluation prompts.
    calib_prompts_path : str | Path
        JSONL file with calibration prompts (for scoring).
    budget : float
        FP16 budget ratio.
    block_size : int
        Block granularity.
    out_dir : str | Path
        Directory to write results.

    Returns
    -------
    SweepResult
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    eval_prompts = read_prompts_jsonl(eval_prompts_path)
    calib_prompts = read_prompts_jsonl(calib_prompts_path)

    results: Dict[str, Dict[str, Any]] = {}

    for mid in tqdm(model_ids, desc="Cross-model sweep"):
        model_key = mid.replace("/", "_")
        model_results: Dict[str, Any] = {"model_id": mid}

        # -- FP16 baseline -------------------------------------------------
        print(f"[cross_model] {mid}: FP16 baseline ...")
        loaded = load_fp_model(mid, dtype=dtype, device_map=device_map)
        model_results["fp16"] = _evaluate_model(
            loaded.model, loaded.tokenizer, eval_prompts, device=loaded.device,
        )
        del loaded
        torch.cuda.empty_cache()

        # -- Full quant ----------------------------------------------------
        print(f"[cross_model] {mid}: Full quant ...")
        loaded = load_fp_model(mid, dtype=dtype, device_map=device_map)
        _quantize_full(loaded.model)
        model_results["full_quant"] = _evaluate_model(
            loaded.model, loaded.tokenizer, eval_prompts, device=loaded.device,
        )
        del loaded
        torch.cuda.empty_cache()

        # -- Scoring -------------------------------------------------------
        print(f"[cross_model] {mid}: Computing scores ...")
        loaded = load_fp_model(mid, dtype=dtype, device_map=device_map)
        scores = compute_block_scores(
            loaded.model,
            loaded.tokenizer,
            calib_prompts,
            block_size=block_size,
            device=loaded.device,
            show_progress=True,
        )
        scores_path = out_dir / f"{model_key}_scores.csv"
        scores.to_csv(scores_path, index=False)

        # -- SSMP ----------------------------------------------------------
        print(f"[cross_model] {mid}: SSMP (budget={budget}) ...")
        plan = select_top_blocks(scores, budget_ratio=budget, block_size=block_size)
        # Reload model fresh for clean quantization.
        del loaded
        torch.cuda.empty_cache()
        loaded = load_fp_model(mid, dtype=dtype, device_map=device_map)
        _apply_ssmp_and_quantize(loaded.model, plan, block_size)
        model_results["ssmp"] = _evaluate_model(
            loaded.model, loaded.tokenizer, eval_prompts, device=loaded.device,
        )
        model_results["ssmp"]["budget"] = budget
        model_results["ssmp"]["protected_params"] = plan.protected_params
        model_results["ssmp"]["total_params"] = plan.total_params

        del loaded
        torch.cuda.empty_cache()

        results[model_key] = model_results

    # -- Save results ------------------------------------------------------
    summary_parts = []
    for model_key, mr in results.items():
        summary_parts.append(
            f"{mr['model_id']}: FP16={mr['fp16']['refusal_rate']:.3f}, "
            f"FullQ={mr['full_quant']['refusal_rate']:.3f}, "
            f"SSMP={mr['ssmp']['refusal_rate']:.3f}"
        )

    result = SweepResult(
        experiment="cross_model_sweep",
        model_id=",".join(model_ids),
        sweep_param="model",
        results=results,
        summary=" | ".join(summary_parts),
    )
    save_json(out_dir / "cross_model_sweep_results.json", result.to_dict())
    print(f"[cross_model] Done. {result.summary}")
    return result


# ---------------------------------------------------------------------------
# 3. Block-Size Sweep
# ---------------------------------------------------------------------------

def block_size_sweep(
    model_id: str,
    calib_prompts: List[str],
    eval_prompts: List[str],
    *,
    block_sizes: Optional[List[int]] = None,
    budget: float = 0.02,
    out_dir: str | Path,
    device_map: str = "auto",
    dtype: str = "float16",
) -> SweepResult:
    """
    Sweep block granularity and measure how it affects safety preservation.

    For each block size: re-compute scores -> build SSMP -> evaluate.

    Parameters
    ----------
    model_id : str
        HuggingFace model ID or local path.
    calib_prompts : list[str]
        Calibration prompts for scoring.
    eval_prompts : list[str]
        Harmful prompts for refusal-rate evaluation.
    block_sizes : list[int], optional
        Block sizes to sweep.  Defaults to ``[32, 64, 128, 256]``.
    budget : float
        FP16 budget ratio (constant across the sweep).
    out_dir : str | Path
        Directory to write results.

    Returns
    -------
    SweepResult
    """
    if block_sizes is None:
        block_sizes = [32, 64, 128, 256]

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results: Dict[str, Dict[str, Any]] = {}

    # -- FP16 baseline (block-size independent) ----------------------------
    print("[block_size_sweep] Evaluating FP16 baseline ...")
    loaded = load_fp_model(model_id, dtype=dtype, device_map=device_map)
    results["fp16_baseline"] = _evaluate_model(
        loaded.model, loaded.tokenizer, eval_prompts, device=loaded.device,
    )
    del loaded
    torch.cuda.empty_cache()

    # -- Sweep each block size ---------------------------------------------
    for bs in tqdm(block_sizes, desc="Block-size sweep"):
        label = f"block_size_{bs}"
        print(f"[block_size_sweep] Scoring with block_size={bs} ...")

        # Score
        loaded = load_fp_model(model_id, dtype=dtype, device_map=device_map)
        scores = compute_block_scores(
            loaded.model,
            loaded.tokenizer,
            calib_prompts,
            block_size=bs,
            device=loaded.device,
            show_progress=True,
        )
        scores_path = out_dir / f"scores_bs{bs}.csv"
        scores.to_csv(scores_path, index=False)

        # Build SSMP -- need a fresh model for clean quantization.
        del loaded
        torch.cuda.empty_cache()
        loaded = load_fp_model(model_id, dtype=dtype, device_map=device_map)
        plan = select_top_blocks(scores, budget_ratio=budget, block_size=bs)
        _apply_ssmp_and_quantize(loaded.model, plan, bs)

        # Evaluate
        eval_out = _evaluate_model(
            loaded.model, loaded.tokenizer, eval_prompts, device=loaded.device,
        )
        eval_out["block_size"] = bs
        eval_out["protected_params"] = plan.protected_params
        eval_out["total_params"] = plan.total_params
        eval_out["num_blocks"] = len(scores)
        results[label] = eval_out

        del loaded
        torch.cuda.empty_cache()

    # -- Save results ------------------------------------------------------
    summary_parts = [f"FP16={results['fp16_baseline']['refusal_rate']:.3f}"]
    for bs in block_sizes:
        label = f"block_size_{bs}"
        summary_parts.append(f"bs={bs}:{results[label]['refusal_rate']:.3f}")

    result = SweepResult(
        experiment="block_size_sweep",
        model_id=model_id,
        sweep_param="block_size",
        results=results,
        summary=", ".join(summary_parts),
    )
    save_json(out_dir / "block_size_sweep_results.json", result.to_dict())
    print(f"[block_size_sweep] Done. {result.summary}")
    return result
