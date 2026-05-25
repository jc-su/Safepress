"""
Causal experiments for validating safety-critical block identification.

These experiments establish causality (not just correlation) between
identified blocks and safety behavior:
1. Targeted Quantize: only quantize critical blocks vs only non-critical
2. Precision Rollback: in a fully quantized model, restore critical blocks to FP16
3. Random Control: compare SSMP selection vs random/heuristic baselines
"""
from __future__ import annotations

import copy
import json
import random
from dataclasses import asdict, dataclass, field
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
from safepress.model.protect import ProtectPlan, select_top_blocks
from safepress.model.quantize import quantize_bnb4
from safepress.model.split_linear import apply_block_splitting
from safepress.utils.logging import save_json


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class ExperimentResult:
    """Container for any causal experiment result."""
    experiment: str
    model_id: str
    conditions: Dict[str, Dict[str, Any]]
    summary: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_scores(scores_csv: str | Path) -> pd.DataFrame:
    """Load block scores from a CSV file."""
    return pd.read_csv(scores_csv)


def _build_inverted_plan(
    scores: pd.DataFrame,
    *,
    budget_ratio: float,
    block_size: int,
) -> ProtectPlan:
    """
    Build a 'protect plan' that protects the *lowest* scoring blocks instead
    of the highest -- i.e., invert the normal SSMP selection so that only
    the least safety-critical blocks are kept in FP16.
    """
    inverted = scores.copy()
    # Flip scores so the lowest-scoring blocks sort to the top.
    inverted["score"] = -inverted["score"]
    return select_top_blocks(inverted, budget_ratio=budget_ratio, block_size=block_size)


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
    plan: ProtectPlan,
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
# 1. Targeted Quantize Experiment
# ---------------------------------------------------------------------------

def targeted_quantize_experiment(
    model_id: str,
    scores_csv: str | Path,
    eval_prompts: List[str],
    *,
    budget: float = 0.02,
    block_size: int = 64,
    bits: int = 4,
    group_size: int = 128,
    out_dir: str | Path,
    device_map: str = "auto",
    dtype: str = "float16",
) -> ExperimentResult:
    """
    Targeted quantize experiment.

    Creates four model variants and measures refusal rate for each:
      - fp16_baseline:          Original FP16 model (no quantization).
      - full_quant:             All linear blocks quantized to 4-bit.
      - critical_only_quant:    Only the TOP scoring (safety-critical) blocks
                                are quantized; the rest remain FP16.
      - noncritical_only_quant: Only the BOTTOM scoring blocks are quantized;
                                the critical blocks remain FP16 (normal SSMP).

    The hypothesis: ``critical_only_quant`` should show much worse safety
    (lower refusal rate) than ``noncritical_only_quant``, proving that the
    identified blocks are genuinely safety-critical.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scores = _load_scores(scores_csv)

    conditions: Dict[str, Dict[str, Any]] = {}

    # -- Condition 1: FP16 baseline ----------------------------------------
    print("[targeted_quantize] Evaluating FP16 baseline ...")
    loaded = load_fp_model(model_id, dtype=dtype, device_map=device_map)
    conditions["fp16_baseline"] = _evaluate_model(
        loaded.model, loaded.tokenizer, eval_prompts, device=loaded.device,
    )
    del loaded
    torch.cuda.empty_cache()

    # -- Condition 2: Full 4-bit (no protection) ---------------------------
    print("[targeted_quantize] Evaluating full 4-bit quantization ...")
    loaded = load_fp_model(model_id, dtype=dtype, device_map=device_map)
    _quantize_full(loaded.model)
    conditions["full_quant"] = _evaluate_model(
        loaded.model, loaded.tokenizer, eval_prompts, device=loaded.device,
    )
    del loaded
    torch.cuda.empty_cache()

    # -- Condition 3: Critical-only quantize -------------------------------
    # Protect the BOTTOM blocks (non-critical) in FP16, quantize the TOP
    # (critical) blocks.  This is the *inverse* of SSMP.
    print("[targeted_quantize] Evaluating critical-only quantization ...")
    loaded = load_fp_model(model_id, dtype=dtype, device_map=device_map)
    inverted_plan = _build_inverted_plan(scores, budget_ratio=1.0 - budget, block_size=block_size)
    _apply_ssmp_and_quantize(loaded.model, inverted_plan, block_size)
    conditions["critical_only_quant"] = _evaluate_model(
        loaded.model, loaded.tokenizer, eval_prompts, device=loaded.device,
    )
    del loaded
    torch.cuda.empty_cache()

    # -- Condition 4: Non-critical-only quantize (normal SSMP) -------------
    print("[targeted_quantize] Evaluating non-critical-only quantization (SSMP) ...")
    loaded = load_fp_model(model_id, dtype=dtype, device_map=device_map)
    ssmp_plan = select_top_blocks(scores, budget_ratio=budget, block_size=block_size)
    _apply_ssmp_and_quantize(loaded.model, ssmp_plan, block_size)
    conditions["noncritical_only_quant"] = _evaluate_model(
        loaded.model, loaded.tokenizer, eval_prompts, device=loaded.device,
    )
    del loaded
    torch.cuda.empty_cache()

    # -- Save results ------------------------------------------------------
    result = ExperimentResult(
        experiment="targeted_quantize",
        model_id=model_id,
        conditions=conditions,
        summary=(
            f"FP16={conditions['fp16_baseline']['refusal_rate']:.3f}, "
            f"FullQ={conditions['full_quant']['refusal_rate']:.3f}, "
            f"CritOnlyQ={conditions['critical_only_quant']['refusal_rate']:.3f}, "
            f"NonCritOnlyQ={conditions['noncritical_only_quant']['refusal_rate']:.3f}"
        ),
    )
    save_json(out_dir / "targeted_quantize_results.json", result.to_dict())
    print(f"[targeted_quantize] Done. {result.summary}")
    return result


# ---------------------------------------------------------------------------
# 2. Precision Rollback Experiment
# ---------------------------------------------------------------------------

def precision_rollback_experiment(
    model_id: str,
    scores_csv: str | Path,
    eval_prompts: List[str],
    *,
    budget: float = 0.02,
    block_size: int = 64,
    out_dir: str | Path,
    device_map: str = "auto",
    dtype: str = "float16",
) -> ExperimentResult:
    """
    Precision rollback experiment.

    Start with a *fully quantized* model (all 4-bit, safety broken).
    Then selectively restore the top-K blocks to FP16 using scores.
    Compare with restoring random blocks and bottom-K blocks.

    Conditions:
      - full_quant:        Fully quantized baseline.
      - rollback_top_k:    Restore the highest-scoring blocks to FP16.
      - rollback_random_k: Restore a random set of blocks to FP16.
      - rollback_bottom_k: Restore the lowest-scoring blocks to FP16.
      - fp16_baseline:     Original FP16 model.

    The hypothesis: ``rollback_top_k`` should recover most of the safety
    (refusal rate increases back toward FP16 baseline), while random and
    bottom-k rollback should show little recovery.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scores = _load_scores(scores_csv)

    conditions: Dict[str, Dict[str, Any]] = {}

    # -- Condition 1: FP16 baseline ----------------------------------------
    print("[precision_rollback] Evaluating FP16 baseline ...")
    loaded = load_fp_model(model_id, dtype=dtype, device_map=device_map)
    conditions["fp16_baseline"] = _evaluate_model(
        loaded.model, loaded.tokenizer, eval_prompts, device=loaded.device,
    )
    del loaded
    torch.cuda.empty_cache()

    # -- Condition 2: Full quant (no rollback) -----------------------------
    print("[precision_rollback] Evaluating full 4-bit quantization ...")
    loaded = load_fp_model(model_id, dtype=dtype, device_map=device_map)
    _quantize_full(loaded.model)
    conditions["full_quant"] = _evaluate_model(
        loaded.model, loaded.tokenizer, eval_prompts, device=loaded.device,
    )
    del loaded
    torch.cuda.empty_cache()

    # -- Condition 3: Rollback top-K (SSMP selection) ----------------------
    # "Rollback" is achieved by protecting the top-K blocks *before*
    # quantization, which is equivalent to quantizing and then restoring
    # those blocks -- the end state is the same.
    print("[precision_rollback] Evaluating rollback top-K (SSMP) ...")
    loaded = load_fp_model(model_id, dtype=dtype, device_map=device_map)
    plan_top = select_top_blocks(scores, budget_ratio=budget, block_size=block_size)
    _apply_ssmp_and_quantize(loaded.model, plan_top, block_size)
    conditions["rollback_top_k"] = _evaluate_model(
        loaded.model, loaded.tokenizer, eval_prompts, device=loaded.device,
    )
    del loaded
    torch.cuda.empty_cache()

    # -- Condition 4: Rollback random-K ------------------------------------
    print("[precision_rollback] Evaluating rollback random-K ...")
    loaded = load_fp_model(model_id, dtype=dtype, device_map=device_map)
    shuffled = scores.copy()
    shuffled["score"] = random.sample(range(len(shuffled)), len(shuffled))
    plan_random = select_top_blocks(shuffled, budget_ratio=budget, block_size=block_size)
    _apply_ssmp_and_quantize(loaded.model, plan_random, block_size)
    conditions["rollback_random_k"] = _evaluate_model(
        loaded.model, loaded.tokenizer, eval_prompts, device=loaded.device,
    )
    del loaded
    torch.cuda.empty_cache()

    # -- Condition 5: Rollback bottom-K ------------------------------------
    print("[precision_rollback] Evaluating rollback bottom-K ...")
    loaded = load_fp_model(model_id, dtype=dtype, device_map=device_map)
    plan_bottom = _build_inverted_plan(scores, budget_ratio=budget, block_size=block_size)
    _apply_ssmp_and_quantize(loaded.model, plan_bottom, block_size)
    conditions["rollback_bottom_k"] = _evaluate_model(
        loaded.model, loaded.tokenizer, eval_prompts, device=loaded.device,
    )
    del loaded
    torch.cuda.empty_cache()

    # -- Save results ------------------------------------------------------
    result = ExperimentResult(
        experiment="precision_rollback",
        model_id=model_id,
        conditions=conditions,
        summary=(
            f"FP16={conditions['fp16_baseline']['refusal_rate']:.3f}, "
            f"FullQ={conditions['full_quant']['refusal_rate']:.3f}, "
            f"TopK={conditions['rollback_top_k']['refusal_rate']:.3f}, "
            f"RandK={conditions['rollback_random_k']['refusal_rate']:.3f}, "
            f"BotK={conditions['rollback_bottom_k']['refusal_rate']:.3f}"
        ),
    )
    save_json(out_dir / "precision_rollback_results.json", result.to_dict())
    print(f"[precision_rollback] Done. {result.summary}")
    return result


# ---------------------------------------------------------------------------
# 3. Control Experiment (selection strategy comparison)
# ---------------------------------------------------------------------------

def _magnitude_scores(model: torch.nn.Module, block_size: int) -> pd.DataFrame:
    """
    Heuristic baseline: rank blocks by L2 magnitude of their weight slice.
    """
    from safepress.model.blocks import chunk_indices, iter_linear_modules

    rows: List[Dict[str, Any]] = []
    for mod_name, mod in iter_linear_modules(model):
        w = mod.weight.detach()
        out_features, in_features = w.shape
        blocks = chunk_indices(out_features, block_size)
        for b_idx, (s, e) in enumerate(blocks):
            w_blk = w[s:e, :]
            mag = w_blk.float().norm().item()
            rows.append(dict(
                module=mod_name,
                block_idx=b_idx,
                out_start=s,
                out_end=e,
                in_features=in_features,
                num_params=(e - s) * in_features,
                score=mag,
            ))
    return pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)


def _gradient_only_scores(
    model: torch.nn.Module,
    tokenizer,
    prompts: List[str],
    *,
    block_size: int,
    device: torch.device | str | None = None,
) -> pd.DataFrame:
    """
    Gradient-only baseline: score = sum |grad| per block (no quant-error term).
    """
    from safepress.model.blocks import chunk_indices, iter_linear_modules
    from safepress.model.score import build_refusal_supervision

    if device is None:
        try:
            device = next(p.device for p in model.parameters() if p.is_floating_point())
        except StopIteration:
            device = torch.device("cpu")
    device = torch.device(device)

    model.eval()
    for p in model.parameters():
        p.requires_grad_(True)

    linear_modules = list(iter_linear_modules(model))
    accum: Dict[str, float] = {}
    meta: Dict[str, Dict[str, Any]] = {}

    for mod_name, mod in linear_modules:
        out_features, in_features = mod.weight.shape
        blocks = chunk_indices(out_features, block_size)
        for b_idx, (s, e) in enumerate(blocks):
            key = f"{mod_name}:{b_idx}"
            accum[key] = 0.0
            meta[key] = dict(
                module=mod_name,
                block_idx=b_idx,
                out_start=s,
                out_end=e,
                in_features=in_features,
                num_params=(e - s) * in_features,
            )

    batch_inputs = build_refusal_supervision(
        tokenizer, prompts, refusal_template="I'm sorry, but I can't help with that.",
    )
    batch_inputs = {k: v.to(device) for k, v in batch_inputs.items()}

    model.zero_grad(set_to_none=True)
    out = model(**batch_inputs)
    loss = out.loss
    if loss is not None:
        loss.backward()

    with torch.no_grad():
        for mod_name, mod in linear_modules:
            g = mod.weight.grad
            if g is None:
                continue
            out_features, in_features = mod.weight.shape
            blocks = chunk_indices(out_features, block_size)
            for b_idx, (s, e) in enumerate(blocks):
                key = f"{mod_name}:{b_idx}"
                accum[key] += g[s:e, :].float().abs().sum().item()

    rows = []
    for key, score in accum.items():
        m = meta[key]
        m["score"] = score
        rows.append(m)
    return pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)


def _layer_uniform_scores(
    model: torch.nn.Module,
    *,
    block_size: int,
) -> pd.DataFrame:
    """
    Heuristic baseline: protect first + last transformer layers uniformly.

    Assigns high score to blocks in the first and last ~10% of layers,
    zero to everything else.
    """
    from safepress.model.blocks import chunk_indices, iter_linear_modules

    linear_modules = list(iter_linear_modules(model))
    # Extract layer indices from module names (e.g. "model.layers.0.self_attn.q_proj")
    layer_indices: Dict[str, Optional[int]] = {}
    all_layer_nums: List[int] = []
    for mod_name, _ in linear_modules:
        parts = mod_name.split(".")
        layer_num = None
        for part in parts:
            if part.isdigit():
                layer_num = int(part)
                break
        layer_indices[mod_name] = layer_num
        if layer_num is not None:
            all_layer_nums.append(layer_num)

    if not all_layer_nums:
        # Fallback: assign equal scores
        rows = []
        for mod_name, mod in linear_modules:
            out_features, in_features = mod.weight.shape
            blocks = chunk_indices(out_features, block_size)
            for b_idx, (s, e) in enumerate(blocks):
                rows.append(dict(
                    module=mod_name, block_idx=b_idx, out_start=s, out_end=e,
                    in_features=in_features, num_params=(e - s) * in_features,
                    score=1.0,
                ))
        return pd.DataFrame(rows)

    max_layer = max(all_layer_nums)
    boundary = max(1, int(max_layer * 0.1))

    rows: List[Dict[str, Any]] = []
    for mod_name, mod in linear_modules:
        out_features, in_features = mod.weight.shape
        blocks = chunk_indices(out_features, block_size)
        lnum = layer_indices[mod_name]
        if lnum is not None and (lnum <= boundary or lnum >= max_layer - boundary):
            score = 1.0
        else:
            score = 0.0
        for b_idx, (s, e) in enumerate(blocks):
            rows.append(dict(
                module=mod_name, block_idx=b_idx, out_start=s, out_end=e,
                in_features=in_features, num_params=(e - s) * in_features,
                score=score,
            ))
    return pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)


def control_experiment(
    model_id: str,
    eval_prompts: List[str],
    *,
    budget: float = 0.02,
    block_size: int = 64,
    out_dir: str | Path,
    baselines: Optional[List[str]] = None,
    device_map: str = "auto",
    dtype: str = "float16",
) -> ExperimentResult:
    """
    Control experiment: compare SSMP against alternative selection strategies.

    All strategies protect the same *budget* of parameters in FP16 and
    quantize the rest.  Only the selection criterion differs.

    Strategies:
      1. ssmp          -- drift-score ranking (our method).
      2. random        -- random block selection.
      3. magnitude     -- weight L2 norm ranking.
      4. gradient_only -- gradient magnitude without the quant-error term.
      5. layer_uniform -- protect first + last ~10 % of layers.

    The hypothesis: SSMP should achieve higher refusal rate than all
    baselines at the same budget, proving that *which* blocks are
    protected matters more than simply protecting *some* blocks.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_strategies = ["ssmp", "random", "magnitude", "gradient_only", "layer_uniform"]
    if baselines is not None:
        strategies = [s for s in baselines if s in all_strategies]
    else:
        strategies = all_strategies

    conditions: Dict[str, Dict[str, Any]] = {}

    # We need the model to compute various score variants.
    # First, compute the SSMP drift scores (requires forward+backward).
    print("[control] Loading model for scoring ...")
    loaded = load_fp_model(model_id, dtype=dtype, device_map=device_map)

    score_cache: Dict[str, pd.DataFrame] = {}

    # -- Compute scores that need the model --
    if "ssmp" in strategies:
        from safepress.model.score import compute_block_scores

        print("[control] Computing SSMP drift scores ...")
        score_cache["ssmp"] = compute_block_scores(
            loaded.model,
            loaded.tokenizer,
            eval_prompts,
            block_size=block_size,
            device=loaded.device,
            show_progress=True,
        )

    if "magnitude" in strategies:
        print("[control] Computing magnitude scores ...")
        score_cache["magnitude"] = _magnitude_scores(loaded.model, block_size)

    if "gradient_only" in strategies:
        print("[control] Computing gradient-only scores ...")
        score_cache["gradient_only"] = _gradient_only_scores(
            loaded.model, loaded.tokenizer, eval_prompts,
            block_size=block_size, device=loaded.device,
        )

    if "layer_uniform" in strategies:
        print("[control] Computing layer-uniform scores ...")
        score_cache["layer_uniform"] = _layer_uniform_scores(
            loaded.model, block_size=block_size,
        )

    if "random" in strategies:
        # Use any existing score df as template, randomize scores.
        template_key = next(iter(score_cache), None)
        if template_key is not None:
            rand_df = score_cache[template_key].copy()
        else:
            # Need to create one from model structure.
            rand_df = _magnitude_scores(loaded.model, block_size)
        rand_df["score"] = [random.random() for _ in range(len(rand_df))]
        score_cache["random"] = rand_df

    del loaded
    torch.cuda.empty_cache()

    # -- Evaluate each strategy -------------------------------------------
    for strategy in tqdm(strategies, desc="Control strategies"):
        print(f"[control] Evaluating strategy: {strategy} ...")
        loaded = load_fp_model(model_id, dtype=dtype, device_map=device_map)
        plan = select_top_blocks(
            score_cache[strategy], budget_ratio=budget, block_size=block_size,
        )
        _apply_ssmp_and_quantize(loaded.model, plan, block_size)
        conditions[strategy] = _evaluate_model(
            loaded.model, loaded.tokenizer, eval_prompts, device=loaded.device,
        )
        del loaded
        torch.cuda.empty_cache()

    # -- Save results ------------------------------------------------------
    summary_parts = [
        f"{s}={conditions[s]['refusal_rate']:.3f}" for s in strategies
    ]
    result = ExperimentResult(
        experiment="control",
        model_id=model_id,
        conditions=conditions,
        summary=", ".join(summary_parts),
    )
    save_json(out_dir / "control_results.json", result.to_dict())
    print(f"[control] Done. {result.summary}")
    return result
