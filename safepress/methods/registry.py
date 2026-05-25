"""
Unified method registry for SafePress.

Each method is described by a ``MethodSpec`` and dispatched through
``build_protect_plan()`` which returns a ``ProtectPlan``.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import pandas as pd

from safepress.model.blocks import compute_block_metadata, guess_layer_index
from safepress.model.protect import ProtectPlan, select_top_blocks
from safepress.model.score import compute_block_scores


@dataclass
class MethodSpec:
    """A runnable method/baseline specification."""

    name: str
    needs_scoring: bool = False
    needs_utility_prompts: bool = False


def list_methods() -> List[MethodSpec]:
    """Return all available method specs."""
    return [
        MethodSpec("fp16", needs_scoring=False),
        MethodSpec("int4", needs_scoring=False),
        MethodSpec("ssmp", needs_scoring=True),
        MethodSpec("random", needs_scoring=False),
        MethodSpec("magnitude", needs_scoring=False),
        MethodSpec("gradient_only", needs_scoring=True),
        MethodSpec("snip", needs_scoring=True),
        MethodSpec("wanda", needs_scoring=True),
        MethodSpec("lastn", needs_scoring=False),
        MethodSpec("layer_uniform", needs_scoring=False),
        MethodSpec("qresafe_noft", needs_scoring=True),
        # qresafe (proper) runs SNIP scoring + LoRA-DPO patch on the safety-
        # critical modules; the registry only produces the ProtectPlan (empty,
        # because Q-resafe quantizes the whole model). The DPO patch itself is
        # applied by ``safepress.methods.qresafe.qresafe_patch`` post-quantize.
        MethodSpec("qresafe", needs_scoring=True),
        MethodSpec("cwp", needs_scoring=True, needs_utility_prompts=True),
        # CWP at the published 60% budget with the I_safe - beta * I_gen
        # combined score using Dolly as the general-capability calibration set.
        # Dispatched by `cwp_published` which forces budget_ratio=0.60 inside.
        MethodSpec("cwp_published", needs_scoring=True, needs_utility_prompts=True),
    ]


# -------------------------------------------------------------------
# Internal helpers for non-scoring methods
# -------------------------------------------------------------------

def _plan_empty(*, block_size: int) -> ProtectPlan:
    """Empty plan for fp16 (no quant) or int4 (full quant) baselines."""
    return ProtectPlan(
        block_size=block_size,
        budget_ratio=0.0,
        total_params=0,
        protected_params=0,
        protect_map={},
    )


def _plan_from_random(
    meta: pd.DataFrame,
    *,
    budget_ratio: float,
    block_size: int,
    seed: int = 0,
) -> ProtectPlan:
    """Random block selection under a budget."""
    if not (0.0 < budget_ratio < 1.0):
        raise ValueError("budget_ratio must be in (0,1)")
    total_params = int(meta["num_params"].sum())
    budget_params = int(total_params * float(budget_ratio))
    rng = random.Random(seed)

    indices = list(range(len(meta)))
    rng.shuffle(indices)
    protect_map: Dict[str, List[int]] = {}
    protected = 0
    for idx in indices:
        row = meta.iloc[idx]
        mod = str(row["module"])
        b = int(row["block_idx"])
        n = int(row["num_params"])
        if protected + n > budget_params:
            continue
        protect_map.setdefault(mod, []).append(b)
        protected += n
        if protected >= budget_params:
            break

    for mod in list(protect_map.keys()):
        protect_map[mod] = sorted(set(int(x) for x in protect_map[mod]))

    return ProtectPlan(
        block_size=block_size,
        budget_ratio=float(budget_ratio),
        total_params=total_params,
        protected_params=protected,
        protect_map=protect_map,
    )


def _plan_lastn_layers(
    meta: pd.DataFrame,
    *,
    budget_ratio: float,
    block_size: int,
    last_n: int,
) -> ProtectPlan:
    """Protect blocks in the last N transformer layers."""
    if not (0.0 < budget_ratio < 1.0):
        raise ValueError("budget_ratio must be in (0,1)")
    total_params = int(meta["num_params"].sum())
    budget_params = int(total_params * float(budget_ratio))

    df = meta.copy()
    df["layer"] = [guess_layer_index(str(m)) for m in df["module"]]
    df["layer"] = df["layer"].fillna(-1).astype(int)
    max_layer = int(df["layer"].max())
    cutoff = max_layer - int(last_n) + 1
    df = df.sort_values(["layer", "module", "block_idx"], ascending=[False, True, True])

    protect_map: Dict[str, List[int]] = {}
    protected = 0
    for _, row in df.iterrows():
        if int(row["layer"]) < cutoff:
            continue
        mod = str(row["module"])
        b = int(row["block_idx"])
        n = int(row["num_params"])
        if protected + n > budget_params:
            continue
        protect_map.setdefault(mod, []).append(b)
        protected += n
        if protected >= budget_params:
            break

    for mod in list(protect_map.keys()):
        protect_map[mod] = sorted(set(int(x) for x in protect_map[mod]))

    return ProtectPlan(
        block_size=block_size,
        budget_ratio=float(budget_ratio),
        total_params=total_params,
        protected_params=protected,
        protect_map=protect_map,
    )


def _combine_cwp(
    safety_scores: pd.DataFrame,
    utility_scores: pd.DataFrame,
    *,
    beta: float,
) -> pd.DataFrame:
    """Combine safety and utility scores: score = safety - beta * utility."""
    req = {"module", "block_idx", "num_params", "score"}
    for df, name in [(safety_scores, "safety"), (utility_scores, "utility")]:
        missing = req - set(df.columns)
        if missing:
            raise ValueError(f"{name} scores missing columns: {missing}")

    merged = pd.merge(
        safety_scores,
        utility_scores[["module", "block_idx", "score"]].rename(columns={"score": "utility_score"}),
        on=["module", "block_idx"],
        how="inner",
    )
    merged["score"] = merged["score"].astype(float) - float(beta) * merged["utility_score"].astype(float)
    merged = merged.drop(columns=["utility_score"]).sort_values("score", ascending=False).reset_index(drop=True)
    return merged


# -------------------------------------------------------------------
# Main dispatcher
# -------------------------------------------------------------------

def build_protect_plan(
    *,
    method: str,
    model,
    tokenizer,
    safety_prompts: Sequence[str] | None = None,
    utility_prompts: Sequence[str] | None = None,
    block_size: int = 64,
    budget_ratio: float = 0.02,
    group_size: int = 128,
    bits: int = 4,
    refusal_template: str = "I'm sorry, but I can't help with that.",
    max_length: int = 2048,
    batch_size: int = 1,
    seed: int = 0,
    last_n_layers: int = 4,
    cwp_beta: float = 1.0,
    device=None,
) -> ProtectPlan:
    """Build a ProtectPlan for a given method name.

    Methods
    -------
    fp16 : No quantization (empty plan).
    int4 : Full quantization (empty plan).
    ssmp : Taylor drift proxy (metric=taylor_abs, prompt_mode=refusal).
    random : Random block selection.
    magnitude : Weight magnitude heuristic.
    gradient_only : Diagonal Fisher (grad^2) on refusal supervision.
    snip : SNIP connection sensitivity |w*g| on refusal supervision.
    wanda : Wanda |w|*||X||_2 activation-aware scoring.
    lastn : Protect last N transformer layers.
    layer_uniform : Protect first/last layers.
    qresafe_noft : Q-resafe without finetuning (SNIP on refusal).
    cwp : Critical Weight Protection (safety - beta * utility).
    """
    method = str(method).lower()

    # --- No-op baselines ---
    if method in {"fp16", "int4"}:
        return _plan_empty(block_size=block_size)

    # --- Random baseline ---
    if method == "random":
        meta = compute_block_metadata(model, block_size=block_size)
        return _plan_from_random(meta, budget_ratio=budget_ratio, block_size=block_size, seed=seed)

    # --- Last-N layers baseline ---
    if method == "lastn":
        meta = compute_block_metadata(model, block_size=block_size)
        return _plan_lastn_layers(meta, budget_ratio=budget_ratio, block_size=block_size, last_n=last_n_layers)

    # --- Magnitude baseline (no prompts needed) ---
    if method == "magnitude":
        from safepress.model.baselines import score_magnitude
        scores = score_magnitude(model, block_size=block_size)
        return select_top_blocks(scores, budget_ratio=budget_ratio, block_size=block_size)

    # --- Layer-uniform baseline (no prompts needed) ---
    if method == "layer_uniform":
        from safepress.model.baselines import score_layer_uniform
        scores = score_layer_uniform(model, block_size=block_size)
        return select_top_blocks(scores, budget_ratio=budget_ratio, block_size=block_size)

    # --- Wanda (needs prompts, forward hooks) ---
    if method == "wanda":
        if safety_prompts is None:
            raise ValueError("wanda requires safety_prompts")
        from safepress.model.baselines import score_wanda
        scores = score_wanda(
            model, tokenizer, list(safety_prompts),
            refusal_template=refusal_template,
            block_size=block_size,
            max_length=max_length,
            batch_size=batch_size,
            device=device,
        )
        return select_top_blocks(scores, budget_ratio=budget_ratio, block_size=block_size)

    # --- Methods using the unified compute_block_scores ---
    if method == "ssmp":
        if safety_prompts is None:
            raise ValueError("ssmp requires safety_prompts")
        scores = compute_block_scores(
            model, tokenizer, list(safety_prompts),
            refusal_template=refusal_template,
            metric="taylor_abs",
            prompt_mode="refusal",
            bits=bits, group_size=group_size, block_size=block_size,
            max_length=max_length, batch_size=batch_size, device=device,
        )
        return select_top_blocks(scores, budget_ratio=budget_ratio, block_size=block_size)

    if method == "gradient_only":
        if safety_prompts is None:
            raise ValueError("gradient_only requires safety_prompts")
        scores = compute_block_scores(
            model, tokenizer, list(safety_prompts),
            refusal_template=refusal_template,
            metric="grad_sq",
            prompt_mode="refusal",
            bits=bits, group_size=group_size, block_size=block_size,
            max_length=max_length, batch_size=batch_size, device=device,
        )
        return select_top_blocks(scores, budget_ratio=budget_ratio, block_size=block_size)

    if method == "snip":
        if safety_prompts is None:
            raise ValueError("snip requires safety_prompts")
        scores = compute_block_scores(
            model, tokenizer, list(safety_prompts),
            refusal_template=refusal_template,
            metric="snip",
            prompt_mode="refusal",
            bits=bits, group_size=group_size, block_size=block_size,
            max_length=max_length, batch_size=batch_size, device=device,
        )
        return select_top_blocks(scores, budget_ratio=budget_ratio, block_size=block_size)

    if method == "qresafe_noft":
        if safety_prompts is None:
            raise ValueError("qresafe_noft requires safety_prompts")
        scores = compute_block_scores(
            model, tokenizer, list(safety_prompts),
            refusal_template=refusal_template,
            metric="snip",
            prompt_mode="refusal",
            bits=bits, group_size=group_size, block_size=block_size,
            max_length=max_length, batch_size=batch_size, device=device,
        )
        return select_top_blocks(scores, budget_ratio=budget_ratio, block_size=block_size)

    if method == "qresafe":
        # Proper Q-resafe quantizes the whole model (no FP16 protection),
        # then re-aligns via LoRA-DPO on the safety-critical submodules. The
        # ProtectPlan here is empty -- the safety patch is applied by the
        # caller via ``safepress.methods.qresafe.qresafe_patch`` after the
        # base quantization.
        return _plan_empty(block_size=block_size)

    if method == "cwp":
        if safety_prompts is None or utility_prompts is None:
            raise ValueError("cwp requires both safety_prompts and utility_prompts")
        s_scores = compute_block_scores(
            model, tokenizer, list(safety_prompts),
            refusal_template=refusal_template,
            metric="grad_sq",
            prompt_mode="refusal",
            bits=bits, group_size=group_size, block_size=block_size,
            max_length=max_length, batch_size=batch_size, device=device,
        )
        # IMPORTANT: free per-weight gradient tensors stored by the first
        # backward pass. Without this, the second compute_block_scores call
        # OOMs because the prior grads occupy ~16 GB on an 8 B FP16 model.
        # (Real bug observed 2026-05-19 on cuda:1 with 48 GB A6000.)
        model.zero_grad(set_to_none=True)
        import gc as _gc
        _gc.collect()
        try:
            import torch as _torch
            if _torch.cuda.is_available():
                _torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass
        u_scores = compute_block_scores(
            model, tokenizer, list(utility_prompts),
            metric="grad_sq",
            prompt_mode="lm",
            bits=bits, group_size=group_size, block_size=block_size,
            max_length=max_length, batch_size=batch_size, device=device,
        )
        model.zero_grad(set_to_none=True)
        combined = _combine_cwp(s_scores, u_scores, beta=cwp_beta)
        return select_top_blocks(combined, budget_ratio=budget_ratio, block_size=block_size)

    if method == "cwp_published":
        # Critical Weight Protection (Yoo et al., arXiv 2601.12033) at the
        # published 60% protection budget. The combined score is
        # I_safe(theta) - beta * I_gen(theta) using squared-gradient Fisher,
        # with Dolly as the general-capability calibration set when
        # ``utility_prompts`` is supplied by the caller.
        if safety_prompts is None or utility_prompts is None:
            raise ValueError(
                "cwp_published requires both safety_prompts (AdvBench) and "
                "utility_prompts (Dolly is the default in the paper)."
            )
        s_scores = compute_block_scores(
            model, tokenizer, list(safety_prompts),
            refusal_template=refusal_template,
            metric="grad_sq",
            prompt_mode="refusal",
            bits=bits, group_size=group_size, block_size=block_size,
            max_length=max_length, batch_size=batch_size, device=device,
        )
        # Same OOM fix as the small-budget cwp branch above: free per-weight
        # gradients before the second compute_block_scores call.
        model.zero_grad(set_to_none=True)
        import gc as _gc
        _gc.collect()
        try:
            import torch as _torch
            if _torch.cuda.is_available():
                _torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass
        u_scores = compute_block_scores(
            model, tokenizer, list(utility_prompts),
            metric="grad_sq",
            prompt_mode="lm",
            bits=bits, group_size=group_size, block_size=block_size,
            max_length=max_length, batch_size=batch_size, device=device,
        )
        model.zero_grad(set_to_none=True)
        combined = _combine_cwp(s_scores, u_scores, beta=cwp_beta)
        # Override budget to the published 60% if caller passed something else.
        published_budget = 0.60
        return select_top_blocks(combined, budget_ratio=published_budget, block_size=block_size)

    available = [m.name for m in list_methods()]
    raise ValueError(f"Unknown method: {method}. Available: {available}")
