"""
Theoretical / diagnostic utilities for quantization-induced safety drift.

Provides:

* Per-module Cauchy-Schwarz bound  ``|ΔL| ≤ ‖∇L‖₂ · ‖ΔW‖₂`` and the
  first-order drift inner product ``Σ ∇L · ΔW``.
* An empirical validation routine that pairs the predicted bound with the
  *measured* safety-loss change between the FP16 model and a quantized model.
  This produces the scatter data needed for the paper's theory figure
  (predicted-vs-measured drift, fit R²).

The measured drift is obtained by:

    ΔL_safe ≈ L_safe(W_quant) − L_safe(W)

where both losses use the same refusal-supervision batch as the gradient
computation, so that the linearization of the loss around W is a sensible
basis for the comparison.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd
import torch

from safepress.model.blocks import iter_linear_modules
from safepress.model.score import _quant_dequant_symmetric_groupwise, build_refusal_supervision

logger = logging.getLogger(__name__)


@dataclass
class DriftBoundRow:
    module: str
    grad_l2: float
    delta_l2: float
    cs_bound: float
    inner_prod: float


def compute_module_level_bounds(
    model,
    tokenizer,
    prompts: Sequence[str],
    *,
    refusal_template: str = "I'm sorry, but I can't help with that.",
    bits: int = 4,
    group_size: int = 128,
    max_length: int = 2048,
    batch_size: int = 2,
    device: torch.device | str | None = None,
) -> pd.DataFrame:
    """
    Compute per-linear-module Cauchy-Schwarz bounds and first-order drift estimates.

    Intended for the theory/analysis section of the paper.

    Returns a DataFrame with columns:
      module, grad_l2, delta_l2, cs_bound, inner_prod
    """
    if device is None:
        try:
            device = next(p.device for p in model.parameters() if p.is_floating_point())
        except StopIteration:
            device = torch.device("cpu")
    device = torch.device(device)

    model.zero_grad(set_to_none=True)

    batch = build_refusal_supervision(
        tokenizer,
        list(prompts),
        refusal_template=refusal_template,
        max_length=max_length,
    )
    batch = {k: v.to(device) for k, v in batch.items()}
    out = model(**batch)
    loss = out.loss
    loss.backward()

    rows: List[DriftBoundRow] = []
    for name, mod in iter_linear_modules(model):
        if mod.weight.grad is None:
            continue
        g = mod.weight.grad.detach()
        w = mod.weight.detach()
        w_hat = _quant_dequant_symmetric_groupwise(w, bits=bits, group_size=group_size)
        delta = (w_hat - w).detach()

        grad_l2 = float(torch.norm(g.float()).item())
        delta_l2 = float(torch.norm(delta.float()).item())
        inner = float(torch.sum(g.float() * delta.float()).item())
        bound = grad_l2 * delta_l2

        rows.append(
            DriftBoundRow(
                module=str(name),
                grad_l2=grad_l2,
                delta_l2=delta_l2,
                cs_bound=bound,
                inner_prod=inner,
            )
        )

    model.zero_grad(set_to_none=True)

    df = pd.DataFrame([r.__dict__ for r in rows]).sort_values("cs_bound", ascending=False)
    return df


# ---------------------------------------------------------------------------
# Empirical validation of the drift bound
# ---------------------------------------------------------------------------

@dataclass
class DriftValidationRow:
    """One measurement point in the predicted-vs-measured drift scatter.

    The theorem in PLAN §1 Pillar 2 bounds the *absolute* safety-loss drift by
    a sum of *per-block absolute* dot products: ``|ΔL| ≤ Σ_b |g_b · δw_b|``
    (triangle inequality). To validate the theorem we therefore need the
    absolute-block-sum and the signed inner-product side-by-side. We report
    both, plus the per-module Cauchy-Schwarz bound, so figures and fits can
    pick the relevant quantity.

    * ``predicted_inner_signed``    -- Σ g·δw at element granularity = first-
                                       order Taylor estimate (the exact signed
                                       drift, matches ΔL not |ΔL|).
    * ``predicted_abs_block``       -- Σ_b |g_b · δw_b| at SSMP block
                                       granularity (block_size output rows).
                                       **This is the theorem's upper bound.**
    * ``predicted_abs_module``      -- Σ_module |g_module · δw_module|,
                                       coarser; useful as a sanity comparison.
    * ``predicted_cs_module``       -- Σ_module ‖g_module‖₂ ‖δw_module‖₂ (CS,
                                       coarser still; matches the original
                                       diagnostic bound).
    * ``measured_dL_signed``        -- L1 - L0 (signed).
    * ``measured_dL_abs``           -- |L1 - L0| (for upper-bound R² fit).
    """

    label: str
    bits: int
    group_size: int
    block_size: int
    predicted_inner_signed: float
    predicted_abs_block: float
    predicted_abs_module: float
    predicted_cs_module: float
    measured_dL_signed: float
    measured_dL_abs: float
    relative_error_signed: float


def _safety_loss(
    model,
    tokenizer,
    prompts: Sequence[str],
    *,
    refusal_template: str,
    max_length: int,
    device,
) -> float:
    """Evaluate the cross-entropy safety loss on the same supervision batch
    used by :func:`compute_module_level_bounds`. Returns a Python float.
    """
    batch = build_refusal_supervision(
        tokenizer, list(prompts),
        refusal_template=refusal_template,
        max_length=max_length,
    )
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        out = model(**batch)
    return float(out.loss.item())


@torch.no_grad()
def _apply_quant_perturbation_in_place(
    model,
    *,
    bits: int,
    group_size: int,
) -> Dict[str, torch.Tensor]:
    """Replace every linear weight with its symmetric group-wise quantize-
    dequantize image, returning the original weights so they can be restored.

    This is a *theoretical-bound* measurement helper -- it produces the exact
    weight perturbation that the score module uses to predict the drift, with
    no model copy.
    """
    saved: Dict[str, torch.Tensor] = {}
    for name, mod in iter_linear_modules(model):
        w = mod.weight.detach()
        saved[name] = w.clone()
        w_hat = _quant_dequant_symmetric_groupwise(w, bits=bits, group_size=group_size)
        mod.weight.copy_(w_hat)
    return saved


@torch.no_grad()
def _restore_weights(model, saved: Dict[str, torch.Tensor]) -> None:
    for name, mod in iter_linear_modules(model):
        if name in saved:
            mod.weight.copy_(saved[name])


def validate_drift_bound(
    model,
    tokenizer,
    prompts: Sequence[str],
    *,
    bit_widths: Sequence[int] = (8, 4, 3, 2),
    group_size: int = 128,
    block_size: int = 64,
    refusal_template: str = "I'm sorry, but I can't help with that.",
    max_length: int = 2048,
    device: torch.device | str | None = None,
) -> pd.DataFrame:
    """Validate the Taylor drift proxy against the measured safety-loss change.

    For each bit-width in *bit_widths*, we:

    1.  Compute the FP16 safety loss ``L0`` and the per-module gradient ``g``.
    2.  Replace every linear weight in place with its quantize-dequantize
        image at the given bit-width to obtain ``W' = W + ΔW``.
    3.  Compute the quantized safety loss ``L1``.
    4.  Restore the original weights.
    5.  Compare ``L1 − L0`` (measured) against ``Σ g · ΔW`` (predicted Taylor)
        and ``Σ ‖g‖ · ‖ΔW‖`` (Cauchy-Schwarz upper bound).

    Returns one row per bit-width with columns:

    ``label, bits, group_size, predicted_inner, predicted_cs_bound,
    measured_dL, relative_error``.

    Notes
    -----
    Step (2) mutates ``model.weight`` in place but restores it before returning
    -- the model on disk and in subsequent calls is unchanged.
    """
    if device is None:
        try:
            device = next(p.device for p in model.parameters() if p.is_floating_point())
        except StopIteration:
            device = torch.device("cpu")
    device = torch.device(device)

    # Step 1: gradient on FP16
    model.zero_grad(set_to_none=True)
    batch = build_refusal_supervision(
        tokenizer, list(prompts),
        refusal_template=refusal_template,
        max_length=max_length,
    )
    batch = {k: v.to(device) for k, v in batch.items()}
    out = model(**batch)
    L0 = float(out.loss.item())
    out.loss.backward()

    grads: Dict[str, torch.Tensor] = {}
    for name, mod in iter_linear_modules(model):
        if mod.weight.grad is not None:
            grads[name] = mod.weight.grad.detach().to("cpu", non_blocking=False)

    model.zero_grad(set_to_none=True)
    try:
        torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass

    rows: List[DriftValidationRow] = []
    for bits in bit_widths:
        # Step 2: in-place quant perturbation -- wrap in try/finally so we
        # always restore the original weights, even if a forward pass at the
        # perturbed weights raises (e.g. on dtype-sensitive code paths).
        saved = _apply_quant_perturbation_in_place(
            model, bits=int(bits), group_size=int(group_size),
        )
        try:
            # Predicted drift components using the saved (original) weights.
            # We use FP32 *accumulators* (via ``sum(..., dtype=torch.float32)``)
            # to avoid FP16 overflow on large dot products, but we keep the
            # source tensors in their native dtype. Previously this loop
            # called ``.float()`` on saved/grads/weight which materialised
            # three full FP32 copies of every linear (~3x memory), and
            # OOM'd on the lm_head of Llama-3.1-8B (vocab=128k).
            signed_total = 0.0
            abs_block = 0.0
            abs_module = 0.0
            cs_module = 0.0
            bs = max(1, int(block_size))
            for name, mod in iter_linear_modules(model):
                if name not in saved or name not in grads:
                    continue
                # All three tensors stay in their native dtype (typically FP16).
                w_orig = saved[name]
                w_new = mod.weight.detach()
                # grads[name] lives on CPU (offloaded after backward to avoid
                # the 9B-model GPU peak); fetch back per-module so only one
                # gradient is on GPU at a time.
                g = grads[name].to(device, non_blocking=False)
                # Element-wise delta in the source dtype (one extra tensor
                # the size of the weight, not doubled).
                delta = w_new - w_orig

                # Module-level signed inner product, accumulated in FP32.
                module_signed = float((g * delta).sum(dtype=torch.float32).item())
                signed_total += module_signed
                abs_module += abs(module_signed)

                # CS module bound: ‖g‖ * ‖δw‖. Compute the norm as
                # sum-of-squares with an FP32 accumulator, chunked along dim 0
                # so we never need to materialise an FP32 copy of the full
                # weight. (The earlier shortcut called ``g.float()`` which
                # materialised the FP32 copy and triggered the OOM that
                # bumped us off Llama-3.1-8B's lm_head.)
                def _l2_norm_chunked(t: torch.Tensor, chunk_rows: int = 4096) -> float:
                    if t.ndim != 2 or t.shape[0] <= chunk_rows:
                        return float((t * t).sum(dtype=torch.float32).sqrt().item())
                    acc = 0.0
                    for i in range(0, t.shape[0], chunk_rows):
                        ch = t[i:i + chunk_rows]
                        acc += float((ch * ch).sum(dtype=torch.float32).item())
                    return acc ** 0.5
                cs_module += _l2_norm_chunked(g) * _l2_norm_chunked(delta)

                # SSMP block granularity along dim 0 (output rows).
                out_features = delta.shape[0]
                if out_features <= bs:
                    abs_block += abs(module_signed)
                else:
                    n_full = out_features // bs
                    # Per-block signed dot products by reshape; the reshape
                    # is a view, no extra memory. Multiply is one tensor the
                    # size of (n_full*bs, in_features); we accumulate in
                    # FP32 via ``sum(dtype=...)``.
                    g_main = g[: n_full * bs].view(n_full, bs, -1)
                    d_main = delta[: n_full * bs].view(n_full, bs, -1)
                    block_signed = (g_main * d_main).sum(dim=(1, 2), dtype=torch.float32)
                    abs_block += float(block_signed.abs().sum().item())
                    if n_full * bs < out_features:
                        g_tail = g[n_full * bs:]
                        d_tail = delta[n_full * bs:]
                        tail_signed = float((g_tail * d_tail).sum(dtype=torch.float32).item())
                        abs_block += abs(tail_signed)

                # Free per-iter intermediates aggressively so the next
                # layer (potentially much larger, e.g. lm_head) has space.
                del delta, g
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # Step 3: measured loss after perturbation
            L1 = _safety_loss(
                model, tokenizer, prompts,
                refusal_template=refusal_template,
                max_length=max_length, device=device,
            )
        finally:
            # Step 4: restore -- always, even on exception.
            _restore_weights(model, saved)
            del saved

        measured_signed = L1 - L0
        measured_abs = abs(measured_signed)
        eps = 1e-8
        rel_err_signed = abs(measured_signed - signed_total) / max(abs(measured_signed), eps)

        rows.append(
            DriftValidationRow(
                label=f"bits={int(bits)}",
                bits=int(bits),
                group_size=int(group_size),
                block_size=int(block_size),
                predicted_inner_signed=float(signed_total),
                predicted_abs_block=float(abs_block),
                predicted_abs_module=float(abs_module),
                predicted_cs_module=float(cs_module),
                measured_dL_signed=float(measured_signed),
                measured_dL_abs=float(measured_abs),
                relative_error_signed=float(rel_err_signed),
            )
        )
        logger.info(
            "[drift-bound] bits=%d  signed: measured=%.4f predicted=%.4f rel_err=%.3f  |  "
            "absolute upper bound (theorem): abs_block=%.4f abs_module=%.4f cs=%.4f  |  "
            "|measured_dL|=%.4f",
            int(bits), measured_signed, signed_total, rel_err_signed,
            abs_block, abs_module, cs_module, measured_abs,
        )

    return pd.DataFrame([r.__dict__ for r in rows]).sort_values("bits", ascending=False)


def _ols_fit(x, y) -> Dict[str, float]:
    """Simple OLS: returns slope, intercept, R²."""
    import numpy as np

    x = np.asarray(x, dtype="float64")
    y = np.asarray(y, dtype="float64")
    n = len(x)
    if n < 2:
        return {"slope": float("nan"), "intercept": float("nan"), "r_squared": float("nan")}
    xm = float(x.mean())
    ym = float(y.mean())
    sxx = float(((x - xm) ** 2).sum())
    sxy = float(((x - xm) * (y - ym)).sum())
    syy = float(((y - ym) ** 2).sum())
    slope = sxy / sxx if sxx > 0 else 0.0
    intercept = ym - slope * xm
    r2 = (sxy * sxy) / (sxx * syy) if sxx > 0 and syy > 0 else 0.0
    return {"slope": slope, "intercept": intercept, "r_squared": r2}


def fit_drift_scatter_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """Return slope / intercept / R² for the drift-validation scatter.

    Two fits are reported, each addressing a different claim:

    1.  **First-order Taylor (signed)**: ``predicted_inner_signed`` vs
        ``measured_dL_signed``. A perfect first-order Taylor approximation
        gives slope=1, intercept=0, R²→1.

    2.  **Upper-bound tightness (absolute)**: ``predicted_abs_block`` vs
        ``measured_dL_abs``. The theorem (PLAN §1.2) only claims the absolute
        block-sum is an *upper bound* on |ΔL|, so slope ≤ 1 (typically), and
        R² ≈ 1 indicates the bound tracks the magnitude closely (= "magnitude
        -tight"). If R² is low but Spearman rank correlation between the two
        is high, the bound is "ranking-tight, magnitude-loose" -- still
        defensible per the §16 G1✗ recovery path.

    The G1 gate (R² ≥ 0.85) is evaluated on the upper-bound fit (#2).
    """
    if df.empty:
        return {"n": 0}

    n = len(df)
    if n < 2:
        return {"n": int(n)}

    signed_fit = _ols_fit(
        df["predicted_inner_signed"], df["measured_dL_signed"],
    )
    upper_bound_fit = _ols_fit(
        df["predicted_abs_block"], df["measured_dL_abs"],
    )

    # Spearman rank correlation as a robust ranking-tightness signal.
    try:
        x = df["predicted_abs_block"].to_numpy(dtype="float64")
        y = df["measured_dL_abs"].to_numpy(dtype="float64")
        rx = x.argsort().argsort()
        ry = y.argsort().argsort()
        rx_m = rx - rx.mean()
        ry_m = ry - ry.mean()
        spearman = float((rx_m * ry_m).sum() / ((rx_m ** 2).sum() ** 0.5 * (ry_m ** 2).sum() ** 0.5))
    except Exception:  # noqa: BLE001
        spearman = float("nan")

    return {
        "n": int(n),
        # Backwards-compat: keep top-level "slope/intercept/r_squared" pointing
        # at the signed first-order Taylor fit (what older callers expected).
        "slope": signed_fit["slope"],
        "intercept": signed_fit["intercept"],
        "r_squared": signed_fit["r_squared"],
        # Explicit named fits for the paper figure.
        "signed_taylor": signed_fit,
        "upper_bound": upper_bound_fit,
        "upper_bound_spearman_rho": spearman,
    }
