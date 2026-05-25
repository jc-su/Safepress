"""Per-layer and per-module quantization error analysis."""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
from scipy import stats
from tqdm import tqdm

from safepress.model.blocks import chunk_indices, iter_linear_modules
from safepress.model.score import _quant_dequant_symmetric_groupwise


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_module_name(name: str) -> Tuple[Optional[int], str]:
    """
    Extract a layer index and module type from a fully-qualified module name.

    Examples
    --------
    >>> _parse_module_name("model.layers.5.self_attn.q_proj")
    (5, 'attn')
    >>> _parse_module_name("model.layers.12.mlp.gate_proj")
    (12, 'mlp')
    >>> _parse_module_name("lm_head")
    (None, 'other')
    """
    # Try to extract layer index from patterns like "layers.5" or "h.12"
    layer_match = re.search(r"(?:layers|h)\.(\d+)", name)
    layer_idx: Optional[int] = int(layer_match.group(1)) if layer_match else None

    # Determine module type
    name_lower = name.lower()
    if any(k in name_lower for k in ("self_attn", "attention", "attn", "q_proj", "k_proj", "v_proj", "o_proj")):
        module_type = "attn"
    elif any(k in name_lower for k in ("mlp", "gate_proj", "up_proj", "down_proj", "fc1", "fc2", "dense_h_to_4h", "dense_4h_to_h")):
        module_type = "mlp"
    else:
        module_type = "other"

    return layer_idx, module_type


# ---------------------------------------------------------------------------
# Per-module quantization error
# ---------------------------------------------------------------------------


@torch.no_grad()
def compute_layer_quant_error(
    model: torch.nn.Module,
    *,
    bits: int = 4,
    group_size: int = 128,
    show_progress: bool = True,
) -> pd.DataFrame:
    """
    For every ``torch.nn.Linear`` module, compute the quantization error.

    Returns
    -------
    DataFrame with columns:
        module, layer_idx, module_type, mse, frobenius, relative_error
    """
    linear_modules = list(iter_linear_modules(model))
    rows: List[Dict[str, object]] = []

    it = linear_modules
    if show_progress:
        it = tqdm(linear_modules, desc="Layer quant error", leave=False)

    for mod_name, mod in it:
        w = mod.weight  # (out_features, in_features)
        w_hat = _quant_dequant_symmetric_groupwise(w, bits=bits, group_size=group_size)

        delta = (w_hat - w).float()
        w_f = w.float()

        mse = float((delta ** 2).mean().item())
        frob = float(delta.norm().item())
        w_norm = float(w_f.norm().item())
        relative_error = float(frob / w_norm) if w_norm > 0 else 0.0

        layer_idx, module_type = _parse_module_name(mod_name)

        rows.append(
            dict(
                module=mod_name,
                layer_idx=layer_idx,
                module_type=module_type,
                mse=mse,
                frobenius=frob,
                relative_error=relative_error,
            )
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Per-block quantization error
# ---------------------------------------------------------------------------


@torch.no_grad()
def compute_block_quant_error(
    model: torch.nn.Module,
    *,
    bits: int = 4,
    group_size: int = 128,
    block_size: int = 64,
    show_progress: bool = True,
) -> pd.DataFrame:
    """
    Per-block quantization error, matching the block structure in ``score.py``.

    Returns
    -------
    DataFrame with columns:
        module, block_idx, out_start, out_end, mse, frobenius, relative_error
    """
    linear_modules = list(iter_linear_modules(model))
    rows: List[Dict[str, object]] = []

    it = linear_modules
    if show_progress:
        it = tqdm(linear_modules, desc="Block quant error", leave=False)

    for mod_name, mod in it:
        w = mod.weight  # (out_features, in_features)
        out_features = w.shape[0]
        blocks = chunk_indices(out_features, block_size)

        for b_idx, (s, e) in enumerate(blocks):
            w_blk = w[s:e, :]
            w_hat_blk = _quant_dequant_symmetric_groupwise(
                w_blk, bits=bits, group_size=group_size,
            )

            delta = (w_hat_blk - w_blk).float()
            w_blk_f = w_blk.float()

            mse = float((delta ** 2).mean().item())
            frob = float(delta.norm().item())
            w_norm = float(w_blk_f.norm().item())
            relative_error = float(frob / w_norm) if w_norm > 0 else 0.0

            rows.append(
                dict(
                    module=mod_name,
                    block_idx=b_idx,
                    out_start=s,
                    out_end=e,
                    mse=mse,
                    frobenius=frob,
                    relative_error=relative_error,
                )
            )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Correlation between drift scores and quant error
# ---------------------------------------------------------------------------


def error_score_correlation(
    scores_df: pd.DataFrame,
    errors_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Join drift scores with quantization errors and compute correlations.

    Both DataFrames are expected to share ``module`` and ``block_idx`` columns.
    ``scores_df`` should have a ``score`` column (from ``compute_block_scores``).
    ``errors_df`` should have ``mse``, ``frobenius``, and ``relative_error``
    columns (from ``compute_block_quant_error``).

    Returns
    -------
    DataFrame with one row per error metric and columns:
        error_metric, pearson_r, pearson_p, spearman_r, spearman_p, n_blocks
    """
    merged = pd.merge(
        scores_df[["module", "block_idx", "score"]],
        errors_df[["module", "block_idx", "mse", "frobenius", "relative_error"]],
        on=["module", "block_idx"],
        how="inner",
    )

    if len(merged) < 3:
        # Not enough data points for meaningful correlation
        return pd.DataFrame(
            columns=["error_metric", "pearson_r", "pearson_p", "spearman_r", "spearman_p", "n_blocks"]
        )

    error_metrics = ["mse", "frobenius", "relative_error"]
    rows: List[Dict[str, object]] = []

    score_values = merged["score"].values

    for metric in error_metrics:
        error_values = merged[metric].values

        # Pearson correlation
        pearson_result = stats.pearsonr(score_values, error_values)
        pearson_r = float(pearson_result.statistic)
        pearson_p = float(pearson_result.pvalue)

        # Spearman rank correlation
        spearman_result = stats.spearmanr(score_values, error_values)
        spearman_r = float(spearman_result.statistic)
        spearman_p = float(spearman_result.pvalue)

        rows.append(
            dict(
                error_metric=metric,
                pearson_r=pearson_r,
                pearson_p=pearson_p,
                spearman_r=spearman_r,
                spearman_p=spearman_p,
                n_blocks=len(merged),
            )
        )

    return pd.DataFrame(rows)
