"""
Paper-quality visualization functions for SafePress experiments.

Generates the 5 key figures for the paper:
1. Safety phase transition curve
2. Block importance heatmap
3. Refusal signal propagation curve
4. Causal experiment bar charts
5. Safety-performance Pareto curve
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

try:
    import seaborn as sns

    _HAS_SEABORN = True
except ImportError:  # pragma: no cover
    _HAS_SEABORN = False


# ---------------------------------------------------------------------------
# Colour palette (consistent across all plots)
# ---------------------------------------------------------------------------

COLORS: Dict[str, str] = {
    "fp16": "#2196F3",        # blue
    "full_quant": "#F44336",  # red
    "ssmp": "#4CAF50",        # green (our method)
    "random": "#9E9E9E",      # grey
    "magnitude": "#FF9800",   # orange
    "gradient": "#9C27B0",    # purple
    "layer": "#795548",       # brown
}

# Ordered list for consistent legend ordering in causal bar charts.
_METHOD_ORDER: List[str] = [
    "fp16",
    "full_quant",
    "ssmp",
    "random",
    "magnitude",
    "gradient",
    "layer",
]


# ---------------------------------------------------------------------------
# Paper style
# ---------------------------------------------------------------------------

def set_paper_style() -> None:
    """Configure matplotlib rcParams for publication-quality figures.

    Call once (already called at module import time) to set serif fonts,
    size-11 text, high-DPI output, and tight layout defaults.
    """
    plt.rcParams.update(
        {
            # Font
            "font.family": "serif",
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9,
            # Layout
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "figure.autolayout": True,
            "figure.constrained_layout.use": False,
            # Lines & markers
            "lines.linewidth": 1.5,
            "lines.markersize": 5,
            # Axes
            "axes.linewidth": 0.8,
            "axes.grid": False,
            # Save
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
        }
    )


# Apply on import so every figure created after ``import safepress.viz.plots``
# automatically uses the paper style.
set_paper_style()


# ---------------------------------------------------------------------------
# Public-API aliases for backwards compatibility
# ---------------------------------------------------------------------------
#
# Historical scripts (e.g. ``scripts/generate_figures.py``) and earlier
# revisions of ``cli.py`` imported short names ``plot_heatmap`` and
# ``plot_causal``. The canonical functions in this module are the more
# descriptive ``plot_block_heatmap`` and ``plot_causal_experiment``. To avoid
# breaking those callers we expose both names at module scope; the aliases
# are defined just after the canonical functions below (forward references
# would fail), via a small ``__init_aliases__`` deferred block at the bottom
# of the module.


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _maybe_save(fig: plt.Figure, save_path: Optional[Union[str, Path]]) -> None:
    """Save *fig* to *save_path* if the argument is not ``None``."""
    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(path), dpi=300, bbox_inches="tight")


def _color_for(name: str) -> str:
    """Return the palette colour for *name*, falling back to a grey."""
    key = name.lower().replace(" ", "_").replace("-", "_")
    # Try exact match first, then substring match.
    if key in COLORS:
        return COLORS[key]
    for palette_key, colour in COLORS.items():
        if palette_key in key or key in palette_key:
            return colour
    return "#607D8B"  # blue-grey fallback


def _extract_layer_index(module_name: str) -> Optional[int]:
    """Extract numeric layer index from a dotted module path.

    Examples::

        model.layers.12.self_attn.q_proj  -> 12
        transformer.h.0.mlp.dense_4h_to_h -> 0
    """
    match = re.search(r"\.(\d+)\.", module_name)
    if match:
        return int(match.group(1))
    return None


def _extract_module_type(module_name: str) -> str:
    """Extract the leaf module type (e.g. ``q_proj``) from a dotted path."""
    return module_name.rsplit(".", 1)[-1] if "." in module_name else module_name


# ---------------------------------------------------------------------------
# 1. Safety phase transition
# ---------------------------------------------------------------------------

def plot_phase_transition(
    bit_widths: Sequence[float],
    safety_scores: Sequence[float],
    utility_scores: Sequence[float],
    *,
    title: Optional[str] = None,
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Plot the safety phase-transition curve (Figure 1 in the paper).

    Parameters
    ----------
    bit_widths:
        Bit-widths for each measurement point (e.g. ``[16, 8, 4, 3, 2]``).
        The FP16 baseline is taken as the first element when it equals 16.
    safety_scores:
        Safety metric at each bit-width (e.g. refusal rate in [0, 1]).
    utility_scores:
        Utility metric at each bit-width (e.g. MMLU accuracy in [0, 1]).
    title:
        Optional figure title.
    save_path:
        If given, save figure to this path.

    Returns
    -------
    matplotlib.figure.Figure
    """
    bit_widths = list(bit_widths)
    safety_scores = list(safety_scores)
    utility_scores = list(utility_scores)

    fig, ax1 = plt.subplots(figsize=(5.5, 3.8))

    color_safety = COLORS["ssmp"]
    color_utility = COLORS["fp16"]

    # --- Left y-axis: safety ---
    ax1.set_xlabel("Bit-width")
    ax1.set_ylabel("Safety (refusal rate)", color=color_safety)
    ln1 = ax1.plot(
        bit_widths,
        safety_scores,
        "o-",
        color=color_safety,
        label="Safety",
        zorder=3,
    )
    ax1.tick_params(axis="y", labelcolor=color_safety)
    ax1.set_ylim(-0.05, 1.05)

    # --- Right y-axis: utility ---
    ax2 = ax1.twinx()
    ax2.set_ylabel("Utility (MMLU accuracy)", color=color_utility)
    ln2 = ax2.plot(
        bit_widths,
        utility_scores,
        "s--",
        color=color_utility,
        label="Utility",
        zorder=3,
    )
    ax2.tick_params(axis="y", labelcolor=color_utility)
    ax2.set_ylim(-0.05, 1.05)

    # --- FP16 baseline dashed lines ---
    # Use the first entry if it corresponds to FP16 (bit_width == 16).
    if bit_widths and bit_widths[0] >= 16:
        ax1.axhline(
            safety_scores[0],
            color=color_safety,
            linestyle=":",
            linewidth=0.9,
            alpha=0.5,
        )
        ax2.axhline(
            utility_scores[0],
            color=color_utility,
            linestyle=":",
            linewidth=0.9,
            alpha=0.5,
        )

    # Invert x-axis so higher precision is on the left.
    ax1.invert_xaxis()
    ax1.set_xticks(bit_widths)
    ax1.set_xticklabels([str(b) for b in bit_widths])

    # Unified legend
    lines = ln1 + ln2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="center left", frameon=True, framealpha=0.9)

    if title:
        ax1.set_title(title)
    else:
        ax1.set_title("Safety degrades faster than utility under quantization")

    fig.tight_layout()
    _maybe_save(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# 2. Block importance heatmap
# ---------------------------------------------------------------------------

def plot_block_heatmap(
    scores_df: pd.DataFrame,
    *,
    model_name: Optional[str] = None,
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Heatmap of per-layer, per-module-type safety-drift scores.

    Parameters
    ----------
    scores_df:
        DataFrame with at least columns ``module``, ``block_idx``, ``score``
        (as produced by :func:`safepress.model.score.compute_block_scores`).
        The ``module`` column is parsed to extract layer index and module type.
    model_name:
        Optional model name for the title.
    save_path:
        If given, save figure to this path.

    Returns
    -------
    matplotlib.figure.Figure
    """
    df = scores_df.copy()

    # Derive layer_idx and module_type from the module name.
    df["layer_idx"] = df["module"].apply(_extract_layer_index)
    df["module_type"] = df["module"].apply(_extract_module_type)

    # Drop rows where layer index could not be parsed.
    df = df.dropna(subset=["layer_idx"])
    df["layer_idx"] = df["layer_idx"].astype(int)

    # Aggregate score per (layer_idx, module_type) -- sum over block_idx.
    pivot_data = (
        df.groupby(["layer_idx", "module_type"])["score"]
        .sum()
        .reset_index()
    )

    pivot = pivot_data.pivot(
        index="layer_idx", columns="module_type", values="score"
    )
    pivot = pivot.sort_index(ascending=True)

    # Order columns: attention projections first, then MLP projections.
    attn_order = ["q_proj", "k_proj", "v_proj", "o_proj"]
    mlp_order = ["gate_proj", "up_proj", "down_proj"]
    preferred_order = attn_order + mlp_order
    present_cols = [c for c in preferred_order if c in pivot.columns]
    remaining = [c for c in pivot.columns if c not in present_cols]
    pivot = pivot[present_cols + remaining]

    n_layers = len(pivot)
    n_cols = len(pivot.columns)
    fig_height = max(4.0, 0.22 * n_layers + 1.5)
    fig_width = max(4.5, 0.75 * n_cols + 2.0)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    if _HAS_SEABORN:
        sns.heatmap(
            pivot,
            ax=ax,
            cmap="RdYlBu_r",
            linewidths=0.3,
            linecolor="white",
            cbar_kws={"label": "Safety-drift score", "shrink": 0.75},
        )
    else:
        im = ax.imshow(
            pivot.values,
            aspect="auto",
            cmap="RdYlBu_r",
            interpolation="nearest",
        )
        ax.set_xticks(range(n_cols))
        ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
        ax.set_yticks(range(n_layers))
        ax.set_yticklabels(pivot.index)
        cbar = fig.colorbar(im, ax=ax, shrink=0.75)
        cbar.set_label("Safety-drift score")

    ax.set_xlabel("Module type")
    ax.set_ylabel("Layer index")

    title_parts = ["Block safety-drift scores"]
    if model_name:
        title_parts.append(f"({model_name})")
    ax.set_title(" ".join(title_parts))

    fig.tight_layout()
    _maybe_save(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# 3. Refusal signal propagation
# ---------------------------------------------------------------------------

def plot_refusal_signal(
    profile_df: pd.DataFrame,
    *,
    quantized_profile_df: Optional[pd.DataFrame] = None,
    title: Optional[str] = None,
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Plot refusal-signal separation across layers.

    Parameters
    ----------
    profile_df:
        DataFrame with columns ``layer``, ``harmful_projection``,
        ``harmless_projection``, ``separation`` (from
        :func:`safepress.analysis.refusal_direction.refusal_signal_profile`).
    quantized_profile_df:
        Optional second profile (e.g. a quantized model) to overlay.
    title:
        Optional figure title.
    save_path:
        If given, save figure to this path.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(6, 3.8))

    layers = profile_df["layer"].values
    sep = profile_df["separation"].values

    ax.plot(
        layers,
        sep,
        "o-",
        color=COLORS["fp16"],
        label="FP16",
        markersize=4,
    )

    if quantized_profile_df is not None:
        q_layers = quantized_profile_df["layer"].values
        q_sep = quantized_profile_df["separation"].values
        ax.plot(
            q_layers,
            q_sep,
            "s--",
            color=COLORS["full_quant"],
            label="Quantized",
            markersize=4,
        )

        # Shade the region of maximum degradation.
        # Use the shared set of layers for the fill.
        shared_layers = sorted(set(layers) & set(q_layers))
        if shared_layers:
            fp_map = dict(zip(layers, sep))
            qt_map = dict(zip(q_layers, q_sep))
            shared = np.array(shared_layers)
            fp_vals = np.array([fp_map[l] for l in shared])
            qt_vals = np.array([qt_map[l] for l in shared])
            ax.fill_between(
                shared,
                fp_vals,
                qt_vals,
                alpha=0.15,
                color=COLORS["full_quant"],
                label="Degradation",
            )

    ax.set_xlabel("Layer index")
    ax.set_ylabel("Refusal-signal separation")
    ax.legend(frameon=True, framealpha=0.9)

    if title:
        ax.set_title(title)
    else:
        ax.set_title("Refusal signal propagation across layers")

    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    fig.tight_layout()
    _maybe_save(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# 4. Causal experiment bar charts
# ---------------------------------------------------------------------------

# Default condition lists for each experiment type.
_CAUSAL_CONDITIONS: Dict[str, List[str]] = {
    "targeted": [
        "FP16",
        "Full 4-bit",
        "Critical-only quant",
        "Non-critical-only quant",
    ],
    "rollback": [
        "FP16",
        "Full 4-bit",
        "Rollback top-K",
        "Rollback random-K",
        "Rollback bottom-K",
    ],
    "control": [
        "FP16",
        "Full 4-bit",
        "SSMP (ours)",
        "Random",
        "Magnitude",
        "Gradient-only",
        "Layer-uniform",
    ],
}


def _condition_color(name: str, experiment_type: str) -> str:
    """Map a condition name to a colour, highlighting 'ours'."""
    lower = name.lower()
    if lower == "fp16":
        return COLORS["fp16"]
    if "full" in lower and ("4" in lower or "quant" in lower):
        return COLORS["full_quant"]
    if "ssmp" in lower or "ours" in lower:
        return COLORS["ssmp"]
    if "random" in lower:
        return COLORS["random"]
    if "magnitude" in lower:
        return COLORS["magnitude"]
    if "gradient" in lower:
        return COLORS["gradient"]
    if "layer" in lower or "uniform" in lower:
        return COLORS["layer"]
    # Experiment-specific fallbacks.
    if "critical" in lower and "non" in lower:
        return "#FF9800"   # orange for non-critical
    if "critical" in lower:
        return COLORS["ssmp"]  # green for critical-only
    if "rollback" in lower and "top" in lower:
        return COLORS["ssmp"]
    if "rollback" in lower and "bottom" in lower:
        return "#FF9800"
    return "#607D8B"


def plot_causal_experiment(
    results_dict: Dict[str, Any],
    *,
    experiment_type: str = "targeted",
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Bar chart for causal ablation experiments.

    Parameters
    ----------
    results_dict:
        Mapping ``condition_name -> value`` **or**
        ``condition_name -> {"mean": float, "std": float}``.
        Values are typically refusal rates.
    experiment_type:
        One of ``"targeted"``, ``"rollback"``, ``"control"``.
        Controls default ordering and colouring.
    save_path:
        If given, save figure to this path.

    Returns
    -------
    matplotlib.figure.Figure
    """
    # Determine ordering.
    default_order = _CAUSAL_CONDITIONS.get(experiment_type, [])
    # Use the default order for keys that appear; then append any extras.
    ordered_keys: List[str] = [k for k in default_order if k in results_dict]
    extras = [k for k in results_dict if k not in ordered_keys]
    ordered_keys.extend(extras)

    means: List[float] = []
    stds: List[float] = []
    colors: List[str] = []

    for key in ordered_keys:
        val = results_dict[key]
        if isinstance(val, dict):
            means.append(float(val.get("mean", val.get("value", 0.0))))
            stds.append(float(val.get("std", val.get("error", 0.0))))
        else:
            means.append(float(val))
            stds.append(0.0)
        colors.append(_condition_color(key, experiment_type))

    x = np.arange(len(ordered_keys))
    width = 0.6

    fig, ax = plt.subplots(figsize=(max(5.0, 0.9 * len(ordered_keys) + 2), 3.8))

    has_errors = any(s > 0 for s in stds)
    bars = ax.bar(
        x,
        means,
        width,
        color=colors,
        edgecolor="white",
        linewidth=0.5,
        yerr=stds if has_errors else None,
        capsize=3,
        error_kw={"linewidth": 1.0},
        zorder=3,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(ordered_keys, rotation=30, ha="right")
    ax.set_ylabel("Refusal rate")
    ax.set_ylim(0, max(means) * 1.25 if means else 1.0)

    # Annotate bars with values.
    for bar_obj, val in zip(bars, means):
        ax.text(
            bar_obj.get_x() + bar_obj.get_width() / 2,
            bar_obj.get_height() + (max(means) * 0.02 if means else 0.01),
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    title_map = {
        "targeted": "Targeted quantization experiment",
        "rollback": "Rollback experiment",
        "control": "Scoring-method comparison",
    }
    ax.set_title(title_map.get(experiment_type, f"Causal experiment ({experiment_type})"))
    fig.tight_layout()
    _maybe_save(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# 5. Safety-performance Pareto curve
# ---------------------------------------------------------------------------

def plot_pareto_curve(
    budget_sweep_results: Dict[str, List[Tuple[float, float]]],
    *,
    x_metric: str = "memory_mb",
    y_metric: str = "refusal_rate",
    baseline_methods: Optional[List[str]] = None,
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Plot safety-performance Pareto frontier for SSMP vs baselines.

    Parameters
    ----------
    budget_sweep_results:
        Mapping ``method_name -> [(x_value, y_value), ...]``.
    x_metric:
        Label for the x-axis (e.g. ``"memory_mb"``, ``"latency_ms"``).
    y_metric:
        Label for the y-axis (e.g. ``"refusal_rate"``, ``"safety_score"``).
    baseline_methods:
        Subset of method names to treat as baselines (drawn as dashed).
        If ``None``, every method except ``"ssmp"`` is treated as baseline.
    save_path:
        If given, save figure to this path.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if baseline_methods is None:
        baseline_methods = [m for m in budget_sweep_results if m.lower() != "ssmp"]

    fig, ax = plt.subplots(figsize=(5.5, 4.0))

    for method, points in budget_sweep_results.items():
        if not points:
            continue
        pts = sorted(points, key=lambda p: p[0])
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        is_ours = method.lower() in ("ssmp", "ssmp (ours)")
        is_baseline = method in baseline_methods

        color = _color_for(method)
        style = "o-" if is_ours else ("x--" if is_baseline else "^:")
        lw = 2.0 if is_ours else 1.2
        ms = 7 if is_ours else 5

        ax.plot(xs, ys, style, color=color, label=method, linewidth=lw, markersize=ms, zorder=4 if is_ours else 2)

        # Highlight Pareto-optimal points for our method.
        if is_ours and len(pts) > 1:
            pareto_xs, pareto_ys = _pareto_front(xs, ys, higher_is_better_y=True)
            ax.scatter(
                pareto_xs,
                pareto_ys,
                s=100,
                facecolors="none",
                edgecolors=COLORS["ssmp"],
                linewidths=1.5,
                zorder=5,
                label="Pareto-optimal (SSMP)",
            )

    ax.set_xlabel(x_metric.replace("_", " ").title())
    ax.set_ylabel(y_metric.replace("_", " ").title())
    ax.set_title("Safety vs. overhead Pareto frontier")
    ax.legend(frameon=True, framealpha=0.9, fontsize=8)
    fig.tight_layout()
    _maybe_save(fig, save_path)
    return fig


def _pareto_front(
    xs: List[float],
    ys: List[float],
    *,
    higher_is_better_y: bool = True,
) -> Tuple[List[float], List[float]]:
    """Return the Pareto-optimal subset (minimise x, maximise/minimise y)."""
    points = sorted(zip(xs, ys), key=lambda p: p[0])
    pareto_x: List[float] = []
    pareto_y: List[float] = []
    best_y = float("-inf") if higher_is_better_y else float("inf")

    for x, y in points:
        if higher_is_better_y:
            if y >= best_y:
                best_y = y
                pareto_x.append(x)
                pareto_y.append(y)
        else:
            if y <= best_y:
                best_y = y
                pareto_x.append(x)
                pareto_y.append(y)

    return pareto_x, pareto_y


# ---------------------------------------------------------------------------
# Paper figure: predicted vs measured drift (theory validation)
# ---------------------------------------------------------------------------

def plot_drift_bound_scatter(
    df: pd.DataFrame,
    *,
    mode: str = "upper_bound",
    title: Optional[str] = None,
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Predicted-vs-measured safety-loss drift scatter for the theory section.

    Two complementary modes:

    * ``mode="upper_bound"`` (default, THEOREM-aligned): plots
      ``predicted_abs_block`` (the per-block absolute-sum upper bound
      Σ |g_b · δw_b|) against ``measured_dL_abs`` (|ΔL_safe|). High R² here
      means the bound is magnitude-tight; low R² with high Spearman ρ means
      "ranking-tight" (G1 recovery path per PLAN §16).
    * ``mode="signed"`` : plots ``predicted_inner_signed`` against
      ``measured_dL_signed`` -- the first-order Taylor fit. Slope ≈ 1 means
      the Taylor approximation is accurate at this δW scale.

    Parameters
    ----------
    df:
        DataFrame from
        :func:`safepress.analysis.drift_bound.validate_drift_bound` with
        columns ``bits``, ``predicted_inner_signed``, ``predicted_abs_block``,
        ``predicted_cs_module``, ``measured_dL_signed``, ``measured_dL_abs``.
    """
    fig, ax = plt.subplots(figsize=(5.0, 4.5))

    # Pick X / Y columns based on mode. Tolerate legacy column names so older
    # output CSVs still render.
    if mode == "upper_bound":
        x_col = "predicted_abs_block" if "predicted_abs_block" in df.columns else "predicted_cs_bound"
        y_col = "measured_dL_abs" if "measured_dL_abs" in df.columns else "measured_dL"
        x_label = r"Predicted upper bound: $\sum_b |g_b \cdot \delta w_b|$"
        y_label = r"Measured $|\Delta L_{\mathrm{safe}}|$"
        ref_label = "y = x (magnitude-tight bound)"
    elif mode == "signed":
        x_col = "predicted_inner_signed" if "predicted_inner_signed" in df.columns else "predicted_inner"
        y_col = "measured_dL_signed" if "measured_dL_signed" in df.columns else "measured_dL"
        x_label = r"Predicted Taylor drift: $\sum_b g_b \cdot \delta w_b$"
        y_label = r"Measured $\Delta L_{\mathrm{safe}}$"
        ref_label = "y = x (exact Taylor)"
    else:
        raise ValueError(f"plot_drift_bound_scatter: unknown mode '{mode}'")

    x = df[x_col].to_numpy(dtype="float64")
    y = df[y_col].to_numpy(dtype="float64")

    # Colour-code points by bit-width.
    if "bits" in df.columns:
        bits_unique = sorted(set(df["bits"].astype(int).tolist()))
        cmap = plt.cm.viridis
        for bit in bits_unique:
            sel = df["bits"].astype(int) == bit
            ax.scatter(
                df.loc[sel, x_col],
                df.loc[sel, y_col],
                color=cmap((bit - min(bits_unique)) / max(1, (max(bits_unique) - min(bits_unique)))),
                s=55, edgecolor="white", linewidth=0.5,
                label=f"{bit}-bit", zorder=3,
            )
    else:
        ax.scatter(x, y, s=55, color=COLORS["ssmp"], edgecolor="white", zorder=3)

    # Reference line
    finite = np.isfinite(x) & np.isfinite(y)
    if finite.any():
        lo = float(min(x[finite].min(), y[finite].min()))
        hi = float(max(x[finite].max(), y[finite].max()))
        pad = 0.05 * (hi - lo + 1e-8)
        ref = np.linspace(lo - pad, hi + pad, 64)
        ax.plot(ref, ref, "--", color="grey", linewidth=1.0, label=ref_label, zorder=2)

    # Show the per-module Cauchy-Schwarz envelope on the same axes as a looser
    # upper-bound comparison, when available.
    cs_col = "predicted_cs_module" if "predicted_cs_module" in df.columns else "predicted_cs_bound"
    if mode == "upper_bound" and cs_col in df.columns:
        ax.scatter(
            df[cs_col], df[y_col],
            marker="x", s=35, color=COLORS["full_quant"], alpha=0.6,
            label="CS module-level upper bound", zorder=2,
        )

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title or ("Drift bound (upper-bound fit)" if mode == "upper_bound" else "Drift bound (signed Taylor fit)"))
    ax.legend(frameon=True, framealpha=0.9, fontsize=8)
    fig.tight_layout()
    _maybe_save(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# Paper figure: attack-augmented bar chart
# ---------------------------------------------------------------------------

def plot_attack_bars(
    results: Dict[str, Dict[str, float]],
    *,
    methods_order: Optional[Sequence[str]] = None,
    attacks_order: Sequence[str] = ("direct", "gcg", "autodan", "pair"),
    metric_label: str = "Attack success rate (lower is better)",
    title: Optional[str] = None,
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Grouped bar chart of ASR per (method, attack).

    Parameters
    ----------
    results:
        Mapping ``method -> {attack: asr}``.  Missing ``(method, attack)``
        cells are rendered as zeros so the bar layout stays stable.
    methods_order:
        Method order along the x-axis.  Defaults to the order of *results*.
    attacks_order:
        Attack name order within each group of bars.
    metric_label:
        Y-axis label.
    title:
        Optional figure title.
    save_path:
        If given, save the figure here.
    """
    methods = list(methods_order) if methods_order is not None else list(results.keys())
    attacks = list(attacks_order)

    n_meth = len(methods)
    n_attacks = len(attacks)
    x = np.arange(n_meth)
    total_width = 0.78
    width = total_width / max(n_attacks, 1)

    fig, ax = plt.subplots(figsize=(max(5.0, 1.2 * n_meth + 1.5), 3.8))

    # Use a fixed palette for attacks.
    attack_colors = {
        "direct":  COLORS["fp16"],
        "gcg":     COLORS["full_quant"],
        "autodan": COLORS["magnitude"],
        "pair":    COLORS["gradient"],
    }

    for j, atk in enumerate(attacks):
        vals = [
            float(results.get(m, {}).get(atk, 0.0))
            for m in methods
        ]
        offset = (j - (n_attacks - 1) / 2.0) * width
        ax.bar(
            x + offset, vals,
            width * 0.92,
            color=attack_colors.get(atk, "#607D8B"),
            edgecolor="white", linewidth=0.5,
            label=atk.upper(),
            zorder=3,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=15, ha="right")
    ax.set_ylabel(metric_label)
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=True, framealpha=0.9, fontsize=8, loc="upper right")
    if title:
        ax.set_title(title)
    else:
        ax.set_title("Attack success rate per method")
    fig.tight_layout()
    _maybe_save(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# Paper figure: safety x utility Pareto (or safety x over-refusal)
# ---------------------------------------------------------------------------

def plot_safety_utility_pareto(
    points: Dict[str, Tuple[float, float]],
    *,
    x_label: str = "Utility (1 - normalized PPL)",
    y_label: str = "Safety (HarmBench refusal rate)",
    higher_is_better_x: bool = True,
    higher_is_better_y: bool = True,
    annotate: bool = True,
    save_path: Optional[Union[str, Path]] = None,
    title: Optional[str] = None,
) -> plt.Figure:
    """Scatter plot of (x, y) per method, with the Pareto frontier drawn in.

    Parameters
    ----------
    points:
        Mapping ``label -> (x, y)`` where labels are method names
        (e.g. ``"SSMP@4%"``, ``"Q-resafe"``, ``"CWP@60%"``).
    x_label, y_label:
        Axis labels.
    higher_is_better_x / higher_is_better_y:
        Direction of the Pareto frontier.
    annotate:
        Draw text labels at each point.
    save_path:
        If given, save the figure here.
    """
    fig, ax = plt.subplots(figsize=(5.5, 4.2))

    xs = [p[0] for p in points.values()]
    ys = [p[1] for p in points.values()]

    for label, (x, y) in points.items():
        ours = "ssmp" in label.lower()
        color = COLORS["ssmp"] if ours else _color_for(label)
        size = 90 if ours else 55
        ax.scatter(
            x, y,
            s=size, color=color, edgecolor="black", linewidth=0.6, zorder=3,
        )
        if annotate:
            ax.annotate(
                label,
                xy=(x, y),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=8,
                color=("black" if not ours else "#1B5E20"),
            )

    # Compute Pareto frontier in the chosen orientation.
    # We canonicalize to "minimize x, maximize y" then flip the input as needed.
    flip_x = not higher_is_better_x   # we want higher_is_better_x semantics
    flip_y = higher_is_better_y       # invert: pareto helper takes higher_is_better_y arg
    if higher_is_better_x:
        # Minimize -x equivalent; just negate before passing
        xs_for_pareto = [-v for v in xs]
    else:
        xs_for_pareto = list(xs)
    pareto_x, pareto_y = _pareto_front(xs_for_pareto, ys, higher_is_better_y=higher_is_better_y)
    if higher_is_better_x:
        pareto_x = [-v for v in pareto_x]
    if pareto_x:
        order = sorted(range(len(pareto_x)), key=lambda i: pareto_x[i])
        ax.plot(
            [pareto_x[i] for i in order],
            [pareto_y[i] for i in order],
            "--",
            color=COLORS["ssmp"],
            linewidth=1.4, alpha=0.8,
            label="Pareto frontier",
            zorder=2,
        )

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title or "Safety vs. utility Pareto")
    ax.legend(frameon=True, framealpha=0.9, fontsize=8, loc="best")
    fig.tight_layout()
    _maybe_save(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# Backwards-compat aliases (resolved here, after the canonical functions are
# defined, so older scripts and CLI viz handlers keep working).
# ---------------------------------------------------------------------------

plot_heatmap = plot_block_heatmap
plot_causal = plot_causal_experiment

# ---------------------------------------------------------------------------
# 6. Budget sweep
# ---------------------------------------------------------------------------

def plot_budget_sweep(
    budgets: Sequence[float],
    refusal_rates: Sequence[float],
    *,
    baseline_refusal: Optional[float] = None,
    fullquant_refusal: Optional[float] = None,
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Plot refusal rate as a function of FP16-budget ratio.

    Parameters
    ----------
    budgets:
        Budget ratios (e.g. ``[0.005, 0.01, 0.02, 0.04, 0.08]``).
    refusal_rates:
        Refusal rate at each budget.
    baseline_refusal:
        FP16 baseline refusal rate (drawn as horizontal dashed line).
    fullquant_refusal:
        Full-quantization refusal rate (drawn as horizontal dashed line).
    save_path:
        If given, save figure to this path.

    Returns
    -------
    matplotlib.figure.Figure
    """
    budgets = list(budgets)
    refusal_rates = list(refusal_rates)

    fig, ax = plt.subplots(figsize=(5.5, 3.8))

    ax.plot(
        budgets,
        refusal_rates,
        "o-",
        color=COLORS["ssmp"],
        label="SSMP",
        linewidth=2.0,
        markersize=6,
        zorder=3,
    )

    if baseline_refusal is not None:
        ax.axhline(
            baseline_refusal,
            color=COLORS["fp16"],
            linestyle="--",
            linewidth=1.0,
            label=f"FP16 baseline ({baseline_refusal:.2f})",
            zorder=2,
        )

    if fullquant_refusal is not None:
        ax.axhline(
            fullquant_refusal,
            color=COLORS["full_quant"],
            linestyle="--",
            linewidth=1.0,
            label=f"Full 4-bit ({fullquant_refusal:.2f})",
            zorder=2,
        )

    # Try to identify the "sweet spot": largest marginal gain per budget step.
    if len(budgets) >= 3 and len(refusal_rates) >= 3:
        diffs = [
            refusal_rates[i] - refusal_rates[i - 1]
            for i in range(1, len(refusal_rates))
        ]
        best_idx = int(np.argmax(np.abs(diffs))) + 1  # index into budgets
        ax.annotate(
            "sweet spot",
            xy=(budgets[best_idx], refusal_rates[best_idx]),
            xytext=(budgets[best_idx] + (max(budgets) - min(budgets)) * 0.08,
                    refusal_rates[best_idx] - 0.05),
            arrowprops=dict(arrowstyle="->", color="grey", lw=1.0),
            fontsize=9,
            color="grey",
        )

    ax.set_xlabel("FP16-budget ratio")
    ax.set_ylabel("Refusal rate")
    ax.set_title("Budget sweep: small budgets recover most safety")

    # Format x-axis as percentages if values are < 1.
    if all(b < 1 for b in budgets):
        ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=1))

    ax.legend(frameon=True, framealpha=0.9)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    _maybe_save(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# 7. Cross-model comparison
# ---------------------------------------------------------------------------

def plot_cross_model_comparison(
    results_df: pd.DataFrame,
    *,
    metric: str = "refusal_rate",
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Grouped bar chart comparing methods across model families.

    Parameters
    ----------
    results_df:
        DataFrame with columns ``model``, ``method``, and at least the column
        named *metric*.  Each row is one (model, method) result.
    metric:
        Column name to plot on the y-axis.
    save_path:
        If given, save figure to this path.

    Returns
    -------
    matplotlib.figure.Figure
    """
    df = results_df.copy()

    # Determine groups.
    models = df["model"].unique().tolist()
    methods = df["method"].unique().tolist()

    # Sort methods into a sensible order (FP16 first, then full-quant, then SSMP, rest).
    def _method_sort_key(m: str) -> int:
        lower = m.lower().replace(" ", "_").replace("-", "_")
        for i, ref in enumerate(_METHOD_ORDER):
            if ref in lower or lower in ref:
                return i
        return len(_METHOD_ORDER)

    methods = sorted(methods, key=_method_sort_key)

    n_models = len(models)
    n_methods = len(methods)
    x = np.arange(n_models)
    total_bar_width = 0.75
    bar_width = total_bar_width / max(n_methods, 1)

    fig, ax = plt.subplots(figsize=(max(5.0, 1.2 * n_models * n_methods + 1), 4.0))

    for j, method in enumerate(methods):
        vals: List[float] = []
        for model in models:
            subset = df[(df["model"] == model) & (df["method"] == method)]
            if len(subset) > 0:
                vals.append(float(subset[metric].iloc[0]))
            else:
                vals.append(0.0)

        offset = (j - (n_methods - 1) / 2) * bar_width
        color = _color_for(method)
        ax.bar(
            x + offset,
            vals,
            bar_width * 0.9,
            color=color,
            edgecolor="white",
            linewidth=0.4,
            label=method,
            zorder=3,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(f"Cross-model comparison: {metric.replace('_', ' ')}")
    ax.legend(frameon=True, framealpha=0.9, fontsize=8)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    _maybe_save(fig, save_path)
    return fig
