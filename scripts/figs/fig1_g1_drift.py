#!/usr/bin/env python
"""Figure 1 (T1, full-width 1x5): drift bound predicted vs measured, per model.

One panel per model (5 across), each a log-log scatter of predicted
upper bound vs measured |dL_safe| over 4 bit-widths. Panel title shows
R^2 and Spearman rho. Cross-column (figure*) layout.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PANELS = [
    ("Llama-3.1",   "runs/emnlp_g1/llama31_drift.csv",  "runs/emnlp_g1/llama31_drift.fit.json"),
    ("Qwen3",       "runs/emnlp_g1/qwen3_drift.csv",    "runs/emnlp_g1/qwen3_drift.fit.json"),
    ("GLM-4",       "runs/emnlp_g1/glm4_drift.csv",     "runs/emnlp_g1/glm4_drift.fit.json"),
    ("Ministral",   "runs/emnlp_g1/mistral_drift.csv",  "runs/emnlp_g1/mistral_drift.fit.json"),
    ("DeepSeek",    "runs/emnlp_g1/deepseek_drift.csv", "runs/emnlp_g1/deepseek_drift.fit.json"),
]
BIT_MARKERS = {2: ("D", "#d62728"), 3: ("^", "#ff7f0e"), 4: ("s", "#1f77b4"), 8: ("o", "#2ca02c")}


def main():
    fig, axes = plt.subplots(1, 5, figsize=(7.2, 1.7))

    for ax, (name, csv, fit_p) in zip(axes, PANELS):
        if not Path(csv).exists():
            ax.text(0.5, 0.5, "(missing)", ha="center", va="center", transform=ax.transAxes)
            continue
        df = pd.read_csv(csv)
        fit = json.loads(Path(fit_p).read_text()) if Path(fit_p).exists() else {}
        r2 = fit.get("upper_bound", {}).get("r_squared", float("nan"))
        rho = fit.get("upper_bound_spearman_rho", float("nan"))

        x = df["predicted_abs_block"].abs()
        y_col = "measured_abs_dL" if "measured_abs_dL" in df.columns else "measured_dL_abs"
        y = df[y_col].abs() if y_col in df.columns else pd.Series([], dtype=float)
        if y.empty:
            for col in df.columns:
                if "measured" in col.lower() and df[col].dtype.kind == "f":
                    y = df[col].abs(); break

        for bits in sorted(BIT_MARKERS):
            mask = (df["bits"] == bits) if "bits" in df.columns else (df.get("bit_width", pd.Series()) == bits)
            if mask.any():
                marker, color = BIT_MARKERS[bits]
                ax.scatter(x[mask], y[mask], marker=marker, s=34, facecolors=color,
                           edgecolors="black", linewidths=0.5, label=f"{bits}-bit", zorder=3)
        # connect points in ascending-x order: turns 4 sparse dots into a clean
        # monotone predicted<->measured trajectory; a vertical wiggle honestly
        # signals weaker magnitude-fit (low R^2) while rank stays tight (rho).
        xv, yv = x.values, y.values
        order = np.argsort(xv)
        ax.plot(xv[order], yv[order], color="#9aa0a6", lw=1.1, ls="-", alpha=0.85, zorder=2)

        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_title(f"{name}\n$R^2$={r2:.2f}, $\\rho$={rho:.2f}", fontsize=6.8)
        ax.tick_params(labelsize=5.5)
        ax.grid(True, alpha=0.18, which="major", linewidth=0.4)

    axes[0].set_ylabel("measured drift $|\\Delta L_{\\text{safe}}|$", fontsize=7)
    fig.supxlabel("predicted drift bound (per model, log–log)", fontsize=8, y=0.07)
    # single shared legend for bit-widths, placed tight above the panel row
    handles = [plt.Line2D([0], [0], marker=BIT_MARKERS[b][0], linestyle="", color=BIT_MARKERS[b][1],
                          markeredgecolor="black", markeredgewidth=0.5, markersize=5, label=f"{b}-bit")
               for b in [2, 3, 4, 8]]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=6.5,
               bbox_to_anchor=(0.5, 0.99), framealpha=0.9, handletextpad=0.3, columnspacing=1.0)
    fig.subplots_adjust(left=0.055, right=0.995, top=0.74, bottom=0.23, wspace=0.30)
    out = Path("paper/figs/fig1_g1_drift.pdf")
    fig.savefig(out, bbox_inches="tight", dpi=200)
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=200)
    print(f"[fig1] wrote {out}")


if __name__ == "__main__":
    main()
