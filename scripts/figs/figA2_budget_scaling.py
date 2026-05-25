#!/usr/bin/env python
"""Appendix Figure A2: budget-scaling — refusal rate and GCG ASR vs Fisher budget.

Side-by-side panels: (left) refusal rate, (right) GCG ASR.
X-axis: protection budget (0=INT4, 0.04, 0.08, 0.60, then FP16 as far right).
One line per model.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODEL_DIRS = {
    "Llama-3.1-8B":    ("runs/emnlp_g2_pilot_llama31",  "runs/emnlp_fisher60/llama31",  "llama31_8b"),
    "Qwen3-8B":        ("runs/emnlp_g2_pilot",          "runs/emnlp_fisher60/qwen3",    "qwen3_8b"),
    "GLM-4-9B":        ("runs/emnlp_g2_pilot_glm4",     "runs/emnlp_fisher60/glm4",     "glm4_9b"),
    "Ministral-8B":    ("runs/emnlp_g2_pilot_mistral",  "runs/emnlp_fisher60/mistral",  "mistral_8b"),
    "DeepSeek-R1-8B":  ("runs/emnlp_g2_pilot_deepseek", "runs/emnlp_fisher60/deepseek", "deepseek_r1_distill_llama_8b"),
}

COLORS = {
    "Llama-3.1-8B":   "#1f77b4",
    "Qwen3-8B":       "#ff7f0e",
    "GLM-4-9B":       "#2ca02c",
    "Ministral-8B":   "#d62728",
    "DeepSeek-R1-8B": "#9467bd",
}


def get(d, fn, key):
    p = Path(d) / fn
    if not p.exists(): return None
    try:
        return json.loads(p.read_text()).get(key)
    except Exception:
        return None


def main():
    # Evenly-spaced categorical x-axis (budgets 0/4/8/60/100% are uneven, so
    # equal spacing keeps points readable). INT4 = floor, FP16 = ceiling.
    x_pts = [0, 1, 2, 3, 4]
    x_labels = ["INT4", "4%", "8%", "60%", "FP16"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.0))

    for model, (g2_dir, f60_dir, tag) in MODEL_DIRS.items():
        color = COLORS[model]
        refusals = []
        gcgs = []
        # INT4
        int4_dir = next(Path(g2_dir).glob(f"sweep_{tag}_3bit_g2pilot_int4_b0.0"), None)
        if int4_dir is None:
            # fallback search
            int4_dir = next(Path(g2_dir).glob("sweep_*_int4_b0.0"), None)
        # Fisher@4, @8 from g2_pilot
        f4_dir = next(Path(g2_dir).glob("sweep_*_gradient_only_b0.04"), None)
        f8_dir = next(Path(g2_dir).glob("sweep_*_gradient_only_b0.08"), None)
        # Fisher@60 from fisher60 dir
        f60 = next(Path(f60_dir).glob("sweep_*_gradient_only_b0.6"), None) if Path(f60_dir).exists() else None
        # FP16
        fp16_dir = next(Path(g2_dir).glob("sweep_*_fp16_b0.0"), None)

        for d in [int4_dir, f4_dir, f8_dir, f60, fp16_dir]:
            refusal = get(d, "eval_seed0.json", "refusal_rate") if d else None
            gcg = get(d, "jailbreak_gcg_seed0.json", "asr") if d else None
            refusals.append(refusal)
            gcgs.append(gcg)

        # Plot lines (drop None)
        xs_r = [x for x, y in zip(x_pts, refusals) if y is not None]
        ys_r = [y for y in refusals if y is not None]
        xs_g = [x for x, y in zip(x_pts, gcgs) if y is not None]
        ys_g = [y for y in gcgs if y is not None]

        ax1.plot(xs_r, ys_r, marker="o", color=color, label=model, markersize=6, linewidth=1.6)
        ax2.plot(xs_g, ys_g, marker="o", color=color, label=model, markersize=6, linewidth=1.6)

    for ax, title, ylabel in [
        (ax1, "Refusal rate vs. budget", "Refusal rate (higher safer)"),
        (ax2, "GCG ASR vs. budget", "GCG ASR (lower safer)"),
    ]:
        ax.set_title(title, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=8.5)
        # shade the two endpoints (floor / ceiling) to set them apart from the
        # Fisher budget sweep (4/8/60%) in between.
        ax.axvspan(-0.45, 0.45, color="grey", alpha=0.08, zorder=0)
        ax.axvspan(3.55, 4.45, color="grey", alpha=0.08, zorder=0)
        ax.set_xticks(x_pts)
        ax.set_xticklabels(x_labels, fontsize=8)
        ax.set_xlabel("Protection budget", fontsize=8.5)
        ax.grid(True, axis="y", alpha=0.25, linewidth=0.4)
        ax.set_xlim(-0.5, 4.5)
        ax.set_ylim(-0.03, 1.05)
        ax.tick_params(labelsize=7.5)
    ax1.legend(loc="upper left", fontsize=7, framealpha=0.92)

    out = Path("paper/figs/figA2_budget_scaling.pdf")
    fig.savefig(out, bbox_inches="tight", dpi=200)
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=200)
    print(f"[figA2] wrote {out}")


if __name__ == "__main__":
    main()
