#!/usr/bin/env python
"""Figure 3: Theorem 3 decoupling — refusal_rate vs GCG ASR scatter.

Each point is one (model, scorer, budget) condition. If T3 didn't hold,
points would cluster along the anti-diagonal (high refusal → low ASR).
Instead, protected conditions sit substantially above the diagonal,
demonstrating that benign refusal-rate recovery is decoupled from
adversarial robustness.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

_CONDITION_RE = re.compile(r"sweep_[^/]+_(?P<method>fp16|int4|ssmp|gradient_only|snip|magnitude)_b(?P<budget>[0-9.]+)$")

MODEL_DIRS = {
    "Llama-3.1-8B":          [Path("runs/emnlp_g2_pilot_llama31"),  Path("runs/emnlp_fisher60/llama31")],
    "Qwen3-8B":              [Path("runs/emnlp_g2_pilot"),          Path("runs/emnlp_fisher60/qwen3")],
    "GLM-4-9B":              [Path("runs/emnlp_g2_pilot_glm4"),     Path("runs/emnlp_fisher60/glm4")],
    "Ministral-8B":          [Path("runs/emnlp_g2_pilot_mistral"),  Path("runs/emnlp_fisher60/mistral")],
    "DeepSeek-R1-8B":        [Path("runs/emnlp_g2_pilot_deepseek"), Path("runs/emnlp_fisher60/deepseek")],
}

MODEL_COLORS = {
    "Llama-3.1-8B":   "#1f77b4",
    "Qwen3-8B":       "#ff7f0e",
    "GLM-4-9B":       "#2ca02c",
    "Ministral-8B":   "#d62728",
    "DeepSeek-R1-8B": "#9467bd",
}

METHOD_MARKERS = {
    "fp16":          ("*", 115),  # star
    "int4":          ("v", 62),   # downward triangle
    "ssmp":          ("o", 36),
    "snip":          ("o", 36),
    "magnitude":     ("o", 36),
    "gradient_only": ("D", 54),   # diamond, highlight Fisher
}


def load_conditions():
    rows = []
    for model, dirs in MODEL_DIRS.items():
        for d in dirs:
            if not d.exists():
                continue
            for cond_dir in sorted(d.glob("sweep_*")):
                if not cond_dir.is_dir():
                    continue
                m = _CONDITION_RE.match(cond_dir.name)
                if not m:
                    continue
                eval_p = cond_dir / "eval_seed0.json"
                gcg_p = cond_dir / "jailbreak_gcg_seed0.json"
                if not (eval_p.exists() and gcg_p.exists()):
                    continue
                e = json.loads(eval_p.read_text())
                g = json.loads(gcg_p.read_text())
                rows.append(dict(
                    model=model,
                    method=m.group("method"),
                    budget=float(m.group("budget")),
                    refusal=e.get("refusal_rate"),
                    gcg=g.get("asr"),
                ))
    return rows


def main():
    rows = load_conditions()
    print(f"[fig3] {len(rows)} conditions with both refusal+GCG data")

    fig, ax = plt.subplots(figsize=(3.35, 3.4))  # single column EMNLP width

    # anti-diagonal reference (if refusal restoration implied GCG restoration)
    ax.plot([0, 1], [1, 0], color="grey", linestyle="--", linewidth=1, alpha=0.6,
            label="anti-diagonal (perfect anti-correlation)")

    # Each model gets a color
    for model, dirs in MODEL_DIRS.items():
        model_rows = [r for r in rows if r["model"] == model]
        if not model_rows:
            continue
        color = MODEL_COLORS[model]
        for r in model_rows:
            marker, size = METHOD_MARKERS.get(r["method"], ("o", 80))
            edge_color = "black" if r["method"] == "gradient_only" else color
            edge_w = 1.0 if r["method"] == "gradient_only" else 0.3
            ax.scatter(r["refusal"], r["gcg"], c=color, marker=marker,
                       s=size, edgecolors=edge_color, linewidths=edge_w,
                       alpha=0.8, zorder=3)

    # Annotate the Llama Fisher@60% point (the killer)
    llama_f60 = next((r for r in rows if r["model"] == "Llama-3.1-8B"
                      and r["method"] == "gradient_only" and r["budget"] == 0.6), None)
    if llama_f60:
        ax.annotate(
            "Llama F@60%\nR=0.75, G=0.44",
            xy=(llama_f60["refusal"], llama_f60["gcg"]),
            xytext=(0.32, 0.70),
            fontsize=6.5, ha="left",
            arrowprops=dict(arrowstyle="->", color="black", lw=0.6),
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="black", lw=0.4, alpha=0.95),
        )

    # Annotate Qwen3 Fisher@8% point (exhibit B)
    qwen_f8 = next((r for r in rows if r["model"] == "Qwen3-8B"
                    and r["method"] == "gradient_only" and r["budget"] == 0.08), None)
    if qwen_f8:
        ax.annotate(
            "Qwen3 F@8%\nR=0.46, G=0.87\n(12× ratio)",
            xy=(qwen_f8["refusal"], qwen_f8["gcg"]),
            xytext=(0.02, 0.62),
            fontsize=6.5, ha="left",
            arrowprops=dict(arrowstyle="->", color="black", lw=0.6),
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="black", lw=0.4, alpha=0.95),
        )

    # Legend: models as color patches
    model_handles = [mpatches.Patch(color=c, label=m) for m, c in MODEL_COLORS.items()]
    # Legend: methods as marker shapes
    method_handles = [
        plt.Line2D([0], [0], marker="*", linestyle="", color="grey", markersize=14, label="FP16"),
        plt.Line2D([0], [0], marker="v", linestyle="", color="grey", markersize=10, label="INT4 (no protect)"),
        plt.Line2D([0], [0], marker="o", linestyle="", color="grey", markersize=8, label="SSMP/SNIP/magnitude"),
        plt.Line2D([0], [0], marker="D", linestyle="", color="grey", markersize=9,
                   markeredgecolor="black", markeredgewidth=1.5, label="Fisher (gradient-only)"),
    ]

    leg1 = ax.legend(handles=model_handles, loc="lower right", fontsize=6.0,
                     title="Model", title_fontsize=6.5, framealpha=0.85,
                     handlelength=1.0, handletextpad=0.4)
    ax.add_artist(leg1)
    ax.legend(handles=method_handles, loc="upper right", fontsize=6.0,
              title="Method", title_fontsize=6.5, framealpha=0.85,
              handlelength=1.0, handletextpad=0.4)

    ax.set_xlabel("Refusal rate (benign HarmBench)", fontsize=8.5)
    ax.set_ylabel("GCG attack ASR (lower safer)", fontsize=8.5)
    ax.set_title("T3 decoupling: refusal $\\neq$ adversarial robustness", fontsize=8.5, pad=5)
    ax.tick_params(labelsize=7.5)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.05)
    ax.grid(True, alpha=0.3)

    out = Path("paper/figs/fig3_t3_decoupling.pdf")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", dpi=200)
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=200)
    print(f"[fig3] wrote {out} and .png")


if __name__ == "__main__":
    main()
