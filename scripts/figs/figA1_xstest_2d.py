#!/usr/bin/env python
"""Appendix Figure A1: XSTest 2D map — safe_compliance vs unsafe_refusal.

Top-right corner = ideal (refuses unsafe, complies with safe).
Top-left = under-refuses (complies with unsafe = bad).
Bottom-right = over-refuses (refuses safe = annoying).
Bottom-left = broken.
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
    "Llama-3.1-8B":    [Path("runs/emnlp_g2_pilot_llama31"),  Path("runs/emnlp_fisher60/llama31")],
    "Qwen3-8B":        [Path("runs/emnlp_g2_pilot"),          Path("runs/emnlp_fisher60/qwen3")],
    "GLM-4-9B":        [Path("runs/emnlp_g2_pilot_glm4"),     Path("runs/emnlp_fisher60/glm4")],
    "Ministral-8B":    [Path("runs/emnlp_g2_pilot_mistral"),  Path("runs/emnlp_fisher60/mistral")],
    "DeepSeek-R1-8B":  [Path("runs/emnlp_g2_pilot_deepseek"), Path("runs/emnlp_fisher60/deepseek")],
}

MODEL_COLORS = {
    "Llama-3.1-8B":   "#1f77b4",
    "Qwen3-8B":       "#ff7f0e",
    "GLM-4-9B":       "#2ca02c",
    "Ministral-8B":   "#d62728",
    "DeepSeek-R1-8B": "#9467bd",
}

METHOD_MARKERS = {
    "fp16":          ("*", 220),
    "int4":          ("v", 130),
    "ssmp":          ("o", 80),
    "snip":          ("o", 80),
    "magnitude":     ("o", 80),
    "gradient_only": ("D", 100),
}


def load_data():
    rows = []
    for model, dirs in MODEL_DIRS.items():
        for d in dirs:
            if not d.exists():
                continue
            for cd in sorted(d.glob("sweep_*")):
                if not cd.is_dir(): continue
                m = _CONDITION_RE.match(cd.name)
                if not m: continue
                xp = cd / "xstest_seed0.json"
                if not xp.exists(): continue
                x = json.loads(xp.read_text())
                rows.append(dict(
                    model=model,
                    method=m.group("method"),
                    budget=float(m.group("budget")),
                    safe_compliance=x.get("safe_compliance_rate"),
                    unsafe_refusal=x.get("unsafe_refusal_rate"),
                ))
    return rows


def main():
    rows = load_data()
    print(f"[figA1] {len(rows)} XSTest conditions")

    fig, ax = plt.subplots(figsize=(7.5, 6))

    # Quadrant guides
    ax.axhline(0.5, color="grey", linestyle=":", alpha=0.5, zorder=1)
    ax.axvline(0.5, color="grey", linestyle=":", alpha=0.5, zorder=1)

    # Quadrant labels
    ax.text(0.97, 0.97, "IDEAL\n(refuses unsafe,\ncomplies with safe)",
            ha="right", va="top", fontsize=8.5, transform=ax.transAxes,
            color="#2ca02c",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#e8f5e9", edgecolor="#2ca02c", lw=0.5, alpha=0.7))
    ax.text(0.03, 0.97, "UNDER-REFUSES\n(complies with\nunsafe)",
            ha="left", va="top", fontsize=8.5, transform=ax.transAxes,
            color="#d62728",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#ffebee", edgecolor="#d62728", lw=0.5, alpha=0.7))
    ax.text(0.97, 0.03, "OVER-REFUSES\n(refuses safe)",
            ha="right", va="bottom", fontsize=8.5, transform=ax.transAxes,
            color="#ff9800",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#fff3e0", edgecolor="#ff9800", lw=0.5, alpha=0.7))
    ax.text(0.03, 0.03, "BROKEN",
            ha="left", va="bottom", fontsize=8.5, transform=ax.transAxes,
            color="#999",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#f5f5f5", edgecolor="#999", lw=0.5, alpha=0.7))

    for r in rows:
        color = MODEL_COLORS[r["model"]]
        marker, size = METHOD_MARKERS.get(r["method"], ("o", 70))
        edge = "black" if r["method"] == "gradient_only" else color
        lw = 1.2 if r["method"] == "gradient_only" else 0.3
        ax.scatter(r["safe_compliance"], r["unsafe_refusal"],
                   c=color, marker=marker, s=size,
                   edgecolors=edge, linewidths=lw, alpha=0.85, zorder=3)

    ax.set_xlabel("Safe compliance rate $\\to$ (higher: complies with safe-looking-unsafe queries)", fontsize=10)
    ax.set_ylabel("Unsafe refusal rate $\\to$ (higher: refuses genuinely unsafe queries)", fontsize=10)
    ax.set_title("XSTest 2D structure: refusal balance across conditions", fontsize=11)
    ax.set_xlim(-0.04, 1.04)
    ax.set_ylim(-0.04, 1.04)
    ax.grid(True, alpha=0.3)

    model_handles = [mpatches.Patch(color=c, label=m) for m, c in MODEL_COLORS.items()]
    method_handles = [
        plt.Line2D([0], [0], marker="*", linestyle="", color="grey", markersize=13, label="FP16"),
        plt.Line2D([0], [0], marker="v", linestyle="", color="grey", markersize=10, label="INT4 (no protect)"),
        plt.Line2D([0], [0], marker="o", linestyle="", color="grey", markersize=8, label="SSMP/SNIP/magnitude"),
        plt.Line2D([0], [0], marker="D", linestyle="", color="grey", markersize=9,
                   markeredgecolor="black", markeredgewidth=1.2, label="Fisher"),
    ]
    # Place both legends OUTSIDE the plot area (below) so they never cover data.
    leg1 = ax.legend(handles=model_handles, loc="upper center", bbox_to_anchor=(0.25, -0.12),
                     fontsize=7.5, title="Model", title_fontsize=8, framealpha=0.92, ncol=1)
    ax.add_artist(leg1)
    ax.legend(handles=method_handles, loc="upper center", bbox_to_anchor=(0.75, -0.12),
              fontsize=7.5, title="Method", title_fontsize=8, framealpha=0.92, ncol=1)

    out = Path("paper/figs/figA1_xstest_2d.pdf")
    fig.savefig(out, bbox_inches="tight", dpi=200)
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=200)
    print(f"[figA1] wrote {out}")


if __name__ == "__main__":
    main()
