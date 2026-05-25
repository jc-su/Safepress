#!/usr/bin/env python
"""Figure (T2, full-width 1x5): cluster + Fisher escape, one panel per model.

Each panel: x = budgets {4%, 8%}, y = GCG ASR. The three compound scorers
(SSMP, SNIP, magnitude) shown as a shaded range + mean; Fisher as a red
diamond; FP16 and INT4 as horizontal reference lines. Cross-column.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_CONDITION_RE = re.compile(r"sweep_[^/]+_(?P<method>ssmp|gradient_only|snip|magnitude)_b(?P<budget>[0-9.]+)$")

MODELS = [
    ("Llama-3.1", "runs/emnlp_g2_pilot_llama31"),
    ("Qwen3",     "runs/emnlp_g2_pilot"),
    ("GLM-4",     "runs/emnlp_g2_pilot_glm4"),
    ("Ministral", "runs/emnlp_g2_pilot_mistral"),
    ("DeepSeek",  "runs/emnlp_g2_pilot_deepseek"),
]
COMPOUND = ["ssmp", "snip", "magnitude"]


def load_gcg(root):
    out = {}
    for cd in sorted(Path(root).glob("sweep_*")):
        if not cd.is_dir(): continue
        m = _CONDITION_RE.match(cd.name)
        if not m: continue
        gp = cd / "jailbreak_gcg_seed0.json"
        if not gp.exists(): continue
        out[(m.group("method"), float(m.group("budget")))] = json.loads(gp.read_text()).get("asr")
    # fp16 / int4 references
    for ref in ["fp16", "int4"]:
        for cd in Path(root).glob(f"sweep_*_{ref}_b0.0"):
            gp = cd / "jailbreak_gcg_seed0.json"
            if gp.exists():
                out[(ref, 0.0)] = json.loads(gp.read_text()).get("asr")
    return out


def main():
    data = {model: load_gcg(root) for model, root in MODELS}
    fig, axes = plt.subplots(1, 5, figsize=(7.2, 1.78), sharey=True)

    budgets = [0.04, 0.08]
    xpos = [0, 1]
    for ax, (model, _) in zip(axes, MODELS):
        d = data[model]
        # FP16 / INT4 reference lines
        fp16, int4 = d.get(("fp16", 0.0)), d.get(("int4", 0.0))
        if fp16 is not None:
            ax.axhline(fp16, color="#444", ls="--", lw=0.9, alpha=0.7)
        if int4 is not None:
            ax.axhline(int4, color="#888", ls=":", lw=0.9, alpha=0.7)
        # per-budget cluster range + Fisher
        for x, b in zip(xpos, budgets):
            comp = [d.get((m, b)) for m in COMPOUND]
            comp = [v for v in comp if v is not None]
            if comp:
                lo, hi = min(comp), max(comp)
                ax.plot([x, x], [lo, hi], color="#1f77b4", lw=5, alpha=0.5, solid_capstyle="round", zorder=2)
                ax.plot(x, sum(comp)/len(comp), marker="_", color="#1f77b4", ms=11, mew=2, zorder=3)
            fisher = d.get(("gradient_only", b))
            if fisher is not None:
                ax.scatter(x, fisher, marker="D", s=40, c="#d62728", edgecolors="black", linewidths=0.6, zorder=4)
        ax.set_title(model, fontsize=7)
        ax.set_xticks(xpos); ax.set_xticklabels(["4%", "8%"], fontsize=6.5)
        ax.set_xlim(-0.4, 1.4); ax.set_ylim(-0.04, 1.08)
        ax.tick_params(labelsize=6)
        ax.grid(True, axis="y", alpha=0.25)

    # escape annotation on Llama panel
    fisher_l = data["Llama-3.1"].get(("gradient_only", 0.08))
    if fisher_l is not None:
        axes[0].annotate("Fisher\nescape", xy=(1, fisher_l), xytext=(0.1, fisher_l - 0.30),
                         fontsize=5.8, color="red",
                         arrowprops=dict(arrowstyle="->", color="red", lw=0.7))

    axes[0].set_ylabel("GCG ASR", fontsize=7.5)
    fig.supxlabel("protection budget", fontsize=7.5, y=0.07)
    handles = [
        plt.Line2D([0], [0], color="#1f77b4", lw=5, alpha=0.5, label="compound range"),
        plt.Line2D([0], [0], marker="D", linestyle="", c="#d62728", markeredgecolor="black", markersize=6, label="Fisher"),
        plt.Line2D([0], [0], color="#444", ls="--", lw=0.9, label="FP16"),
        plt.Line2D([0], [0], color="#888", ls=":", lw=0.9, label="INT4"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=6.5,
               bbox_to_anchor=(0.5, 0.99), framealpha=0.9, handletextpad=0.4, columnspacing=1.2)
    fig.subplots_adjust(left=0.06, right=0.995, top=0.80, bottom=0.24, wspace=0.12)
    out = Path("paper/figs/fig2_t2_cluster.pdf")
    fig.savefig(out, bbox_inches="tight", dpi=200)
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=200)
    print(f"[fig2] wrote {out}")


if __name__ == "__main__":
    main()
