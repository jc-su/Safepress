#!/usr/bin/env python3
"""Generate all paper figures from SafePress experiment results.

This script uses data from:
  - 3-bit SSMP experiments (Qwen3, Llama, Yi, Phi)
  - 4-bit SSMP experiments (Qwen3, Llama, Yi, Phi)
  - Phase transition experiments (all 6 models)
  - Block score CSVs (all models)
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from safepress.viz.plots import (
    COLORS,
    plot_block_heatmap,
    plot_budget_sweep,
    plot_causal_experiment,
    plot_cross_model_comparison,
    plot_phase_transition,
    set_paper_style,
)

set_paper_style()

RUNS = Path("runs")
OUT = Path("figures")
OUT.mkdir(parents=True, exist_ok=True)

# All models with their display names and directory suffixes
MODELS = [
    ("Qwen3-8B", "qwen3"),
    ("Llama-3.1-8B", "llama31"),
    ("Yi-1.5-9B", "yi15"),
    ("Phi-3.5-mini", "phi35"),
]

# Models for phase transition (includes low-baseline models as controls)
ALL_PHASE_MODELS = MODELS + [
    ("Mistral-7B", "mistral7b"),
]

SCORE_FILES = {
    "Qwen3-8B": RUNS / "scores" / "qwen_qwen3-8b_scores.csv",
    "Llama-3.1-8B": RUNS / "scores" / "meta-llama_llama-3.1-8b-instruct_scores.csv",
    "Yi-1.5-9B": RUNS / "scores" / "01-ai_yi-1.5-9b-chat_scores.csv",
    "Phi-3.5-mini": RUNS / "scores" / "microsoft_phi-3.5-mini-instruct_scores.csv",
    "Mistral-7B": RUNS / "scores" / "mistralai_mistral-7b-instruct-v0.3_scores.csv",
}

# Model-specific line styles for multi-model plots
MODEL_STYLES = {
    "Qwen3-8B": {"color": "#1f77b4", "marker": "o"},
    "Llama-3.1-8B": {"color": "#ff7f0e", "marker": "s"},
    "Yi-1.5-9B": {"color": "#2ca02c", "marker": "^"},
    "Phi-3.5-mini": {"color": "#d62728", "marker": "D"},
    "Mistral-7B": {"color": "#9467bd", "marker": "v"},
}


def load_json(path: Path) -> dict:
    """Load JSON file, handling NaN values."""
    with open(path) as f:
        content = f.read()
    content = content.replace("NaN", "null")
    return json.loads(content)


def safe_name(name: str) -> str:
    return name.lower().replace(".", "").replace("-", "_").replace(" ", "_")


# -------------------------------------------------------------------------
# Figure 1: Block importance heatmaps (all models)
# -------------------------------------------------------------------------
def gen_heatmaps():
    print("\n=== Figure 1: Block importance heatmaps ===")
    for name, _ in MODELS:
        csv_path = SCORE_FILES.get(name)
        if csv_path is None or not csv_path.exists():
            print(f"  [skip] {name}: scores not found")
            continue
        df = pd.read_csv(csv_path)
        out_path = OUT / f"heatmap_{safe_name(name)}.pdf"
        print(f"  Generating heatmap for {name} -> {out_path}")
        fig = plot_block_heatmap(df, model_name=name, save_path=out_path)
        plt.close(fig)


# -------------------------------------------------------------------------
# Figure 2: 3-bit SSMP budget sweep (refusal rate vs budget, all models)
# -------------------------------------------------------------------------
def gen_budget_sweep():
    print("\n=== Figure 2: 3-bit SSMP budget sweep ===")

    # Individual per-model plots
    for name, suffix in MODELS:
        results_path = RUNS / f"3bit_ssmp_{suffix}" / "ssmp_3bit_results.json"
        if not results_path.exists():
            print(f"  [skip] {results_path} not found")
            continue

        data = load_json(results_path)
        conds = data["conditions"]

        fp16_rr = conds["fp16_baseline"]["refusal_rate"]
        uniform_rr = conds["uniform_3bit"]["refusal_rate"]

        budgets = []
        refusal_rates = []
        for key, val in conds.items():
            if key.startswith("ssmp_"):
                b = val.get("budget")
                if b is not None:
                    budgets.append(b)
                    refusal_rates.append(val["refusal_rate"])

        if not budgets:
            continue

        pairs = sorted(zip(budgets, refusal_rates))
        budgets = [p[0] for p in pairs]
        refusal_rates = [p[1] for p in pairs]

        out_path = OUT / f"budget_sweep_{safe_name(name)}.pdf"
        print(f"  Generating budget sweep for {name} -> {out_path}")
        fig = plot_budget_sweep(
            budgets, refusal_rates,
            baseline_refusal=fp16_rr,
            fullquant_refusal=uniform_rr,
            save_path=out_path,
        )
        fig.axes[0].set_title(f"3-bit SSMP budget sweep ({name})")
        try:
            fig.axes[0].get_legend().texts[-1].set_text(f"Uniform 3-bit ({uniform_rr:.2f})")
        except (AttributeError, IndexError):
            pass
        fig.savefig(str(out_path), dpi=300, bbox_inches="tight")
        plt.close(fig)

    # Combined multi-model budget sweep
    _gen_combined_budget_sweep("3bit_ssmp", 3)
    _gen_combined_budget_sweep("4bit_ssmp", 4)


def _gen_combined_budget_sweep(prefix: str, bits: int):
    """Generate a combined budget sweep plot with all models on one figure."""
    fig, ax = plt.subplots(figsize=(6, 4))
    any_data = False

    for name, suffix in MODELS:
        results_path = RUNS / f"{prefix}_{suffix}" / f"ssmp_{bits}bit_results.json"
        if not results_path.exists():
            continue

        data = load_json(results_path)
        conds = data["conditions"]
        fp16_rr = conds["fp16_baseline"]["refusal_rate"]
        if fp16_rr < 0.05:
            continue  # skip models with negligible baseline

        budgets = []
        refusal_rates = []
        for key, val in conds.items():
            if key.startswith("ssmp_"):
                b = val.get("budget")
                if b is not None:
                    budgets.append(b)
                    refusal_rates.append(val["refusal_rate"])

        if not budgets:
            continue

        pairs = sorted(zip(budgets, refusal_rates))
        budgets_sorted = [p[0] for p in pairs]
        rr_sorted = [p[1] for p in pairs]

        # Normalize: fraction of FP16 baseline recovered
        uniform_rr = conds["uniform_3bit"]["refusal_rate"]
        gap = fp16_rr - uniform_rr
        if gap > 0.01:
            recovery = [(r - uniform_rr) / gap for r in rr_sorted]
        else:
            recovery = rr_sorted

        style = MODEL_STYLES.get(name, {"color": "gray", "marker": "o"})
        ax.plot(
            [b * 100 for b in budgets_sorted], recovery,
            f"{style['marker']}-", color=style["color"],
            linewidth=2, markersize=7, label=name, zorder=3,
        )
        any_data = True

    if not any_data:
        plt.close(fig)
        return

    ax.axhline(0, color="gray", linestyle=":", linewidth=0.8, alpha=0.5, label="Uniform (no protection)")
    ax.axhline(1, color="gray", linestyle="--", linewidth=0.8, alpha=0.5, label="FP16 baseline")
    ax.set_xlabel("SSMP budget (%)")
    ax.set_ylabel("Safety recovery fraction")
    ax.set_title(f"{bits}-bit SSMP: safety recovery vs protection budget")
    ax.legend(fontsize=8, frameon=True, framealpha=0.9)
    ax.set_ylim(-0.1, 1.3)
    fig.tight_layout()

    out_path = OUT / f"combined_budget_sweep_{bits}bit.pdf"
    print(f"  Generating combined {bits}-bit budget sweep -> {out_path}")
    fig.savefig(str(out_path), dpi=300, bbox_inches="tight")
    plt.close(fig)


# -------------------------------------------------------------------------
# Figure 3: 3-bit SSMP ablation bar chart (SSMP vs random vs inverted)
# -------------------------------------------------------------------------
def gen_ablation_bars():
    print("\n=== Figure 3: 3-bit SSMP ablation bar charts ===")
    for name, suffix in MODELS:
        results_path = RUNS / f"3bit_ssmp_{suffix}" / "ssmp_3bit_results.json"
        if not results_path.exists():
            print(f"  [skip] {results_path} not found")
            continue

        data = load_json(results_path)
        conds = data["conditions"]

        results = {}
        label_map = {
            "fp16_baseline": "FP16",
            "uniform_3bit": "Uniform 3-bit",
        }

        for key, val in conds.items():
            if key in label_map:
                results[label_map[key]] = val["refusal_rate"]
            elif key.startswith("ssmp_"):
                budget = val.get("budget", 0)
                results[f"SSMP@{budget*100:.0f}%"] = val["refusal_rate"]
            elif key.startswith("random_"):
                results["Random@4%"] = val["refusal_rate"]
            elif key.startswith("inverted_"):
                results["Inverted@4%"] = val["refusal_rate"]

        out_path = OUT / f"ablation_{safe_name(name)}.pdf"
        print(f"  Generating ablation bars for {name} -> {out_path}")

        fig, ax = plt.subplots(figsize=(8, 4.5))

        ordered_keys = list(results.keys())
        vals = [results[k] for k in ordered_keys]

        bar_colors = []
        for k in ordered_keys:
            if k == "FP16":
                bar_colors.append(COLORS["fp16"])
            elif "Uniform" in k:
                bar_colors.append(COLORS["full_quant"])
            elif "SSMP" in k:
                bar_colors.append(COLORS["ssmp"])
            elif "Random" in k:
                bar_colors.append(COLORS["random"])
            elif "Inverted" in k:
                bar_colors.append(COLORS["magnitude"])
            else:
                bar_colors.append("#607D8B")

        x = np.arange(len(ordered_keys))
        bars = ax.bar(x, vals, 0.6, color=bar_colors, edgecolor="white", linewidth=0.5, zorder=3)

        for bar_obj, v in zip(bars, vals):
            ax.text(
                bar_obj.get_x() + bar_obj.get_width() / 2,
                bar_obj.get_height() + 0.01,
                f"{v:.1%}",
                ha="center", va="bottom", fontsize=9,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(ordered_keys, rotation=30, ha="right")
        ax.set_ylabel("Refusal rate")
        ax.set_ylim(0, max(vals) * 1.25 if vals else 1.0)
        ax.set_title(f"3-bit quantization: SSMP vs ablations ({name})")
        fig.tight_layout()
        fig.savefig(str(out_path), dpi=300, bbox_inches="tight")
        plt.close(fig)


# -------------------------------------------------------------------------
# Figure 4: Phase transition curves (individual + combined)
# -------------------------------------------------------------------------
def _parse_phase_transition(phase_dir: Path):
    """Parse phase transition results from a directory, returns (bit_widths, refusal_rates)."""
    results_path = phase_dir / "phase_transition_results.json"
    if not results_path.exists():
        results_path = phase_dir / "phase_transition.json"
    if not results_path.exists():
        return None, None

    data = load_json(results_path)
    results = data.get("results", data)
    bit_widths = []
    safety_scores = []
    if isinstance(results, list):
        bit_widths = [r.get("bits", r.get("bit_width")) for r in results]
        safety_scores = [r["refusal_rate"] for r in results]
    elif isinstance(results, dict):
        for k, v in results.items():
            if isinstance(v, dict) and "refusal_rate" in v:
                bw = v.get("bits")
                if bw is None:
                    try:
                        bw = int(k.replace("bits_", "").replace("bit", ""))
                    except ValueError:
                        continue
                bit_widths.append(bw)
                safety_scores.append(v["refusal_rate"])

    if not bit_widths:
        return None, None

    pairs = sorted(zip(bit_widths, safety_scores), reverse=True)
    return [p[0] for p in pairs], [p[1] for p in pairs]


def gen_phase_transition():
    print("\n=== Figure 4: Phase transition curves ===")

    # Individual plots
    for name, suffix in ALL_PHASE_MODELS:
        phase_dir = RUNS / f"phase_transition_{suffix}"
        bws, rrs = _parse_phase_transition(phase_dir)
        if bws is None:
            print(f"  [skip] {name}: no phase transition data")
            continue

        out_path = OUT / f"phase_transition_{safe_name(name)}.pdf"
        print(f"  Generating phase transition for {name} -> {out_path}")

        fig, ax = plt.subplots(figsize=(5.5, 3.8))
        ax.plot(bws, rrs, "o-", color=COLORS["ssmp"], linewidth=2, markersize=7, zorder=3)

        for bw, ss in zip(bws, rrs):
            ax.annotate(f"{ss:.1%}", (bw, ss), textcoords="offset points",
                        xytext=(0, 10), ha="center", fontsize=9)

        ax.axhline(rrs[0], color=COLORS["fp16"], linestyle=":", linewidth=0.9, alpha=0.5,
                    label=f"FP16 ({rrs[0]:.1%})")
        ax.invert_xaxis()
        ax.set_xticks(bws)
        ax.set_xticklabels([f"{b}-bit" if b < 16 else "FP16" for b in bws])
        ax.set_xlabel("Precision")
        ax.set_ylabel("Refusal rate (safety)")
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(f"Safety phase transition ({name})")
        ax.legend(frameon=True, framealpha=0.9)
        fig.tight_layout()
        fig.savefig(str(out_path), dpi=300, bbox_inches="tight")
        plt.close(fig)

    # Combined multi-model phase transition
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, suffix in ALL_PHASE_MODELS:
        phase_dir = RUNS / f"phase_transition_{suffix}"
        bws, rrs = _parse_phase_transition(phase_dir)
        if bws is None:
            continue
        style = MODEL_STYLES.get(name, {"color": "gray", "marker": "o"})
        ax.plot(bws, rrs, f"{style['marker']}-", color=style["color"],
                linewidth=2, markersize=6, label=name, zorder=3)

    ax.invert_xaxis()
    ax.set_xticks([16, 8, 4, 3, 2])
    ax.set_xticklabels(["FP16", "8-bit", "4-bit", "3-bit", "2-bit"])
    ax.set_xlabel("Precision")
    ax.set_ylabel("Refusal rate (safety)")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Safety phase transition under quantization (all models)")
    ax.legend(fontsize=8, frameon=True, framealpha=0.9)
    fig.tight_layout()
    out_path = OUT / "phase_transition_combined.pdf"
    print(f"  Generating combined phase transition -> {out_path}")
    fig.savefig(str(out_path), dpi=300, bbox_inches="tight")
    plt.close(fig)


# -------------------------------------------------------------------------
# Figure 5: Cross-model comparison (combined 3-bit SSMP)
# -------------------------------------------------------------------------
def _parse_ssmp_conditions(conds: dict) -> list:
    """Parse SSMP conditions into rows with human-readable labels."""
    rows = []
    for key, val in conds.items():
        label = key
        if key == "fp16_baseline":
            label = "FP16"
        elif key == "uniform_3bit":
            label = "Uniform 3-bit"
        elif key.startswith("ssmp_") and "b" in key:
            budget = val.get("budget", 0)
            label = f"SSMP@{budget*100:.0f}%"
        elif key.startswith("random_"):
            label = "Random@4%"
        elif key.startswith("inverted_"):
            label = "Inverted@4%"
        rows.append({"method": label, "refusal_rate": val["refusal_rate"]})
    return rows


def gen_cross_model():
    print("\n=== Figure 5: Cross-model comparison ===")
    rows = []
    for name, suffix in MODELS:
        results_path = RUNS / f"3bit_ssmp_{suffix}" / "ssmp_3bit_results.json"
        if not results_path.exists():
            continue
        data = load_json(results_path)
        conds = data["conditions"]
        for parsed in _parse_ssmp_conditions(conds):
            parsed["model"] = name
            rows.append(parsed)

    if len(rows) == 0:
        print("  [skip] No cross-model data available")
        return

    df = pd.DataFrame(rows)
    key_methods = ["FP16", "Uniform 3-bit", "SSMP@4%", "SSMP@8%", "Random@4%"]
    df_filtered = df[df["method"].isin(key_methods)]

    n_models = len(df_filtered["model"].unique())
    if n_models < 2:
        print("  [skip] Need at least 2 models for cross-model comparison")
        return

    out_path = OUT / "cross_model_3bit.pdf"
    print(f"  Generating cross-model comparison ({n_models} models) -> {out_path}")

    # Custom grouped bar chart for more flexibility
    methods = key_methods
    models = [m for m, _ in MODELS if m in df_filtered["model"].unique()]
    n_methods = len(methods)
    n_models = len(models)

    fig, ax = plt.subplots(figsize=(10, 5))
    width = 0.8 / n_models
    x = np.arange(n_methods)

    for i, model in enumerate(models):
        vals = []
        for method in methods:
            subset = df_filtered[(df_filtered["model"] == model) & (df_filtered["method"] == method)]
            vals.append(subset["refusal_rate"].values[0] if len(subset) > 0 else 0)
        style = MODEL_STYLES.get(model, {"color": "gray"})
        offset = (i - n_models / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=model, color=style["color"],
                       edgecolor="white", linewidth=0.5, zorder=3)
        for bar_obj, v in zip(bars, vals):
            if v > 0.01:
                ax.text(bar_obj.get_x() + bar_obj.get_width() / 2, bar_obj.get_height() + 0.01,
                        f"{v:.0%}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=15, ha="right")
    ax.set_ylabel("Refusal rate")
    ax.set_title("3-bit quantization: cross-model comparison")
    ax.legend(fontsize=8, frameon=True, framealpha=0.9)
    ax.set_ylim(0, 1.15)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=300, bbox_inches="tight")
    plt.close(fig)


# -------------------------------------------------------------------------
# Figure 6: 4-bit SSMP comparison
# -------------------------------------------------------------------------
def gen_4bit_ssmp():
    print("\n=== Figure 6: 4-bit SSMP results ===")
    rows = []
    for name, suffix in MODELS:
        results_path = RUNS / f"4bit_ssmp_{suffix}" / "ssmp_4bit_results.json"
        if not results_path.exists():
            continue
        data = load_json(results_path)
        conds = data["conditions"]
        for parsed in _parse_ssmp_conditions(conds):
            parsed["model"] = name
            # Fix mislabeled "uniform_3bit" key in 4-bit results
            if parsed["method"] == "Uniform 3-bit":
                parsed["method"] = "Uniform 4-bit"
            rows.append(parsed)

    if not rows:
        print("  [skip] No 4-bit SSMP data")
        return

    df = pd.DataFrame(rows)
    key_methods = ["FP16", "Uniform 4-bit", "SSMP@4%", "SSMP@8%", "Random@4%"]
    df_filtered = df[df["method"].isin(key_methods)]
    models = [m for m, _ in MODELS if m in df_filtered["model"].unique()]

    if len(models) < 2:
        print("  [skip] Need at least 2 models for 4-bit comparison")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    methods = [m for m in key_methods if m in df_filtered["method"].values]
    n_methods = len(methods)
    n_models = len(models)
    width = 0.8 / n_models
    x = np.arange(n_methods)

    for i, model in enumerate(models):
        vals = []
        for method in methods:
            subset = df_filtered[(df_filtered["model"] == model) & (df_filtered["method"] == method)]
            vals.append(subset["refusal_rate"].values[0] if len(subset) > 0 else 0)
        style = MODEL_STYLES.get(model, {"color": "gray"})
        offset = (i - n_models / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=model, color=style["color"],
                       edgecolor="white", linewidth=0.5, zorder=3)
        for bar_obj, v in zip(bars, vals):
            if v > 0.01:
                ax.text(bar_obj.get_x() + bar_obj.get_width() / 2, bar_obj.get_height() + 0.01,
                        f"{v:.0%}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=15, ha="right")
    ax.set_ylabel("Refusal rate")
    ax.set_title("4-bit quantization: cross-model comparison")
    ax.legend(fontsize=8, frameon=True, framealpha=0.9)
    ax.set_ylim(0, 1.15)
    fig.tight_layout()

    out_path = OUT / "cross_model_4bit.pdf"
    print(f"  Generating 4-bit cross-model comparison ({len(models)} models) -> {out_path}")
    fig.savefig(str(out_path), dpi=300, bbox_inches="tight")
    plt.close(fig)


# -------------------------------------------------------------------------
# Figure 7: Utility retention
# -------------------------------------------------------------------------
def gen_utility():
    print("\n=== Figure 7: Utility retention ===")
    # Collect utility data
    model_data = {}
    for name, suffix in MODELS:
        results_path = RUNS / f"utility_{suffix}" / "utility_results.json"
        if not results_path.exists():
            continue
        data = load_json(results_path)
        model_data[name] = data

    if not model_data:
        print("  [skip] No utility data")
        return

    # Figure 7a: Perplexity comparison (grouped bars)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    conditions = [("fp16", "FP16"), ("uniform_3bit", "Uniform 3-bit"), ("ssmp_3bit_b0.04", "SSMP@4%")]
    models = list(model_data.keys())
    n_models = len(models)
    n_conds = len(conditions)

    # 7a: Perplexity
    ax = axes[0]
    width = 0.8 / n_conds
    x = np.arange(n_models)
    cond_colors = [COLORS.get("fp16", "#4477AA"), COLORS.get("uniform", "#CC6677"), COLORS.get("ssmp", "#228833")]
    for j, (cond_key, cond_label) in enumerate(conditions):
        vals = []
        for m in models:
            d = model_data[m]
            ppl = d.get(cond_key, {}).get("perplexity", {}).get("perplexity", 0)
            vals.append(ppl)
        offset = (j - n_conds / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=cond_label, color=cond_colors[j],
                       edgecolor="white", linewidth=0.5, zorder=3)
        for bar_obj, v in zip(bars, vals):
            if v > 0:
                ax.text(bar_obj.get_x() + bar_obj.get_width() / 2, bar_obj.get_height() + 0.3,
                        f"{v:.1f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right", fontsize=8)
    ax.set_ylabel("Perplexity (lower is better)")
    ax.set_title("Perplexity (WikiText-2)")
    ax.legend(fontsize=7, frameon=True, framealpha=0.9)

    # 7b: MMLU
    ax = axes[1]
    for j, (cond_key, cond_label) in enumerate(conditions):
        vals = []
        for m in models:
            d = model_data[m]
            mmlu = d.get(cond_key, {}).get("mmlu", {}).get("accuracy", 0)
            vals.append(mmlu)
        offset = (j - n_conds / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=cond_label, color=cond_colors[j],
                       edgecolor="white", linewidth=0.5, zorder=3)
        for bar_obj, v in zip(bars, vals):
            if v > 0:
                ax.text(bar_obj.get_x() + bar_obj.get_width() / 2, bar_obj.get_height() + 0.005,
                        f"{v:.2f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right", fontsize=8)
    ax.set_ylabel("Accuracy (higher is better)")
    ax.set_title("MMLU-lite (200 questions)")
    ax.legend(fontsize=7, frameon=True, framealpha=0.9)
    ax.set_ylim(0, 0.8)

    # 7c: TruthfulQA
    ax = axes[2]
    for j, (cond_key, cond_label) in enumerate(conditions):
        vals = []
        for m in models:
            d = model_data[m]
            tqa = d.get(cond_key, {}).get("truthfulqa", {}).get("accuracy", 0)
            vals.append(tqa)
        offset = (j - n_conds / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=cond_label, color=cond_colors[j],
                       edgecolor="white", linewidth=0.5, zorder=3)
        for bar_obj, v in zip(bars, vals):
            if v > 0:
                ax.text(bar_obj.get_x() + bar_obj.get_width() / 2, bar_obj.get_height() + 0.005,
                        f"{v:.2f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right", fontsize=8)
    ax.set_ylabel("Accuracy (higher is better)")
    ax.set_title("TruthfulQA-lite (100 questions)")
    ax.legend(fontsize=7, frameon=True, framealpha=0.9)
    ax.set_ylim(0, 0.5)

    fig.suptitle("Utility retention under 3-bit quantization with SSMP@4%", fontsize=12, y=1.02)
    fig.tight_layout()
    out_path = OUT / "utility_retention.pdf"
    print(f"  Generating utility retention figure -> {out_path}")
    fig.savefig(str(out_path), dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Figure 7d: Safety vs. Utility scatter
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, suffix in MODELS:
        if name not in model_data:
            continue
        d = model_data[name]
        style = MODEL_STYLES.get(name, {"color": "gray", "marker": "o"})

        for cond_key, cond_label, ms in [
            ("fp16", "FP16", 10),
            ("uniform_3bit", "Uniform 3-bit", 7),
            ("ssmp_3bit_b0.04", "SSMP@4%", 9),
        ]:
            if cond_key not in d:
                continue
            mmlu = d[cond_key].get("mmlu", {}).get("accuracy", 0)

            # Get safety refusal rate from SSMP experiments
            ssmp_path = RUNS / f"3bit_ssmp_{suffix}" / "ssmp_3bit_results.json"
            rr = 0
            if ssmp_path.exists():
                ssmp_data = load_json(ssmp_path)
                conds = ssmp_data.get("conditions", {})
                if cond_key == "fp16":
                    rr = conds.get("fp16_baseline", {}).get("refusal_rate", 0)
                elif cond_key == "uniform_3bit":
                    rr = conds.get("uniform_3bit", {}).get("refusal_rate", 0)
                elif cond_key == "ssmp_3bit_b0.04":
                    for k, v in conds.items():
                        if k.startswith("ssmp_") and abs(v.get("budget", 0) - 0.04) < 0.001:
                            rr = v["refusal_rate"]
                            break

            marker = style["marker"]
            if cond_key == "uniform_3bit":
                marker = "x"
            elif cond_key == "ssmp_3bit_b0.04":
                marker = style["marker"]

            label = f"{name} ({cond_label})" if name == list(model_data.keys())[0] or cond_key == "fp16" else None
            ax.scatter(mmlu, rr, color=style["color"], marker=marker, s=ms**2,
                       zorder=3, edgecolors="black", linewidth=0.5)
            ax.annotate(f"{name[:4]}", (mmlu, rr), textcoords="offset points",
                        xytext=(5, 5), fontsize=6, alpha=0.7)

    # Legend for condition markers
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="gray", markersize=8, label="FP16"),
        Line2D([0], [0], marker="x", color="gray", markersize=8, label="Uniform 3-bit", linestyle="None"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="gray", markersize=8, label="SSMP@4%"),
    ]
    ax.legend(handles=legend_elements, fontsize=8, frameon=True, framealpha=0.9, loc="lower right")

    ax.set_xlabel("MMLU accuracy (utility)")
    ax.set_ylabel("Refusal rate (safety)")
    ax.set_title("Safety vs. Utility tradeoff under 3-bit quantization")
    ax.set_xlim(0.15, 0.7)
    ax.set_ylim(-0.05, 1.0)
    fig.tight_layout()
    out_path = OUT / "safety_vs_utility.pdf"
    print(f"  Generating safety vs utility scatter -> {out_path}")
    fig.savefig(str(out_path), dpi=300, bbox_inches="tight")
    plt.close(fig)


# -------------------------------------------------------------------------
# Figure 8: Drift bound correlation
# -------------------------------------------------------------------------
def gen_drift_bounds():
    print("\n=== Figure 8: Drift bound correlation ===")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for idx, (name, suffix) in enumerate(MODELS):
        ax = axes[idx]
        drift_path = RUNS / f"drift_bounds_{suffix}.csv"
        scores_path = list(RUNS.glob(f"scores/*{suffix.replace('_', '*')}*scores.csv"))
        if not drift_path.exists():
            ax.text(0.5, 0.5, f"No drift data for {name}", ha="center", va="center")
            continue

        drift_df = pd.read_csv(drift_path)
        style = MODEL_STYLES.get(name, {"color": "#1f77b4", "marker": "o"})

        ax.scatter(drift_df["cs_bound"], drift_df["inner_prod"],
                   alpha=0.5, s=15, color=style["color"], edgecolors="none")

        # Highlight top-10 modules
        top10 = drift_df.nlargest(10, "cs_bound")
        ax.scatter(top10["cs_bound"], top10["inner_prod"],
                   alpha=0.8, s=40, color="red", edgecolors="black", linewidth=0.5, zorder=5)
        for _, row in top10.iterrows():
            short_name = row["module"].split(".")[-1][:15]
            ax.annotate(short_name, (row["cs_bound"], row["inner_prod"]),
                        textcoords="offset points", xytext=(3, 3), fontsize=5, alpha=0.7)

        ax.set_xlabel("Cauchy-Schwarz bound (|∇L|·|ΔW|)")
        ax.set_ylabel("Inner product (∇L·ΔW)")
        ax.set_title(f"{name}: Drift bounds per module")
        ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")

    fig.suptitle("Per-module safety drift bounds under 3-bit quantization", fontsize=12, y=1.01)
    fig.tight_layout()
    out_path = OUT / "drift_bounds.pdf"
    print(f"  Generating drift bounds figure -> {out_path}")
    fig.savefig(str(out_path), dpi=300, bbox_inches="tight")
    plt.close(fig)


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------
if __name__ == "__main__":
    gen_heatmaps()
    gen_budget_sweep()
    gen_ablation_bars()
    gen_phase_transition()
    gen_cross_model()
    gen_4bit_ssmp()
    gen_utility()
    gen_drift_bounds()
    print(f"\n[done] All figures saved to {OUT}/")
