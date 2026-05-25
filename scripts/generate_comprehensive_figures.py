#!/usr/bin/env python3
"""Generate comprehensive multi-bit-width figures combining all SafePress results.

Produces:
  1. Combined phase transition (both models on one plot)
  2. Multi-bit SSMP comparison (4-bit + 3-bit side by side)
  3. Comprehensive ablation table (all conditions × all bit-widths × both models)
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from safepress.viz.plots import COLORS, set_paper_style
from safepress.viz.tables import results_to_latex, results_to_markdown

set_paper_style()

RUNS = Path("runs")
OUT = Path("figures")
TOUT = Path("tables")
OUT.mkdir(parents=True, exist_ok=True)
TOUT.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


# -------------------------------------------------------------------------
# Figure A: Combined phase transition (both models, one plot)
# -------------------------------------------------------------------------
def gen_combined_phase_transition():
    print("\n=== Combined Phase Transition ===")

    fig, ax = plt.subplots(figsize=(6.5, 4.2))

    model_styles = {
        "Qwen3-8B": {"color": "#E91E63", "marker": "o", "ls": "-"},
        "Llama-3.1-8B": {"color": "#3F51B5", "marker": "s", "ls": "--"},
    }

    for model_name, phase_dir in [
        ("Qwen3-8B", RUNS / "phase_transition_qwen3"),
        ("Llama-3.1-8B", RUNS / "phase_transition_llama31"),
    ]:
        results_path = phase_dir / "phase_transition_results.json"
        if not results_path.exists():
            print(f"  [skip] {results_path}")
            continue

        data = load_json(results_path)
        bit_widths = []
        safety_scores = []
        for k, v in data.get("results", {}).items():
            if isinstance(v, dict) and "refusal_rate" in v:
                bw = v.get("bits", int(k.replace("bits_", "")))
                bit_widths.append(bw)
                safety_scores.append(v["refusal_rate"])

        pairs = sorted(zip(bit_widths, safety_scores), reverse=True)
        bws = [p[0] for p in pairs]
        sss = [p[1] for p in pairs]

        style = model_styles[model_name]
        ax.plot(bws, sss, f"{style['marker']}{style['ls']}",
                color=style["color"], label=model_name,
                linewidth=2.0, markersize=7, zorder=3)

        # Annotate key points
        for bw, ss in zip(bws, sss):
            if bw in [3, 16]:
                ax.annotate(f"{ss:.1%}", (bw, ss),
                           textcoords="offset points", xytext=(8, 5),
                           fontsize=8, color=style["color"])

    # Shade the "danger zone"
    ax.axvspan(1.5, 3.5, alpha=0.08, color="red", label="Danger zone")
    ax.axvspan(3.5, 5, alpha=0.06, color="orange")

    ax.invert_xaxis()
    ax.set_xticks([16, 8, 4, 3, 2])
    ax.set_xticklabels(["FP16", "8-bit", "4-bit", "3-bit", "2-bit"])
    ax.set_xlabel("Quantization precision")
    ax.set_ylabel("Refusal rate (safety)")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Safety phase transition under quantization")
    ax.legend(frameon=True, framealpha=0.9, loc="center left")
    fig.tight_layout()

    out_path = OUT / "combined_phase_transition.pdf"
    fig.savefig(str(out_path), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out_path}")


# -------------------------------------------------------------------------
# Figure B: Multi-bit SSMP comparison (grouped bars)
# -------------------------------------------------------------------------
def gen_multi_bit_ssmp():
    print("\n=== Multi-bit SSMP Comparison ===")

    # Collect data from both 3-bit and 4-bit experiments
    all_data = {}
    for bits, bit_label in [(3, "3-bit"), (4, "4-bit")]:
        for model_name, ssmp_dir in [
            ("Qwen3-8B", RUNS / f"{bits}bit_ssmp_qwen3"),
            ("Llama-3.1-8B", RUNS / f"{bits}bit_ssmp_llama31"),
        ]:
            results_path = ssmp_dir / f"ssmp_{bits}bit_results.json"
            if not results_path.exists():
                continue
            data = load_json(results_path)
            all_data[(model_name, bit_label)] = data["conditions"]

    if not all_data:
        print("  [skip] No SSMP data")
        return

    # Create comprehensive bar chart per model
    for model_name in ["Qwen3-8B", "Llama-3.1-8B"]:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

        for ax_idx, (bits, bit_label) in enumerate([(4, "4-bit"), (3, "3-bit")]):
            ax = axes[ax_idx]
            key = (model_name, bit_label)
            if key not in all_data:
                ax.set_title(f"{bit_label} (no data)")
                continue

            conds = all_data[key]

            # Build ordered results
            results = {}
            for cond_key, val in conds.items():
                if cond_key == "fp16_baseline":
                    results["FP16"] = val["refusal_rate"]
                elif cond_key.startswith("uniform_"):
                    results[f"Uniform\n{bit_label}"] = val["refusal_rate"]
                elif cond_key.startswith("ssmp_"):
                    budget = val.get("budget", 0)
                    results[f"SSMP\n@{budget*100:.0f}%"] = val["refusal_rate"]
                elif cond_key.startswith("random_"):
                    results[f"Random\n@4%"] = val["refusal_rate"]
                elif cond_key.startswith("inverted_"):
                    results[f"Inverted\n@4%"] = val["refusal_rate"]

            labels = list(results.keys())
            vals = list(results.values())

            bar_colors = []
            for k in labels:
                if "FP16" in k:
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

            x = np.arange(len(labels))
            bars = ax.bar(x, vals, 0.6, color=bar_colors, edgecolor="white",
                         linewidth=0.5, zorder=3)

            for bar_obj, v in zip(bars, vals):
                ax.text(bar_obj.get_x() + bar_obj.get_width() / 2,
                       bar_obj.get_height() + 0.01, f"{v:.1%}",
                       ha="center", va="bottom", fontsize=8)

            ax.set_xticks(x)
            ax.set_xticklabels(labels, fontsize=8)
            ax.set_title(f"{bit_label} quantization")
            ax.set_ylim(0, 1.05)
            if ax_idx == 0:
                ax.set_ylabel("Refusal rate")

        fig.suptitle(f"SSMP across bit-widths ({model_name})", fontsize=13, y=1.02)
        fig.tight_layout()

        out_path = OUT / f"multi_bit_ssmp_{model_name.lower().replace('.', '').replace('-', '_')}.pdf"
        fig.savefig(str(out_path), dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  -> {out_path}")


# -------------------------------------------------------------------------
# Figure C: Comprehensive recovery chart
# -------------------------------------------------------------------------
def gen_recovery_chart():
    print("\n=== Safety Recovery Chart ===")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    for ax_idx, model_name in enumerate(["Qwen3-8B", "Llama-3.1-8B"]):
        ax = axes[ax_idx]
        bars_data = []  # (bit_label, condition, refusal_rate)

        for bits in [4, 3]:
            ssmp_dir = RUNS / f"{bits}bit_ssmp_{model_name.lower().replace('.', '').replace('-', '_').replace('_8b', '').replace('llama_31', 'llama31').replace('qwen3', 'qwen3')}"
            # Try different naming patterns
            for dir_name in [
                f"{bits}bit_ssmp_qwen3",
                f"{bits}bit_ssmp_llama31",
            ]:
                results_path = RUNS / dir_name / f"ssmp_{bits}bit_results.json"
                if results_path.exists():
                    data = load_json(results_path)
                    if data.get("model_id", "").endswith(model_name.split("-")[0]) or \
                       model_name.lower().replace("-", "").replace(".", "") in data.get("model_id", "").lower().replace("-", "").replace("/", ""):
                        conds = data["conditions"]
                        fp16 = conds["fp16_baseline"]["refusal_rate"]
                        uniform = conds.get(f"uniform_{bits}bit", {}).get("refusal_rate", 0)
                        ssmp4 = None
                        for k, v in conds.items():
                            if k.startswith("ssmp_") and v.get("budget") == 0.04:
                                ssmp4 = v["refusal_rate"]
                        ssmp8 = None
                        for k, v in conds.items():
                            if k.startswith("ssmp_") and v.get("budget") == 0.08:
                                ssmp8 = v["refusal_rate"]
                        random4 = None
                        for k, v in conds.items():
                            if k.startswith("random_"):
                                random4 = v["refusal_rate"]

                        bars_data.append((f"{bits}-bit", "FP16", fp16))
                        bars_data.append((f"{bits}-bit", "Uniform", uniform))
                        if ssmp4 is not None:
                            bars_data.append((f"{bits}-bit", "SSMP@4%", ssmp4))
                        if ssmp8 is not None:
                            bars_data.append((f"{bits}-bit", "SSMP@8%", ssmp8))
                        if random4 is not None:
                            bars_data.append((f"{bits}-bit", "Random@4%", random4))

        if not bars_data:
            ax.set_title(f"{model_name} (no data)")
            continue

        df = pd.DataFrame(bars_data, columns=["bits", "condition", "refusal_rate"])

        bit_groups = df["bits"].unique()
        conditions = df["condition"].unique()
        n_groups = len(bit_groups)
        n_bars = len(conditions)
        bar_width = 0.15
        group_width = n_bars * bar_width

        cond_colors = {
            "FP16": COLORS["fp16"],
            "Uniform": COLORS["full_quant"],
            "SSMP@4%": COLORS["ssmp"],
            "SSMP@8%": "#2E7D32",
            "Random@4%": COLORS["random"],
        }

        for j, cond in enumerate(conditions):
            vals = []
            for bg in bit_groups:
                sub = df[(df["bits"] == bg) & (df["condition"] == cond)]
                vals.append(sub["refusal_rate"].iloc[0] if len(sub) > 0 else 0)

            x = np.arange(n_groups)
            offset = (j - (n_bars - 1) / 2) * bar_width
            ax.bar(x + offset, vals, bar_width * 0.9,
                  color=cond_colors.get(cond, "#607D8B"),
                  label=cond if ax_idx == 0 else "", zorder=3)

        ax.set_xticks(np.arange(n_groups))
        ax.set_xticklabels(bit_groups)
        ax.set_xlabel("Quantization precision")
        ax.set_ylabel("Refusal rate" if ax_idx == 0 else "")
        ax.set_ylim(0, 1.05)
        ax.set_title(model_name)

    axes[0].legend(frameon=True, framealpha=0.9, fontsize=8, loc="upper right")
    fig.suptitle("Safety preservation across bit-widths and methods", fontsize=13, y=1.02)
    fig.tight_layout()

    out_path = OUT / "recovery_chart_combined.pdf"
    fig.savefig(str(out_path), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out_path}")


# -------------------------------------------------------------------------
# Table: Comprehensive results (all conditions × bit-widths × models)
# -------------------------------------------------------------------------
def gen_comprehensive_table():
    print("\n=== Comprehensive Results Table ===")

    rows = []

    for bits in [4, 3]:
        for model_name, dir_suffix in [
            ("Qwen3-8B", "qwen3"),
            ("Llama-3.1-8B", "llama31"),
        ]:
            results_path = RUNS / f"{bits}bit_ssmp_{dir_suffix}" / f"ssmp_{bits}bit_results.json"
            if not results_path.exists():
                continue
            data = load_json(results_path)
            conds = data["conditions"]

            for key, val in conds.items():
                label = key
                if key == "fp16_baseline":
                    label = "FP16"
                elif key.startswith("uniform_"):
                    label = f"Uniform {bits}-bit"
                elif key.startswith("ssmp_"):
                    budget = val.get("budget", 0)
                    label = f"SSMP@{budget*100:.0f}%"
                elif key.startswith("random_"):
                    label = "Random@4%"
                elif key.startswith("inverted_"):
                    label = "Inverted@4%"

                rows.append({
                    "Model": model_name,
                    "Bits": bits,
                    "Condition": label,
                    "Refusal Rate": val["refusal_rate"],
                    "Avg Words": val.get("avg_response_words", 0),
                })

    if not rows:
        print("  [skip] No data")
        return

    df = pd.DataFrame(rows)

    # Create pivot: Model × Bits → columns, Condition → rows
    df["Model_Bits"] = df["Model"] + " " + df["Bits"].astype(str) + "-bit"

    pivot = df.pivot_table(
        index="Condition",
        columns="Model_Bits",
        values="Refusal Rate",
        aggfunc="first",
    ).reset_index()

    # Reorder conditions
    order = ["FP16", "Uniform 4-bit", "Uniform 3-bit",
             "SSMP@2%", "SSMP@4%", "SSMP@8%",
             "Random@4%", "Inverted@4%"]
    pivot["sort_key"] = pivot["Condition"].apply(lambda x: order.index(x) if x in order else 99)
    pivot = pivot.sort_values("sort_key").drop("sort_key", axis=1)

    # Reorder columns
    col_order = ["Condition"]
    for model in ["Qwen3-8B", "Llama-3.1-8B"]:
        for bits in [4, 3]:
            col = f"{model} {bits}-bit"
            if col in pivot.columns:
                col_order.append(col)
    pivot = pivot[[c for c in col_order if c in pivot.columns]]

    latex = results_to_latex(
        pivot,
        caption="Comprehensive SSMP results across models and bit-widths. Higher refusal rate indicates better safety preservation.",
        label="tab:comprehensive",
        bold_best=True,
        higher_is_better={c: True for c in pivot.columns if c != "Condition"},
    )
    md = results_to_markdown(pivot)

    (TOUT / "comprehensive_results.tex").write_text(latex, encoding="utf-8")
    (TOUT / "comprehensive_results.md").write_text(md, encoding="utf-8")

    print(f"  -> {TOUT}/comprehensive_results.tex")
    print("\nMarkdown:")
    print(md)


if __name__ == "__main__":
    gen_combined_phase_transition()
    gen_multi_bit_ssmp()
    gen_recovery_chart()
    gen_comprehensive_table()
    print(f"\n[done] Comprehensive outputs saved to {OUT}/ and {TOUT}/")
