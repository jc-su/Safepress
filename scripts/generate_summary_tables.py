#!/usr/bin/env python3
"""Generate summary tables from all SafePress experiment results."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from safepress.viz.tables import results_to_latex, results_to_markdown

RUNS = Path("runs")
OUT = Path("tables")
OUT.mkdir(parents=True, exist_ok=True)

# All models with SSMP data
MODELS = [
    ("Qwen3-8B", "qwen3"),
    ("Llama-3.1-8B", "llama31"),
    ("Yi-1.5-9B", "yi15"),
    ("Phi-3.5-mini", "phi35"),
]

# All models with phase transition data (includes controls)
ALL_PHASE_MODELS = MODELS + [
    ("Mistral-7B", "mistral7b"),
]


def load_json(path: Path) -> dict:
    """Load JSON file, handling NaN values."""
    with open(path) as f:
        content = f.read()
    content = content.replace("NaN", "null")
    return json.loads(content)


# -------------------------------------------------------------------------
# Table 1: 3-bit SSMP results (all models)
# -------------------------------------------------------------------------
def gen_ssmp_table():
    print("\n=== Table 1: 3-bit SSMP Results ===")
    rows = []
    for model_name, suffix in MODELS:
        results_path = RUNS / f"3bit_ssmp_{suffix}" / "ssmp_3bit_results.json"
        if not results_path.exists():
            continue
        data = load_json(results_path)
        conds = data["conditions"]
        for key, val in conds.items():
            label = key
            if key == "fp16_baseline":
                label = "FP16"
            elif key == "uniform_3bit":
                label = "Uniform 3-bit"
            elif key.startswith("ssmp_3bit_b") or key.startswith("ssmp_"):
                budget = val.get("budget", 0)
                label = f"SSMP@{budget*100:.0f}%"
            elif key.startswith("random_"):
                label = "Random@4%"
            elif key.startswith("inverted_"):
                label = "Inverted@4%"
            rows.append({
                "Model": model_name,
                "Condition": label,
                "Refusal Rate": val["refusal_rate"],
                "Avg Words": val.get("avg_response_words", 0),
            })

    if not rows:
        print("  [skip] No data")
        return

    df = pd.DataFrame(rows)

    pivot = df.pivot_table(
        index="Condition",
        columns="Model",
        values="Refusal Rate",
        aggfunc="first",
    ).reset_index()

    order = ["FP16", "Uniform 3-bit", "SSMP@2%", "SSMP@4%", "SSMP@8%", "Random@4%", "Inverted@4%"]
    pivot["sort_key"] = pivot["Condition"].apply(lambda x: order.index(x) if x in order else 99)
    pivot = pivot.sort_values("sort_key").drop("sort_key", axis=1)

    # Reorder columns
    model_cols = [m for m, _ in MODELS if m in pivot.columns]
    pivot = pivot[["Condition"] + model_cols]

    latex = results_to_latex(
        pivot,
        caption="Refusal rate under 3-bit quantization with SSMP protection. Higher is better (model refuses harmful prompts).",
        label="tab:ssmp_3bit",
        bold_best=True,
        higher_is_better={c: True for c in pivot.columns if c != "Condition"},
    )
    md = results_to_markdown(pivot)

    (OUT / "table1_ssmp_3bit.tex").write_text(latex, encoding="utf-8")
    (OUT / "table1_ssmp_3bit.md").write_text(md, encoding="utf-8")

    print(f"  Table 1: {OUT}/table1_ssmp_3bit.tex")
    print("\nMarkdown preview:")
    print(md)


# -------------------------------------------------------------------------
# Table 2: Phase transition results
# -------------------------------------------------------------------------
def gen_phase_table():
    print("\n=== Table 2: Phase Transition ===")
    rows = []
    for model_name, suffix in ALL_PHASE_MODELS:
        results_path = RUNS / f"phase_transition_{suffix}" / "phase_transition_results.json"
        if not results_path.exists():
            continue
        data = load_json(results_path)
        for key, val in data.get("results", {}).items():
            if isinstance(val, dict) and "refusal_rate" in val:
                bw = val.get("bits", int(key.replace("bits_", "")))
                rows.append({
                    "Model": model_name,
                    "Bit-Width": bw,
                    "Refusal Rate": val["refusal_rate"],
                    "Avg Words": val.get("avg_response_words", 0),
                })

    if not rows:
        print("  [skip] No phase transition data")
        return

    df = pd.DataFrame(rows)

    pivot = df.pivot_table(
        index="Bit-Width",
        columns="Model",
        values="Refusal Rate",
        aggfunc="first",
    ).reset_index()
    pivot = pivot.sort_values("Bit-Width", ascending=False)

    model_cols = [m for m, _ in ALL_PHASE_MODELS if m in pivot.columns]
    pivot = pivot[["Bit-Width"] + model_cols]

    latex = results_to_latex(
        pivot,
        caption="Phase transition: refusal rate at different quantization bit-widths. Sharp degradation occurs between 4-bit and 3-bit.",
        label="tab:phase_transition",
        bold_best=True,
        higher_is_better={c: True for c in pivot.columns if c != "Bit-Width"},
    )
    md = results_to_markdown(pivot)

    (OUT / "table2_phase_transition.tex").write_text(latex, encoding="utf-8")
    (OUT / "table2_phase_transition.md").write_text(md, encoding="utf-8")

    print(f"  Table 2: {OUT}/table2_phase_transition.tex")
    print("\nMarkdown preview:")
    print(md)


# -------------------------------------------------------------------------
# Table 3: 4-bit SSMP results
# -------------------------------------------------------------------------
def gen_4bit_table():
    print("\n=== Table 3: 4-bit SSMP Results ===")
    rows = []
    for model_name, suffix in MODELS:
        results_path = RUNS / f"4bit_ssmp_{suffix}" / "ssmp_4bit_results.json"
        if not results_path.exists():
            continue
        data = load_json(results_path)
        conds = data["conditions"]
        for key, val in conds.items():
            label = key
            if key == "fp16_baseline":
                label = "FP16"
            elif key == "uniform_3bit":
                label = "Uniform 4-bit"
            elif key.startswith("ssmp_4bit_b") or key.startswith("ssmp_"):
                budget = val.get("budget", 0)
                label = f"SSMP@{budget*100:.0f}%"
            elif key.startswith("random_"):
                label = "Random@4%"
            elif key.startswith("inverted_"):
                label = "Inverted@4%"
            rows.append({
                "Model": model_name,
                "Condition": label,
                "Refusal Rate": val["refusal_rate"],
            })

    if not rows:
        print("  [skip] No 4-bit SSMP data")
        return

    df = pd.DataFrame(rows)

    pivot = df.pivot_table(
        index="Condition",
        columns="Model",
        values="Refusal Rate",
        aggfunc="first",
    ).reset_index()

    order = ["FP16", "Uniform 4-bit", "SSMP@2%", "SSMP@4%", "SSMP@8%", "Random@4%", "Inverted@4%"]
    pivot["sort_key"] = pivot["Condition"].apply(lambda x: order.index(x) if x in order else 99)
    pivot = pivot.sort_values("sort_key").drop("sort_key", axis=1)

    model_cols = [m for m, _ in MODELS if m in pivot.columns]
    pivot = pivot[["Condition"] + model_cols]

    latex = results_to_latex(
        pivot,
        caption="Refusal rate under 4-bit quantization with SSMP protection.",
        label="tab:ssmp_4bit",
        bold_best=True,
        higher_is_better={c: True for c in pivot.columns if c != "Condition"},
    )
    md = results_to_markdown(pivot)

    (OUT / "table3_ssmp_4bit.tex").write_text(latex, encoding="utf-8")
    (OUT / "table3_ssmp_4bit.md").write_text(md, encoding="utf-8")

    print(f"  Table 3: {OUT}/table3_ssmp_4bit.tex")
    print("\nMarkdown preview:")
    print(md)


# -------------------------------------------------------------------------
# Table 4: Sweep summary (legacy)
# -------------------------------------------------------------------------
def gen_sweep_table():
    print("\n=== Table 4: 4-bit Sweep Summary ===")
    rows = []

    for label, sweep_dir in [
        ("Qwen3-8B", RUNS / "sweep"),
        ("Llama-3.1-8B", RUNS / "sweep_llama31"),
    ]:
        if not sweep_dir.exists():
            continue
        for subdir in sorted(sweep_dir.iterdir()):
            eval_path = subdir / "eval.json"
            if eval_path.exists():
                data = load_json(eval_path)
                rows.append({
                    "Model": label,
                    "Condition": subdir.name,
                    "Refusal Rate": data.get("refusal_rate", 0),
                })

    if not rows:
        print("  [skip] No sweep data")
        return

    df = pd.DataFrame(rows)
    md = results_to_markdown(df)
    (OUT / "table4_sweep.md").write_text(md, encoding="utf-8")
    print(f"  Table 4: {OUT}/table4_sweep.md")
    print("\nMarkdown preview (first 20 rows):")
    print("\n".join(md.split("\n")[:22]))


# -------------------------------------------------------------------------
# Summary statistics
# -------------------------------------------------------------------------
def gen_summary():
    print("\n" + "=" * 70)
    print("COMPLETE EXPERIMENT SUMMARY")
    print("=" * 70)

    for bits_label, prefix in [("3-bit", "3bit_ssmp"), ("4-bit", "4bit_ssmp")]:
        print(f"\n{'─'*50}")
        print(f"  {bits_label} SSMP Results")
        print(f"{'─'*50}")
        for model_name, suffix in MODELS:
            bits = int(bits_label[0])
            results_path = RUNS / f"{prefix}_{suffix}" / f"ssmp_{bits}bit_results.json"
            if not results_path.exists():
                continue
            data = load_json(results_path)
            conds = data["conditions"]
            fp16 = conds.get("fp16_baseline", {}).get("refusal_rate", 0)
            uniform_key = "uniform_3bit"  # script uses this key name even for 4-bit
            uniform = conds.get(uniform_key, {}).get("refusal_rate", 0)
            gap = fp16 - uniform

            print(f"\n--- {model_name} ({bits_label} SSMP) ---")
            print(f"  FP16 baseline:    {fp16:.1%}")
            print(f"  Uniform {bits_label}:  {uniform:.1%}")
            print(f"  Safety gap:       {gap:.1%} ({gap*100:.1f}pp)")

            for key, val in conds.items():
                if key.startswith("ssmp_"):
                    budget = val.get("budget", 0)
                    rr = val["refusal_rate"]
                    recovery = rr - uniform
                    recovery_pct = recovery / gap * 100 if gap > 0.001 else 0
                    print(f"  SSMP@{budget*100:.0f}%:         {rr:.1%} (+{recovery*100:.1f}pp, {recovery_pct:.0f}% recovery)")

            for key in [f"random_{bits}bit_b0.04", f"inverted_{bits}bit_b0.04"]:
                if key in conds:
                    rr = conds[key]["refusal_rate"]
                    recovery = rr - uniform
                    label = "Random@4%" if "random" in key else "Inverted@4%"
                    print(f"  {label}:     {rr:.1%} (+{recovery*100:.1f}pp)")

    # Phase transition summary
    print(f"\n{'─'*50}")
    print("  Phase Transition Summary")
    print(f"{'─'*50}")
    for model_name, suffix in ALL_PHASE_MODELS:
        results_path = RUNS / f"phase_transition_{suffix}" / "phase_transition_results.json"
        if not results_path.exists():
            continue
        data = load_json(results_path)
        print(f"\n--- {model_name} ---")
        for key in sorted(data.get("results", {}).keys()):
            val = data["results"][key]
            if isinstance(val, dict) and "refusal_rate" in val:
                bw = val.get("bits", key)
                print(f"  {bw}-bit: {val['refusal_rate']:.1%}")


# -------------------------------------------------------------------------
# Table 5: Utility retention (Perplexity, MMLU, TruthfulQA)
# -------------------------------------------------------------------------
def gen_utility_table():
    print("\n=== Table 5: Utility Retention ===")
    rows = []
    for model_name, suffix in MODELS:
        results_path = RUNS / f"utility_{suffix}" / "utility_results.json"
        if not results_path.exists():
            continue
        data = load_json(results_path)
        for cond_key, cond_label in [
            ("fp16", "FP16"),
            ("uniform_3bit", "Uniform 3-bit"),
            ("ssmp_3bit_b0.04", "SSMP@4\\%"),
        ]:
            if cond_key not in data:
                continue
            cond = data[cond_key]
            ppl = cond.get("perplexity", {}).get("perplexity", None)
            mmlu = cond.get("mmlu", {}).get("accuracy", None)
            tqa = cond.get("truthfulqa", {}).get("accuracy", None)
            rows.append({
                "Model": model_name,
                "Condition": cond_label,
                "PPL": ppl,
                "MMLU": mmlu,
                "TruthfulQA": tqa,
            })

    if not rows:
        print("  [skip] No utility data")
        return

    df = pd.DataFrame(rows)

    # Create a multi-metric pivot
    # For LaTeX: grouped by model with PPL/MMLU/TQA columns per condition
    lines = []
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append("\\caption{Utility retention under 3-bit quantization with SSMP@4\\% protection. PPL = perplexity (lower is better); MMLU and TruthfulQA = accuracy (higher is better).}")
    lines.append("\\label{tab:utility}")
    lines.append("\\begin{tabular}{ll" + "c" * 3 + "}")
    lines.append("\\toprule")
    lines.append("Model & Condition & PPL ($\\downarrow$) & MMLU ($\\uparrow$) & TruthfulQA ($\\uparrow$) \\\\")
    lines.append("\\midrule")

    for i, (model_name, suffix) in enumerate(MODELS):
        model_rows = [r for r in rows if r["Model"] == model_name]
        if not model_rows:
            continue
        if i > 0:
            lines.append("\\midrule")
        for r in model_rows:
            ppl_str = f"{r['PPL']:.1f}" if r['PPL'] is not None else "---"
            mmlu_str = f"{r['MMLU']:.3f}" if r['MMLU'] is not None else "---"
            tqa_str = f"{r['TruthfulQA']:.2f}" if r['TruthfulQA'] is not None else "---"
            lines.append(f"{r['Model']} & {r['Condition']} & {ppl_str} & {mmlu_str} & {tqa_str} \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    latex = "\n".join(lines)

    # Markdown version
    md_lines = []
    md_lines.append("| Model | Condition | PPL | MMLU | TruthfulQA |")
    md_lines.append("| ----- | --------- | ---:| ----:| ----------:|")
    for r in rows:
        ppl_str = f"{r['PPL']:.2f}" if r['PPL'] is not None else "---"
        mmlu_str = f"{r['MMLU']:.3f}" if r['MMLU'] is not None else "---"
        tqa_str = f"{r['TruthfulQA']:.2f}" if r['TruthfulQA'] is not None else "---"
        md_lines.append(f"| {r['Model']} | {r['Condition']} | {ppl_str} | {mmlu_str} | {tqa_str} |")
    md = "\n".join(md_lines)

    (OUT / "table5_utility.tex").write_text(latex, encoding="utf-8")
    (OUT / "table5_utility.md").write_text(md, encoding="utf-8")

    print(f"  Table 5: {OUT}/table5_utility.tex")
    print("\nMarkdown preview:")
    print(md)


# -------------------------------------------------------------------------
# Table 6: StrongREJECT scores for key conditions
# -------------------------------------------------------------------------
def gen_strongreject_table():
    print("\n=== Table 6: StrongREJECT Scores ===")
    sr_path = OUT / "strongreject_scores.json"
    if not sr_path.exists():
        print("  [skip] No StrongREJECT scores")
        return

    sr_data = load_json(sr_path)

    # Collect key conditions for each model
    rows = []
    for model_name, suffix in MODELS:
        for bits_label, prefix in [("3-bit", "3bit_ssmp"), ("4-bit", "4bit_ssmp")]:
            key_conditions = {
                "fp16_baseline": "FP16",
                f"uniform_{bits_label[0]}bit": f"Uniform {bits_label}",
            }
            for bud in ["0.02", "0.04", "0.08"]:
                key_conditions[f"ssmp_{bits_label[0]}bit_b{bud}"] = f"SSMP@{float(bud)*100:.0f}%"
            key_conditions[f"random_{bits_label[0]}bit_b0.04"] = "Random@4%"
            key_conditions[f"inverted_{bits_label[0]}bit_b0.04"] = "Inverted@4%"

            for cond_key, cond_label in key_conditions.items():
                sr_key = f"{prefix}_{suffix}/{cond_key}"
                if sr_key in sr_data:
                    v = sr_data[sr_key]
                    if v.get("ok") and v.get("n_scored", 0) > 0:
                        rows.append({
                            "Model": model_name,
                            "Bits": bits_label,
                            "Condition": cond_label,
                            "StrongREJECT": v["avg_score"],
                            "N_scored": v["n_scored"],
                        })

    if not rows:
        print("  [skip] No valid StrongREJECT data")
        return

    df = pd.DataFrame(rows)

    # Focus on 3-bit results for the paper table
    df_3bit = df[df["Bits"] == "3-bit"]
    if len(df_3bit) == 0:
        print("  [skip] No 3-bit StrongREJECT data")
        return

    pivot = df_3bit.pivot_table(
        index="Condition",
        columns="Model",
        values="StrongREJECT",
        aggfunc="first",
    ).reset_index()

    order = ["FP16", "Uniform 3-bit", "SSMP@2%", "SSMP@4%", "SSMP@8%", "Random@4%", "Inverted@4%"]
    pivot["sort_key"] = pivot["Condition"].apply(lambda x: order.index(x) if x in order else 99)
    pivot = pivot.sort_values("sort_key").drop("sort_key", axis=1)

    model_cols = [m for m, _ in MODELS if m in pivot.columns]
    pivot = pivot[["Condition"] + model_cols]

    latex = results_to_latex(
        pivot,
        caption="StrongREJECT scores under 3-bit quantization (lower score = safer model). Scale 0--1.",
        label="tab:strongreject",
        bold_best=True,
        higher_is_better={c: False for c in pivot.columns if c != "Condition"},
    )
    md = results_to_markdown(pivot)

    (OUT / "table6_strongreject.tex").write_text(latex, encoding="utf-8")
    (OUT / "table6_strongreject.md").write_text(md, encoding="utf-8")

    print(f"  Table 6: {OUT}/table6_strongreject.tex")
    print("\nMarkdown preview:")
    print(md)


if __name__ == "__main__":
    gen_ssmp_table()
    gen_phase_table()
    gen_4bit_table()
    gen_sweep_table()
    gen_utility_table()
    gen_strongreject_table()
    gen_summary()
    print(f"\n[done] Tables saved to {OUT}/")
