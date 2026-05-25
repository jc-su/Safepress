#!/usr/bin/env python3
"""Generate LaTeX and Markdown tables from experiment results.

Usage
-----
    python scripts/generate_tables.py --results_dir runs/ --output_dir tables/
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from safepress.viz.tables import results_to_latex, results_to_markdown


# ---------------------------------------------------------------------------
# Table 1: Method × Budget sweep (main results)
# ---------------------------------------------------------------------------

def generate_sweep_table(results_dir: Path, output_dir: Path) -> None:
    """Build the main method-comparison table from sweep_summary.csv."""
    sweep_csv = results_dir / "sweep" / "sweep_summary.csv"
    if not sweep_csv.exists():
        print("  [skip] sweep_summary.csv not found.")
        return

    df = pd.read_csv(sweep_csv)

    # Pivot: rows = method, columns = budget, values = refusal_rate
    if "refusal_rate" in df.columns and "budget" in df.columns:
        pivot = df.pivot_table(
            index="method",
            columns="budget",
            values="refusal_rate",
            aggfunc="first",
        )
        pivot = pivot.reset_index()
        pivot.columns = ["Method"] + [f"b={c}" for c in pivot.columns[1:]]

        latex = results_to_latex(
            pivot,
            caption="Refusal rate (\\%) across methods and FP16 budgets. "
                    "Higher is better (model refuses harmful prompts).",
            label="tab:sweep",
            bold_best=True,
            higher_is_better={c: True for c in pivot.columns if c != "Method"},
        )
        md = results_to_markdown(pivot)

        (output_dir / "table1_sweep.tex").write_text(latex, encoding="utf-8")
        (output_dir / "table1_sweep.md").write_text(md, encoding="utf-8")
        print(f"  Table 1 (sweep): {output_dir}/table1_sweep.tex")

    # Also write flat CSV for reference
    df.to_csv(output_dir / "table1_sweep_flat.csv", index=False)


# ---------------------------------------------------------------------------
# Table 2: Causal experiment results
# ---------------------------------------------------------------------------

def generate_causal_table(results_dir: Path, output_dir: Path) -> None:
    """Build the causal experiment results table."""
    causal_dir = results_dir / "causal"
    if not causal_dir.is_dir():
        print("  [skip] No causal results directory found.")
        return

    rows: List[Dict[str, Any]] = []
    for json_path in sorted(causal_dir.glob("*.json")):
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue

        # Expect structure: {"condition_name": {"refusal_rate": float, ...}}
        # or flat: {"refusal_rate": float, ...}
        if isinstance(data, dict):
            if "refusal_rate" in data:
                rows.append({
                    "Condition": json_path.stem,
                    "Refusal Rate": data["refusal_rate"],
                })
            else:
                for condition, metrics in data.items():
                    if isinstance(metrics, dict) and "refusal_rate" in metrics:
                        rows.append({
                            "Condition": condition,
                            "Refusal Rate": metrics["refusal_rate"],
                        })

    if not rows:
        print("  [skip] No causal results parsed.")
        return

    df = pd.DataFrame(rows)

    latex = results_to_latex(
        df,
        caption="Causal experiment: refusal rate under targeted quantization conditions.",
        label="tab:causal",
        bold_best=True,
        higher_is_better={"Refusal Rate": True},
    )
    md = results_to_markdown(df)

    (output_dir / "table2_causal.tex").write_text(latex, encoding="utf-8")
    (output_dir / "table2_causal.md").write_text(md, encoding="utf-8")
    print(f"  Table 2 (causal): {output_dir}/table2_causal.tex")


# ---------------------------------------------------------------------------
# Table 3: Cross-model comparison
# ---------------------------------------------------------------------------

def generate_cross_model_table(results_dir: Path, output_dir: Path) -> None:
    """Build a cross-model comparison table from per-model eval JSONs."""
    eval_dir = results_dir / "eval"
    if not eval_dir.is_dir():
        print("  [skip] No eval directory found.")
        return

    rows: List[Dict[str, Any]] = []
    for json_path in sorted(eval_dir.glob("*.json")):
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue

        model_name = json_path.stem.replace("_eval", "").replace("_ssmp", " (SSMP)")
        row: Dict[str, Any] = {"Model": model_name}

        # Flat eval format
        if "refusal_rate" in data:
            row["Refusal Rate"] = data["refusal_rate"]
        if "strongreject" in data and isinstance(data["strongreject"], dict):
            if "avg_score" in data["strongreject"]:
                row["StrongREJECT"] = data["strongreject"]["avg_score"]

        # Comprehensive eval format
        safety = data.get("safety", {})
        utility = data.get("utility", {})
        if "refusal" in safety:
            row["Refusal Rate"] = safety["refusal"].get("refusal_rate", "")
        if "harmbench" in safety and "asr" in safety["harmbench"]:
            row["HarmBench ASR"] = safety["harmbench"]["asr"]
        if "perplexity" in utility and "perplexity" in utility["perplexity"]:
            row["PPL"] = utility["perplexity"]["perplexity"]
        if "mmlu" in utility and "accuracy" in utility["mmlu"]:
            row["MMLU"] = utility["mmlu"]["accuracy"]

        if len(row) > 1:
            rows.append(row)

    if not rows:
        print("  [skip] No eval results parsed.")
        return

    df = pd.DataFrame(rows)

    latex = results_to_latex(
        df,
        caption="Cross-model evaluation: safety and utility metrics.",
        label="tab:cross_model",
        bold_best=True,
        higher_is_better={
            "Refusal Rate": True,
            "StrongREJECT": False,
            "HarmBench ASR": False,
            "PPL": False,
            "MMLU": True,
        },
    )
    md = results_to_markdown(df)

    (output_dir / "table3_cross_model.tex").write_text(latex, encoding="utf-8")
    (output_dir / "table3_cross_model.md").write_text(md, encoding="utf-8")
    print(f"  Table 3 (cross-model): {output_dir}/table3_cross_model.tex")


# ---------------------------------------------------------------------------
# Table 4: Phase transition
# ---------------------------------------------------------------------------

def generate_phase_table(results_dir: Path, output_dir: Path) -> None:
    """Build phase-transition table (bit-width vs refusal rate)."""
    phase_dir = results_dir / "phase_transition"
    if not phase_dir.is_dir():
        print("  [skip] No phase_transition directory found.")
        return

    rows: List[Dict[str, Any]] = []
    for json_path in sorted(phase_dir.glob("*.json")):
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue

        # Expect: {"results": [{"bits": 8, "refusal_rate": 0.95}, ...]}
        # or flat list
        items = data if isinstance(data, list) else data.get("results", [])
        for item in items:
            if isinstance(item, dict) and "refusal_rate" in item:
                rows.append({
                    "Bit-Width": item.get("bits", item.get("bit_width", "?")),
                    "Refusal Rate": item["refusal_rate"],
                })

    if not rows:
        print("  [skip] No phase-transition results parsed.")
        return

    df = pd.DataFrame(rows)

    latex = results_to_latex(
        df,
        caption="Phase transition: refusal rate at different quantization bit-widths.",
        label="tab:phase",
        bold_best=True,
        higher_is_better={"Refusal Rate": True},
    )
    md = results_to_markdown(df)

    (output_dir / "table4_phase.tex").write_text(latex, encoding="utf-8")
    (output_dir / "table4_phase.md").write_text(md, encoding="utf-8")
    print(f"  Table 4 (phase): {output_dir}/table4_phase.tex")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

TABLE_GENERATORS = {
    "sweep": generate_sweep_table,
    "causal": generate_causal_table,
    "cross_model": generate_cross_model_table,
    "phase": generate_phase_table,
}


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(
        description="Generate LaTeX and Markdown tables from experiment results.",
    )
    ap.add_argument(
        "--results_dir", type=str, default="runs",
        help="Directory containing experiment results (default: runs/).",
    )
    ap.add_argument(
        "--output_dir", type=str, default="tables",
        help="Directory where table files will be saved (default: tables/).",
    )
    ap.add_argument(
        "--tables", type=str, nargs="+", default=None,
        choices=list(TABLE_GENERATORS.keys()),
        help="Which tables to generate (default: all).",
    )
    args = ap.parse_args(argv)

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not results_dir.is_dir():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    table_types = args.tables or list(TABLE_GENERATORS.keys())

    for tbl_type in table_types:
        print(f"\n--- Generating {tbl_type} table ---")
        TABLE_GENERATORS[tbl_type](results_dir, output_dir)

    print(f"\n[done] Tables written to {output_dir}/")


if __name__ == "__main__":
    main()
