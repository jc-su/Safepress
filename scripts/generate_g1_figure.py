#!/usr/bin/env python
"""Generate the G1 drift-bound theory figure from the per-model CSVs.

Reads ``runs/emnlp_g1/{model_tag}_drift.csv`` for each model in the panel,
concatenates the rows (one row per (model, bits)), and produces a single
predicted-vs-measured scatter with one colour per model and one marker
shape per bit-width. R² and Spearman ρ from each model's ``.fit.json`` are
shown in the caption block.

Pure CPU; no GPU contention.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from safepress.viz.plots import plot_drift_bound_scatter


def main() -> None:
    runs = Path("runs/emnlp_g1")
    pairs = [
        ("qwen3_8b", runs / "qwen3_drift.csv", runs / "qwen3_drift.fit.json"),
        ("llama31_8b", runs / "llama31_drift.csv", runs / "llama31_drift.fit.json"),
        ("mistral_8b", runs / "mistral_drift.csv", runs / "mistral_drift.fit.json"),
        ("glm4_9b", runs / "glm4_drift.csv", runs / "glm4_drift.fit.json"),
        ("deepseek_8b", runs / "deepseek_drift.csv", runs / "deepseek_drift.fit.json"),
    ]
    frames = []
    captions = []
    for tag, csv_path, fit_path in pairs:
        if not csv_path.exists():
            print(f"[g1-fig] skip {tag}: {csv_path} missing")
            continue
        df = pd.read_csv(csv_path)
        df["model_tag"] = tag
        frames.append(df)
        if fit_path.exists():
            fit = json.loads(fit_path.read_text())
            ub = fit.get("upper_bound", {})
            captions.append(
                f"{tag}: R²={ub.get('r_squared', float('nan')):.3f}  "
                f"slope={ub.get('slope', float('nan')):.3f}  "
                f"Spearman={fit.get('upper_bound_spearman_rho', float('nan')):.2f}"
            )

    if not frames:
        print("[g1-fig] no input CSVs found")
        return
    merged = pd.concat(frames, ignore_index=True)
    out_path = runs / "g1_drift_scatter.pdf"
    plot_drift_bound_scatter(
        merged,
        mode="upper_bound",
        title="G1: predicted upper bound vs measured |ΔL_safe| (theorem fit)",
        save_path=str(out_path),
    )
    print(f"[g1-fig] wrote {out_path}")
    for c in captions:
        print(f"  {c}")


if __name__ == "__main__":
    main()
