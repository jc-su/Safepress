#!/usr/bin/env python3
"""
Bootstrap confidence intervals for refusal rate estimates.

Reads saved responses.jsonl files from experiment directories and computes
95% CIs via bootstrap resampling.

Usage:
  python scripts/compute_confidence_intervals.py \
      --runs_dir runs/ \
      --out tables/confidence_intervals.json \
      --n_bootstrap 10000
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from safepress.eval.basic import is_refusal


def load_responses(responses_path: Path) -> List[str]:
    responses = []
    with responses_path.open() as f:
        for line in f:
            row = json.loads(line)
            responses.append(row["response"])
    return responses


def bootstrap_refusal_rate(
    responses: List[str],
    n_bootstrap: int = 10000,
    confidence: float = 0.95,
    seed: int = 42,
) -> Dict[str, float]:
    rng = np.random.RandomState(seed)
    refusals = np.array([1.0 if is_refusal(r) else 0.0 for r in responses])
    n = len(refusals)
    point_estimate = float(refusals.mean())

    boot_means = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        sample = rng.choice(refusals, size=n, replace=True)
        boot_means[i] = sample.mean()

    alpha = (1 - confidence) / 2
    ci_low = float(np.percentile(boot_means, 100 * alpha))
    ci_high = float(np.percentile(boot_means, 100 * (1 - alpha)))
    se = float(boot_means.std())

    return {
        "refusal_rate": point_estimate,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "se": se,
        "n": n,
        "n_bootstrap": n_bootstrap,
        "confidence": confidence,
    }


def find_response_files(runs_dir: Path) -> List[Tuple[str, Path]]:
    """Find all responses.jsonl files under runs_dir and label them."""
    results = []
    for rfile in sorted(runs_dir.rglob("responses.jsonl")):
        # Build a label from the path relative to runs_dir
        rel = rfile.relative_to(runs_dir)
        label = str(rel.parent).replace("/", " / ")
        results.append((label, rfile))
    return results


def main():
    parser = argparse.ArgumentParser(description="Bootstrap confidence intervals for refusal rates")
    parser.add_argument("--runs_dir", default="runs", help="Root directory containing experiment results")
    parser.add_argument("--out", default="tables/confidence_intervals.json", help="Output JSON file")
    parser.add_argument("--n_bootstrap", type=int, default=10000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    response_files = find_response_files(runs_dir)

    if not response_files:
        print(f"No responses.jsonl files found under {runs_dir}")
        return

    print(f"Found {len(response_files)} response files")

    all_results: Dict[str, Any] = {}
    for label, rfile in response_files:
        print(f"  Processing: {label} ...")
        responses = load_responses(rfile)
        ci = bootstrap_refusal_rate(
            responses,
            n_bootstrap=args.n_bootstrap,
            confidence=args.confidence,
            seed=args.seed,
        )
        all_results[label] = ci
        rr = ci["refusal_rate"]
        lo = ci["ci_low"]
        hi = ci["ci_high"]
        print(f"    refusal_rate={rr:.4f}  95% CI=[{lo:.4f}, {hi:.4f}]  SE={ci['se']:.4f}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(all_results, indent=2) + "\n")
    print(f"\nSaved confidence intervals to {out_path}")

    # Also produce a LaTeX table
    tex_path = out_path.with_suffix(".tex")
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Bootstrap 95\% confidence intervals for refusal rates.}",
        r"\label{tab:confidence_intervals}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Condition & Refusal Rate & 95\% CI Low & 95\% CI High & SE \\",
        r"\midrule",
    ]
    for label, ci in all_results.items():
        short_label = label.replace("_", r"\_")
        lines.append(
            f"{short_label} & {ci['refusal_rate']:.4f} & {ci['ci_low']:.4f} & {ci['ci_high']:.4f} & {ci['se']:.4f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    tex_path.write_text("\n".join(lines) + "\n")
    print(f"LaTeX table saved to {tex_path}")


if __name__ == "__main__":
    main()
