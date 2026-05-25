#!/usr/bin/env python3
"""Generate all paper figures from experiment results.

Usage
-----
    python scripts/generate_figures.py --results_dir runs/ --output_dir figures/
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional


def load_json(path: Path) -> dict:
    """Load a JSON results file."""
    with open(path, "r") as f:
        return json.load(f)


def generate_heatmap(results_dir: Path, output_dir: Path) -> None:
    """Generate score heatmap figure if scores CSV exists."""
    from safepress.viz.plots import plot_heatmap

    # Look for scores CSV files in results subdirectories
    for scores_path in sorted(results_dir.rglob("scores.csv")):
        stem = scores_path.parent.name
        out_path = output_dir / f"heatmap_{stem}.pdf"
        print(f"  Generating heatmap: {scores_path} -> {out_path}")
        plot_heatmap(scores_path=str(scores_path), out_path=str(out_path))


def generate_phase_transition(results_dir: Path, output_dir: Path) -> None:
    """Generate phase-transition figure if results exist."""
    from safepress.viz.plots import plot_phase_transition

    phase_dir = results_dir / "phase_transition"
    if not phase_dir.is_dir():
        print("  [skip] No phase_transition results directory found.")
        return

    for results_path in sorted(phase_dir.glob("*.json")):
        stem = results_path.stem
        out_path = output_dir / f"phase_transition_{stem}.pdf"
        print(f"  Generating phase-transition: {results_path} -> {out_path}")
        plot_phase_transition(results_path=str(results_path), out_path=str(out_path))


def generate_causal(results_dir: Path, output_dir: Path) -> None:
    """Generate causal experiment figures if results exist."""
    from safepress.viz.plots import plot_causal

    # Look for causal result directories
    for causal_dir in sorted(results_dir.glob("causal_*")):
        if not causal_dir.is_dir():
            continue
        for results_path in sorted(causal_dir.glob("*.json")):
            stem = f"{causal_dir.name}_{results_path.stem}"
            out_path = output_dir / f"causal_{stem}.pdf"
            print(f"  Generating causal: {results_path} -> {out_path}")
            plot_causal(results_path=str(results_path), out_path=str(out_path))


def generate_sweep(results_dir: Path, output_dir: Path) -> None:
    """Generate sweep figures if results exist."""
    from safepress.viz.plots import plot_phase_transition

    sweep_dir = results_dir / "sweep"
    if not sweep_dir.is_dir():
        print("  [skip] No sweep results directory found.")
        return

    for results_path in sorted(sweep_dir.glob("*.json")):
        stem = results_path.stem
        out_path = output_dir / f"sweep_{stem}.pdf"
        print(f"  Generating sweep: {results_path} -> {out_path}")
        # Sweep data can be plotted with the same phase-transition style
        plot_phase_transition(results_path=str(results_path), out_path=str(out_path))


FIGURE_GENERATORS = {
    "heatmap": generate_heatmap,
    "phase_transition": generate_phase_transition,
    "causal": generate_causal,
    "sweep": generate_sweep,
}


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(
        description="Generate all paper figures from experiment results.",
    )
    ap.add_argument(
        "--results_dir",
        type=str,
        default="runs",
        help="Directory containing experiment results (default: runs/).",
    )
    ap.add_argument(
        "--output_dir",
        type=str,
        default="figures",
        help="Directory where PDF figures will be saved (default: figures/).",
    )
    ap.add_argument(
        "--figures",
        type=str,
        nargs="+",
        default=None,
        choices=list(FIGURE_GENERATORS.keys()),
        help="Which figures to generate (default: all).",
    )
    args = ap.parse_args(argv)

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not results_dir.is_dir():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    figure_types = args.figures or list(FIGURE_GENERATORS.keys())

    for fig_type in figure_types:
        print(f"\n--- Generating {fig_type} figures ---")
        generator = FIGURE_GENERATORS[fig_type]
        generator(results_dir, output_dir)

    print(f"\n[done] Figures written to {output_dir}/")


if __name__ == "__main__":
    main()
