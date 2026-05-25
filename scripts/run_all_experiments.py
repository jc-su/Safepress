#!/usr/bin/env python3
"""Run all SafePress experiments from a config directory.

Usage
-----
    python scripts/run_all_experiments.py --config_dir configs/ --experiment_type all
    python scripts/run_all_experiments.py --config_dir configs/ --experiment_type causal
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional

import yaml


EXPERIMENT_CONFIG_MAP = {
    "causal": "experiment_causal.yaml",
    "sweep": "experiment_sweep.yaml",
    "phase": "experiment_phase.yaml",
}


def load_config(path: Path) -> dict:
    """Load a YAML config file."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def run_causal(cfg: dict) -> dict:
    """Run the causal experiment."""
    from safepress.experiments.causal import run_causal as _run_causal

    out_dir = cfg.get("out_dir", "runs/causal")
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    result = _run_causal(
        model_id=cfg["model_id"],
        scores=cfg.get("calib_prompts"),
        eval_prompts=cfg.get("eval_prompts"),
        out_dir=out_dir,
        dtype=cfg.get("dtype", "float16"),
        device_map=cfg.get("device_map", "auto"),
        bits=cfg.get("bits", 4),
        group_size=cfg.get("group_size", 128),
        block_size=cfg.get("block_size", 64),
        budget=cfg.get("budget", 0.02),
        max_new_tokens=cfg.get("max_new_tokens", 256),
    )
    print(f"[causal] Results saved to {out_dir}")
    return result


def run_sweep(cfg: dict) -> dict:
    """Run the budget-sweep experiment."""
    from safepress.experiments.sweep import run_sweep as _run_sweep

    out_dir = cfg.get("out_dir", "runs/sweep")
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    result = _run_sweep(
        model_id=cfg.get("model_ids", [cfg.get("model_id")])[0]
        if isinstance(cfg.get("model_ids"), list)
        else cfg["model_id"],
        budgets=cfg.get("budgets", [0.005, 0.01, 0.02, 0.04, 0.08]),
        out_dir=out_dir,
        dtype=cfg.get("dtype", "float16"),
        device_map=cfg.get("device_map", "auto"),
        bits=cfg.get("bits", 4),
        group_size=cfg.get("group_size", 128),
        block_size=cfg.get("block_size", 64),
        max_new_tokens=cfg.get("max_new_tokens", 256),
    )
    print(f"[sweep] Results saved to {out_dir}")
    return result


def run_phase(cfg: dict) -> dict:
    """Run the phase-transition experiment."""
    from safepress.experiments.phase_transition import run_phase_transition

    out_dir = cfg.get("out_dir", "runs/phase_transition")
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    result = run_phase_transition(
        model_id=cfg["model_id"],
        eval_prompts=cfg.get("eval_prompts"),
        out_dir=out_dir,
        dtype=cfg.get("dtype", "float16"),
        device_map=cfg.get("device_map", "auto"),
        bit_widths=cfg.get("bit_widths", [8, 4, 3, 2]),
        group_size=cfg.get("group_size", 128),
        max_new_tokens=cfg.get("max_new_tokens", 256),
    )
    print(f"[phase] Results saved to {out_dir}")
    return result


RUNNERS = {
    "causal": run_causal,
    "sweep": run_sweep,
    "phase": run_phase,
}


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(
        description="Run all SafePress experiments from a config directory.",
    )
    ap.add_argument(
        "--config_dir",
        type=str,
        default="configs",
        help="Directory containing YAML experiment configs (default: configs/).",
    )
    ap.add_argument(
        "--experiment_type",
        type=str,
        default="all",
        choices=["causal", "sweep", "phase", "all"],
        help="Which experiment to run (default: all).",
    )
    args = ap.parse_args(argv)

    config_dir = Path(args.config_dir)
    if not config_dir.is_dir():
        raise FileNotFoundError(f"Config directory not found: {config_dir}")

    if args.experiment_type == "all":
        experiment_types = ["causal", "sweep", "phase"]
    else:
        experiment_types = [args.experiment_type]

    all_results = {}
    for exp_type in experiment_types:
        config_file = config_dir / EXPERIMENT_CONFIG_MAP[exp_type]
        if not config_file.exists():
            print(f"[WARN] Config file not found, skipping: {config_file}")
            continue

        print(f"\n{'='*60}")
        print(f"Running experiment: {exp_type}")
        print(f"Config: {config_file}")
        print(f"{'='*60}\n")

        cfg = load_config(config_file)
        runner = RUNNERS[exp_type]
        result = runner(cfg)
        all_results[exp_type] = result

    # Save combined results summary
    summary_path = Path(config_dir).parent / "runs" / "experiment_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(
            {k: str(v) if not isinstance(v, (dict, list, type(None))) else v for k, v in all_results.items()},
            f,
            indent=2,
            default=str,
        )
    print(f"\n[done] Experiment summary saved to {summary_path}")


if __name__ == "__main__":
    main()
