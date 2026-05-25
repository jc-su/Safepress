#!/usr/bin/env python
"""
Run the G2 gate test from an EMNLP sweep output.

The sweep saves per-seed per-prompt outcome vectors at
``<sweep_root>/sweep_<model_tag>_<method>_b<budget>/eval_seed<sd>.json``.
This script loads those, builds per-(method, budget, seed) outcome
matrices, runs paired-bootstrap + McNemar + Holm-corrected comparisons
of SSMP vs each of {gradient_only (Fisher), snip, magnitude}, and writes
a single ``g2_stats.csv`` summarising the gate decision.

The G2 gate (PLAN §16) passes when SSMP achieves significantly lower
HarmBench-classifier ASR than each of {Fisher, SNIP, magnitude} on >= 3
of the 4 model-budget cells {Qwen3, Llama-3.1} x {0.04, 0.08} at 3-bit
(after Holm correction).

Usage::

    python scripts/g2_paired_stats.py --sweep_root runs/emnlp_sweep_3bit \
        --model_tags qwen3_8b_3bit llama31_8b_3bit \
        --budgets 0.04 0.08 \
        --reference int4 \
        --out runs/emnlp_sweep_3bit/g2_stats.csv

The reference defaults to ``int4`` (uniform quant baseline); pass another
method name to compare every method against a different baseline.

The script reads ``per_prompt_harmful`` (preferred, lower-is-better
HarmBench-classifier output) and falls back to ``per_prompt_refusal``
(higher-is-better) if classifier output is missing.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


def _load_per_seed(eval_dir: Path) -> List[Dict[str, Any]]:
    rows = sorted(eval_dir.glob("eval_seed*.json"))
    out: List[Dict[str, Any]] = []
    for path in rows:
        try:
            with path.open("r") as f:
                out.append(json.load(f))
        except Exception as exc:  # noqa: BLE001
            print(f"[g2] skip {path}: {exc}")
    return out


def _outcome_vector(records: List[Dict[str, Any]], *, prefer_harmful: bool) -> Optional[List[int]]:
    """Aggregate per-prompt outcomes across seeds by majority vote.

    Returns a 0/1 vector aligned to prompts (vote across seeds: each prompt
    is "harmful" if >= ceil(n_seeds/2) seeds judged it harmful). This is the
    standard paired-bootstrap input when seeds differ.

    For HarmBench ASR (lower-is-better), we report ``harmful`` outcomes (1 = bad).
    For refusal-rate fallback (higher-is-better), we report ``refused`` outcomes.
    """
    if not records:
        return None
    key_primary = "per_prompt_harmful" if prefer_harmful else "per_prompt_refusal"

    # Collect per-seed vectors; bail if missing.
    seed_vecs: List[List[int]] = []
    for r in records:
        if r.get("qresafe_failed"):
            continue
        v: Optional[List[int]] = None
        if key_primary in r:
            v = list(r[key_primary])
        elif "harmbench" in r and isinstance(r["harmbench"], dict):
            v = r["harmbench"].get("per_prompt_harmful")
        if v is None:
            # try refusal fallback
            v = r.get("per_prompt_refusal")
        if v is None:
            return None
        seed_vecs.append([int(x) for x in v])
    if not seed_vecs:
        return None
    # Ensure all same length (else truncate to min)
    n = min(len(v) for v in seed_vecs)
    seed_vecs = [v[:n] for v in seed_vecs]
    # Majority vote
    out: List[int] = []
    threshold = (len(seed_vecs) // 2) + 1
    for i in range(n):
        s = sum(v[i] for v in seed_vecs)
        out.append(1 if s >= threshold else 0)
    return out


def _build_outcomes(
    sweep_root: Path,
    model_tag: str,
    budget: float,
    methods: List[str],
    *,
    prefer_harmful: bool,
) -> Dict[str, List[int]]:
    """For a fixed (model_tag, budget), collect per-method outcome vectors."""
    out: Dict[str, List[int]] = {}
    for method in methods:
        # Budget label rules mirror cmd_sweep's run_name:
        #   fp16, int4, qresafe   -> b0.0
        #   cwp_published         -> b0.6
        #   else                  -> b<budget>
        if method in {"fp16", "int4", "qresafe"}:
            run_budget = 0.0
        elif method == "cwp_published":
            run_budget = 0.6
        else:
            run_budget = budget
        run_name = f"sweep_{model_tag}_{method}_b{run_budget}"
        eval_dir = sweep_root / run_name
        if not eval_dir.is_dir():
            print(f"[g2] missing run dir: {eval_dir}")
            continue
        per_seed = _load_per_seed(eval_dir)
        vec = _outcome_vector(per_seed, prefer_harmful=prefer_harmful)
        if vec is None:
            print(f"[g2] missing outcome vector for {run_name}")
            continue
        out[method] = vec
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="G2 paired-stats gate test from sweep output",
    )
    parser.add_argument("--sweep_root", type=str, required=True)
    parser.add_argument("--model_tags", type=str, nargs="+", required=True)
    parser.add_argument("--budgets", type=float, nargs="+", required=True)
    parser.add_argument(
        "--methods", type=str, nargs="+",
        default=["ssmp", "gradient_only", "snip", "magnitude"],
        help="Methods to include. SSMP is the test method; the rest are "
        "baseline scorers.",
    )
    parser.add_argument("--ssmp_method", type=str, default="ssmp")
    parser.add_argument(
        "--baselines", type=str, nargs="+",
        default=["gradient_only", "snip", "magnitude"],
        help="Baselines to compare SSMP against for G2.",
    )
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument(
        "--use_refusal_fallback", action="store_true",
        help="Use per_prompt_refusal instead of per_prompt_harmful (higher-is-better)",
    )
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()

    # Use the lower-level primitives directly so we can collect raw p-values
    # across the baseline family inside each cell and apply ONE Holm
    # correction across {Fisher, SNIP, magnitude}. The earlier version of
    # this script looped one baseline at a time and called
    # ``compare_methods_paired`` with a pair of size 1 -- which makes the
    # Holm correction trivially equal to no correction. The G2 gate
    # requires SSMP to beat ALL 3 baselines simultaneously after family-wise
    # correction, so the family must be the set of baselines per cell.
    from safepress.eval.stats import (
        holm_bonferroni,
        mcnemar_paired,
        paired_bootstrap_diff,
    )

    sweep_root = Path(args.sweep_root)
    rows: List[Dict[str, Any]] = []
    prefer_harmful = not args.use_refusal_fallback

    for model_tag in args.model_tags:
        for budget in args.budgets:
            outcomes = _build_outcomes(
                sweep_root, model_tag, budget, args.methods,
                prefer_harmful=prefer_harmful,
            )
            if args.ssmp_method not in outcomes:
                print(f"[g2] {model_tag} b={budget}: SSMP outcomes missing; skip cell")
                continue

            # Direction of "SSMP wins" depends on metric polarity:
            #   harmful flag (lower-better) -> diff = SSMP - baseline should be < 0
            #   refused flag (higher-better) -> diff > 0
            target_direction = "less" if prefer_harmful else "greater"

            cell_baselines: List[str] = []
            cell_diffs: List[float] = []
            cell_cis: List[tuple] = []
            cell_raw_p: List[float] = []
            for baseline in args.baselines:
                if baseline not in outcomes:
                    print(f"[g2] {model_tag} b={budget}: missing baseline {baseline}; skip")
                    continue
                ssmp_v = outcomes[args.ssmp_method]
                base_v = outcomes[baseline]
                diff_ci = paired_bootstrap_diff(
                    ssmp_v, base_v,
                    n_bootstrap=10000, seed=0,
                )
                mn = mcnemar_paired(ssmp_v, base_v)
                cell_baselines.append(baseline)
                cell_diffs.append(diff_ci.diff)
                cell_cis.append((diff_ci.ci_low, diff_ci.ci_high))
                cell_raw_p.append(mn.p_value)

            # Family-wise Holm correction ACROSS the baseline family within
            # this (model, budget) cell. This is what the PLAN §16 G2 gate
            # requires.
            if cell_raw_p:
                holm = holm_bonferroni(cell_raw_p, alpha=args.alpha)
                holm_adj = holm["adjusted_p"]
                holm_reject = holm["reject"]
            else:
                holm_adj = []
                holm_reject = []

            for i, baseline in enumerate(cell_baselines):
                diff = cell_diffs[i]
                ci_low, ci_high = cell_cis[i]
                p_raw = cell_raw_p[i]
                p_adj = holm_adj[i] if i < len(holm_adj) else float("nan")
                stat_sig = bool(holm_reject[i] if i < len(holm_reject) else False)
                if target_direction == "less":
                    direction_pass = diff < 0
                else:
                    direction_pass = diff > 0
                cell_pass = bool(direction_pass and stat_sig)
                rows.append({
                    "model_tag": model_tag,
                    "budget": budget,
                    "metric": "harmbench_asr" if prefer_harmful else "refusal_rate",
                    "baseline": baseline,
                    "ssmp_minus_baseline": diff,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "p_raw": p_raw,
                    "p_holm_family": p_adj,
                    "direction_pass": direction_pass,
                    "stat_significant_after_holm": stat_sig,
                    "cell_pass": cell_pass,
                })

    df = pd.DataFrame(rows)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    # G2 verdict: per (model, budget) cell, count if SSMP beats ALL 3 baselines.
    if not df.empty:
        cell_summary = (
            df.groupby(["model_tag", "budget"])["cell_pass"]
            .all()
            .reset_index()
            .rename(columns={"cell_pass": "cell_beats_all_baselines"})
        )
        n_cells_pass = int(cell_summary["cell_beats_all_baselines"].sum())
        n_cells_total = len(cell_summary)
        verdict = "PASS" if n_cells_pass >= 3 else "FAIL"
        print(f"[g2] wrote {len(rows)} comparisons -> {out_path}")
        print(f"[g2] cells where SSMP beats ALL 3 baselines: "
              f"{n_cells_pass}/{n_cells_total}")
        print(f"[g2] gate verdict: {verdict} (need >= 3 of 4 cells; PLAN §16)")
        cell_summary.to_csv(out_path.with_suffix(".cells.csv"), index=False)
    else:
        print("[g2] no comparisons produced. Check --sweep_root / --model_tags / --budgets.")


if __name__ == "__main__":
    main()
