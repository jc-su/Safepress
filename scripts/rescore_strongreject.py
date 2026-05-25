#!/usr/bin/env python3
"""Batch re-score all saved responses with StrongREJECT.

This reads responses.jsonl files from experiment directories and re-scores them
using the StrongREJECT evaluator (requires OPENAI_API_KEY env var).

Usage:
  export OPENAI_API_KEY=...
  python scripts/rescore_strongreject.py --runs_dir runs --out tables/strongreject_scores.json
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from safepress.eval.basic import try_strongreject_eval
from safepress.utils.logging import save_json


def find_response_files(runs_dir: Path):
    """Find all responses.jsonl files recursively."""
    files = sorted(runs_dir.rglob("responses.jsonl"))
    return files


def load_responses(path: Path):
    """Load prompts and responses from a responses.jsonl file."""
    prompts = []
    responses = []
    with path.open() as f:
        for line in f:
            row = json.loads(line)
            prompts.append(row["prompt"])
            responses.append(row["response"])
    return prompts, responses


def main():
    parser = argparse.ArgumentParser(description="Batch StrongREJECT re-scoring")
    parser.add_argument("--runs_dir", default="runs", help="Root directory of experiment runs")
    parser.add_argument("--out", default="tables/strongreject_scores.json", help="Output JSON file")
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip experiments that already have valid StrongREJECT scores")
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    response_files = find_response_files(runs_dir)
    print(f"Found {len(response_files)} response files")

    if not os.environ.get("OPENAI_API_KEY"):
        print("WARNING: OPENAI_API_KEY not set. StrongREJECT rubric evaluator will fail.")
        print("Set it with: export OPENAI_API_KEY=...")
        return

    results = {}
    for i, resp_file in enumerate(response_files):
        # Build a label from the path: e.g., "3bit_ssmp_qwen3/ssmp_3bit_b0.04"
        rel = resp_file.relative_to(runs_dir)
        parts = list(rel.parts)
        # Remove "responses.jsonl" from the end
        label = "/".join(parts[:-1])

        # Check if existing eval.json has valid StrongREJECT score
        if args.skip_existing:
            eval_path = resp_file.parent / "eval.json"
            if eval_path.exists():
                try:
                    with open(eval_path) as f:
                        content = f.read().replace("NaN", "null")
                    eval_data = json.loads(content)
                    sr = eval_data.get("strongreject", {})
                    if sr.get("ok") and sr.get("n_scored", 0) > 0:
                        print(f"  [{i+1}/{len(response_files)}] {label}: already scored ({sr['n_scored']} scored), skipping")
                        results[label] = sr
                        continue
                except Exception:
                    pass

        print(f"  [{i+1}/{len(response_files)}] Scoring: {label} ...")
        try:
            prompts, responses = load_responses(resp_file)
            sr = try_strongreject_eval(prompts, responses)
            results[label] = sr
            print(f"    -> avg_score={sr.get('avg_score', 'N/A')}, n_scored={sr.get('n_scored', 0)}/{sr.get('n_total', 0)}")

            # Also update the eval.json in-place
            eval_path = resp_file.parent / "eval.json"
            if eval_path.exists():
                try:
                    with open(eval_path) as f:
                        content = f.read().replace("NaN", "null")
                    eval_data = json.loads(content)
                    eval_data["strongreject"] = sr
                    save_json(eval_path, eval_data)
                except Exception as e:
                    print(f"    WARNING: Could not update eval.json: {e}")

        except Exception as e:
            print(f"    ERROR: {e}")
            results[label] = {"ok": False, "error": str(e)}

    # Save aggregate results
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(out_path, results)
    print(f"\nSaved StrongREJECT scores to {out_path}")


if __name__ == "__main__":
    main()
