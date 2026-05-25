#!/usr/bin/env python3
"""Generate utility prompts from Alpaca dataset for CWP baseline.

Usage:
    python scripts/prepare_utility_prompts.py --out data/utility_alpaca.jsonl --n 1000
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate utility prompts from Alpaca for CWP baseline.",
    )
    ap.add_argument("--out", type=str, required=True, help="Output JSONL path")
    ap.add_argument("--n", type=int, default=1000, help="Number of prompts")
    ap.add_argument("--seed", type=int, default=0, help="Random seed for shuffle")
    args = ap.parse_args()

    try:
        from datasets import load_dataset
    except Exception as e:
        raise SystemExit(
            "This script requires the `datasets` package. "
            f"Import error: {e}"
        )

    ds = load_dataset("tatsu-lab/alpaca", split="train")
    ds = ds.shuffle(seed=int(args.seed))
    ds = ds.select(range(min(int(args.n), len(ds))))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for ex in ds:
            inst = ex.get("instruction") or ""
            inp = ex.get("input") or ""
            prompt = inst if not inp else f"{inst}\n\n{inp}"
            f.write(json.dumps({"prompt": str(prompt)}, ensure_ascii=False) + "\n")

    print(f"Wrote {len(ds)} prompts -> {out}")


if __name__ == "__main__":
    main()
