#!/usr/bin/env python3
"""
CLI entry-point for preparing SafePress datasets.

Usage
-----
    python scripts/prepare_data.py --data_dir data/ \
        --sources advbench harmbench strongreject \
        --calib_source c4 --n_calib 128
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

from safepress.data.prepare import prepare_all


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Download and prepare safety / calibration datasets for SafePress.",
    )
    p.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Root directory where JSONL files will be written (default: data/).",
    )
    p.add_argument(
        "--sources",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Safety-prompt sources to download. "
            "Choices: advbench, harmbench, strongreject. "
            "Default: all."
        ),
    )
    p.add_argument(
        "--calib_source",
        type=str,
        default="c4",
        choices=["c4", "wikitext"],
        help="Calibration corpus (default: c4).",
    )
    p.add_argument(
        "--n_calib",
        type=int,
        default=128,
        help="Number of calibration text samples (default: 128).",
    )
    p.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Override HuggingFace / download cache directory.",
    )
    return p


def main(argv: Optional[List[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    prepare_all(
        data_dir=args.data_dir,
        sources=args.sources,
        calib_source=args.calib_source,
        n_calib=args.n_calib,
        cache_dir=args.cache_dir,
    )


if __name__ == "__main__":
    main()
