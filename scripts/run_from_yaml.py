#!/usr/bin/env python3
from __future__ import annotations

import argparse
from safepress.utils.io import read_yaml
from safepress.cli import main as safepress_main


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--cmd", type=str, default="pipeline", choices=["score", "build", "eval", "pipeline"])
    args = ap.parse_args()

    cfg = read_yaml(args.config)

    # Convert config dict to CLI argv.
    argv = [args.cmd]
    for k, v in cfg.items():
        if v is None:
            continue
        flag = f"--{k}"
        if isinstance(v, bool):
            if v:
                argv.append(flag)
        else:
            argv.extend([flag, str(v)])
    safepress_main(argv)


if __name__ == "__main__":
    main()
