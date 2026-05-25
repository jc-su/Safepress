#!/usr/bin/env python
"""Reconstruct each sweep condition's protected+quantized model and measure
WikiText-2 perplexity + MMLU-lite accuracy.

Why this script exists
----------------------
The G2 pilots saved ``protect_map_seed0.json`` + responses but never wrote
utility numbers (the sweep config had ``use_perplexity / use_mmlu`` off).
Without utility data we cannot tell whether at 3-bit Qwen3 the protection
methods recover coherent capability or just produce more "I'm sorry…"-style
text on top of incoherent generations. Re-running the full sweep with
utility on would cost another 10-15 h per model; rebuilding each condition's
model JUST for forward-pass utility metrics (no generation) is much
cheaper (~5 min per condition).

Per condition we:
  1. Load a FRESH FP16 base model.
  2. If protect_map is non-empty, apply ``apply_block_splitting``.
  3. Quantize the unprotected sub-modules via ``quantize_for_bits(bits=3,
     group_size=128)`` -- the same path the sweep uses.
  4. Compute WikiText-2 PPL via ``safepress.eval.utility.eval_perplexity``.
  5. Compute MMLU-lite via ``safepress.eval.utility.eval_mmlu_lite``.
  6. Save ``utility_seed{seed}.json`` to the same condition directory.

For ``fp16`` conditions we skip steps 2-3. For ``int4`` (uniform 3-bit) the
saved protect_map is empty by construction; only step 3 applies.

Usage::

    python scripts/utility_eval_from_protect_maps.py \\
        --sweep_root runs/emnlp_g2_pilot \\
        --model_id Qwen/Qwen3-8B \\
        --bits 3 --group_size 128 --block_size 64 \\
        --seed 0
"""
from __future__ import annotations

import argparse
import gc
import json
import re
from pathlib import Path
from typing import Optional


_CONDITION_RE = re.compile(r"sweep_[^/]+_(?P<method>[a-z0-9_]+?)_b(?P<budget>[0-9.]+)$")


def _parse_condition(dir_name: str) -> tuple[str, float]:
    m = _CONDITION_RE.match(dir_name)
    if not m:
        raise ValueError(f"could not parse condition from {dir_name!r}")
    return m.group("method"), float(m.group("budget"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep_root", type=str, required=True)
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bits", type=int, default=3)
    parser.add_argument("--group_size", type=int, default=128)
    parser.add_argument("--block_size", type=int, default=64)
    parser.add_argument("--dtype", type=str, default="float16",
                        choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--device_map", type=str, default="auto")
    # PPL settings (small to keep total wall time low)
    parser.add_argument("--ppl_n_samples", type=int, default=64)
    parser.add_argument("--ppl_max_length", type=int, default=1024)
    parser.add_argument("--ppl_stride", type=int, default=512)
    # MMLU settings
    parser.add_argument("--mmlu_n", type=int, default=200)
    parser.add_argument("--mmlu_subjects", type=str, nargs="+", default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    import torch

    from safepress.model.load import load_fp_model
    from safepress.model.protect import load_protect_plan
    from safepress.model.split_linear import apply_block_splitting
    from safepress.model.quantize import quantize_for_bits
    from safepress.eval.utility import eval_perplexity, eval_mmlu_lite
    from safepress.utils.logging import save_json

    sweep_root = Path(args.sweep_root)
    plan_files = sorted(sweep_root.glob(f"sweep_*/protect_map_seed{args.seed}.json"))
    if not plan_files:
        raise SystemExit(f"No protect_map_seed{args.seed}.json files under {sweep_root}")
    print(f"[util] found {len(plan_files)} condition protect_maps")

    pending = []
    for pf in plan_files:
        out_path = pf.parent / f"utility_seed{args.seed}.json"
        if out_path.exists() and not args.overwrite:
            print(f"[util] skip (already done): {out_path.relative_to(sweep_root)}")
            continue
        pending.append(pf)
    if not pending:
        print("[util] all conditions already have utility_seed*.json; pass --overwrite to redo")
        return
    print(f"[util] {len(pending)} pending")

    for pf in pending:
        cond_dir = pf.parent
        cond_name = cond_dir.name
        try:
            method, budget = _parse_condition(cond_name)
        except ValueError:
            print(f"[util] skip un-parseable: {cond_name}")
            continue
        print(f"[util] === {cond_name}  (method={method}, budget={budget}) ===")

        # 1) load FRESH FP16 base
        loaded = load_fp_model(
            args.model_id,
            dtype=args.dtype,
            device_map=args.device_map,
        )
        try:
            # 2) for non-fp16 methods: load + apply protect_map (if any), then quantize
            if method.lower() != "fp16":
                plan = load_protect_plan(pf)
                if plan.protect_map:
                    apply_block_splitting(
                        loaded.model, plan.protect_map, block_size=args.block_size,
                    )
                    print(f"[util]   applied block_splitting on {len(plan.protect_map)} modules")
                # always quantize the unprotected modules
                skip_modules: list[str] = []
                if plan.protect_map:
                    # mimic the sweep's logic: split_report.protected_modules_to_skip
                    # is what's emitted by apply_block_splitting; we need to derive it
                    # here. Easier: just rebuild via the SplitLinear-aware quantizer
                    # which already knows to skip protected sub-modules by suffix.
                    # We don't have the split_report dataclass handy, so we
                    # heuristically skip any module whose name ends with
                    # ``.protected`` (the SplitLinear sub-module naming).
                    pass
                # Skip-list: any Linear whose full name ends with ``.protected``
                # (created by apply_block_splitting).
                from safepress.model.blocks import iter_linear_modules
                skip_modules = [
                    name for name, _ in iter_linear_modules(loaded.model)
                    if name.endswith(".protected")
                ]
                qrep = quantize_for_bits(
                    loaded.model,
                    bits=args.bits,
                    group_size=args.group_size,
                    modules_to_not_convert=skip_modules,
                )
                print(f"[util]   quantized ({qrep.backend}): {qrep.note}")

            # 3) PPL
            ppl = eval_perplexity(
                loaded.model, loaded.tokenizer,
                n_samples=args.ppl_n_samples,
                max_length=args.ppl_max_length,
                stride=args.ppl_stride,
                device=loaded.device,
            )
            # 4) MMLU-lite
            try:
                mmlu = eval_mmlu_lite(
                    loaded.model, loaded.tokenizer,
                    n_questions=args.mmlu_n,
                    subjects=args.mmlu_subjects,
                    device=loaded.device,
                )
            except Exception as exc:
                mmlu = {"ok": False, "error": str(exc)}

            out = {
                "method": method,
                "budget": budget,
                "seed": args.seed,
                "bits": args.bits,
                "perplexity": ppl,
                "mmlu": mmlu,
            }
            out_path = cond_dir / f"utility_seed{args.seed}.json"
            save_json(out_path, out)
            ppl_val = ppl.get("perplexity") if isinstance(ppl, dict) else None
            mmlu_acc = mmlu.get("accuracy") if isinstance(mmlu, dict) else None
            print(f"[util]   PPL={ppl_val}  MMLU={mmlu_acc}  -> {out_path.relative_to(sweep_root)}")
        finally:
            del loaded
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print("[util] done")


if __name__ == "__main__":
    main()
