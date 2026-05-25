#!/usr/bin/env python
"""Run JBB-GCG attack ASR + XSTest over-refusal on each sweep condition.

For each protect_map_seed*.json under a sweep_root, rebuild the
protected+quantized model and evaluate against:
  1. ``data/harmbench_attacks_gcg.jsonl`` (GCG adversarial prompts) -> ASR
  2. ``data/xstest.jsonl`` (XSTest safe/unsafe pairs) -> over-refusal

Uses heuristic refusal detector for both (avoids loading the HarmBench
classifier alongside the target model on a single 48GB GPU). The heuristic
matches the standard `is_refusal` definition used elsewhere in this repo.

Usage::

    python scripts/attack_xstest_eval_from_protect_maps.py \\
        --sweep_root runs/emnlp_fisher60/qwen3 \\
        --model_id Qwen/Qwen3-8B \\
        --bits 3
"""
from __future__ import annotations

import argparse
import gc
import re
from pathlib import Path

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
    # Eval prompts
    parser.add_argument("--gcg_attacks", type=str, default="data/harmbench_attacks_gcg.jsonl")
    parser.add_argument("--xstest_prompts", type=str, default="data/xstest.jsonl")
    parser.add_argument("--harmbench_behaviors", type=str, default="data/harmbench.jsonl")
    # Generation settings
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--max_prompts", type=int, default=100,
                        help="cap for both GCG (~100 available) and XSTest per-label")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip_gcg", action="store_true")
    parser.add_argument("--skip_xstest", action="store_true")
    args = parser.parse_args()

    import torch

    from safepress.model.load import load_fp_model
    from safepress.model.protect import load_protect_plan
    from safepress.model.split_linear import apply_block_splitting
    from safepress.model.quantize import quantize_for_bits
    from safepress.eval.jailbreaks import JailbreakConfig, run_jailbreak_eval
    from safepress.eval.xstest import XSTestConfig, run_xstest
    from safepress.utils.logging import save_json
    from safepress.utils.seed import set_seed

    sweep_root = Path(args.sweep_root)
    plan_files = sorted(sweep_root.glob(f"sweep_*/protect_map_seed{args.seed}.json"))
    if not plan_files:
        raise SystemExit(f"No protect_map_seed{args.seed}.json files under {sweep_root}")
    print(f"[atk] found {len(plan_files)} condition protect_maps")

    pending = []
    for pf in plan_files:
        cond_dir = pf.parent
        gcg_out = cond_dir / f"jailbreak_gcg_seed{args.seed}.json"
        xs_out = cond_dir / f"xstest_seed{args.seed}.json"
        need_gcg = not args.skip_gcg and (not gcg_out.exists() or args.overwrite)
        need_xs = not args.skip_xstest and (not xs_out.exists() or args.overwrite)
        if need_gcg or need_xs:
            pending.append((pf, need_gcg, need_xs))
    if not pending:
        print("[atk] nothing pending; pass --overwrite to redo")
        return
    print(f"[atk] {len(pending)} pending")

    for pf, need_gcg, need_xs in pending:
        cond_dir = pf.parent
        cond_name = cond_dir.name
        try:
            method, budget = _parse_condition(cond_name)
        except ValueError:
            print(f"[atk] skip un-parseable: {cond_name}")
            continue
        print(f"[atk] === {cond_name}  (method={method}, budget={budget}) ===")

        set_seed(int(args.seed), deterministic=True)

        loaded = load_fp_model(
            args.model_id, dtype=args.dtype, device_map=args.device_map,
        )
        try:
            if method.lower() != "fp16":
                plan = load_protect_plan(pf)
                if plan.protect_map:
                    apply_block_splitting(
                        loaded.model, plan.protect_map, block_size=args.block_size,
                    )
                    print(f"[atk]   applied block_splitting on {len(plan.protect_map)} modules")
                from safepress.model.blocks import iter_linear_modules
                skip_modules = [
                    name for name, _ in iter_linear_modules(loaded.model)
                    if name.endswith(".protected")
                ]
                qrep = quantize_for_bits(
                    loaded.model, bits=args.bits, group_size=args.group_size,
                    modules_to_not_convert=skip_modules,
                )
                print(f"[atk]   quantized ({qrep.backend}): {qrep.note}")

            # GCG jailbreak
            if need_gcg:
                jcfg = JailbreakConfig(
                    attack_prompts_path=args.gcg_attacks,
                    behaviors_path=args.harmbench_behaviors,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    do_sample=False,
                    max_length=args.max_length,
                    max_prompts=args.max_prompts,
                    use_harmbench_classifier=False,  # heuristic ASR
                )
                jres = run_jailbreak_eval(loaded.model, loaded.tokenizer, jcfg,
                                          device=loaded.device, show_progress=False)
                jres.pop("records", None)
                jres["method"] = method
                jres["budget"] = budget
                jres["seed"] = args.seed
                jres["bits"] = args.bits
                gcg_out = cond_dir / f"jailbreak_gcg_seed{args.seed}.json"
                save_json(gcg_out, jres)
                print(f"[atk]   GCG ASR={jres.get('asr', float('nan')):.3f}  n={jres.get('n')}  -> {gcg_out.relative_to(sweep_root)}")

            # XSTest
            if need_xs:
                xcfg = XSTestConfig(
                    prompts_path=args.xstest_prompts,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    do_sample=False,
                    max_length=args.max_length,
                    max_prompts_per_label=args.max_prompts,
                )
                xres = run_xstest(loaded.model, loaded.tokenizer, xcfg,
                                  device=loaded.device, show_progress=False)
                xres["method"] = method
                xres["budget"] = budget
                xres["seed"] = args.seed
                xres["bits"] = args.bits
                xs_out = cond_dir / f"xstest_seed{args.seed}.json"
                save_json(xs_out, xres)
                print(f"[atk]   XSTest safe_compliance={xres.get('safe_compliance_rate', float('nan')):.3f}  unsafe_refusal={xres.get('unsafe_refusal_rate', float('nan')):.3f}  score={xres.get('safety_score', float('nan')):.3f}")
        finally:
            del loaded
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print("[atk] done")


if __name__ == "__main__":
    main()
