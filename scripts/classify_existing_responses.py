#!/usr/bin/env python
"""Score existing responses with the HarmBench classifier — no regeneration.

The prior Qwen3 G2 pilot (``runs/emnlp_g2_pilot/``) wrote
``responses_seed0.jsonl`` for every (method, budget) condition. We never
got HarmBench-classifier ASR for those because the original config had
``use_harmbench_classifier: false`` (heuristic refusal only). Re-running the
sweep with the classifier ON OOMs because scoring needs ~48 GB and the 13B
classifier already takes ~26 GB.

This script side-steps the problem: it loads the HarmBench classifier
ONCE and iterates the existing responses files, computing ASR per
condition. Output: ``<run_dir>/harmbench_asr_seed{seed}.json`` with the
same shape as the ``harmbench`` sub-dict that
``cmd_eval``/``cmd_sweep`` would have produced inline.

Usage::

    python scripts/classify_existing_responses.py \
        --sweep_root runs/emnlp_g2_pilot \
        --seed 0

By default uses the prompt text as the "behavior" (matches the original
sweep's HarmBench classifier path, where ``data/harmbench.jsonl`` prompts
ARE the behavior descriptions). Override with ``--behaviors_jsonl`` if you
want to join via behavior_id instead.

Compute: classifier load ~2 min once, then ~60s per 400-prompt condition
× 10 conditions ≈ 12 min total. Saves 12-15 hours vs re-running the sweep.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional


def _read_responses(path: Path, prompt_key: str = "prompt", response_key: str = "response") -> tuple:
    prompts: List[str] = []
    responses: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            prompts.append(str(row[prompt_key]))
            responses.append(str(row[response_key]))
    return prompts, responses


def main() -> None:
    parser = argparse.ArgumentParser(description="Score existing responses with HarmBench classifier")
    parser.add_argument("--sweep_root", type=str, required=True,
                        help="Sweep root directory (contains sweep_*/responses_seed{seed}.jsonl)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--classifier_id", type=str,
                        default="cais/HarmBench-Llama-2-13b-cls")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-classify even if harmbench_asr_seed{seed}.json already exists")
    parser.add_argument("--device_map", type=str, default="auto")
    args = parser.parse_args()

    from safepress.eval.harmbench import (
        harmbench_attack_success_rate,
        load_harmbench_classifier,
    )
    from safepress.utils.logging import save_json

    sweep_root = Path(args.sweep_root)
    response_files = sorted(sweep_root.glob(f"sweep_*/responses_seed{args.seed}.jsonl"))
    if not response_files:
        raise SystemExit(
            f"No response files matched {sweep_root}/sweep_*/responses_seed{args.seed}.jsonl"
        )
    print(f"[classify] found {len(response_files)} condition response files")

    # Filter to ones that aren't already classified.
    pending = []
    for rf in response_files:
        out_path = rf.parent / f"harmbench_asr_seed{args.seed}.json"
        if out_path.exists() and not args.overwrite:
            print(f"[classify] skip (already done): {out_path.relative_to(sweep_root)}")
            continue
        pending.append(rf)
    if not pending:
        print("[classify] all conditions already have harmbench_asr_seed*.json; pass --overwrite to redo")
        return

    print(f"[classify] {len(pending)} pending; loading classifier...")
    classifier_model, classifier_tokenizer = load_harmbench_classifier(
        model_id=args.classifier_id,
        device_map=args.device_map,
    )

    for rf in pending:
        cond_dir = rf.parent
        cond_name = cond_dir.name
        prompts, responses = _read_responses(rf)
        print(f"[classify] {cond_name}: n={len(prompts)}")
        hb = harmbench_attack_success_rate(
            prompts, responses,
            classifier_model=classifier_model,
            classifier_tokenizer=classifier_tokenizer,
            batch_size=args.batch_size,
            show_progress=True,
        )
        out_path = cond_dir / f"harmbench_asr_seed{args.seed}.json"
        save_json(out_path, {
            "asr": hb["asr"],
            "n_harmful": hb["n_harmful"],
            "n_total": hb["n_total"],
            "per_prompt_harmful": [int(x) for x in hb["per_behavior"]],
            "classifier_id": args.classifier_id,
            "source_responses": str(rf.relative_to(sweep_root)),
        })
        print(f"[classify]   ASR={hb['asr']:.4f}  ({hb['n_harmful']}/{hb['n_total']})  -> {out_path.relative_to(sweep_root)}")

    print("[classify] done")


if __name__ == "__main__":
    main()
