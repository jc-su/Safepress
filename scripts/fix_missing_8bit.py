#!/usr/bin/env python3
"""Fix missing 8-bit SSMP experiments for phi35 and gemma2_9b."""
import gc
import sys
import traceback
from pathlib import Path

import torch

RUNS = Path("/home/jis23009/Dev/safepress_repo/runs")
DATA = Path("/home/jis23009/Dev/safepress_repo/data")

MODELS = {
    "phi35": {
        "model_id": "microsoft/Phi-3.5-mini-instruct",
        "scores": str(RUNS / "scores/microsoft_phi-3.5-mini-instruct_scores.csv"),
    },
    "gemma2_9b": {
        "model_id": "google/gemma-2-9b-it",
        "scores": str(RUNS / "scores/google_gemma-2-9b-it_scores.csv"),
    },
}


def _cleanup():
    gc.collect()
    torch.cuda.empty_cache()


def load_eval_prompts():
    from safepress.utils.io import read_prompts_jsonl
    return read_prompts_jsonl(DATA / "harmbench_128.jsonl", key="prompt")


def run_ssmp_8bit(model_short):
    """Run 8-bit SSMP experiment for a model."""
    m = MODELS[model_short]
    out_dir = RUNS / f"8bit_ssmp_{model_short}"
    result_file = out_dir / "ssmp_8bit_results.json"

    if result_file.exists():
        print(f"=== {model_short} 8-bit SSMP: ALREADY DONE ===")
        return

    scores_path = Path(m["scores"])
    if not scores_path.exists():
        print(f"!!! {model_short} 8-bit SSMP: scores not found at {scores_path}", flush=True)
        return

    print(f"=== Running {model_short} 8-bit SSMP ===", flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    from safepress.model.load import load_fp_model
    from safepress.model.protect import select_top_blocks
    from safepress.model.split_linear import apply_block_splitting
    from safepress.experiments.phase_transition import _simulate_quantize_inplace, _evaluate_model
    from safepress.utils.logging import save_json
    import pandas as pd

    prompts = load_eval_prompts()
    results = {}
    bits = 8

    # FP16 baseline
    print(f"  [8bit] FP16 baseline...", flush=True)
    loaded = load_fp_model(m["model_id"], dtype="float16", device_map="auto",
                           trust_remote_code=True)
    results["fp16_baseline"] = _evaluate_model(
        loaded.model, loaded.tokenizer, prompts, device=loaded.device)
    results["fp16_baseline"]["bits"] = 16
    del loaded; _cleanup()

    # Uniform 8-bit
    print(f"  [8bit] Uniform 8-bit...", flush=True)
    loaded = load_fp_model(m["model_id"], dtype="float16", device_map="auto",
                           trust_remote_code=True)
    _simulate_quantize_inplace(loaded.model, bits=bits)
    results[f"uniform_8bit"] = _evaluate_model(
        loaded.model, loaded.tokenizer, prompts, device=loaded.device)
    results[f"uniform_8bit"]["bits"] = bits
    del loaded; _cleanup()

    # SSMP at budgets 2%, 4%, 8%
    scores_df = pd.read_csv(scores_path)
    for budget in [0.02, 0.04, 0.08]:
        label = f"ssmp_8bit_b{budget}"
        print(f"  [8bit] SSMP 8-bit budget={budget}...", flush=True)
        loaded = load_fp_model(m["model_id"], dtype="float16", device_map="auto",
                               trust_remote_code=True)

        plan = select_top_blocks(scores_df, budget_ratio=budget, block_size=64)
        apply_block_splitting(loaded.model, plan.protect_map, block_size=64)
        _simulate_quantize_inplace(loaded.model, bits=bits)

        results[label] = _evaluate_model(
            loaded.model, loaded.tokenizer, prompts, device=loaded.device)
        results[label]["bits"] = bits
        results[label]["budget"] = budget
        del loaded; _cleanup()

    save_json(result_file, {"model_id": m["model_id"], "conditions": results})
    print(f"  Done: {result_file}", flush=True)


def main():
    print("=" * 60)
    print("  Fixing Missing 8-bit SSMP Experiments")
    print("=" * 60)

    failed = []
    for model_short in ["phi35", "gemma2_9b"]:
        print(f"\n{'='*60}", flush=True)
        print(f"  Running 8-bit SSMP for {model_short}", flush=True)
        print(f"{'='*60}", flush=True)
        try:
            run_ssmp_8bit(model_short)
        except Exception as e:
            print(f"!!! FAILED: {model_short}: {e}", file=sys.stderr, flush=True)
            traceback.print_exc()
            failed.append(model_short)
            _cleanup()

    print(f"\n{'='*60}", flush=True)
    if failed:
        print(f"=== FAILED: {failed} ===", flush=True)
    else:
        print("=== COMPLETED: All 8-bit experiments fixed ===", flush=True)
    print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    main()
