#!/usr/bin/env python3
"""Run all experiments for Llama-3.2-3B-Instruct (replacing Phi-4-Mini)."""
import gc
import sys
import traceback
from pathlib import Path

import torch

RUNS = Path("/home/jis23009/Dev/safepress_repo/runs")
DATA = Path("/home/jis23009/Dev/safepress_repo/data")
MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"
SHORT = "llama32_3b"
SCORES_PATH = RUNS / f"scores/meta-llama_llama-3.2-3b-instruct_scores.csv"


def _cleanup():
    gc.collect()
    torch.cuda.empty_cache()


def load_eval_prompts():
    from safepress.utils.io import read_prompts_jsonl
    return read_prompts_jsonl(DATA / "harmbench_128.jsonl", key="prompt")


def run_scores():
    if SCORES_PATH.exists():
        print(f"=== {SHORT} scores: ALREADY DONE ===")
        return
    print(f"=== Running {SHORT} scores ===", flush=True)
    SCORES_PATH.parent.mkdir(parents=True, exist_ok=True)

    from safepress.model.load import load_fp_model
    from safepress.model.score import compute_block_scores
    from safepress.utils.io import read_prompts_jsonl

    prompts = read_prompts_jsonl(DATA / "advbench.jsonl", key="prompt")[:128]
    loaded = load_fp_model(MODEL_ID, dtype="float16", device_map="auto")

    df = compute_block_scores(
        loaded.model, loaded.tokenizer, prompts,
        bits=4, group_size=128, block_size=64,
        batch_size=1, max_length=512, device=loaded.device,
    )
    df.to_csv(SCORES_PATH, index=False)
    print(f"  Scored {len(df)} blocks", flush=True)
    del loaded; _cleanup()


def run_phase_transition():
    out_dir = RUNS / f"phase_transition_{SHORT}"
    if (out_dir / "phase_transition_results.json").exists():
        print(f"=== {SHORT} phase transition: ALREADY DONE ===")
        return
    print(f"=== Running {SHORT} phase transition ===", flush=True)
    from safepress.experiments.phase_transition import phase_transition_curve
    result = phase_transition_curve(
        model_id=MODEL_ID, eval_prompts=load_eval_prompts(),
        bit_widths=[8, 4, 3, 2], out_dir=str(out_dir),
        device_map="auto", dtype="float16",
    )
    print(f"  Result: {result.summary}", flush=True)
    _cleanup()


def run_ssmp(bits):
    out_dir = RUNS / f"{bits}bit_ssmp_{SHORT}"
    result_file = out_dir / f"ssmp_{bits}bit_results.json"
    if result_file.exists():
        print(f"=== {SHORT} {bits}-bit SSMP: ALREADY DONE ===")
        return
    if not SCORES_PATH.exists():
        print(f"!!! {SHORT} {bits}-bit SSMP: no scores, skipping", flush=True)
        return
    print(f"=== Running {SHORT} {bits}-bit SSMP ===", flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    from safepress.model.load import load_fp_model
    from safepress.model.protect import select_top_blocks
    from safepress.model.split_linear import apply_block_splitting
    from safepress.experiments.phase_transition import _simulate_quantize_inplace, _evaluate_model
    from safepress.utils.logging import save_json
    import pandas as pd

    prompts = load_eval_prompts()
    results = {}

    print(f"  [{bits}bit] FP16 baseline...", flush=True)
    loaded = load_fp_model(MODEL_ID, dtype="float16", device_map="auto")
    results["fp16_baseline"] = _evaluate_model(loaded.model, loaded.tokenizer, prompts, device=loaded.device)
    results["fp16_baseline"]["bits"] = 16
    del loaded; _cleanup()

    print(f"  [{bits}bit] Uniform {bits}-bit...", flush=True)
    loaded = load_fp_model(MODEL_ID, dtype="float16", device_map="auto")
    _simulate_quantize_inplace(loaded.model, bits=bits)
    results[f"uniform_{bits}bit"] = _evaluate_model(loaded.model, loaded.tokenizer, prompts, device=loaded.device)
    results[f"uniform_{bits}bit"]["bits"] = bits
    del loaded; _cleanup()

    scores_df = pd.read_csv(SCORES_PATH)
    for budget in [0.02, 0.04, 0.08]:
        label = f"ssmp_{bits}bit_b{budget}"
        print(f"  [{bits}bit] SSMP budget={budget}...", flush=True)
        loaded = load_fp_model(MODEL_ID, dtype="float16", device_map="auto")
        plan = select_top_blocks(scores_df, budget_ratio=budget, block_size=64)
        apply_block_splitting(loaded.model, plan.protect_map, block_size=64)
        _simulate_quantize_inplace(loaded.model, bits=bits)
        results[label] = _evaluate_model(loaded.model, loaded.tokenizer, prompts, device=loaded.device)
        results[label]["bits"] = bits
        results[label]["budget"] = budget
        del loaded; _cleanup()

    save_json(result_file, {"model_id": MODEL_ID, "conditions": results})
    print(f"  Done: {result_file}", flush=True)


def run_utility():
    out_dir = RUNS / f"utility_{SHORT}"
    result_file = out_dir / "utility_results.json"
    if result_file.exists():
        print(f"=== {SHORT} utility: ALREADY DONE ===")
        return
    print(f"=== Running {SHORT} utility ===", flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    from safepress.model.load import load_fp_model
    from safepress.model.protect import select_top_blocks
    from safepress.model.split_linear import apply_block_splitting
    from safepress.experiments.phase_transition import _simulate_quantize_inplace
    from safepress.eval.utility import eval_perplexity, eval_mmlu_lite, eval_truthfulqa_lite
    from safepress.utils.logging import save_json
    import pandas as pd

    results = {}

    print("  [utility] FP16 baseline...", flush=True)
    loaded = load_fp_model(MODEL_ID, dtype="float16", device_map="auto")
    results["fp16"] = {
        "perplexity": eval_perplexity(loaded.model, loaded.tokenizer, device=loaded.device),
        "mmlu": eval_mmlu_lite(loaded.model, loaded.tokenizer, device=loaded.device, n_questions=200),
        "truthfulqa": eval_truthfulqa_lite(loaded.model, loaded.tokenizer, device=loaded.device),
    }
    del loaded; _cleanup()

    print("  [utility] Uniform 3-bit...", flush=True)
    loaded = load_fp_model(MODEL_ID, dtype="float16", device_map="auto")
    _simulate_quantize_inplace(loaded.model, bits=3)
    results["uniform_3bit"] = {
        "perplexity": eval_perplexity(loaded.model, loaded.tokenizer, device=loaded.device),
        "mmlu": eval_mmlu_lite(loaded.model, loaded.tokenizer, device=loaded.device, n_questions=200),
        "truthfulqa": eval_truthfulqa_lite(loaded.model, loaded.tokenizer, device=loaded.device),
    }
    del loaded; _cleanup()

    if SCORES_PATH.exists():
        scores_df = pd.read_csv(SCORES_PATH)
        print("  [utility] SSMP 3-bit @ 4%...", flush=True)
        loaded = load_fp_model(MODEL_ID, dtype="float16", device_map="auto")
        plan = select_top_blocks(scores_df, budget_ratio=0.04, block_size=64)
        apply_block_splitting(loaded.model, plan.protect_map, block_size=64)
        _simulate_quantize_inplace(loaded.model, bits=3)
        results["ssmp_3bit_b0.04"] = {
            "perplexity": eval_perplexity(loaded.model, loaded.tokenizer, device=loaded.device),
            "mmlu": eval_mmlu_lite(loaded.model, loaded.tokenizer, device=loaded.device, n_questions=200),
            "truthfulqa": eval_truthfulqa_lite(loaded.model, loaded.tokenizer, device=loaded.device),
        }
        del loaded; _cleanup()

    save_json(result_file, results)
    print(f"  Done: {result_file}", flush=True)


def run_drift_bounds():
    out_file = RUNS / f"drift_bounds_{SHORT}.csv"
    if out_file.exists():
        print(f"=== {SHORT} drift bounds: ALREADY DONE ===")
        return
    print(f"=== Running {SHORT} drift bounds ===", flush=True)

    from safepress.model.load import load_fp_model
    from safepress.analysis.drift_bound import compute_module_level_bounds
    from safepress.utils.io import read_prompts_jsonl

    prompts = read_prompts_jsonl(DATA / "advbench.jsonl", key="prompt")[:64]
    loaded = load_fp_model(MODEL_ID, dtype="float16", device_map="auto")
    df = compute_module_level_bounds(
        loaded.model, loaded.tokenizer, prompts,
        bits=4, group_size=128, batch_size=1, max_length=512,
        device=loaded.device,
    )
    df.to_csv(out_file, index=False)
    del loaded; _cleanup()
    print(f"  Done: {out_file}", flush=True)


def main():
    experiments = [
        ("Scores", run_scores),
        ("Phase Transition", run_phase_transition),
        ("8-bit SSMP", lambda: run_ssmp(8)),
        ("3-bit SSMP", lambda: run_ssmp(3)),
        ("4-bit SSMP", lambda: run_ssmp(4)),
        ("Utility", run_utility),
        ("Drift Bounds", run_drift_bounds),
    ]
    total = len(experiments)
    for i, (name, fn) in enumerate(experiments, 1):
        print(f"\n{'='*60}", flush=True)
        print(f"  [{i}/{total}] {SHORT}: {name}", flush=True)
        print(f"{'='*60}", flush=True)
        try:
            fn()
        except Exception as e:
            print(f"!!! FAILED: {name}: {e}", file=sys.stderr, flush=True)
            traceback.print_exc()
            _cleanup()

    print(f"\n=== {SHORT} ALL DONE ===", flush=True)


if __name__ == "__main__":
    main()
