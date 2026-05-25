#!/usr/bin/env python3
"""Run all missing experiments for Qwen2.5-14B and Gemma-3-12B."""
import gc
import json
import sys
import traceback
from pathlib import Path

import torch

RUNS = Path("/home/jis23009/Dev/safepress_repo/runs")
DATA = Path("/home/jis23009/Dev/safepress_repo/data")


def load_eval_prompts():
    """Load 128 HarmBench prompts from pre-prepared JSONL."""
    from safepress.utils.io import read_prompts_jsonl
    return read_prompts_jsonl(DATA / "harmbench_128.jsonl", key="prompt")


# ── Phase transition ─────────────────────────────────────────────────────
def run_phase_transition_qwen25():
    out_dir = RUNS / "phase_transition_qwen25_14b"
    final = out_dir / "phase_transition_results.json"
    if final.exists():
        print("=== Qwen2.5-14B phase transition: ALREADY DONE ===")
        return
    print("=== Running Qwen2.5-14B phase transition ===", flush=True)
    from safepress.experiments.phase_transition import phase_transition_curve
    prompts = load_eval_prompts()
    result = phase_transition_curve(
        model_id="Qwen/Qwen2.5-14B-Instruct",
        eval_prompts=prompts,
        bit_widths=[8, 4, 3, 2],
        out_dir=str(out_dir),
        device_map="auto",
        dtype="float16",
    )
    print(f"  Result: {result.summary}", flush=True)
    gc.collect(); torch.cuda.empty_cache()


# ── 3-bit SSMP ───────────────────────────────────────────────────────────
def run_3bit_ssmp(model_id, model_short, scores_csv):
    out_dir = RUNS / f"3bit_ssmp_{model_short}"
    result_file = out_dir / "ssmp_3bit_results.json"
    if result_file.exists():
        print(f"=== {model_short} 3-bit SSMP: ALREADY DONE ===")
        return
    print(f"=== Running {model_short} 3-bit SSMP ===", flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    from safepress.model.load import load_fp_model
    from safepress.model.protect import select_top_blocks
    from safepress.model.split_linear import apply_block_splitting
    from safepress.experiments.phase_transition import _simulate_quantize_inplace, _evaluate_model
    from safepress.utils.logging import save_json
    import pandas as pd

    prompts = load_eval_prompts()
    results = {}

    # FP16 baseline
    print("  [3bit] FP16 baseline...", flush=True)
    loaded = load_fp_model(model_id, dtype="float16", device_map="auto")
    results["fp16_baseline"] = _evaluate_model(loaded.model, loaded.tokenizer, prompts, device=loaded.device)
    results["fp16_baseline"]["bits"] = 16
    del loaded; gc.collect(); torch.cuda.empty_cache()

    # Uniform 3-bit
    print("  [3bit] Uniform 3-bit...", flush=True)
    loaded = load_fp_model(model_id, dtype="float16", device_map="auto")
    _simulate_quantize_inplace(loaded.model, bits=3)
    results["uniform_3bit"] = _evaluate_model(loaded.model, loaded.tokenizer, prompts, device=loaded.device)
    results["uniform_3bit"]["bits"] = 3
    del loaded; gc.collect(); torch.cuda.empty_cache()

    # SSMP 3-bit at budgets 2%, 4%, 8%
    scores_df = pd.read_csv(scores_csv)
    for budget in [0.02, 0.04, 0.08]:
        label = f"ssmp_3bit_b{budget}"
        print(f"  [3bit] SSMP 3-bit budget={budget}...", flush=True)
        loaded = load_fp_model(model_id, dtype="float16", device_map="auto")

        plan = select_top_blocks(scores_df, budget_ratio=budget, block_size=64)
        apply_block_splitting(loaded.model, plan.protect_map, block_size=64)
        _simulate_quantize_inplace(loaded.model, bits=3)

        results[label] = _evaluate_model(loaded.model, loaded.tokenizer, prompts, device=loaded.device)
        results[label]["bits"] = 3
        results[label]["budget"] = budget
        del loaded; gc.collect(); torch.cuda.empty_cache()

    save_json(result_file, {"model_id": model_id, "conditions": results})
    print(f"  Done: {result_file}", flush=True)


# ── 4-bit SSMP ───────────────────────────────────────────────────────────
def run_4bit_ssmp(model_id, model_short, scores_csv):
    out_dir = RUNS / f"4bit_ssmp_{model_short}"
    result_file = out_dir / "ssmp_4bit_results.json"
    if result_file.exists():
        print(f"=== {model_short} 4-bit SSMP: ALREADY DONE ===")
        return
    print(f"=== Running {model_short} 4-bit SSMP ===", flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    from safepress.model.load import load_fp_model
    from safepress.model.protect import select_top_blocks
    from safepress.model.split_linear import apply_block_splitting
    from safepress.experiments.phase_transition import _simulate_quantize_inplace, _evaluate_model
    from safepress.utils.logging import save_json
    import pandas as pd

    prompts = load_eval_prompts()
    results = {}

    # FP16 baseline
    print("  [4bit] FP16 baseline...", flush=True)
    loaded = load_fp_model(model_id, dtype="float16", device_map="auto")
    results["fp16_baseline"] = _evaluate_model(loaded.model, loaded.tokenizer, prompts, device=loaded.device)
    results["fp16_baseline"]["bits"] = 16
    del loaded; gc.collect(); torch.cuda.empty_cache()

    # Uniform 4-bit
    print("  [4bit] Uniform 4-bit...", flush=True)
    loaded = load_fp_model(model_id, dtype="float16", device_map="auto")
    _simulate_quantize_inplace(loaded.model, bits=4)
    results["uniform_4bit"] = _evaluate_model(loaded.model, loaded.tokenizer, prompts, device=loaded.device)
    results["uniform_4bit"]["bits"] = 4
    del loaded; gc.collect(); torch.cuda.empty_cache()

    # SSMP 4-bit at budgets 2%, 4%, 8%
    scores_df = pd.read_csv(scores_csv)
    for budget in [0.02, 0.04, 0.08]:
        label = f"ssmp_4bit_b{budget}"
        print(f"  [4bit] SSMP 4-bit budget={budget}...", flush=True)
        loaded = load_fp_model(model_id, dtype="float16", device_map="auto")

        plan = select_top_blocks(scores_df, budget_ratio=budget, block_size=64)
        apply_block_splitting(loaded.model, plan.protect_map, block_size=64)
        _simulate_quantize_inplace(loaded.model, bits=4)

        results[label] = _evaluate_model(loaded.model, loaded.tokenizer, prompts, device=loaded.device)
        results[label]["bits"] = 4
        results[label]["budget"] = budget
        del loaded; gc.collect(); torch.cuda.empty_cache()

    save_json(result_file, {"model_id": model_id, "conditions": results})
    print(f"  Done: {result_file}", flush=True)


# ── Utility evaluation ───────────────────────────────────────────────────
def run_utility(model_id, model_short, scores_csv):
    out_dir = RUNS / f"utility_{model_short}"
    result_file = out_dir / "utility_results.json"
    if result_file.exists():
        print(f"=== {model_short} utility: ALREADY DONE ===")
        return
    print(f"=== Running {model_short} utility ===", flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    from safepress.model.load import load_fp_model
    from safepress.model.protect import select_top_blocks
    from safepress.model.split_linear import apply_block_splitting
    from safepress.experiments.phase_transition import _simulate_quantize_inplace
    from safepress.eval.utility import eval_perplexity, eval_mmlu_lite, eval_truthfulqa_lite
    from safepress.utils.logging import save_json
    import pandas as pd

    results = {}

    # FP16 baseline
    print("  [utility] FP16 baseline...", flush=True)
    loaded = load_fp_model(model_id, dtype="float16", device_map="auto")
    results["fp16"] = {
        "perplexity": eval_perplexity(loaded.model, loaded.tokenizer, device=loaded.device),
        "mmlu": eval_mmlu_lite(loaded.model, loaded.tokenizer, device=loaded.device, n_questions=200),
        "truthfulqa": eval_truthfulqa_lite(loaded.model, loaded.tokenizer, device=loaded.device),
    }
    del loaded; gc.collect(); torch.cuda.empty_cache()

    # Uniform 3-bit
    print("  [utility] Uniform 3-bit...", flush=True)
    loaded = load_fp_model(model_id, dtype="float16", device_map="auto")
    _simulate_quantize_inplace(loaded.model, bits=3)
    results["uniform_3bit"] = {
        "perplexity": eval_perplexity(loaded.model, loaded.tokenizer, device=loaded.device),
        "mmlu": eval_mmlu_lite(loaded.model, loaded.tokenizer, device=loaded.device, n_questions=200),
        "truthfulqa": eval_truthfulqa_lite(loaded.model, loaded.tokenizer, device=loaded.device),
    }
    del loaded; gc.collect(); torch.cuda.empty_cache()

    # SSMP 3-bit at 4% budget
    if scores_csv and Path(scores_csv).exists():
        scores_df = pd.read_csv(scores_csv)
        print("  [utility] SSMP 3-bit @ 4% budget...", flush=True)
        loaded = load_fp_model(model_id, dtype="float16", device_map="auto")
        plan = select_top_blocks(scores_df, budget_ratio=0.04, block_size=64)
        apply_block_splitting(loaded.model, plan.protect_map, block_size=64)
        _simulate_quantize_inplace(loaded.model, bits=3)
        results["ssmp_3bit_b0.04"] = {
            "perplexity": eval_perplexity(loaded.model, loaded.tokenizer, device=loaded.device),
            "mmlu": eval_mmlu_lite(loaded.model, loaded.tokenizer, device=loaded.device, n_questions=200),
            "truthfulqa": eval_truthfulqa_lite(loaded.model, loaded.tokenizer, device=loaded.device),
        }
        del loaded; gc.collect(); torch.cuda.empty_cache()

    save_json(result_file, results)
    print(f"  Done: {result_file}", flush=True)


# ── Drift bounds ─────────────────────────────────────────────────────────
def run_drift_bounds(model_id, model_short):
    out_file = RUNS / f"drift_bounds_{model_short}.csv"
    if out_file.exists():
        print(f"=== {model_short} drift bounds: ALREADY DONE ===")
        return
    print(f"=== Running {model_short} drift bounds ===", flush=True)

    from safepress.model.load import load_fp_model
    from safepress.analysis.drift_bound import compute_module_level_bounds
    from safepress.utils.io import read_prompts_jsonl

    prompts = read_prompts_jsonl(DATA / "advbench.jsonl", key="prompt")[:64]
    loaded = load_fp_model(model_id, dtype="float16", device_map="auto")

    df = compute_module_level_bounds(
        loaded.model,
        loaded.tokenizer,
        prompts,
        bits=4,
        group_size=128,
        batch_size=1,
        max_length=512,
        device=loaded.device,
    )
    df.to_csv(out_file, index=False)
    del loaded; gc.collect(); torch.cuda.empty_cache()
    print(f"  Done: {out_file}", flush=True)


# ── Main ─────────────────────────────────────────────────────────────────
def main():
    models = {
        "qwen25_14b": {
            "model_id": "Qwen/Qwen2.5-14B-Instruct",
            "scores": str(RUNS / "scores/qwen_qwen2.5-14b-instruct_scores.csv"),
        },
        "gemma12b": {
            "model_id": "google/gemma-3-12b-it",
            "scores": str(RUNS / "scores/google_gemma-3-12b-it_scores.csv"),
        },
    }

    experiments = [
        ("Phase Transition Qwen2.5-14B", lambda: run_phase_transition_qwen25()),
        ("3-bit SSMP Qwen2.5-14B", lambda: run_3bit_ssmp(
            models["qwen25_14b"]["model_id"], "qwen25_14b", models["qwen25_14b"]["scores"])),
        ("4-bit SSMP Qwen2.5-14B", lambda: run_4bit_ssmp(
            models["qwen25_14b"]["model_id"], "qwen25_14b", models["qwen25_14b"]["scores"])),
        ("Utility Qwen2.5-14B", lambda: run_utility(
            models["qwen25_14b"]["model_id"], "qwen25_14b", models["qwen25_14b"]["scores"])),
        ("Drift Bounds Qwen2.5-14B", lambda: run_drift_bounds(
            models["qwen25_14b"]["model_id"], "qwen25_14b")),
        ("Utility Gemma-3-12B", lambda: run_utility(
            models["gemma12b"]["model_id"], "gemma12b", models["gemma12b"]["scores"])),
        ("Drift Bounds Gemma-3-12B", lambda: run_drift_bounds(
            models["gemma12b"]["model_id"], "gemma12b")),
    ]

    for name, fn in experiments:
        print(f"\n{'='*60}", flush=True)
        print(f"  EXPERIMENT: {name}", flush=True)
        print(f"{'='*60}", flush=True)
        try:
            fn()
        except Exception as e:
            print(f"!!! FAILED: {name}: {e}", file=sys.stderr, flush=True)
            traceback.print_exc()
            gc.collect()
            torch.cuda.empty_cache()
            continue

    print("\n=== ALL EXPERIMENTS COMPLETE ===", flush=True)


if __name__ == "__main__":
    main()
