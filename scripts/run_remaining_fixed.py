#!/usr/bin/env python3
"""
Run remaining experiments after fixes:
  - Gemma2 3/4-bit SSMP (fixed multi-GPU SplitLinear)
  - Phi4-mini full pipeline (fixed with transformers 5.1.0)
  - SmolLM3-3B full pipeline (new model)
  - 8-bit SSMP for phi4_mini
"""
import gc
import sys
import traceback
from pathlib import Path

import torch

RUNS = Path("/home/jis23009/Dev/safepress_repo/runs")
DATA = Path("/home/jis23009/Dev/safepress_repo/data")

MODELS = {
    "gemma2_9b": {
        "model_id": "google/gemma-2-9b-it",
        "scores": str(RUNS / "scores/google_gemma-2-9b-it_scores.csv"),
    },
    "phi4_mini": {
        "model_id": "microsoft/Phi-4-mini-instruct",
        "scores": str(RUNS / "scores/microsoft_phi-4-mini-instruct_scores.csv"),
    },
    "smollm3_3b": {
        "model_id": "HuggingFaceTB/SmolLM3-3B-Instruct",
        "scores": str(RUNS / "scores/huggingfacetb_smollm3-3b-instruct_scores.csv"),
    },
}


def _cleanup():
    gc.collect()
    torch.cuda.empty_cache()


def load_eval_prompts():
    from safepress.utils.io import read_prompts_jsonl
    return read_prompts_jsonl(DATA / "harmbench_128.jsonl", key="prompt")


def run_scores(model_short):
    m = MODELS[model_short]
    scores_path = Path(m["scores"])
    if scores_path.exists():
        print(f"=== {model_short} scores: ALREADY DONE ===")
        return

    print(f"=== Running {model_short} scores ===", flush=True)
    scores_path.parent.mkdir(parents=True, exist_ok=True)

    from safepress.model.load import load_fp_model
    from safepress.model.score import compute_block_scores
    from safepress.utils.io import read_prompts_jsonl

    prompts = read_prompts_jsonl(DATA / "advbench.jsonl", key="prompt")[:128]
    loaded = load_fp_model(m["model_id"], dtype="float16", device_map="auto",
                           trust_remote_code=True)

    df = compute_block_scores(
        loaded.model, loaded.tokenizer, prompts,
        bits=4, group_size=128, block_size=64,
        batch_size=1, max_length=512,
        device=loaded.device,
    )
    df.to_csv(scores_path, index=False)
    print(f"  Scored {len(df)} blocks → {scores_path}", flush=True)
    del loaded; _cleanup()


def run_phase_transition(model_short):
    m = MODELS[model_short]
    out_dir = RUNS / f"phase_transition_{model_short}"
    final = out_dir / "phase_transition_results.json"
    if final.exists():
        print(f"=== {model_short} phase transition: ALREADY DONE ===")
        return

    print(f"=== Running {model_short} phase transition ===", flush=True)
    from safepress.experiments.phase_transition import phase_transition_curve

    prompts = load_eval_prompts()
    result = phase_transition_curve(
        model_id=m["model_id"],
        eval_prompts=prompts,
        bit_widths=[8, 4, 3, 2],
        out_dir=str(out_dir),
        device_map="auto",
        dtype="float16",
    )
    print(f"  Result: {result.summary}", flush=True)
    _cleanup()


def run_ssmp(model_short, bits):
    m = MODELS[model_short]
    out_dir = RUNS / f"{bits}bit_ssmp_{model_short}"
    result_file = out_dir / f"ssmp_{bits}bit_results.json"

    # Force re-run for gemma2 3/4-bit (was broken)
    if model_short == "gemma2_9b" and bits in [3, 4]:
        if result_file.exists():
            result_file.unlink()

    if result_file.exists():
        print(f"=== {model_short} {bits}-bit SSMP: ALREADY DONE ===")
        return

    scores_path = Path(m["scores"])
    if not scores_path.exists():
        print(f"!!! {model_short} {bits}-bit SSMP: scores not found, skipping", flush=True)
        return

    print(f"=== Running {model_short} {bits}-bit SSMP ===", flush=True)
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
    print(f"  [{bits}bit] FP16 baseline...", flush=True)
    loaded = load_fp_model(m["model_id"], dtype="float16", device_map="auto",
                           trust_remote_code=True)
    results["fp16_baseline"] = _evaluate_model(
        loaded.model, loaded.tokenizer, prompts, device=loaded.device)
    results["fp16_baseline"]["bits"] = 16
    del loaded; _cleanup()

    # Uniform N-bit
    print(f"  [{bits}bit] Uniform {bits}-bit...", flush=True)
    loaded = load_fp_model(m["model_id"], dtype="float16", device_map="auto",
                           trust_remote_code=True)
    _simulate_quantize_inplace(loaded.model, bits=bits)
    results[f"uniform_{bits}bit"] = _evaluate_model(
        loaded.model, loaded.tokenizer, prompts, device=loaded.device)
    results[f"uniform_{bits}bit"]["bits"] = bits
    del loaded; _cleanup()

    # SSMP at budgets 2%, 4%, 8%
    scores_df = pd.read_csv(scores_path)
    for budget in [0.02, 0.04, 0.08]:
        label = f"ssmp_{bits}bit_b{budget}"
        print(f"  [{bits}bit] SSMP {bits}-bit budget={budget}...", flush=True)
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


def run_utility(model_short):
    m = MODELS[model_short]
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
    loaded = load_fp_model(m["model_id"], dtype="float16", device_map="auto",
                           trust_remote_code=True)
    results["fp16"] = {
        "perplexity": eval_perplexity(loaded.model, loaded.tokenizer, device=loaded.device),
        "mmlu": eval_mmlu_lite(loaded.model, loaded.tokenizer, device=loaded.device, n_questions=200),
        "truthfulqa": eval_truthfulqa_lite(loaded.model, loaded.tokenizer, device=loaded.device),
    }
    del loaded; _cleanup()

    # Uniform 3-bit
    print("  [utility] Uniform 3-bit...", flush=True)
    loaded = load_fp_model(m["model_id"], dtype="float16", device_map="auto",
                           trust_remote_code=True)
    _simulate_quantize_inplace(loaded.model, bits=3)
    results["uniform_3bit"] = {
        "perplexity": eval_perplexity(loaded.model, loaded.tokenizer, device=loaded.device),
        "mmlu": eval_mmlu_lite(loaded.model, loaded.tokenizer, device=loaded.device, n_questions=200),
        "truthfulqa": eval_truthfulqa_lite(loaded.model, loaded.tokenizer, device=loaded.device),
    }
    del loaded; _cleanup()

    # SSMP 3-bit at 4% budget
    scores_path = Path(m["scores"])
    if scores_path.exists():
        scores_df = pd.read_csv(scores_path)
        print("  [utility] SSMP 3-bit @ 4% budget...", flush=True)
        loaded = load_fp_model(m["model_id"], dtype="float16", device_map="auto",
                               trust_remote_code=True)
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


def run_drift_bounds(model_short):
    m = MODELS[model_short]
    out_file = RUNS / f"drift_bounds_{model_short}.csv"
    if out_file.exists():
        print(f"=== {model_short} drift bounds: ALREADY DONE ===")
        return

    print(f"=== Running {model_short} drift bounds ===", flush=True)

    from safepress.model.load import load_fp_model
    from safepress.analysis.drift_bound import compute_module_level_bounds
    from safepress.utils.io import read_prompts_jsonl

    prompts = read_prompts_jsonl(DATA / "advbench.jsonl", key="prompt")[:64]
    loaded = load_fp_model(m["model_id"], dtype="float16", device_map="auto",
                           trust_remote_code=True)

    df = compute_module_level_bounds(
        loaded.model, loaded.tokenizer, prompts,
        bits=4, group_size=128, batch_size=1, max_length=512,
        device=loaded.device,
    )
    df.to_csv(out_file, index=False)
    del loaded; _cleanup()
    print(f"  Done: {out_file}", flush=True)


def main():
    experiments = []

    # 1. Score new models (phi4_mini, smollm3_3b)
    for ms in ["phi4_mini", "smollm3_3b"]:
        experiments.append((f"Scores {ms}", lambda ms=ms: run_scores(ms)))

    # 2. Phase transition for new models
    for ms in ["phi4_mini", "smollm3_3b"]:
        experiments.append((f"Phase Trans {ms}", lambda ms=ms: run_phase_transition(ms)))

    # 3. Re-run Gemma2 3/4-bit SSMP (now fixed)
    experiments.append(("3-bit SSMP gemma2_9b", lambda: run_ssmp("gemma2_9b", 3)))
    experiments.append(("4-bit SSMP gemma2_9b", lambda: run_ssmp("gemma2_9b", 4)))

    # 4. 8-bit SSMP for phi4_mini
    experiments.append(("8-bit SSMP phi4_mini", lambda: run_ssmp("phi4_mini", 8)))

    # 5. 3/4/8-bit SSMP for smollm3_3b
    for bits in [3, 4, 8]:
        experiments.append((f"{bits}-bit SSMP smollm3_3b", lambda b=bits: run_ssmp("smollm3_3b", b)))

    # 6. Utility for new models
    for ms in ["phi4_mini", "smollm3_3b"]:
        experiments.append((f"Utility {ms}", lambda ms=ms: run_utility(ms)))

    # 7. Drift bounds for new models
    for ms in ["phi4_mini", "smollm3_3b"]:
        experiments.append((f"Drift {ms}", lambda ms=ms: run_drift_bounds(ms)))

    total = len(experiments)
    failed = []
    for i, (name, fn) in enumerate(experiments, 1):
        print(f"\n{'='*60}", flush=True)
        print(f"  [{i}/{total}] {name}", flush=True)
        print(f"{'='*60}", flush=True)
        try:
            fn()
        except Exception as e:
            print(f"!!! FAILED: {name}: {e}", file=sys.stderr, flush=True)
            traceback.print_exc()
            failed.append(name)
            _cleanup()
            continue

    print(f"\n{'='*60}", flush=True)
    print(f"=== COMPLETED: {total - len(failed)}/{total} experiments ===", flush=True)
    if failed:
        print(f"=== FAILED: {failed} ===", flush=True)
    print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    main()
