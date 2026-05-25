#!/usr/bin/env python3
"""Fix 8-bit SSMP experiment for phi35 with transformers compatibility fix."""
import gc
import sys
import traceback
from pathlib import Path

import torch

RUNS = Path("/home/jis23009/Dev/safepress_repo/runs")
DATA = Path("/home/jis23009/Dev/safepress_repo/data")

MODEL_ID = "microsoft/Phi-3.5-mini-instruct"
SCORES_PATH = RUNS / "scores/microsoft_phi-3.5-mini-instruct_scores.csv"
OUT_DIR = RUNS / "8bit_ssmp_phi35"


def _cleanup():
    gc.collect()
    torch.cuda.empty_cache()


def run_phi35_8bit():
    """Run 8-bit SSMP experiment for phi35."""
    import pandas as pd
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from safepress.model.protect import select_top_blocks
    from safepress.model.split_linear import apply_block_splitting
    from safepress.experiments.phase_transition import _simulate_quantize_inplace
    from safepress.eval.basic import refusal_rate, generate_completions, GenConfig, try_strongreject_eval
    from safepress.utils.io import read_prompts_jsonl
    from safepress.utils.logging import save_json

    result_file = OUT_DIR / "ssmp_8bit_results.json"
    if result_file.exists():
        print("=== phi35 8-bit SSMP: ALREADY DONE ===")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load prompts
    prompts = read_prompts_jsonl(DATA / "harmbench_128.jsonl", key="prompt")

    def eval_model(model, tokenizer, desc=""):
        """Evaluate a model."""
        print(f"    Evaluating {desc}...", flush=True)
        gen = GenConfig()
        responses = generate_completions(
            model, tokenizer, prompts,
            gen=gen, device="cuda", show_progress=True
        )
        rr = refusal_rate(responses)
        sr = try_strongreject_eval(prompts, responses)
        return {
            "refusal_rate": rr,
            "n": len(prompts),
            "strongreject": sr,
        }

    results = {}
    bits = 8

    # FP16 baseline
    print("  [8bit] FP16 baseline...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager",  # Avoid flash attention issues
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    results["fp16_baseline"] = eval_model(model, tokenizer, "FP16")
    results["fp16_baseline"]["bits"] = 16
    del model; _cleanup()

    # Uniform 8-bit
    print("  [8bit] Uniform 8-bit...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager",
    )
    _simulate_quantize_inplace(model, bits=bits)
    results["uniform_8bit"] = eval_model(model, tokenizer, "Uniform 8-bit")
    results["uniform_8bit"]["bits"] = bits
    del model; _cleanup()

    # SSMP at budgets 2%, 4%, 8%
    scores_df = pd.read_csv(SCORES_PATH)
    for budget in [0.02, 0.04, 0.08]:
        label = f"ssmp_8bit_b{budget}"
        print(f"  [8bit] SSMP 8-bit budget={budget}...", flush=True)

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="eager",
        )

        plan = select_top_blocks(scores_df, budget_ratio=budget, block_size=64)
        apply_block_splitting(model, plan.protect_map, block_size=64)
        _simulate_quantize_inplace(model, bits=bits)

        results[label] = eval_model(model, tokenizer, f"SSMP b={budget}")
        results[label]["bits"] = bits
        results[label]["budget"] = budget
        del model; _cleanup()

    save_json(result_file, {"model_id": MODEL_ID, "conditions": results})
    print(f"  Done: {result_file}", flush=True)


def main():
    print("=" * 60)
    print("  Fixing Phi-3.5 8-bit SSMP")
    print("=" * 60)

    try:
        run_phi35_8bit()
        print("\n" + "=" * 60)
        print("=== COMPLETED: Phi-3.5 8-bit fixed ===")
        print("=" * 60)
    except Exception as e:
        print(f"!!! FAILED: {e}", file=sys.stderr, flush=True)
        traceback.print_exc()


if __name__ == "__main__":
    main()
