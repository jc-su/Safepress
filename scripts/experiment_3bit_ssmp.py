#!/usr/bin/env python3
"""
3-bit SSMP Experiment: Demonstrate that SSMP preserves safety under
aggressive (3-bit) quantization where uniform quantization fails.

Conditions tested:
  1. fp16_baseline:     Original FP16 model (no quantization).
  2. uniform_3bit:      All linear blocks quantized to 3-bit (simulated).
  3. ssmp_3bit@2%:      Top 2% safety-critical blocks kept FP16, rest 3-bit.
  4. ssmp_3bit@4%:      Top 4% safety-critical blocks kept FP16, rest 3-bit.
  5. ssmp_3bit@8%:      Top 8% safety-critical blocks kept FP16, rest 3-bit.
  6. random_3bit@4%:    Random 4% blocks kept FP16, rest 3-bit (ablation).
  7. inverted_3bit@4%:  Bottom 4% scoring blocks kept FP16, rest 3-bit (ablation).

Usage:
  python scripts/experiment_3bit_ssmp.py \
      --model_id Qwen/Qwen3-8B \
      --scores runs/scores/qwen_qwen3-8b_scores.csv \
      --eval_prompts data/harmbench.jsonl \
      --out_dir runs/3bit_ssmp \
      --max_prompts 128
"""
from __future__ import annotations

import argparse
import gc
import json
import random as stdlib_random
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd
import torch
from tqdm import tqdm

from safepress.eval.basic import (
    GenConfig,
    generate_completions,
    is_refusal,
    refusal_rate,
    try_strongreject_eval,
)
from safepress.model.blocks import chunk_indices, iter_linear_modules
from safepress.model.load import load_fp_model
from safepress.model.protect import ProtectPlan, select_top_blocks
from safepress.model.score import _quant_dequant_symmetric_groupwise
from safepress.utils.io import read_prompts_jsonl
from safepress.utils.logging import save_json


# ---------------------------------------------------------------------------
# Selective simulated quantization
# ---------------------------------------------------------------------------

@torch.no_grad()
def simulate_selective_3bit(
    model: torch.nn.Module,
    protect_map: Dict[str, List[int]],
    *,
    bits: int = 3,
    block_size: int = 64,
    group_size: int = 128,
) -> Dict[str, Any]:
    """
    Apply simulated quantization at `bits` to all blocks EXCEPT those in
    `protect_map`, which stay in FP16.

    Returns stats about how many blocks were quantized vs protected.
    """
    n_quantized = 0
    n_protected = 0
    n_total = 0

    for mod_name, mod in iter_linear_modules(model):
        w = mod.weight
        out_features = w.shape[0]
        blocks = chunk_indices(out_features, block_size)
        protected_indices: Set[int] = set(protect_map.get(mod_name, []))

        for b_idx, (s, e) in enumerate(blocks):
            n_total += 1
            if b_idx in protected_indices:
                n_protected += 1
                continue  # Leave in FP16

            # Quantize this block's rows
            block_w = w[s:e, :]
            block_w_hat = _quant_dequant_symmetric_groupwise(
                block_w, bits=bits, group_size=group_size,
            )
            w[s:e, :] = block_w_hat.to(w.dtype)
            n_quantized += 1

    return {
        "total_blocks": n_total,
        "quantized_blocks": n_quantized,
        "protected_blocks": n_protected,
        "bits": bits,
    }


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------

def evaluate_model(
    model: torch.nn.Module,
    tokenizer,
    prompts: List[str],
    *,
    device=None,
    gen: GenConfig,
    save_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Generate completions and compute safety metrics."""
    responses = generate_completions(
        model, tokenizer, prompts, gen=gen, device=device, show_progress=True,
    )
    rr = refusal_rate(responses)
    sr = try_strongreject_eval(prompts, responses)
    avg_len = sum(len(r.split()) for r in responses) / max(len(responses), 1)

    result = {
        "refusal_rate": rr,
        "n": len(prompts),
        "strongreject": sr,
        "avg_response_words": avg_len,
    }

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        with (save_dir / "responses.jsonl").open("w") as f:
            for p, r in zip(prompts, responses):
                f.write(json.dumps({"prompt": p, "response": r}, ensure_ascii=False) + "\n")
        save_json(save_dir / "eval.json", result)

    return result


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_experiment(
    model_id: str,
    scores_csv: str,
    eval_prompts: List[str],
    *,
    out_dir: Path,
    bits: int = 3,
    block_size: int = 64,
    group_size: int = 128,
    dtype: str = "float16",
    device_map: str = "auto",
    budgets: Optional[List[float]] = None,
    seed: int = 0,
) -> Dict[str, Any]:
    if budgets is None:
        budgets = [0.02, 0.04, 0.08]

    out_dir.mkdir(parents=True, exist_ok=True)
    scores = pd.read_csv(scores_csv)
    gen = GenConfig(max_new_tokens=256, temperature=0.0)

    conditions: Dict[str, Dict[str, Any]] = {}

    # -- Condition 1: FP16 baseline --
    print("\n[3bit_ssmp] Condition 1: FP16 baseline")
    loaded = load_fp_model(model_id, dtype=dtype, device_map=device_map)
    conditions["fp16_baseline"] = evaluate_model(
        loaded.model, loaded.tokenizer, eval_prompts,
        device=loaded.device, gen=gen,
        save_dir=out_dir / "fp16_baseline",
    )
    print(f"  -> refusal_rate = {conditions['fp16_baseline']['refusal_rate']:.4f}")
    del loaded; gc.collect(); torch.cuda.empty_cache()

    # -- Condition 2: Uniform 3-bit --
    print(f"\n[3bit_ssmp] Condition 2: Uniform {bits}-bit (no protection)")
    loaded = load_fp_model(model_id, dtype=dtype, device_map=device_map)
    stats = simulate_selective_3bit(
        loaded.model, protect_map={},
        bits=bits, block_size=block_size, group_size=group_size,
    )
    conditions["uniform_3bit"] = evaluate_model(
        loaded.model, loaded.tokenizer, eval_prompts,
        device=loaded.device, gen=gen,
        save_dir=out_dir / f"uniform_{bits}bit",
    )
    conditions["uniform_3bit"]["quant_stats"] = stats
    print(f"  -> refusal_rate = {conditions['uniform_3bit']['refusal_rate']:.4f}")
    del loaded; gc.collect(); torch.cuda.empty_cache()

    # -- Conditions 3-5: SSMP at various budgets --
    for budget in budgets:
        label = f"ssmp_{bits}bit_b{budget}"
        print(f"\n[3bit_ssmp] Condition: SSMP {bits}-bit @ {budget*100:.1f}% budget")
        loaded = load_fp_model(model_id, dtype=dtype, device_map=device_map)
        plan = select_top_blocks(scores, budget_ratio=budget, block_size=block_size)
        stats = simulate_selective_3bit(
            loaded.model, protect_map=plan.protect_map,
            bits=bits, block_size=block_size, group_size=group_size,
        )
        conditions[label] = evaluate_model(
            loaded.model, loaded.tokenizer, eval_prompts,
            device=loaded.device, gen=gen,
            save_dir=out_dir / label,
        )
        conditions[label]["quant_stats"] = stats
        conditions[label]["budget"] = budget
        conditions[label]["protected_params"] = plan.protected_params
        conditions[label]["total_params"] = plan.total_params
        print(f"  -> refusal_rate = {conditions[label]['refusal_rate']:.4f}")
        print(f"     protected: {stats['protected_blocks']}/{stats['total_blocks']} blocks")
        del loaded; gc.collect(); torch.cuda.empty_cache()

    # -- Condition 6: Random protection (ablation) --
    ablation_budget = budgets[len(budgets) // 2]  # middle budget
    print(f"\n[3bit_ssmp] Condition: Random {bits}-bit @ {ablation_budget*100:.1f}% budget")
    loaded = load_fp_model(model_id, dtype=dtype, device_map=device_map)
    # Build random protect map with same number of params as SSMP
    ssmp_plan = select_top_blocks(scores, budget_ratio=ablation_budget, block_size=block_size)
    n_protected = sum(len(v) for v in ssmp_plan.protect_map.values())

    # Randomly select the same number of blocks
    all_blocks = [(row["module"], int(row["block_idx"])) for _, row in scores.iterrows()]
    stdlib_random.seed(seed)
    stdlib_random.shuffle(all_blocks)
    random_selected = all_blocks[:n_protected]
    random_map: Dict[str, List[int]] = {}
    for mod, bidx in random_selected:
        random_map.setdefault(mod, []).append(bidx)

    stats = simulate_selective_3bit(
        loaded.model, protect_map=random_map,
        bits=bits, block_size=block_size, group_size=group_size,
    )
    label = f"random_{bits}bit_b{ablation_budget}"
    conditions[label] = evaluate_model(
        loaded.model, loaded.tokenizer, eval_prompts,
        device=loaded.device, gen=gen,
        save_dir=out_dir / label,
    )
    conditions[label]["quant_stats"] = stats
    print(f"  -> refusal_rate = {conditions[label]['refusal_rate']:.4f}")
    del loaded; gc.collect(); torch.cuda.empty_cache()

    # -- Condition 7: Inverted protection (ablation) --
    print(f"\n[3bit_ssmp] Condition: Inverted {bits}-bit @ {ablation_budget*100:.1f}% budget")
    loaded = load_fp_model(model_id, dtype=dtype, device_map=device_map)
    inv_scores = scores.copy()
    inv_scores["score"] = -inv_scores["score"]
    inv_plan = select_top_blocks(inv_scores, budget_ratio=ablation_budget, block_size=block_size)
    stats = simulate_selective_3bit(
        loaded.model, protect_map=inv_plan.protect_map,
        bits=bits, block_size=block_size, group_size=group_size,
    )
    label = f"inverted_{bits}bit_b{ablation_budget}"
    conditions[label] = evaluate_model(
        loaded.model, loaded.tokenizer, eval_prompts,
        device=loaded.device, gen=gen,
        save_dir=out_dir / label,
    )
    conditions[label]["quant_stats"] = stats
    print(f"  -> refusal_rate = {conditions[label]['refusal_rate']:.4f}")
    del loaded; gc.collect(); torch.cuda.empty_cache()

    # -- Summary --
    print(f"\n{'='*60}")
    print(f"3-BIT SSMP EXPERIMENT SUMMARY ({model_id})")
    print(f"{'='*60}")
    for cond_name, cond_data in conditions.items():
        rr = cond_data["refusal_rate"]
        print(f"  {cond_name:30s}  refusal_rate={rr:.4f}")
    print()

    result = {
        "experiment": f"ssmp_{bits}bit",
        "model_id": model_id,
        "bits": bits,
        "conditions": conditions,
    }
    save_json(out_dir / f"ssmp_{bits}bit_results.json", result)
    print(f"Results saved to {out_dir / f'ssmp_{bits}bit_results.json'}")
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="3-bit SSMP experiment")
    parser.add_argument("--model_id", required=True)
    parser.add_argument("--scores", required=True, help="Block scores CSV")
    parser.add_argument("--eval_prompts", required=True, help="Harmful prompts JSONL")
    parser.add_argument("--out_dir", default="runs/3bit_ssmp")
    parser.add_argument("--bits", type=int, default=3)
    parser.add_argument("--block_size", type=int, default=64)
    parser.add_argument("--group_size", type=int, default=128)
    parser.add_argument("--max_prompts", type=int, default=128)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--budgets", nargs="+", type=float, default=[0.02, 0.04, 0.08])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prompt_key", default="prompt")
    args = parser.parse_args()

    prompts = read_prompts_jsonl(args.eval_prompts, key=args.prompt_key)
    if args.max_prompts:
        prompts = prompts[:args.max_prompts]

    run_experiment(
        model_id=args.model_id,
        scores_csv=args.scores,
        eval_prompts=prompts,
        out_dir=Path(args.out_dir),
        bits=args.bits,
        block_size=args.block_size,
        group_size=args.group_size,
        dtype=args.dtype,
        device_map=args.device_map,
        budgets=args.budgets,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
