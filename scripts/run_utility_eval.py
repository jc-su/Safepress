#!/usr/bin/env python3
"""
Utility evaluation driver: measures capability retention under quantization.

Runs perplexity (WikiText-2), MMLU-lite, and TruthfulQA-lite on:
  - FP16 baseline
  - Simulated uniform N-bit quantized model
  - Simulated SSMP N-bit quantized model (if --scores provided)

Usage:
  python scripts/run_utility_eval.py \
      --model_id Qwen/Qwen3-8B \
      --out_dir runs/utility_qwen3 \
      --bits 3 \
      --scores runs/scores/qwen_qwen3-8b_scores.csv \
      --budget 0.04
"""
from __future__ import annotations

import argparse
import gc
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import torch

from safepress.eval.utility import eval_all_utility
from safepress.model.blocks import chunk_indices, iter_linear_modules
from safepress.model.load import load_fp_model
from safepress.model.protect import select_top_blocks
from safepress.model.score import _quant_dequant_symmetric_groupwise
from safepress.utils.logging import save_json


@torch.no_grad()
def apply_simulated_quantization(
    model: torch.nn.Module,
    protect_map: Dict[str, List[int]],
    *,
    bits: int = 3,
    block_size: int = 64,
    group_size: int = 128,
) -> Dict[str, int]:
    n_quantized = 0
    n_protected = 0
    n_total = 0
    for mod_name, mod in iter_linear_modules(model):
        w = mod.weight
        out_features = w.shape[0]
        blocks = chunk_indices(out_features, block_size)
        protected = set(protect_map.get(mod_name, []))
        for b_idx, (s, e) in enumerate(blocks):
            n_total += 1
            if b_idx in protected:
                n_protected += 1
                continue
            block_w = w[s:e, :]
            block_w_hat = _quant_dequant_symmetric_groupwise(block_w, bits=bits, group_size=group_size)
            w[s:e, :] = block_w_hat.to(w.dtype)
            n_quantized += 1
    return {"total_blocks": n_total, "quantized_blocks": n_quantized, "protected_blocks": n_protected}


def run_utility(
    model_id: str,
    out_dir: Path,
    *,
    bits: int = 3,
    scores_csv: Optional[str] = None,
    budget: float = 0.04,
    block_size: int = 64,
    group_size: int = 128,
    dtype: str = "float16",
    device_map: str = "auto",
) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    results: Dict[str, Any] = {"model_id": model_id, "bits": bits}

    # -- FP16 baseline --
    print(f"\n[utility] FP16 baseline for {model_id}")
    loaded = load_fp_model(model_id, dtype=dtype, device_map=device_map)
    fp16_util = eval_all_utility(loaded.model, loaded.tokenizer)
    results["fp16"] = fp16_util
    save_json(out_dir / "fp16_utility.json", fp16_util)
    print(f"  PPL={fp16_util.get('perplexity', {}).get('perplexity', 'N/A')}")
    print(f"  MMLU={fp16_util.get('mmlu', {}).get('accuracy', 'N/A')}")
    del loaded; gc.collect(); torch.cuda.empty_cache()

    # -- Uniform N-bit --
    print(f"\n[utility] Uniform {bits}-bit for {model_id}")
    loaded = load_fp_model(model_id, dtype=dtype, device_map=device_map)
    apply_simulated_quantization(loaded.model, {}, bits=bits, block_size=block_size, group_size=group_size)
    uniform_util = eval_all_utility(loaded.model, loaded.tokenizer)
    results[f"uniform_{bits}bit"] = uniform_util
    save_json(out_dir / f"uniform_{bits}bit_utility.json", uniform_util)
    print(f"  PPL={uniform_util.get('perplexity', {}).get('perplexity', 'N/A')}")
    print(f"  MMLU={uniform_util.get('mmlu', {}).get('accuracy', 'N/A')}")
    del loaded; gc.collect(); torch.cuda.empty_cache()

    # -- SSMP N-bit (if scores provided) --
    if scores_csv:
        print(f"\n[utility] SSMP {bits}-bit @ {budget*100:.0f}% for {model_id}")
        loaded = load_fp_model(model_id, dtype=dtype, device_map=device_map)
        scores = pd.read_csv(scores_csv)
        plan = select_top_blocks(scores, budget_ratio=budget, block_size=block_size)
        apply_simulated_quantization(
            loaded.model, plan.protect_map,
            bits=bits, block_size=block_size, group_size=group_size,
        )
        ssmp_util = eval_all_utility(loaded.model, loaded.tokenizer)
        results[f"ssmp_{bits}bit_b{budget}"] = ssmp_util
        save_json(out_dir / f"ssmp_{bits}bit_b{budget}_utility.json", ssmp_util)
        print(f"  PPL={ssmp_util.get('perplexity', {}).get('perplexity', 'N/A')}")
        print(f"  MMLU={ssmp_util.get('mmlu', {}).get('accuracy', 'N/A')}")
        del loaded; gc.collect(); torch.cuda.empty_cache()

    save_json(out_dir / "utility_results.json", results)
    print(f"\nResults saved to {out_dir / 'utility_results.json'}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Utility evaluation (PPL, MMLU, TruthfulQA)")
    parser.add_argument("--model_id", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--bits", type=int, default=3)
    parser.add_argument("--scores", default=None, help="Block scores CSV for SSMP condition")
    parser.add_argument("--budget", type=float, default=0.04)
    parser.add_argument("--block_size", type=int, default=64)
    parser.add_argument("--group_size", type=int, default=128)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--device_map", default="auto")
    args = parser.parse_args()

    run_utility(
        model_id=args.model_id,
        out_dir=Path(args.out_dir),
        bits=args.bits,
        scores_csv=args.scores,
        budget=args.budget,
        block_size=args.block_size,
        group_size=args.group_size,
        dtype=args.dtype,
        device_map=args.device_map,
    )


if __name__ == "__main__":
    main()
