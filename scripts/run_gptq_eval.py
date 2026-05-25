#!/usr/bin/env python3
"""
GPTQ quantization + safety evaluation.

Quantizes a model to N-bit GPTQ format, then evaluates safety (refusal rate)
and compares with the FP16 baseline.

Usage:
  python scripts/run_gptq_eval.py \
      --model_id Qwen/Qwen3-8B \
      --eval_prompts data/harmbench_128.jsonl \
      --out_dir runs/gptq_3bit_qwen3 \
      --bits 3
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any, Dict, List

import torch

from safepress.eval.basic import (
    GenConfig,
    generate_completions,
    refusal_rate,
    try_strongreject_eval,
)
from safepress.model.load import load_fp_model
from safepress.model.quantize_gptq import load_gptq_model, quantize_gptq
from safepress.utils.io import read_prompts_jsonl
from safepress.utils.logging import save_json
from transformers import AutoTokenizer


def evaluate_safety(
    model,
    tokenizer,
    prompts: List[str],
    *,
    device=None,
    save_dir: Path,
) -> Dict[str, Any]:
    gen = GenConfig(max_new_tokens=256, temperature=0.0)
    save_dir.mkdir(parents=True, exist_ok=True)

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

    with (save_dir / "responses.jsonl").open("w") as f:
        for p, r in zip(prompts, responses):
            f.write(json.dumps({"prompt": p, "response": r}, ensure_ascii=False) + "\n")
    save_json(save_dir / "eval.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description="GPTQ quantization + safety evaluation")
    parser.add_argument("--model_id", required=True)
    parser.add_argument("--eval_prompts", required=True, help="Harmful prompts JSONL")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--bits", type=int, default=3)
    parser.add_argument("--group_size", type=int, default=128)
    parser.add_argument("--max_prompts", type=int, default=128)
    parser.add_argument("--prompt_key", default="prompt")
    parser.add_argument("--skip_quantize", action="store_true", help="Load pre-quantized model from out_dir")
    parser.add_argument("--skip_fp16", action="store_true", help="Skip FP16 baseline eval (use existing)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prompts = read_prompts_jsonl(args.eval_prompts, key=args.prompt_key)
    if args.max_prompts:
        prompts = prompts[:args.max_prompts]

    results: Dict[str, Any] = {"model_id": args.model_id, "bits": args.bits}

    # -- FP16 baseline --
    fp16_eval_path = out_dir / "fp16_baseline" / "eval.json"
    if args.skip_fp16 and fp16_eval_path.exists():
        print(f"\n[gptq] Skipping FP16 baseline (loading from {fp16_eval_path})")
        with open(fp16_eval_path) as f:
            fp16_result = json.load(f)
        results["fp16_baseline"] = fp16_result
        print(f"  -> refusal_rate = {fp16_result['refusal_rate']:.4f}")
        tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    else:
        print(f"\n[gptq] FP16 baseline for {args.model_id}")
        loaded = load_fp_model(args.model_id, dtype="float16", device_map="auto")
        fp16_result = evaluate_safety(
            loaded.model, loaded.tokenizer, prompts,
            device=loaded.device, save_dir=out_dir / "fp16_baseline",
        )
        results["fp16_baseline"] = fp16_result
        print(f"  -> refusal_rate = {fp16_result['refusal_rate']:.4f}")
        tokenizer = loaded.tokenizer  # keep tokenizer for later
        del loaded; gc.collect(); torch.cuda.empty_cache()

    # -- GPTQ quantization --
    quant_model_dir = out_dir / f"gptq_{args.bits}bit_model"
    if not args.skip_quantize:
        print(f"\n[gptq] Quantizing {args.model_id} to {args.bits}-bit GPTQ...")
        tokenizer_fresh = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
        report = quantize_gptq(
            args.model_id,
            tokenizer_fresh,
            out_dir=quant_model_dir,
            bits=args.bits,
            group_size=args.group_size,
        )
        print(f"  -> {report.note}")
        del tokenizer_fresh; gc.collect(); torch.cuda.empty_cache()

    # -- Evaluate GPTQ model --
    print(f"\n[gptq] Evaluating GPTQ {args.bits}-bit model...")
    gptq_model = load_gptq_model(quant_model_dir)
    gptq_tokenizer = AutoTokenizer.from_pretrained(str(quant_model_dir), trust_remote_code=True)
    gptq_result = evaluate_safety(
        gptq_model, gptq_tokenizer, prompts,
        save_dir=out_dir / f"gptq_{args.bits}bit",
    )
    results[f"gptq_{args.bits}bit"] = gptq_result
    print(f"  -> refusal_rate = {gptq_result['refusal_rate']:.4f}")
    del gptq_model; gc.collect(); torch.cuda.empty_cache()

    # -- Summary --
    fp16_rr = fp16_result["refusal_rate"]
    gptq_rr = gptq_result["refusal_rate"]
    delta = fp16_rr - gptq_rr
    print(f"\n{'='*50}")
    print(f"GPTQ {args.bits}-bit EVALUATION SUMMARY ({args.model_id})")
    print(f"{'='*50}")
    print(f"  FP16:  refusal_rate = {fp16_rr:.4f}")
    print(f"  GPTQ:  refusal_rate = {gptq_rr:.4f}")
    print(f"  Delta: {delta:+.4f}")

    save_json(out_dir / "gptq_results.json", results)
    print(f"\nResults saved to {out_dir / 'gptq_results.json'}")


if __name__ == "__main__":
    main()
