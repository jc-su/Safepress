#!/usr/bin/env python
"""Dump raw per-block scores for all 4 scorers in a SINGLE backward pass.

Replicates the exact per-block accumulation of compute_block_scores
(score.py) for taylor_abs (SSMP), snip, grad_sq (Fisher), grad_abs, and
additionally records pure magnitude (|w|) and the group scale s_b. Output
is one CSV per model with all per-block quantities, enabling:
  - full-ranking Spearman correlation between scorers (vs only top-beta Jaccard)
  - the escape-clause correlation corr(||g||, magnitude) on top-beta blocks
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from safepress.model.score import (
    _quant_dequant_symmetric_groupwise,
    build_refusal_supervision,
)
from safepress.model.blocks import iter_linear_modules, chunk_indices
from safepress.model.load import load_fp_model
from safepress.utils.io import read_prompts_jsonl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", required=True)
    ap.add_argument("--calib_prompts", default="data/advbench.jsonl")
    ap.add_argument("--prompt_key", default="prompt")
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--max_prompts", type=int, default=128)
    ap.add_argument("--bits", type=int, default=3)
    ap.add_argument("--group_size", type=int, default=128)
    ap.add_argument("--block_size", type=int, default=64)
    ap.add_argument("--max_length", type=int, default=1024)
    ap.add_argument("--refusal_template", default="I'm sorry, but I can't help with that.")
    args = ap.parse_args()

    prompts = read_prompts_jsonl(args.calib_prompts, key=args.prompt_key)[: args.max_prompts]
    loaded = load_fp_model(args.model_id, dtype="float16", device_map="auto")
    model, tok, device = loaded.model, loaded.tokenizer, loaded.device
    model.eval()
    for p in model.parameters():
        p.requires_grad_(True)

    linears = list(iter_linear_modules(model))

    # accumulators keyed by (module, block_idx)
    acc = {}      # gradient-based, summed over batches
    static = {}   # batch-independent: magnitude, group scale, meta
    for name, mod in linears:
        out_f, in_f = mod.weight.shape
        for b_idx, (s, e) in enumerate(chunk_indices(out_f, args.block_size)):
            acc[(name, b_idx)] = dict(ssmp=0.0, snip=0.0, fisher=0.0, grad_abs=0.0)

    # static quantities (computed once, no gradient needed)
    with torch.no_grad():
        for name, mod in linears:
            w = mod.weight
            out_f, in_f = w.shape
            for b_idx, (s, e) in enumerate(chunk_indices(out_f, args.block_size)):
                w_blk = w[s:e, :].float()
                w_hat = _quant_dequant_symmetric_groupwise(w[s:e, :], bits=args.bits, group_size=args.group_size)
                # group scale proxy: max|w| / q_max over the block
                q_max = 2 ** (args.bits - 1) - 1
                s_b = float(w_blk.abs().max().item()) / q_max
                static[(name, b_idx)] = dict(
                    module=name, block_idx=b_idx,
                    magnitude=float(w_blk.abs().sum().item()),
                    group_scale=s_b,
                    grad_norm=0.0,  # filled below from accumulated grad_abs proxy
                    num_params=(e - s) * in_f,
                )

    # batch loop: one backward pass per batch, accumulate all gradient metrics
    for start in tqdm(range(0, len(prompts), 1), desc="scoring"):
        batch = build_refusal_supervision(tok, prompts[start:start+1],
                                           refusal_template=args.refusal_template,
                                           max_length=args.max_length)
        batch = {k: v.to(device) for k, v in batch.items()}
        model.zero_grad(set_to_none=True)
        out = model(**batch)
        out.loss.backward()
        with torch.no_grad():
            for name, mod in linears:
                w = mod.weight
                g = mod.weight.grad
                if g is None:
                    continue
                out_f, in_f = w.shape
                for b_idx, (s, e) in enumerate(chunk_indices(out_f, args.block_size)):
                    w_blk = w[s:e, :]
                    g_blk = g[s:e, :].float()
                    w_hat = _quant_dequant_symmetric_groupwise(w_blk, bits=args.bits, group_size=args.group_size)
                    delta = (w_hat - w_blk).float()
                    a = acc[(name, b_idx)]
                    a["ssmp"]     += float((g_blk * delta).abs().sum().item())
                    a["snip"]     += float((g_blk * w_blk.float()).abs().sum().item())
                    a["fisher"]   += float((g_blk ** 2).sum().item())
                    a["grad_abs"] += float(g_blk.abs().sum().item())

    # assemble dataframe
    rows = []
    for key, a in acc.items():
        st = static[key]
        rows.append(dict(
            module=st["module"], block_idx=st["block_idx"],
            ssmp=a["ssmp"], snip=a["snip"], fisher=a["fisher"],
            grad_abs=a["grad_abs"], magnitude=st["magnitude"],
            group_scale=st["group_scale"], num_params=st["num_params"],
        ))
    df = pd.DataFrame(rows)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f"[dump] wrote {len(df)} blocks -> {args.out_csv}")


if __name__ == "__main__":
    main()
