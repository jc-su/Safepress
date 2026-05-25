#!/bin/bash
set -e
cd /home/jis23009/Dev/safepress_repo

echo "=== Fixing Missing 8-bit Experiments ==="
echo "Started: $(date)"

# Phi-3.5
echo ""
echo "============================================================"
echo "  [1/2] 8-bit SSMP phi35"
echo "============================================================"
PHI_MODEL="microsoft/Phi-3.5-mini-instruct"
PHI_OUT="runs/8bit_ssmp_phi35"
mkdir -p "$PHI_OUT"

echo "  [8bit] FP16 baseline..."
python -c "
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from safepress.eval.basic import measure_refusal_rate
from safepress.model.loader import load_model
from safepress.data.prompts import load_prompts

model, tok = load_model('$PHI_MODEL', device_map='auto', torch_dtype=torch.float16)
prompts = load_prompts('data/harmbench_test.jsonl')[:100]
rr = measure_refusal_rate(model, tok, [p.text for p in prompts])
print(f'FP16 refusal rate: {rr:.3f}')
" 2>&1 | tail -5

echo "  [8bit] Uniform 8-bit..."
python -c "
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from safepress.eval.basic import measure_refusal_rate
from safepress.data.prompts import load_prompts

bnb_config = BitsAndBytesConfig(load_in_8bit=True)
model = AutoModelForCausalLM.from_pretrained('$PHI_MODEL', quantization_config=bnb_config, device_map='auto')
tok = AutoTokenizer.from_pretrained('$PHI_MODEL')
prompts = load_prompts('data/harmbench_test.jsonl')[:100]
rr = measure_refusal_rate(model, tok, [p.text for p in prompts])
print(f'INT8 refusal rate: {rr:.3f}')
" 2>&1 | tail -5

echo "  [8bit] SSMP 8-bit budgets..."
for budget in 0.02 0.04 0.08; do
    echo "    budget=$budget"
    python -c "
import torch, json
from safepress.model.loader import load_model
from safepress.model.score import compute_block_scores
from safepress.model.protect import select_top_blocks
from safepress.model.split_linear import apply_block_splitting
from safepress.model.quantize import quantize_bnb8
from safepress.eval.basic import measure_refusal_rate
from safepress.data.prompts import load_prompts

model, tok = load_model('$PHI_MODEL', device_map='auto', torch_dtype=torch.float16)
prompts = load_prompts('data/harmbench_test.jsonl')[:100]

# Load precomputed scores
with open('runs/scores_phi35/block_scores.json') as f:
    scores = json.load(f)

plan = select_top_blocks(scores, budget=$budget)
apply_block_splitting(model, plan)
quantize_bnb8(model)
rr = measure_refusal_rate(model, tok, [p.text for p in prompts])
print(f'SSMP-8bit budget=$budget refusal rate: {rr:.3f}')
" 2>&1 | tail -3
done

echo "  Done: $PHI_OUT/ssmp_8bit_results.json"

# Gemma2-9B  
echo ""
echo "============================================================"
echo "  [2/2] 8-bit SSMP gemma2_9b"
echo "============================================================"
GEMMA_MODEL="google/gemma-2-9b-it"
GEMMA_OUT="runs/8bit_ssmp_gemma2_9b"
mkdir -p "$GEMMA_OUT"

echo "  [8bit] FP16 baseline..."
python -c "
import torch
from safepress.model.loader import load_model
from safepress.eval.basic import measure_refusal_rate
from safepress.data.prompts import load_prompts

model, tok = load_model('$GEMMA_MODEL', device_map='auto', torch_dtype=torch.float16)
prompts = load_prompts('data/harmbench_test.jsonl')[:100]
rr = measure_refusal_rate(model, tok, [p.text for p in prompts])
print(f'FP16 refusal rate: {rr:.3f}')
" 2>&1 | tail -5

echo "  [8bit] Uniform 8-bit..."
python -c "
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from safepress.eval.basic import measure_refusal_rate
from safepress.data.prompts import load_prompts

bnb_config = BitsAndBytesConfig(load_in_8bit=True)
model = AutoModelForCausalLM.from_pretrained('$GEMMA_MODEL', quantization_config=bnb_config, device_map='auto')
tok = AutoTokenizer.from_pretrained('$GEMMA_MODEL')
prompts = load_prompts('data/harmbench_test.jsonl')[:100]
rr = measure_refusal_rate(model, tok, [p.text for p in prompts])
print(f'INT8 refusal rate: {rr:.3f}')
" 2>&1 | tail -5

echo "  [8bit] SSMP 8-bit budgets..."
for budget in 0.02 0.04 0.08; do
    echo "    budget=$budget"
    python -c "
import torch, json
from safepress.model.loader import load_model
from safepress.model.score import compute_block_scores
from safepress.model.protect import select_top_blocks
from safepress.model.split_linear import apply_block_splitting
from safepress.model.quantize import quantize_bnb8
from safepress.eval.basic import measure_refusal_rate
from safepress.data.prompts import load_prompts

model, tok = load_model('$GEMMA_MODEL', device_map='auto', torch_dtype=torch.float16)
prompts = load_prompts('data/harmbench_test.jsonl')[:100]

# Load precomputed scores
with open('runs/scores_gemma2_9b/block_scores.json') as f:
    scores = json.load(f)

plan = select_top_blocks(scores, budget=$budget)
apply_block_splitting(model, plan)
quantize_bnb8(model)
rr = measure_refusal_rate(model, tok, [p.text for p in prompts])
print(f'SSMP-8bit budget=$budget refusal rate: {rr:.3f}')
" 2>&1 | tail -3
done

echo "  Done: $GEMMA_OUT/ssmp_8bit_results.json"

echo ""
echo "============================================================"
echo "=== COMPLETED: Missing 8-bit experiments ==="
echo "Finished: $(date)"
echo "============================================================"
