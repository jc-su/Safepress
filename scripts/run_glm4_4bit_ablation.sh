#!/bin/bash
# GLM-4 4-bit ablation on cuda:0 (parallel with Track Z GLM-4 attacks on cuda:1).
# Completes the bit-width x regime grid (Qwen3 moderate, Llama strong-fragile,
# GLM-4 strong-robust).

set -u
cd /home/jis23009/Dev/safepress_repo

LOG=runs/emnlp_phase4b/glm4_4bit_ablation.log
mkdir -p runs/emnlp_4bit_ablation/glm4

echo "===== GLM-4 4-bit ablation (cuda:0) started $(date -Iseconds) =====" > $LOG
echo "PID=$$ PPID=$PPID" >> $LOG

# 1) 4-bit ablation sweep
echo "" >> $LOG
echo "----- 4-bit GLM-4 sweep start $(date -Iseconds) -----" >> $LOG
CUDA_VISIBLE_DEVICES=0 safepress sweep --config configs/fisher60_glm4_4bit.yaml >> $LOG 2>&1
echo "----- 4-bit GLM-4 sweep done $(date -Iseconds) -----" >> $LOG

# 2) Offline classifier
echo "" >> $LOG
echo "----- 4-bit GLM-4 classifier start $(date -Iseconds) -----" >> $LOG
CUDA_VISIBLE_DEVICES=0 python scripts/classify_existing_responses.py \
    --sweep_root runs/emnlp_4bit_ablation/glm4 --seed 0 --overwrite >> $LOG 2>&1 || \
    echo "[classifier failed]" >> $LOG
echo "----- 4-bit GLM-4 classifier done $(date -Iseconds) -----" >> $LOG

# 3) Offline utility
echo "" >> $LOG
echo "----- 4-bit GLM-4 utility start $(date -Iseconds) -----" >> $LOG
CUDA_VISIBLE_DEVICES=0 python scripts/utility_eval_from_protect_maps.py \
    --sweep_root runs/emnlp_4bit_ablation/glm4 --model_id THUDM/GLM-4-9B-0414 \
    --bits 4 --group_size 128 --block_size 64 \
    --ppl_n_samples 64 --ppl_max_length 1024 --ppl_stride 512 \
    --mmlu_n 200 --seed 0 --overwrite >> $LOG 2>&1 || \
    echo "[utility failed]" >> $LOG
echo "----- 4-bit GLM-4 utility done $(date -Iseconds) -----" >> $LOG

echo "" >> $LOG
echo "===== GLM4_4BIT_DONE $(date -Iseconds) =====" >> $LOG
