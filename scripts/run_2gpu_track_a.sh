#!/bin/bash
# Track A (cuda:0): G1 drift-bound extension + Qwen3 Fisher@60% retry + offline.
# Parallelizes with Track B on cuda:1. Total ETA ~4.25h.

set -u
cd /home/jis23009/Dev/safepress_repo

LOG=runs/emnlp_fisher60/track_a_cuda0.log
mkdir -p runs/emnlp_fisher60 runs/emnlp_g1

echo "===== Track A (cuda:0) started $(date -Iseconds) =====" > $LOG
echo "PID=$$ PPID=$PPID" >> $LOG
echo "" >> $LOG

# --------------------------------------------------------------------
# Step A1: G1 drift-bound on the 3 models missing it
# --------------------------------------------------------------------
declare -A G1_MODELS=(
    [mistral]="mistralai/Ministral-8B-Instruct-2410"
    [glm4]="THUDM/GLM-4-9B-0414"
    [deepseek]="deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
)
for tag in mistral glm4 deepseek; do
    model_id=${G1_MODELS[$tag]}
    out=runs/emnlp_g1/${tag}_drift.csv
    echo "" >> $LOG
    echo "----- G1 $tag start $(date -Iseconds) -----" >> $LOG
    CUDA_VISIBLE_DEVICES=0 safepress analyze drift-validate \
        --model_id "$model_id" \
        --calib_prompts data/advbench.jsonl \
        --prompt_key prompt \
        --max_prompts 64 \
        --max_length 1024 \
        --bit_widths 8 4 3 2 \
        --group_size 128 \
        --block_size 64 \
        --out $out >> $LOG 2>&1
    echo "----- G1 $tag done $(date -Iseconds) -----" >> $LOG
done
echo "" >> $LOG
echo "===== G1_EXTENSION_DONE $(date -Iseconds) =====" >> $LOG

# --------------------------------------------------------------------
# Step A2: Qwen3 Fisher@60% sweep (the one that failed earlier)
# --------------------------------------------------------------------
rm -rf runs/emnlp_fisher60/qwen3/sweep_qwen3_8b_3bit_fisher60_gradient_only_b0.6
echo "" >> $LOG
echo "----- Qwen3 Fisher@60% start $(date -Iseconds) -----" >> $LOG
CUDA_VISIBLE_DEVICES=0 safepress sweep --config configs/fisher60_qwen3_3bit.yaml >> $LOG 2>&1
echo "----- Qwen3 Fisher@60% done $(date -Iseconds) -----" >> $LOG

# --------------------------------------------------------------------
# Step A3: Offline classifier + utility for Qwen3 and Llama
# (Llama Fisher@60% sweep already completed under earlier run)
# --------------------------------------------------------------------
for tag in qwen3 llama31; do
    root=runs/emnlp_fisher60/$tag
    if [ -d "$root" ]; then
        echo "" >> $LOG
        echo "----- Offline classifier $tag start $(date -Iseconds) -----" >> $LOG
        CUDA_VISIBLE_DEVICES=0 python scripts/classify_existing_responses.py \
            --sweep_root $root --seed 0 --overwrite >> $LOG 2>&1 || \
            echo "[classifier $tag failed but continuing]" >> $LOG
        echo "----- Offline classifier $tag done $(date -Iseconds) -----" >> $LOG
    fi
done

declare -A UTIL_MODELS=(
    [qwen3]="Qwen/Qwen3-8B"
    [llama31]="meta-llama/Llama-3.1-8B-Instruct"
)
for tag in qwen3 llama31; do
    model_id=${UTIL_MODELS[$tag]}
    root=runs/emnlp_fisher60/$tag
    if [ -d "$root" ]; then
        echo "" >> $LOG
        echo "----- Offline utility $tag start $(date -Iseconds) -----" >> $LOG
        CUDA_VISIBLE_DEVICES=0 python scripts/utility_eval_from_protect_maps.py \
            --sweep_root $root --model_id "$model_id" \
            --bits 3 --group_size 128 --block_size 64 \
            --ppl_n_samples 64 --ppl_max_length 1024 --ppl_stride 512 \
            --mmlu_n 200 --seed 0 --overwrite >> $LOG 2>&1 || \
            echo "[utility $tag failed but continuing]" >> $LOG
        echo "----- Offline utility $tag done $(date -Iseconds) -----" >> $LOG
    fi
done

echo "" >> $LOG
echo "===== TRACK_A_DONE $(date -Iseconds) =====" >> $LOG
