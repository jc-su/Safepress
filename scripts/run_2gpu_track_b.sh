#!/bin/bash
# Track B (cuda:1): Mistral + GLM-4 + DeepSeek Fisher@60% sweeps + offline.
# Parallelizes with Track A on cuda:0. Total ETA ~5h.

set -u
cd /home/jis23009/Dev/safepress_repo

LOG=runs/emnlp_fisher60/track_b_cuda1.log
mkdir -p runs/emnlp_fisher60

echo "===== Track B (cuda:1) started $(date -Iseconds) =====" > $LOG
echo "PID=$$ PPID=$PPID" >> $LOG
echo "" >> $LOG

# --------------------------------------------------------------------
# Step B1: Fisher@60% sweeps on Mistral, GLM-4, DeepSeek (restart Mistral)
# --------------------------------------------------------------------
# Clean Mistral partial output from the killed run, if any
rm -rf runs/emnlp_fisher60/mistral/sweep_mistral_8b_3bit_fisher60_gradient_only_b0.6

for cfg in \
    configs/fisher60_mistral_3bit.yaml \
    configs/fisher60_glm4_3bit.yaml \
    configs/fisher60_deepseek_3bit.yaml ; do
    tag=$(basename $cfg .yaml | sed 's/fisher60_//')
    echo "" >> $LOG
    echo "----- Fisher@60% $tag start $(date -Iseconds) -----" >> $LOG
    CUDA_VISIBLE_DEVICES=1 safepress sweep --config $cfg >> $LOG 2>&1
    echo "----- Fisher@60% $tag done $(date -Iseconds) -----" >> $LOG
done
echo "" >> $LOG
echo "===== FISHER60_TRACKB_DONE $(date -Iseconds) =====" >> $LOG

# --------------------------------------------------------------------
# Step B2: Offline classifier + utility for Mistral / GLM-4 / DeepSeek
# --------------------------------------------------------------------
for tag in mistral glm4 deepseek; do
    root=runs/emnlp_fisher60/$tag
    if [ -d "$root" ]; then
        echo "" >> $LOG
        echo "----- Offline classifier $tag start $(date -Iseconds) -----" >> $LOG
        CUDA_VISIBLE_DEVICES=1 python scripts/classify_existing_responses.py \
            --sweep_root $root --seed 0 --overwrite >> $LOG 2>&1 || \
            echo "[classifier $tag failed but continuing]" >> $LOG
        echo "----- Offline classifier $tag done $(date -Iseconds) -----" >> $LOG
    fi
done

declare -A UTIL_MODELS=(
    [mistral]="mistralai/Ministral-8B-Instruct-2410"
    [glm4]="THUDM/GLM-4-9B-0414"
    [deepseek]="deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
)
for tag in mistral glm4 deepseek; do
    model_id=${UTIL_MODELS[$tag]}
    root=runs/emnlp_fisher60/$tag
    if [ -d "$root" ]; then
        echo "" >> $LOG
        echo "----- Offline utility $tag start $(date -Iseconds) -----" >> $LOG
        CUDA_VISIBLE_DEVICES=1 python scripts/utility_eval_from_protect_maps.py \
            --sweep_root $root --model_id "$model_id" \
            --bits 3 --group_size 128 --block_size 64 \
            --ppl_n_samples 64 --ppl_max_length 1024 --ppl_stride 512 \
            --mmlu_n 200 --seed 0 --overwrite >> $LOG 2>&1 || \
            echo "[utility $tag failed but continuing]" >> $LOG
        echo "----- Offline utility $tag done $(date -Iseconds) -----" >> $LOG
    fi
done

echo "" >> $LOG
echo "===== TRACK_B_DONE $(date -Iseconds) =====" >> $LOG
