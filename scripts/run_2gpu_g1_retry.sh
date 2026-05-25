#!/bin/bash
# G1 retry on cuda:0 — runs AFTER Track A finishes (so cuda:0 is free).
# Uses max_length=512 instead of 1024 to avoid the logits.float() OOM that
# hit Mistral/GLM-4/DeepSeek on the first attempt (vocab_size up to 131k
# means logits upcast = 512 MB tipping the 48 GB cliff).

set -u
cd /home/jis23009/Dev/safepress_repo

LOG=runs/emnlp_fisher60/g1_retry_cuda0.log
mkdir -p runs/emnlp_g1 runs/emnlp_fisher60

A_LOG=runs/emnlp_fisher60/track_a_cuda0.log
A_SENTINEL="===== TRACK_A_DONE"

echo "===== G1 retry watcher armed $(date -Iseconds), waiting for $A_SENTINEL =====" > $LOG

while ! grep -F "$A_SENTINEL" "$A_LOG" > /dev/null 2>&1; do
    sleep 120
done

echo "" >> $LOG
echo "===== Track A done; G1 retry starting $(date -Iseconds) =====" >> $LOG

declare -A G1_MODELS=(
    [mistral]="mistralai/Ministral-8B-Instruct-2410"
    [glm4]="THUDM/GLM-4-9B-0414"
    [deepseek]="deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
)
for tag in mistral glm4 deepseek; do
    model_id=${G1_MODELS[$tag]}
    out=runs/emnlp_g1/${tag}_drift.csv
    echo "" >> $LOG
    echo "----- G1 retry $tag start $(date -Iseconds) -----" >> $LOG
    # max_length=512 (was 1024) + max_prompts=32 (was 64) to fit under the
    # 48GB ceiling for models with vocab >= 131k
    CUDA_VISIBLE_DEVICES=0 safepress analyze drift-validate \
        --model_id "$model_id" \
        --calib_prompts data/advbench.jsonl \
        --prompt_key prompt \
        --max_prompts 32 \
        --max_length 512 \
        --bit_widths 8 4 3 2 \
        --group_size 128 \
        --block_size 64 \
        --out $out >> $LOG 2>&1
    echo "----- G1 retry $tag done $(date -Iseconds) -----" >> $LOG
done

echo "" >> $LOG
echo "===== G1_RETRY_DONE $(date -Iseconds) =====" >> $LOG
