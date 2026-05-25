#!/bin/bash
# Retry Qwen3 Fisher@60% after main Fisher60+G1 launcher completes.
# Qwen3 failed in the first attempt because a leftover Mistral baseline
# orphan from an earlier killed watcher was holding 15GB on cuda:1.
# That orphan has been killed; Qwen3 should succeed on retry.

set -u
cd /home/jis23009/Dev/safepress_repo

MAIN_LOG=runs/emnlp_fisher60/run_log.log
RETRY_LOG=runs/emnlp_fisher60/qwen3_retry.log
SENTINEL="===== FISHER60_AND_G1_ALL_DONE"

mkdir -p runs/emnlp_fisher60
echo "===== Qwen3 retry watcher armed $(date -Iseconds), waiting for $SENTINEL =====" > $RETRY_LOG

# Poll for the all-done sentinel
while ! grep -F "$SENTINEL" "$MAIN_LOG" > /dev/null 2>&1; do
    sleep 120
done

echo "=== Sentinel detected at $(date -Iseconds); cleaning failed Qwen3 artifacts and retrying ===" >> $RETRY_LOG

# Clean previous failed Qwen3 sweep dir (created when sweep OOM'd)
rm -rf runs/emnlp_fisher60/qwen3/sweep_qwen3_8b_3bit_fisher60_gradient_only_b0.6
echo "[cleaned $(date -Iseconds)]" >> $RETRY_LOG

# Re-run Qwen3 Fisher@60%
echo "" >> $RETRY_LOG
echo "===== Qwen3 Fisher@60% retry start $(date -Iseconds) =====" >> $RETRY_LOG
CUDA_VISIBLE_DEVICES=1 safepress sweep --config configs/fisher60_qwen3_3bit.yaml >> $RETRY_LOG 2>&1
echo "===== Qwen3 Fisher@60% retry done $(date -Iseconds) =====" >> $RETRY_LOG

# Offline classifier + utility on the retry
echo "" >> $RETRY_LOG
CUDA_VISIBLE_DEVICES=1 python scripts/classify_existing_responses.py \
    --sweep_root runs/emnlp_fisher60/qwen3 --seed 0 --overwrite >> $RETRY_LOG 2>&1 || \
    echo "[classifier failed]" >> $RETRY_LOG
CUDA_VISIBLE_DEVICES=1 python scripts/utility_eval_from_protect_maps.py \
    --sweep_root runs/emnlp_fisher60/qwen3 --model_id Qwen/Qwen3-8B \
    --bits 3 --group_size 128 --block_size 64 --ppl_n_samples 64 --ppl_max_length 1024 --ppl_stride 512 --mmlu_n 200 \
    --seed 0 --overwrite >> $RETRY_LOG 2>&1 || \
    echo "[utility failed]" >> $RETRY_LOG

echo "" >> $RETRY_LOG
echo "===== QWEN3_FISHER60_RETRY_DONE $(date -Iseconds) =====" >> $RETRY_LOG
