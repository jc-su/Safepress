#!/bin/bash
# Steal DeepSeek Fisher@60% from Track B onto cuda:0 (currently idle).
# When this finishes, kill any running Track B DeepSeek subprocess so
# Track B advances to its offline phase without re-running the sweep.

set -u
cd /home/jis23009/Dev/safepress_repo

LOG=runs/emnlp_fisher60/deepseek_steal_cuda0.log
echo "===== DeepSeek steal (cuda:0) started $(date -Iseconds) =====" > $LOG

# Clean any partial output (none expected, but be safe)
rm -rf runs/emnlp_fisher60/deepseek/sweep_deepseek_r1_distill_llama_8b_3bit_fisher60_gradient_only_b0.6

echo "----- DeepSeek Fisher@60% start $(date -Iseconds) -----" >> $LOG
CUDA_VISIBLE_DEVICES=0 safepress sweep --config configs/fisher60_deepseek_3bit.yaml >> $LOG 2>&1
echo "----- DeepSeek Fisher@60% done $(date -Iseconds) -----" >> $LOG

# Kill any Track B DeepSeek subprocess that may have started (avoid overwrite)
echo "" >> $LOG
echo "----- Killing any concurrent Track B DeepSeek subprocess $(date -Iseconds) -----" >> $LOG
pkill -f "safepress sweep --config configs/fisher60_deepseek_3bit.yaml" >> $LOG 2>&1 || \
    echo "[no Track B DeepSeek subprocess to kill]" >> $LOG

# Offline classifier + utility for DeepSeek (also frees Track B from doing it)
echo "" >> $LOG
echo "----- Offline classifier deepseek start $(date -Iseconds) -----" >> $LOG
CUDA_VISIBLE_DEVICES=0 python scripts/classify_existing_responses.py \
    --sweep_root runs/emnlp_fisher60/deepseek --seed 0 --overwrite >> $LOG 2>&1 || \
    echo "[classifier failed]" >> $LOG
echo "----- Offline classifier deepseek done $(date -Iseconds) -----" >> $LOG

echo "" >> $LOG
echo "----- Offline utility deepseek start $(date -Iseconds) -----" >> $LOG
CUDA_VISIBLE_DEVICES=0 python scripts/utility_eval_from_protect_maps.py \
    --sweep_root runs/emnlp_fisher60/deepseek \
    --model_id "deepseek-ai/DeepSeek-R1-Distill-Llama-8B" \
    --bits 3 --group_size 128 --block_size 64 \
    --ppl_n_samples 64 --ppl_max_length 1024 --ppl_stride 512 \
    --mmlu_n 200 --seed 0 --overwrite >> $LOG 2>&1 || \
    echo "[utility failed]" >> $LOG
echo "----- Offline utility deepseek done $(date -Iseconds) -----" >> $LOG

echo "" >> $LOG
echo "===== DEEPSEEK_STEAL_DONE $(date -Iseconds) =====" >> $LOG
