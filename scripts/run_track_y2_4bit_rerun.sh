#!/bin/bash
# Track Y2: re-run the 3 failed 4-bit conditions (int4 + Fisher@8/60%)
# after the bnb→simulated fallback fix in quantize.py.
# Waits for TRACK_Y_DONE (cuda:1 frees) then runs.

set -u
cd /home/jis23009/Dev/safepress_repo

LOG=runs/emnlp_phase4b/track_y2_4bit_rerun.log
TRACK_Y_LOG=runs/emnlp_phase4b/track_y_4bit.log

echo "===== Track Y2 (4-bit rerun) armed $(date -Iseconds) =====" > $LOG
echo "Waiting for TRACK_Y_DONE on $TRACK_Y_LOG..." >> $LOG

while ! grep -F "===== TRACK_Y_DONE" "$TRACK_Y_LOG" > /dev/null 2>&1; do
    sleep 60
done

echo "" >> $LOG
echo "===== TRACK_Y_DONE detected; starting rerun $(date -Iseconds) =====" >> $LOG

# Clean previous failed condition dirs (preserves fp16 which succeeded)
for cond in sweep_qwen3_8b_4bit_fisher60_int4_b0.0 \
            sweep_qwen3_8b_4bit_fisher60_gradient_only_b0.08 \
            sweep_qwen3_8b_4bit_fisher60_gradient_only_b0.6 ; do
    rm -rf runs/emnlp_4bit_ablation/qwen3/$cond
done

# Re-run sweep with fixed quantize.py (bnb→simulated fallback)
echo "" >> $LOG
echo "----- 4-bit rerun sweep start $(date -Iseconds) -----" >> $LOG
CUDA_VISIBLE_DEVICES=1 safepress sweep \
    --config configs/fisher60_qwen3_4bit_remaining.yaml >> $LOG 2>&1
echo "----- 4-bit rerun sweep done $(date -Iseconds) -----" >> $LOG

# Offline classifier on the 3 new conditions
echo "" >> $LOG
echo "----- Offline classifier (new conditions) $(date -Iseconds) -----" >> $LOG
CUDA_VISIBLE_DEVICES=1 python scripts/classify_existing_responses.py \
    --sweep_root runs/emnlp_4bit_ablation/qwen3 --seed 0 --overwrite >> $LOG 2>&1 || \
    echo "[classifier failed]" >> $LOG

# Offline utility on the 3 new conditions
echo "" >> $LOG
echo "----- Offline utility (new conditions) $(date -Iseconds) -----" >> $LOG
CUDA_VISIBLE_DEVICES=1 python scripts/utility_eval_from_protect_maps.py \
    --sweep_root runs/emnlp_4bit_ablation/qwen3 --model_id Qwen/Qwen3-8B \
    --bits 4 --group_size 128 --block_size 64 \
    --ppl_n_samples 64 --ppl_max_length 1024 --ppl_stride 512 \
    --mmlu_n 200 --seed 0 --overwrite >> $LOG 2>&1 || \
    echo "[utility failed]" >> $LOG

# Run final synthesis (includes 4-bit conditions in the table)
python scripts/synthesize_g2_pilot_table.py >> $LOG 2>&1 || true

echo "" >> $LOG
echo "===== TRACK_Y2_DONE $(date -Iseconds) =====" >> $LOG
