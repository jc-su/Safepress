#!/bin/bash
# Track W (cuda:0): steal DeepSeek attack/XSTest from Track Z.
# Track Z (cuda:1) does Mistral -> GLM-4 -> DeepSeek serially; we run
# DeepSeek in parallel here. The eval script's existence-check will make
# Track Z skip DeepSeek when it gets there.

set -u
cd /home/jis23009/Dev/safepress_repo

LOG=runs/emnlp_phase4b/track_w_deepseek_attacks.log
echo "===== Track W (cuda:0, DeepSeek attacks) started $(date -Iseconds) =====" > $LOG
echo "PID=$$ PPID=$PPID" >> $LOG

for pair in \
    "runs/emnlp_g2_pilot_deepseek|deepseek-ai/DeepSeek-R1-Distill-Llama-8B" \
    "runs/emnlp_fisher60/deepseek|deepseek-ai/DeepSeek-R1-Distill-Llama-8B" ; do
    root=${pair%|*}
    model_id=${pair#*|}
    echo "" >> $LOG
    echo "----- attacks+xstest $root start $(date -Iseconds) -----" >> $LOG
    CUDA_VISIBLE_DEVICES=0 python scripts/attack_xstest_eval_from_protect_maps.py \
        --sweep_root $root --model_id "$model_id" \
        --bits 3 --group_size 128 --block_size 64 \
        --max_new_tokens 256 --max_length 1024 \
        --max_prompts 100 \
        --seed 0 >> $LOG 2>&1
    echo "----- attacks+xstest $root done $(date -Iseconds) -----" >> $LOG
done

echo "" >> $LOG
echo "===== TRACK_W_DONE $(date -Iseconds) =====" >> $LOG
