#!/bin/bash
# Track Z (cuda:1): complete the 5-model attack panel by running JBB-GCG +
# XSTest on Mistral, GLM-4, DeepSeek g2_pilot conditions + their Fisher@60%.
# Track X (cuda:0) keeps doing Qwen3 + Llama; this runs the other 3 models
# in parallel.

set -u
cd /home/jis23009/Dev/safepress_repo

LOG=runs/emnlp_phase4b/track_z_attacks_3models.log
mkdir -p runs/emnlp_phase4b

echo "===== Track Z (cuda:1) started $(date -Iseconds) =====" > $LOG
echo "PID=$$ PPID=$PPID" >> $LOG

for pair in \
    "runs/emnlp_g2_pilot_mistral|mistralai/Ministral-8B-Instruct-2410" \
    "runs/emnlp_fisher60/mistral|mistralai/Ministral-8B-Instruct-2410" \
    "runs/emnlp_g2_pilot_glm4|THUDM/GLM-4-9B-0414" \
    "runs/emnlp_fisher60/glm4|THUDM/GLM-4-9B-0414" \
    "runs/emnlp_g2_pilot_deepseek|deepseek-ai/DeepSeek-R1-Distill-Llama-8B" \
    "runs/emnlp_fisher60/deepseek|deepseek-ai/DeepSeek-R1-Distill-Llama-8B" ; do
    root=${pair%|*}
    model_id=${pair#*|}
    if [ ! -d "$root" ]; then
        echo "[skip] $root missing" >> $LOG
        continue
    fi
    echo "" >> $LOG
    echo "----- attacks+xstest $root ($model_id) start $(date -Iseconds) -----" >> $LOG
    CUDA_VISIBLE_DEVICES=1 python scripts/attack_xstest_eval_from_protect_maps.py \
        --sweep_root $root --model_id "$model_id" \
        --bits 3 --group_size 128 --block_size 64 \
        --max_new_tokens 256 --max_length 1024 \
        --max_prompts 100 \
        --seed 0 >> $LOG 2>&1
    echo "----- attacks+xstest $root done $(date -Iseconds) -----" >> $LOG
done

echo "" >> $LOG
echo "===== TRACK_Z_DONE $(date -Iseconds) =====" >> $LOG

# Regenerate synthesis tables to include attack/XSTest data from all 5 models
python scripts/synthesize_g2_pilot_table.py >> $LOG 2>&1 || true
