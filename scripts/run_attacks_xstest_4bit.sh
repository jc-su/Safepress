#!/bin/bash
# Phase 4b: attack robustness (JBB-GCG), over-refusal (XSTest), 4-bit ablation.
#
# Track X (cuda:0): JBB-GCG attacks + XSTest on Qwen3 + Llama conditions.
#   Sweep roots: emnlp_g2_pilot (Qwen3 fp16/int4/scorers @ 4%, 8%),
#                emnlp_g2_pilot_llama31 (Llama same),
#                emnlp_fisher60/qwen3 (Fisher@60%),
#                emnlp_fisher60/llama31 (Fisher@60%).
#
# Track Y (cuda:1): Qwen3 4-bit ablation sweep (fp16/int4/Fisher@8%/Fisher@60%).
#
# Done sentinel: PHASE4B_ALL_DONE

set -u
cd /home/jis23009/Dev/safepress_repo

mkdir -p runs/emnlp_phase4b runs/emnlp_4bit_ablation/qwen3

LOG_X=runs/emnlp_phase4b/track_x_attacks_xstest.log
LOG_Y=runs/emnlp_phase4b/track_y_4bit.log
LOG_DONE=runs/emnlp_phase4b/all_done.log

# --- Launch Track X (cuda:0) ---
(
    echo "===== Track X (cuda:0) started $(date -Iseconds) =====" > $LOG_X
    for pair in \
        "runs/emnlp_g2_pilot|Qwen/Qwen3-8B" \
        "runs/emnlp_fisher60/qwen3|Qwen/Qwen3-8B" \
        "runs/emnlp_g2_pilot_llama31|meta-llama/Llama-3.1-8B-Instruct" \
        "runs/emnlp_fisher60/llama31|meta-llama/Llama-3.1-8B-Instruct" ; do
        root=${pair%|*}
        model_id=${pair#*|}
        if [ ! -d "$root" ]; then
            echo "[skip] $root missing" >> $LOG_X
            continue
        fi
        echo "" >> $LOG_X
        echo "----- attacks+xstest $root ($model_id) start $(date -Iseconds) -----" >> $LOG_X
        CUDA_VISIBLE_DEVICES=0 python scripts/attack_xstest_eval_from_protect_maps.py \
            --sweep_root $root --model_id "$model_id" \
            --bits 3 --group_size 128 --block_size 64 \
            --max_new_tokens 256 --max_length 1024 \
            --max_prompts 100 \
            --seed 0 >> $LOG_X 2>&1
        echo "----- attacks+xstest $root done $(date -Iseconds) -----" >> $LOG_X
    done
    echo "" >> $LOG_X
    echo "===== TRACK_X_DONE $(date -Iseconds) =====" >> $LOG_X
) &

# --- Launch Track Y (cuda:1) ---
(
    echo "===== Track Y (cuda:1) started $(date -Iseconds) =====" > $LOG_Y

    # 1) 4-bit Fisher@60% + Fisher@8% + fp16 + int4 sweep on Qwen3
    echo "" >> $LOG_Y
    echo "----- 4-bit ablation sweep start $(date -Iseconds) -----" >> $LOG_Y
    CUDA_VISIBLE_DEVICES=1 safepress sweep --config configs/fisher60_qwen3_4bit.yaml >> $LOG_Y 2>&1
    echo "----- 4-bit ablation sweep done $(date -Iseconds) -----" >> $LOG_Y

    # 2) Offline classifier + utility on the 4-bit ablation
    echo "" >> $LOG_Y
    echo "----- 4-bit offline classifier start $(date -Iseconds) -----" >> $LOG_Y
    CUDA_VISIBLE_DEVICES=1 python scripts/classify_existing_responses.py \
        --sweep_root runs/emnlp_4bit_ablation/qwen3 --seed 0 --overwrite >> $LOG_Y 2>&1 || \
        echo "[classifier failed]" >> $LOG_Y
    echo "----- 4-bit offline classifier done $(date -Iseconds) -----" >> $LOG_Y

    echo "" >> $LOG_Y
    echo "----- 4-bit offline utility start $(date -Iseconds) -----" >> $LOG_Y
    CUDA_VISIBLE_DEVICES=1 python scripts/utility_eval_from_protect_maps.py \
        --sweep_root runs/emnlp_4bit_ablation/qwen3 --model_id Qwen/Qwen3-8B \
        --bits 4 --group_size 128 --block_size 64 \
        --ppl_n_samples 64 --ppl_max_length 1024 --ppl_stride 512 \
        --mmlu_n 200 --seed 0 --overwrite >> $LOG_Y 2>&1 || \
        echo "[utility failed]" >> $LOG_Y
    echo "----- 4-bit offline utility done $(date -Iseconds) -----" >> $LOG_Y

    echo "" >> $LOG_Y
    echo "===== TRACK_Y_DONE $(date -Iseconds) =====" >> $LOG_Y
) &

# --- All-done watcher ---
(
    echo "===== Phase 4b watcher started $(date -Iseconds) =====" > $LOG_DONE
    while true; do
        x_done=0; y_done=0
        [ -f "$LOG_X" ] && grep -F "===== TRACK_X_DONE" "$LOG_X" > /dev/null 2>&1 && x_done=1
        [ -f "$LOG_Y" ] && grep -F "===== TRACK_Y_DONE" "$LOG_Y" > /dev/null 2>&1 && y_done=1
        if [ $x_done -eq 1 ] && [ $y_done -eq 1 ]; then break; fi
        sleep 120
    done
    echo "" >> $LOG_DONE
    echo "===== PHASE4B_ALL_DONE $(date -Iseconds) =====" >> $LOG_DONE
    # Final synthesis includes 4-bit conditions (the synthesis script
    # already scans by glob and picks up any new dirs).
    python scripts/synthesize_g2_pilot_table.py >> $LOG_DONE 2>&1 || true
) &

wait
