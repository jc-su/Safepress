#!/bin/bash
# Fisher@60% across 5 models + G1 drift bound on 3 missing models.
# After Phase 4 baselines hit hardware limits (qresafe DPO + CWP@60% OOM
# on 48GB A6000), this provides the theorem-equivalent comparison:
# Fisher@60% serves as CWP@60% under Theorem 2 (structural collapse).
#
# Single GPU (cuda:1). nohup setsid for SSH-disconnect survival.
# Total ETA: ~12-15 GPU-h on A6000.

set -u
cd /home/jis23009/Dev/safepress_repo

LOG=runs/emnlp_fisher60/run_log.log
mkdir -p runs/emnlp_fisher60 runs/emnlp_g1

echo "===== Fisher60 + G1 launcher started $(date -Iseconds) =====" > $LOG
echo "PID=$$ PPID=$PPID" >> $LOG
echo "" >> $LOG

# --------------------------------------------------------------------
# Step 1: Fisher@60% sweeps across 5 models (serial)
# Each sweep loads ONE model, computes Fisher scoring once at 60% budget,
# quantizes, evals. No compound conditions = ~36GB peak, fits 48GB.
# --------------------------------------------------------------------
echo "===== STEP 1: Fisher@60% sweeps =====" >> $LOG
for cfg in \
    configs/fisher60_qwen3_3bit.yaml \
    configs/fisher60_llama31_3bit.yaml \
    configs/fisher60_mistral_3bit.yaml \
    configs/fisher60_glm4_3bit.yaml \
    configs/fisher60_deepseek_3bit.yaml ; do
    tag=$(basename $cfg .yaml | sed 's/fisher60_//')
    echo "" >> $LOG
    echo "----- Fisher@60% $tag start $(date -Iseconds) -----" >> $LOG
    CUDA_VISIBLE_DEVICES=1 safepress sweep --config $cfg >> $LOG 2>&1
    echo "----- Fisher@60% $tag done  $(date -Iseconds) -----" >> $LOG
done
echo "===== FISHER60_ALL_DONE $(date -Iseconds) =====" >> $LOG

# --------------------------------------------------------------------
# Step 2: G1 drift-bound validation on the 3 missing models
# (Qwen3 + Llama already have G1 in runs/emnlp_g1/)
# --------------------------------------------------------------------
echo "" >> $LOG
echo "===== STEP 2: G1 drift bound extension =====" >> $LOG

declare -A G1_MODELS=(
    [mistral]="mistralai/Ministral-8B-Instruct-2410"
    [glm4]="THUDM/GLM-4-9B-0414"
    [deepseek]="deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
)
for tag in mistral glm4 deepseek; do
    model_id=${G1_MODELS[$tag]}
    out=runs/emnlp_g1/${tag}_drift.csv
    echo "" >> $LOG
    echo "----- G1 $tag ($model_id) start $(date -Iseconds) -----" >> $LOG
    CUDA_VISIBLE_DEVICES=1 safepress analyze drift-validate \
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
echo "===== G1_EXTENSION_DONE $(date -Iseconds) =====" >> $LOG

# --------------------------------------------------------------------
# Step 3: Offline HarmBench-style refusal classifier on Fisher@60% responses
# (use the heuristic classifier as primary; HarmBench classifier optional)
# --------------------------------------------------------------------
echo "" >> $LOG
echo "===== STEP 3: Offline classifier on Fisher@60% =====" >> $LOG
for tag in qwen3 llama31 mistral glm4 deepseek; do
    root=runs/emnlp_fisher60/$tag
    if [ -d "$root" ]; then
        echo "----- classifier $tag $(date -Iseconds) -----" >> $LOG
        CUDA_VISIBLE_DEVICES=1 python scripts/classify_existing_responses.py \
            --sweep_root $root --seed 0 --overwrite >> $LOG 2>&1 || \
            echo "[classifier $tag failed but continuing]" >> $LOG
    fi
done

# --------------------------------------------------------------------
# Step 4: Offline utility eval (PPL + MMLU) from protect maps
# --------------------------------------------------------------------
echo "" >> $LOG
echo "===== STEP 4: Offline utility eval =====" >> $LOG

declare -A UTIL_MODELS=(
    [qwen3]="Qwen/Qwen3-8B"
    [llama31]="meta-llama/Llama-3.1-8B-Instruct"
    [mistral]="mistralai/Ministral-8B-Instruct-2410"
    [glm4]="THUDM/GLM-4-9B-0414"
    [deepseek]="deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
)
for tag in qwen3 llama31 mistral glm4 deepseek; do
    model_id=${UTIL_MODELS[$tag]}
    root=runs/emnlp_fisher60/$tag
    if [ -d "$root" ]; then
        echo "----- utility $tag $(date -Iseconds) -----" >> $LOG
        CUDA_VISIBLE_DEVICES=1 python scripts/utility_eval_from_protect_maps.py \
            --sweep_root $root --model_id "$model_id" \
            --bits 3 --group_size 128 --block_size 64 \
            --ppl_n_samples 64 --ppl_max_length 1024 --ppl_stride 512 \
            --mmlu_n 200 --seed 0 --overwrite >> $LOG 2>&1 || \
            echo "[utility $tag failed but continuing]" >> $LOG
    fi
done

echo "" >> $LOG
echo "===== FISHER60_AND_G1_ALL_DONE $(date -Iseconds) =====" >> $LOG
