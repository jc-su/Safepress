#!/bin/bash
# Dump raw block scores for all 5 models across 2 GPUs (single pass each).
# Enables full-ranking Spearman + escape-clause correlation analysis.
set -u
cd /home/jis23009/Dev/safepress_repo
mkdir -p runs/emnlp_blockscores
LOG=runs/emnlp_blockscores/run.log
echo "===== block-scores dump started $(date -Iseconds) =====" > $LOG

# GPU 0: Qwen3, GLM-4, DeepSeek
(
  for pair in \
    "Qwen/Qwen3-8B|qwen3" \
    "THUDM/GLM-4-9B-0414|glm4" \
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B|deepseek" ; do
    mid=${pair%|*}; tag=${pair#*|}
    echo "[gpu0] $tag start $(date -Iseconds)" >> $LOG
    CUDA_VISIBLE_DEVICES=0 python scripts/dump_block_scores.py \
      --model_id "$mid" --max_prompts 128 --bits 3 \
      --out_csv runs/emnlp_blockscores/${tag}_scores.csv >> $LOG 2>&1
    echo "[gpu0] $tag done $(date -Iseconds)" >> $LOG
  done
  echo "===== GPU0_DONE $(date -Iseconds) =====" >> $LOG
) &

# GPU 1: Llama, Mistral
(
  for pair in \
    "meta-llama/Llama-3.1-8B-Instruct|llama31" \
    "mistralai/Ministral-8B-Instruct-2410|mistral" ; do
    mid=${pair%|*}; tag=${pair#*|}
    echo "[gpu1] $tag start $(date -Iseconds)" >> $LOG
    CUDA_VISIBLE_DEVICES=1 python scripts/dump_block_scores.py \
      --model_id "$mid" --max_prompts 128 --bits 3 \
      --out_csv runs/emnlp_blockscores/${tag}_scores.csv >> $LOG 2>&1
    echo "[gpu1] $tag done $(date -Iseconds)" >> $LOG
  done
  echo "===== GPU1_DONE $(date -Iseconds) =====" >> $LOG
) &

wait
echo "===== BLOCKSCORES_ALL_DONE $(date -Iseconds) =====" >> $LOG
