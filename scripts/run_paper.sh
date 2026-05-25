#!/usr/bin/env bash
# ==========================================================================
# SafePress — Full Paper Reproduction Script
# ==========================================================================
#
# Runs every experiment, evaluation, analysis, and figure generation step
# needed to reproduce all paper results.
#
# Usage:
#   bash scripts/run_paper.sh                     # full run (all 3 models)
#   bash scripts/run_paper.sh --primary-only      # only primary model
#   bash scripts/run_paper.sh --skip-data          # skip data download
#   bash scripts/run_paper.sh --step 3             # resume from step 3
#
# Prerequisites:
#   pip install -e ".[all]"
#   ~48 GB GPU VRAM recommended (A100/A6000) for 8B models
#   Alternatively: ~24 GB for single-model runs with device_map=auto
#
# Output structure:
#   data/                    ← datasets (Step 1)
#   runs/                    ← all experiment outputs (Steps 2-6)
#   figures/                 ← publication PDF figures (Step 7)
#   tables/                  ← LaTeX/Markdown tables (Step 8)
# ==========================================================================

set -euo pipefail

# ── Configuration ──────────────────────────────────────────────────────────
PRIMARY_MODEL="Qwen/Qwen3-8B"
SECONDARY_MODEL="meta-llama/Llama-3.1-8B-Instruct"
TERTIARY_MODEL="google/gemma-3-4b-it"
DTYPE="float16"
DEVICE_MAP="auto"
BLOCK_SIZE=64
BITS=4
GROUP_SIZE=128
BUDGET=0.02
MAX_PROMPTS=128
MAX_NEW_TOKENS=256
BATCH_SIZE=1

DATA_DIR="data"
RUNS_DIR="runs"
FIGURES_DIR="figures"
TABLES_DIR="tables"

# Parse flags
PRIMARY_ONLY=false
SKIP_DATA=false
START_STEP=1
while [[ $# -gt 0 ]]; do
    case "$1" in
        --primary-only) PRIMARY_ONLY=true; shift ;;
        --skip-data)    SKIP_DATA=true; shift ;;
        --step)         START_STEP="$2"; shift 2 ;;
        *) echo "Unknown flag: $1"; exit 1 ;;
    esac
done

if [ "$PRIMARY_ONLY" = true ]; then
    MODELS=("$PRIMARY_MODEL")
else
    MODELS=("$PRIMARY_MODEL" "$SECONDARY_MODEL" "$TERTIARY_MODEL")
fi

log() {
    echo ""
    echo "=================================================================="
    echo "  STEP $1: $2"
    echo "=================================================================="
    echo ""
}

# ══════════════════════════════════════════════════════════════════════════
# STEP 1: Data Preparation
# ══════════════════════════════════════════════════════════════════════════
if [ "$START_STEP" -le 1 ] && [ "$SKIP_DATA" = false ]; then
    log 1 "Downloading and preparing datasets"

    python -m safepress prepare-data \
        --data_dir "$DATA_DIR" \
        --sources advbench harmbench strongreject \
        --calib_source c4 \
        --n_calib "$MAX_PROMPTS"

    # Prepare utility prompts (Alpaca) for CWP baseline and utility evaluation
    python scripts/prepare_utility_prompts.py \
        --out "${DATA_DIR}/utility_alpaca.jsonl" \
        --n 1000 --seed 0

    echo "[Step 1] Datasets ready in ${DATA_DIR}/"
fi

# ══════════════════════════════════════════════════════════════════════════
# STEP 2: Score safety-critical blocks (per model)
# ══════════════════════════════════════════════════════════════════════════
if [ "$START_STEP" -le 2 ]; then
    log 2 "Computing block safety-drift scores"

    for MODEL in "${MODELS[@]}"; do
        TAG=$(echo "$MODEL" | tr '/' '_' | tr '[:upper:]' '[:lower:]')
        SCORE_OUT="${RUNS_DIR}/scores/${TAG}_scores.csv"
        mkdir -p "${RUNS_DIR}/scores"

        echo "[Step 2] Scoring: $MODEL -> $SCORE_OUT"
        python -m safepress score \
            --model_id "$MODEL" \
            --calib_prompts "${DATA_DIR}/advbench.jsonl" \
            --out "$SCORE_OUT" \
            --dtype "$DTYPE" \
            --device_map "$DEVICE_MAP" \
            --bits "$BITS" \
            --group_size "$GROUP_SIZE" \
            --block_size "$BLOCK_SIZE" \
            --batch_size "$BATCH_SIZE" \
            --max_prompts "$MAX_PROMPTS"
    done

    echo "[Step 2] All scores computed."
fi

# ══════════════════════════════════════════════════════════════════════════
# STEP 3: Build SSMP models (protect + split + quantize)
# ══════════════════════════════════════════════════════════════════════════
if [ "$START_STEP" -le 3 ]; then
    log 3 "Building SSMP mixed-precision models"

    for MODEL in "${MODELS[@]}"; do
        TAG=$(echo "$MODEL" | tr '/' '_' | tr '[:upper:]' '[:lower:]')
        SCORE_CSV="${RUNS_DIR}/scores/${TAG}_scores.csv"
        BUILD_OUT="${RUNS_DIR}/build/${TAG}_ssmp"

        echo "[Step 3] Building: $MODEL -> $BUILD_OUT"
        python -m safepress build \
            --model_id "$MODEL" \
            --scores "$SCORE_CSV" \
            --out_dir "$BUILD_OUT" \
            --budget "$BUDGET" \
            --block_size "$BLOCK_SIZE" \
            --dtype "$DTYPE" \
            --device_map "$DEVICE_MAP" \
            --quant_backend bnb4 \
            --overwrite
    done

    echo "[Step 3] All SSMP models built."
fi

# ══════════════════════════════════════════════════════════════════════════
# STEP 4: Paper-style method×budget sweep (primary model)
# ══════════════════════════════════════════════════════════════════════════
if [ "$START_STEP" -le 4 ]; then
    log 4 "Running method × budget sweep (paper Table 1)"

    python -m safepress sweep \
        --config configs/paper_sweep.yaml

    echo "[Step 4] Sweep complete. Summary: ${RUNS_DIR}/sweep/sweep_summary.csv"
fi

# ══════════════════════════════════════════════════════════════════════════
# STEP 5: Causal experiments (primary model)
# ══════════════════════════════════════════════════════════════════════════
if [ "$START_STEP" -le 5 ]; then
    log 5 "Running causal experiments (targeted / rollback / control)"

    TAG=$(echo "$PRIMARY_MODEL" | tr '/' '_' | tr '[:upper:]' '[:lower:]')
    SCORE_CSV="${RUNS_DIR}/scores/${TAG}_scores.csv"

    python -m safepress experiment causal \
        --model_id "$PRIMARY_MODEL" \
        --scores "$SCORE_CSV" \
        --eval_prompts "${DATA_DIR}/harmbench.jsonl" \
        --out_dir "${RUNS_DIR}/causal" \
        --dtype "$DTYPE" \
        --device_map "$DEVICE_MAP" \
        --bits "$BITS" \
        --group_size "$GROUP_SIZE" \
        --block_size "$BLOCK_SIZE" \
        --budget "$BUDGET"

    echo "[Step 5] Causal experiments done."
fi

# ══════════════════════════════════════════════════════════════════════════
# STEP 6: Phase transition experiment (primary model)
# ══════════════════════════════════════════════════════════════════════════
if [ "$START_STEP" -le 6 ]; then
    log 6 "Running phase transition experiment (bit-width vs safety)"

    python -m safepress experiment phase \
        --model_id "$PRIMARY_MODEL" \
        --eval_prompts "${DATA_DIR}/harmbench.jsonl" \
        --out_dir "${RUNS_DIR}/phase_transition" \
        --dtype "$DTYPE" \
        --device_map "$DEVICE_MAP" \
        --bit_widths 8 4 3 2

    echo "[Step 6] Phase transition done."
fi

# ══════════════════════════════════════════════════════════════════════════
# STEP 7: Safety evaluation on all built models
# ══════════════════════════════════════════════════════════════════════════
if [ "$START_STEP" -le 7 ]; then
    log 7 "Running safety evaluation (refusal rate + StrongREJECT)"

    for MODEL in "${MODELS[@]}"; do
        TAG=$(echo "$MODEL" | tr '/' '_' | tr '[:upper:]' '[:lower:]')
        MODEL_PATH="${RUNS_DIR}/build/${TAG}_ssmp/model_quantized"

        if [ -d "$MODEL_PATH" ]; then
            echo "[Step 7] Evaluating: $MODEL_PATH"
            python -m safepress eval \
                --model_path "$MODEL_PATH" \
                --eval_prompts "${DATA_DIR}/harmbench.jsonl" \
                --out "${RUNS_DIR}/eval/${TAG}_ssmp_eval.json" \
                --dtype "$DTYPE" \
                --device_map "$DEVICE_MAP" \
                --max_new_tokens "$MAX_NEW_TOKENS" \
                --try_strongreject \
                --llamaguard
        else
            echo "[Step 7] SKIP: Model not found at $MODEL_PATH"
        fi
    done

    echo "[Step 7] All evaluations done."
fi

# ══════════════════════════════════════════════════════════════════════════
# STEP 8: Analysis — refusal direction + layer error (primary model)
# ══════════════════════════════════════════════════════════════════════════
if [ "$START_STEP" -le 8 ]; then
    log 8 "Running analysis (refusal direction + layer quantization error)"

    # Refusal direction (needs contrastive prompts)
    echo "[Step 8a] Refusal direction analysis..."
    python -m safepress analyze refusal-direction \
        --model_id "$PRIMARY_MODEL" \
        --harmful_prompts "${DATA_DIR}/advbench.jsonl" \
        --harmless_prompts "${DATA_DIR}/utility_alpaca.jsonl" \
        --out_dir "${RUNS_DIR}/analysis/refusal_direction" \
        --dtype "$DTYPE" \
        --device_map "$DEVICE_MAP"

    # Layer-level quantization error
    echo "[Step 8b] Layer quantization error analysis..."
    python -m safepress analyze layer-error \
        --model_id "$PRIMARY_MODEL" \
        --out_dir "${RUNS_DIR}/analysis/layer_error" \
        --dtype "$DTYPE" \
        --device_map "$DEVICE_MAP"

    # Cauchy-Schwarz drift bounds (theory section)
    echo "[Step 8c] Cauchy-Schwarz drift bounds..."
    python -m safepress analyze bounds \
        --model_id "$PRIMARY_MODEL" \
        --calib_prompts "${DATA_DIR}/advbench.jsonl" \
        --out "${RUNS_DIR}/analysis/drift_bounds.csv" \
        --dtype "$DTYPE" \
        --device_map "$DEVICE_MAP" \
        --bits "$BITS" \
        --group_size "$GROUP_SIZE" \
        --max_prompts "$MAX_PROMPTS"

    echo "[Step 8] Analysis done."
fi

# ══════════════════════════════════════════════════════════════════════════
# STEP 9: Generate all figures
# ══════════════════════════════════════════════════════════════════════════
if [ "$START_STEP" -le 9 ]; then
    log 9 "Generating publication figures"

    python scripts/generate_figures.py \
        --results_dir "$RUNS_DIR" \
        --output_dir "$FIGURES_DIR"

    echo "[Step 9] Figures saved to ${FIGURES_DIR}/"
fi

# ══════════════════════════════════════════════════════════════════════════
# STEP 10: Generate tables
# ══════════════════════════════════════════════════════════════════════════
if [ "$START_STEP" -le 10 ]; then
    log 10 "Generating LaTeX and Markdown tables"

    mkdir -p "$TABLES_DIR"
    python scripts/generate_tables.py \
        --results_dir "$RUNS_DIR" \
        --output_dir "$TABLES_DIR"

    echo "[Step 10] Tables saved to ${TABLES_DIR}/"
fi

echo ""
echo "=================================================================="
echo "  ALL DONE — Paper reproduction complete!"
echo "=================================================================="
echo ""
echo "  Datasets:     ${DATA_DIR}/"
echo "  Runs:         ${RUNS_DIR}/"
echo "  Figures:      ${FIGURES_DIR}/"
echo "  Tables:       ${TABLES_DIR}/"
echo ""
echo "  Key outputs:"
echo "    - Sweep summary:  ${RUNS_DIR}/sweep/sweep_summary.csv"
echo "    - Causal results: ${RUNS_DIR}/causal/"
echo "    - Phase curve:    ${RUNS_DIR}/phase_transition/"
echo "    - Eval JSONs:     ${RUNS_DIR}/eval/"
echo "    - Drift bounds:   ${RUNS_DIR}/analysis/drift_bounds.csv"
echo "    - PDF figures:    ${FIGURES_DIR}/"
echo "    - LaTeX tables:   ${TABLES_DIR}/"
echo ""
