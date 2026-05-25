#!/usr/bin/env bash
# scripts/run_paper_emnlp.sh
#
# End-to-end runner for the EMNLP 2026 SafePress paper. Designed for the
# 2x RTX 6000 Ada workstation (96 GB total VRAM). Stages 2-6 each consume a
# fixed model load, so they are safe to interrupt and resume per-stage.
#
# Usage:
#   bash scripts/run_paper_emnlp.sh [stage] [stage ...]
#
# Stages (default: all):
#   data            Download datasets (advbench, harmbench, strongreject,
#                                       xstest, dolly, harmbench_attacks_gcg
#                                       /autodan/pair).
#   g1              Drift-bound theory validation (G1 gate): predicted vs
#                   measured safety-loss drift, R^2 fit. CHEAP. Run first.
#   phase           Phase-transition curves on the main panel (5 models)
#                   at fractional bit-widths.
#   sweep4          4-bit method sweep on the main panel.
#   sweep3          3-bit method sweep on the main panel.
#   snapshot        Build & save SSMP@4% and INT4 model snapshots for each
#                   main-panel model. Required before `attacks` and `xstest`
#                   so those stages can target protected variants, not just
#                   the FP16 base model.
#   attacks         Adversarial-attack eval (GCG/AutoDAN/PAIR) on {FP16,
#                   INT4, SSMP@4%} snapshots (requires `snapshot` stage).
#   xstest          Over-refusal evaluation on {FP16, INT4, SSMP@4%}
#                   snapshots (requires `snapshot` stage).
#   bounds          Per-module Cauchy-Schwarz bounds (descriptive, not the
#                   G1 R^2 gate -- use the `g1` stage for the gate).
#   figures         Regenerate tables + figures from the run outputs.
#
# Environment overrides:
#   N=400                  prompts per condition (HarmBench has 400 behaviors)
#   SEEDS="0 1 2"          seed list (space-separated)
#   MODELS="Qwen/Qwen3-8B meta-llama/Llama-3.1-8B-Instruct ..."

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_DIR"

# Default panel
MODELS_DEFAULT=(
  "Qwen/Qwen3-8B"
  "meta-llama/Llama-3.1-8B-Instruct"
  "Qwen/Qwen2.5-7B-Instruct"
  "google/gemma-2-9b-it"
  "meta-llama/Meta-Llama-3-8B-Instruct"
)
read -ra MODELS_ARR <<< "${MODELS:-${MODELS_DEFAULT[*]}}"
read -ra SEEDS_ARR <<< "${SEEDS:-0 1 2}"
N_PROMPTS="${N:-400}"

STAGES=("$@")
if [ "${#STAGES[@]}" -eq 0 ]; then
  STAGES=(data g1 phase sweep4 sweep3 snapshot attacks xstest bounds figures)
fi

_has_stage() {
  local s
  for s in "${STAGES[@]}"; do [[ "$s" == "$1" ]] && return 0; done
  return 1
}

_safe_tag() {
  echo "$1" | tr '/' '_' | tr ':' '_' | tr -d ' '
}

# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------
if _has_stage data; then
  echo "[paper-emnlp] Stage: data"
  safepress prepare-data \
    --data_dir data \
    --sources advbench harmbench strongreject xstest dolly \
    --harmbench_attacks gcg autodan pair \
    --calib_source c4 --n_calib 128
fi

# --------------------------------------------------------------------------
# g1 (drift-bound theory validation: predicted vs measured drift R^2)
# This is the *theory sanity* gate. It is cheap (~3h per model on Ada 6000)
# and prints a PASS/FAIL line vs the 0.85 R^2 threshold.
# --------------------------------------------------------------------------
if _has_stage g1; then
  echo "[paper-emnlp] Stage: g1 (drift-bound theory validation)"
  # Restrict G1 to the two main models the gate is defined against, per PLAN §16.
  G1_MODELS=("Qwen/Qwen3-8B" "meta-llama/Llama-3.1-8B-Instruct")
  for model in "${G1_MODELS[@]}"; do
    tag=$(_safe_tag "$model")
    out_path="runs/emnlp_g1/${tag}_drift.csv"
    mkdir -p "runs/emnlp_g1"
    safepress analyze drift-validate \
      --model_id "$model" \
      --calib_prompts "data/advbench.jsonl" \
      --max_prompts 64 \
      --bit_widths 8 4 3 2 \
      --group_size 128 \
      --out "$out_path" \
      --dtype float16 --device_map auto
  done
fi

# --------------------------------------------------------------------------
# phase transition (per model)
# --------------------------------------------------------------------------
if _has_stage phase; then
  echo "[paper-emnlp] Stage: phase"
  for model in "${MODELS_ARR[@]}"; do
    tag=$(_safe_tag "$model")
    out_dir="runs/emnlp_phase/$tag"
    mkdir -p "$out_dir"
    safepress experiment phase \
      --model_id "$model" \
      --eval_prompts "data/harmbench.jsonl" \
      --out_dir "$out_dir" \
      --bit_widths 8 5 4 3.5 3 2.5 2 \
      --group_size 128 \
      --dtype float16 --device_map auto
  done
fi

# --------------------------------------------------------------------------
# sweep4 (4-bit method sweep on main panel)
# --------------------------------------------------------------------------
if _has_stage sweep4; then
  echo "[paper-emnlp] Stage: sweep4 (4-bit method sweep)"
  for model in "${MODELS_ARR[@]}"; do
    tag=$(_safe_tag "$model")
    cfg_path="runs/emnlp_sweep/cfg_${tag}.yaml"
    mkdir -p "runs/emnlp_sweep"
    python -c "
import yaml, sys
cfg = yaml.safe_load(open('configs/paper_emnlp.yaml'))
cfg['model_id'] = '$model'
cfg['model_tag'] = '$tag'
cfg['out_root'] = 'runs/emnlp_sweep/$tag'
cfg['summary_csv'] = 'runs/emnlp_sweep/$tag/sweep_summary.csv'
cfg['seeds'] = [int(s) for s in '${SEEDS_ARR[*]}'.split()]
cfg['max_prompts'] = ${N_PROMPTS}
yaml.safe_dump(cfg, open('$cfg_path', 'w'), sort_keys=False)
"
    safepress sweep --config "$cfg_path"
  done
fi

# --------------------------------------------------------------------------
# sweep3 (3-bit method sweep on main panel)
# --------------------------------------------------------------------------
if _has_stage sweep3; then
  echo "[paper-emnlp] Stage: sweep3 (3-bit method sweep)"
  for model in "${MODELS_ARR[@]}"; do
    tag=$(_safe_tag "$model")_3bit
    cfg_path="runs/emnlp_sweep_3bit/cfg_${tag}.yaml"
    mkdir -p "runs/emnlp_sweep_3bit"
    python -c "
import yaml
cfg = yaml.safe_load(open('configs/paper_emnlp_3bit.yaml'))
cfg['model_id'] = '$model'
cfg['model_tag'] = '$tag'
cfg['out_root'] = 'runs/emnlp_sweep_3bit/$tag'
cfg['summary_csv'] = 'runs/emnlp_sweep_3bit/$tag/sweep_summary.csv'
cfg['seeds'] = [int(s) for s in '${SEEDS_ARR[*]}'.split()]
cfg['max_prompts'] = ${N_PROMPTS}
yaml.safe_dump(cfg, open('$cfg_path', 'w'), sort_keys=False)
"
    safepress sweep --config "$cfg_path"
  done
fi

# --------------------------------------------------------------------------
# snapshot: build & save SSMP@4% (and INT4) variants so `attacks` and
# `xstest` can target the protected models rather than the FP16 base.
# Without this stage, the downstream evaluations only measure base-model
# behaviour and the paper claim about "SSMP is more attack-robust" cannot
# be tested.
# --------------------------------------------------------------------------
if _has_stage snapshot; then
  echo "[paper-emnlp] Stage: snapshot (SSMP@4% + INT4 builds per model)"
  for model in "${MODELS_ARR[@]}"; do
    tag=$(_safe_tag "$model")
    snap_dir="runs/emnlp_snapshots/${tag}"
    mkdir -p "$snap_dir"

    # 1. Score blocks (required for both SSMP build and the INT4 protect_map)
    scores_csv="${snap_dir}/scores.csv"
    if [ ! -f "$scores_csv" ]; then
      safepress score \
        --model_id "$model" \
        --calib_prompts data/advbench.jsonl \
        --max_prompts 128 \
        --out "$scores_csv" \
        --bits 4 --group_size 128 --block_size 64 \
        --dtype float16 --device_map auto
    fi

    # 2. Build SSMP @ 4% (overwrite-friendly)
    if [ ! -d "${snap_dir}/ssmp_b0.04/model_quantized" ]; then
      safepress build \
        --model_id "$model" \
        --scores "$scores_csv" \
        --out_dir "${snap_dir}/ssmp_b0.04" \
        --budget 0.04 --block_size 64 --quant_backend bnb4 \
        --dtype float16 --device_map auto --overwrite
    fi

    # 3. Build (near-)INT4 contrast.
    #
    # NOTE: ``safepress build`` calls ``select_top_blocks`` which enforces
    # ``budget > 0``. The closest we get to "pure INT4, no protection" via
    # the build CLI is a single-block budget (~0.1%). We label the directory
    # accordingly (``int4_protect_floor`` rather than misleading ``int4_b0.0``)
    # so downstream attack/xstest results can't be confused with a true
    # unprotected baseline.
    #
    # For a *true* INT4 (zero protection) measurement, use the sweep with
    # ``method: int4`` -- that goes through ``_plan_empty`` and skips the
    # protect_map entirely. The sweep tables are the load-bearing INT4
    # numbers for the paper; this snapshot is just for attacks/xstest.
    if [ ! -d "${snap_dir}/int4_protect_floor/model_quantized" ]; then
      safepress build \
        --model_id "$model" \
        --scores "$scores_csv" \
        --out_dir "${snap_dir}/int4_protect_floor" \
        --budget 0.001 --block_size 64 --quant_backend bnb4 \
        --dtype float16 --device_map auto --overwrite
    fi
  done
fi

# --------------------------------------------------------------------------
# attacks (adversarial robustness against SSMP@4% + INT4 snapshots).
# Iterates the snapshot directories produced by the `snapshot` stage so we
# measure attack ASR against the actually-protected variants.
# --------------------------------------------------------------------------
if _has_stage attacks; then
  echo "[paper-emnlp] Stage: attacks (GCG / AutoDAN / PAIR on snapshots)"
  for model in "${MODELS_ARR[@]}"; do
    tag=$(_safe_tag "$model")
    snap_dir="runs/emnlp_snapshots/${tag}"

    # Three attack targets: FP16 base (= model id), INT4 snapshot,
    # SSMP@4% snapshot. Each is evaluated under the three attacks.
    declare -a TARGETS=(
      "fp16:${model}"
      "int4_pf:${snap_dir}/int4_protect_floor/model_quantized"
      "ssmp_b0.04:${snap_dir}/ssmp_b0.04/model_quantized"
    )
    for target_spec in "${TARGETS[@]}"; do
      target_name="${target_spec%%:*}"
      target_path="${target_spec#*:}"
      if [[ "$target_name" != "fp16" && ! -d "$target_path" ]]; then
        echo "[attacks] skipping ${target_name} for ${tag}: snapshot not built (run 'snapshot' stage first)"
        continue
      fi
      for atk in gcg autodan pair; do
        out_dir="runs/emnlp_attacks/${tag}/${target_name}/${atk}"
        mkdir -p "$out_dir"
        safepress jailbreak \
          --model_path "$target_path" \
          --attack_prompts "data/harmbench_attacks_${atk}.jsonl" \
          --behaviors "data/harmbench.jsonl" \
          --out "${out_dir}/eval.json" \
          --harmbench_classifier \
          --seed 0 --deterministic \
          --n "${N_PROMPTS}"
      done
    done
  done
fi

# --------------------------------------------------------------------------
# xstest (over-refusal) on snapshots (FP16, INT4, SSMP@4%)
# --------------------------------------------------------------------------
if _has_stage xstest; then
  echo "[paper-emnlp] Stage: xstest (over-refusal on snapshots)"
  for model in "${MODELS_ARR[@]}"; do
    tag=$(_safe_tag "$model")
    snap_dir="runs/emnlp_snapshots/${tag}"

    declare -a TARGETS=(
      "fp16:${model}"
      "int4_pf:${snap_dir}/int4_protect_floor/model_quantized"
      "ssmp_b0.04:${snap_dir}/ssmp_b0.04/model_quantized"
    )
    for target_spec in "${TARGETS[@]}"; do
      target_name="${target_spec%%:*}"
      target_path="${target_spec#*:}"
      if [[ "$target_name" != "fp16" && ! -d "$target_path" ]]; then
        echo "[xstest] skipping ${target_name} for ${tag}: snapshot not built (run 'snapshot' stage first)"
        continue
      fi
      out_path="runs/emnlp_xstest/${tag}/${target_name}_eval.json"
      mkdir -p "runs/emnlp_xstest/${tag}"
      safepress xstest \
        --model_path "$target_path" \
        --xstest_prompts "data/xstest.jsonl" \
        --out "$out_path" \
        --seed 0 --deterministic
    done
  done
fi

# --------------------------------------------------------------------------
# bounds (drift-bound theory validation)
# --------------------------------------------------------------------------
if _has_stage bounds; then
  echo "[paper-emnlp] Stage: bounds (per-module CS bounds -- descriptive only)"
  echo "[paper-emnlp] NOTE: this is NOT the G1 R^2 gate; use the 'g1' stage for that."
  for model in "${MODELS_ARR[@]}"; do
    tag=$(_safe_tag "$model")
    out_path="runs/emnlp_bounds/${tag}_bounds.csv"
    mkdir -p "runs/emnlp_bounds"
    safepress analyze bounds \
      --model_id "$model" \
      --calib_prompts "data/advbench.jsonl" \
      --max_prompts 64 \
      --bits 4 --group_size 128 \
      --out "$out_path"
  done
fi

# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------
if _has_stage figures; then
  echo "[paper-emnlp] Stage: figures"
  # Earlier versions piped these to `|| true` to keep the rest of the
  # pipeline running on figure failures. That silently swallowed real
  # bugs (broken viz imports, stale column names). We now fail loud but
  # continue past tables if figures crash, since the table CSV/MD are
  # the load-bearing paper artifacts.
  if ! python scripts/generate_figures.py; then
    echo "[paper-emnlp] WARNING: generate_figures.py FAILED; check viz API alignment."
  fi
  if ! python scripts/generate_tables.py; then
    echo "[paper-emnlp] WARNING: generate_tables.py FAILED."
  fi
fi

echo "[paper-emnlp] Done."
