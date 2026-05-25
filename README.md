# Structural Collapse of Safety-Critical Weight Protection Scorers Under Group-Wise Post-Training Quantization
---

## Pipeline overview

```
1. Prepare data       Download AdvBench / HarmBench / StrongREJECT / calibration data
2. Score blocks       Compute per-block safety-drift scores (Taylor, SNIP, Wanda, ...)
3. Protect + quantize Select top-K blocks for FP16, quantize rest (bnb4 / AWQ / GPTQ)
4. Evaluate           Safety (refusal rate, HarmBench ASR, StrongREJECT) + Utility (PPL, MMLU, TruthfulQA)
5. Analyze            Refusal direction profiling, layer-level error analysis
6. Experiment         Causal ablations, budget sweeps, phase-transition curves
7. Visualize          Publication-quality figures and LaTeX tables
```

---

## What "safety-critical block scoring" means

Given a model weight tensor **W** and a safety-loss **L_safety** (computed on "refusal calibration" prompts), we approximate the safety loss increase under a quantizer **Q(.)** by:

```
score(b) = sum_{i in b} |g_i * delta_w_i|
```

where `g_i = dL/dW_i` and `delta_w_i = Q(W_i) - W_i`. The top-K blocks are kept in higher precision under a user-specified **budget**.

---

## Models

SafePress targets **dense, safety-aligned, instruction-tuned** models. The default configs use:

| Role | Model | Family | Size |
|------|-------|--------|------|
| **Primary** | `Qwen/Qwen3-8B` | Qwen | 8B |
| Secondary | `meta-llama/Llama-3.1-8B-Instruct` | Llama | 8B |
| Cross-family | `google/gemma-3-4b-it` | Gemma | 4B |
| Ablation | `Qwen/Qwen2.5-3B-Instruct` | Qwen | 3B |

All model IDs are configured in `configs/models.yaml`. Every experiment config defaults to `Qwen/Qwen3-8B` as the primary model.

---

## Install

```bash
# editable install (core)
pip install -e .

# with specific extras
pip install -e ".[bnb]"          # bitsandbytes 4-bit
pip install -e ".[awq]"          # AutoAWQ quantization
pip install -e ".[gptq]"        # GPTQ quantization
pip install -e ".[viz]"          # matplotlib + seaborn for figures
pip install -e ".[eval]"         # jailbreakbench
pip install -e ".[agent]"        # agentdojo
pip install -e ".[strongreject]" # StrongREJECT evaluator

# everything (recommended for paper reproduction)
pip install -e ".[all]"
```

---

## Full paper reproduction (one command)

```bash
bash scripts/run_paper.sh
```

This master script runs all 10 steps end-to-end. Flags:

```bash
bash scripts/run_paper.sh --primary-only     # only Qwen3-8B (faster)
bash scripts/run_paper.sh --skip-data         # skip data download if data/ exists
bash scripts/run_paper.sh --step 5            # resume from step 5
```

### The 10 steps

| Step | Command | Output | Paper artifact |
|------|---------|--------|----------------|
| 1 | `safepress prepare-data` | `data/*.jsonl` | -- |
| 2 | `safepress score` (x3 models) | `runs/scores/*_scores.csv` | Heatmap figure |
| 3 | `safepress build` (x3 models) | `runs/build/*/model_quantized/` | -- |
| 4 | `safepress sweep --config configs/paper_sweep.yaml` | `runs/sweep/sweep_summary.csv` | **Table 1** (main) |
| 5 | `safepress experiment causal` | `runs/causal/*.json` | **Table 2** + figure |
| 6 | `safepress experiment phase` | `runs/phase_transition/*.json` | **Table 4** + figure |
| 7 | `safepress eval` (x3 models) | `runs/eval/*_eval.json` | **Table 3** (cross-model) |
| 8 | `safepress analyze refusal-direction` + `layer-error` + `bounds` | `runs/analysis/` | Figures + theory |
| 9 | `python scripts/generate_figures.py` | `figures/*.pdf` | All figures |
| 10 | `python scripts/generate_tables.py` | `tables/*.tex`, `tables/*.md` | All tables |

### Output directory structure

```
data/
  advbench.jsonl              # AdvBench harmful prompts
  harmbench.jsonl             # HarmBench behaviors
  strongreject.jsonl          # StrongREJECT forbidden prompts
  calibration_c4.jsonl        # C4 calibration samples
  utility_alpaca.jsonl         # Alpaca utility prompts (CWP baseline)

runs/
  scores/                     # Step 2: per-block safety-drift scores
  build/                      # Step 3: SSMP mixed-precision models
  sweep/                      # Step 4: method x budget sweep
    sweep_summary.csv         #   main results CSV
  causal/                     # Step 5: causal experiments
  phase_transition/           # Step 6: bit-width vs safety
  eval/                       # Step 7: safety evaluations
  analysis/                   # Step 8: refusal direction + layer error + drift bounds
    refusal_direction/
    layer_error/
    drift_bounds.csv           #   per-module Cauchy-Schwarz bounds

figures/                      # Step 9: publication PDF plots
tables/                       # Step 10: LaTeX + Markdown tables
  table1_sweep.tex
  table2_causal.tex
  table3_cross_model.tex
  table4_phase.tex
```

### Hardware requirements

- **Full run (3 models):** ~48 GB VRAM (A100 / A6000), or `device_map=auto` across multiple GPUs
- **Primary only:** ~24 GB VRAM for Qwen3-8B
- **CPU fallback:** Works but slow -- edit `DEVICE_MAP="cpu"` in `run_paper.sh`

---

## Data preparation

```bash
# Download all safety datasets + calibration data
safepress prepare-data --data_dir data/ --sources advbench harmbench strongreject --calib_source c4

# Or use the standalone script
python scripts/prepare_data.py --data_dir data/
```

---

## Quickstart: full pipeline (single model)

```bash
safepress pipeline \
  --model_id Qwen/Qwen3-8B \
  --calib_prompts data/advbench.jsonl \
  --eval_prompts  data/harmbench.jsonl \
  --out_dir runs/qwen3_8b_ssmp \
  --quant_backend bnb4 \
  --bnb_quant_type nf4 \
  --budget 0.02 \
  --block_size 64 \
  --max_new_tokens 256
```

Artifacts (saved under `--out_dir`):
- `scores.csv` -- per-module block scores
- `protect_map.json` -- selected protected blocks
- `model_quantized/` -- quantized HF model folder
- `eval.json` -- evaluation results + refusal-rate metrics

---

## CLI reference

### Core pipeline

```bash
# 1) Score blocks
safepress score \
  --model_id <hf_model_or_path> \
  --calib_prompts <jsonl> \
  --out scores.csv \
  --block_size 64 --group_size 128 --bits 4

# 2) Build mixed-precision + quantize
safepress build \
  --model_id <hf_model_or_path> \
  --scores scores.csv \
  --out_dir model_quantized \
  --budget 0.02 --quant_backend bnb4

# 3) Evaluate
safepress eval \
  --model_path model_quantized \
  --eval_prompts <jsonl> \
  --out eval.json \
  --llamaguard             # optional: Llama Guard 3 safety classifier

# 4) Full pipeline (score -> build -> eval)
safepress pipeline \
  --model_id <hf_model_or_path> \
  --calib_prompts <jsonl> --eval_prompts <jsonl> \
  --out_dir runs/my_run --budget 0.02
```

### Paper-style sweep

```bash
# Run all methods x all budgets from a YAML config
safepress sweep --config configs/paper_sweep.yaml
```

The sweep config (`configs/paper_sweep.yaml`) specifies:
- 9 methods: fp16, int4, ssmp, random, magnitude, snip, qresafe_noft, lastn, cwp
- 5 budgets: 0.5%, 1%, 2%, 4%, 8%
- Outputs: per-run directories + `sweep_summary.csv`

### Experiments

```bash
# Causal ablation (targeted quant, rollback, control)
safepress experiment causal \
  --model_id Qwen/Qwen3-8B \
  --scores scores.csv \
  --eval_prompts data/harmbench.jsonl \
  --out_dir runs/causal --budget 0.02

# Budget sweep
safepress experiment sweep \
  --model_id Qwen/Qwen3-8B \
  --budgets 0.005 0.01 0.02 0.04 0.08 \
  --out_dir runs/sweep

# Phase-transition curve (bit-width vs safety)
safepress experiment phase \
  --model_id Qwen/Qwen3-8B \
  --eval_prompts data/harmbench.jsonl \
  --bit_widths 8 4 3 2 \
  --out_dir runs/phase
```

### Analysis

```bash
# Refusal direction profiling (Arditi et al.)
safepress analyze refusal-direction \
  --model_id Qwen/Qwen3-8B \
  --harmful_prompts data/harmbench.jsonl \
  --harmless_prompts data/calibration_c4.jsonl \
  --out_dir runs/analysis

# Per-layer quantization error
safepress analyze layer-error \
  --model_id Qwen/Qwen3-8B \
  --out_dir runs/analysis

# Cauchy-Schwarz drift bounds (theory section)
safepress analyze bounds \
  --model_id Qwen/Qwen3-8B \
  --calib_prompts data/advbench.jsonl \
  --out runs/analysis/drift_bounds.csv \
  --bits 4 --group_size 128
```

### Visualization

```bash
# Generate all figures at once
python scripts/generate_figures.py --results_dir runs/ --output_dir figures/

# Generate all tables at once
python scripts/generate_tables.py --results_dir runs/ --output_dir tables/

# Or generate individual figures via CLI
safepress viz heatmap --scores scores.csv --out figures/heatmap.pdf
safepress viz phase-transition --results runs/phase/results.json --out figures/phase.pdf
safepress viz causal --results runs/causal/targeted_results.json --out figures/causal.pdf
```

### AgentDojo integration (optional)

Serve your model with vLLM, then run AgentDojo:

```bash
# Start vLLM server
bash scripts/serve_vllm.sh runs/build/model_quantized 8000

# Run AgentDojo benchmark
OPENAI_BASE_URL=http://localhost:8000/v1 OPENAI_API_KEY=EMPTY \
  bash scripts/run_agentdojo.sh <model-id> tool_filter tool_knowledge
```

---

## Method registry

SafePress provides 12 methods via a unified dispatcher (`safepress/methods/registry.py`):

| Method | Key | Description |
|--------|-----|-------------|
| **SSMP (ours)** | `ssmp` | Taylor drift proxy on refusal supervision |
| FP16 baseline | `fp16` | No quantization (upper bound) |
| INT4 baseline | `int4` | Full 4-bit quantization (lower bound) |
| Random | `random` | Uniform random block selection |
| Magnitude | `magnitude` | Weight magnitude heuristic |
| Gradient-only (Fisher) | `gradient_only` | Diagonal Fisher information |
| SNIP | `snip` | Connection sensitivity `\|w * g\|` |
| Wanda | `wanda` | Activation-aware `\|w\| * \|\|X\|\|_2` |
| Last-N layers | `lastn` | Protect last N transformer layers |
| Layer-uniform | `layer_uniform` | Protect first + last layers |
| Q-resafe (no FT) | `qresafe_noft` | Q-resafe without finetuning |
| CWP-style | `cwp` | `I_safe - beta * I_utility` (critical weight protection) |

All methods are dispatched through `build_protect_plan(method=..., ...)` and return a `ProtectPlan`.

---

## Scoring metrics

The unified `compute_block_scores()` function supports multiple metrics and supervision modes:

**Metrics:** `taylor_abs` (default), `grad_abs`, `grad_sq`, `snip`

**Supervision modes:** `refusal` (safety-loss on refusal template), `lm` (causal LM loss on utility text)

```python
from safepress.model.score import compute_block_scores

df = compute_block_scores(
    model, tokenizer, prompts,
    metric="taylor_abs",     # or grad_abs, grad_sq, snip
    prompt_mode="refusal",   # or lm
    block_size=64, bits=4,
)
```

---

## Quantization backends

| Backend | Method | Module |
|---------|--------|--------|
| **bnb4** | bitsandbytes NF4/FP4 | `model/quantize.py` |
| **awq** | AutoAWQ 4-bit | `model/quantize_awq.py` |
| **gptq** | GPTQ via transformers | `model/quantize_gptq.py` |
| naive | Uniform 4-bit / 8-bit (no calibration) | `model/quantize_naive.py` |

---

## Evaluation benchmarks

**Safety:**
- Heuristic refusal rate (prefix matching)
- HarmBench classifier ASR (`cais/HarmBench-Llama-2-13b-cls`)
- StrongREJECT continuous scoring (0-1 harmfulness)
- Llama Guard 3 (`meta-llama/Llama-Guard-3-1B`) — LLM-based safe/unsafe classifier with category codes
- AgentDojo tool-calling prompt injection

**Utility:**
- Perplexity (WikiText-2, sliding window)
- MMLU-lite (200 questions, next-token log-prob)
- TruthfulQA-lite (MC1, 100 questions)

Comprehensive evaluation:
```python
from safepress.eval.comprehensive import ComprehensiveEvalConfig, run_comprehensive_eval

config = ComprehensiveEvalConfig(
    eval_prompts_path="data/harmbench.jsonl",
    use_harmbench_cls=True,
    use_strongreject=True,
    use_perplexity=True,
    use_mmlu=True,
    use_truthfulqa=True,
)
results = run_comprehensive_eval(model, tokenizer, config)
# results["safety"]["refusal"]["refusal_rate"]
# results["safety"]["harmbench"]["asr"]
# results["utility"]["perplexity"]["perplexity"]
# results["utility"]["mmlu"]["accuracy"]
```

---

## Project structure

```
safepress/
  cli.py                    # CLI entry point (score, build, eval, pipeline, sweep, experiment, analyze, viz, prepare-data)
  methods/
    registry.py             # Unified method dispatcher (12 methods via MethodSpec + build_protect_plan)
    __init__.py
  model/
    load.py                 # Model loading
    score.py                # Block scoring (4 metrics x 2 supervision modes)
    blocks.py               # Block utilities (chunk_indices, iter_linear_modules, ...)
    baselines.py            # Baseline scoring methods
    protect.py              # Top-K block selection under budget
    split_linear.py         # SplitLinear mixed-precision module
    quantize.py             # bitsandbytes 4-bit quantization
    quantize_awq.py         # AWQ quantization
    quantize_gptq.py        # GPTQ quantization
    quantize_naive.py       # Naive uniform quantization
  eval/
    basic.py                # Refusal heuristic + StrongREJECT
    harmbench.py            # HarmBench classifier integration
    llamaguard.py           # Llama Guard 3 safety classifier (safe/unsafe + category codes)
    utility.py              # Perplexity, MMLU, TruthfulQA
    comprehensive.py        # Unified evaluation runner
    agentdojo_runner.py     # AgentDojo subprocess wrapper
  analysis/
    refusal_direction.py    # Refusal direction extraction (Arditi et al.)
    layer_analysis.py       # Quantization error analysis + correlation
    drift_bound.py          # Cauchy-Schwarz drift bounds (theory section)
  experiments/
    causal.py               # Targeted quant, rollback, control experiments
    sweep.py                # Budget, cross-model, block-size sweeps
    phase_transition.py     # Bit-width vs safety phase transition
  data/
    download.py             # Dataset downloaders (AdvBench, HarmBench, StrongREJECT, C4)
    prepare.py              # JSONL conversion
    prompts.py              # PromptRecord dataclass + StrongREJECT/HarmBench loaders
  viz/
    plots.py                # 7 publication-quality plot functions
    tables.py               # LaTeX + Markdown table generators
  utils/
    io.py                   # JSONL/JSON/YAML I/O
    logging.py              # Run directory + save_json
    seed.py                 # RNG seeding

configs/
  models.yaml               # Model registry (Qwen3-8B, Llama-3.1-8B, Gemma-3-4B, Qwen2.5-3B)
  ssmp_bnb4.yaml            # Default SSMP config
  paper_sweep.yaml          # Paper-style method x budget sweep config
  experiment_causal.yaml    # Causal experiment config
  experiment_sweep.yaml     # Budget sweep config (3 models)
  experiment_phase.yaml     # Phase transition config

scripts/
  run_paper.sh              # Master script: runs all 10 steps for paper reproduction
  prepare_data.py           # Dataset preparation
  run_all_experiments.py    # Experiment runner (causal + sweep + phase)
  generate_figures.py       # Paper figure generator (heatmap, phase, causal, sweep)
  generate_tables.py        # Paper table generator (sweep, causal, cross-model, phase)
  prepare_utility_prompts.py # Alpaca utility prompts for CWP baseline
  run_from_yaml.py          # YAML-based pipeline runner
  run_agentdojo.sh          # AgentDojo launcher
  serve_vllm.sh             # vLLM server launcher
```

---

## Repro tips for a paper

- Run `bash scripts/run_paper.sh` for end-to-end reproduction.
- Always include **(FP16 baseline, full 4-bit, SSMP 4-bit)** for each model.
- Sweep budgets (0.5%, 1%, 2%, 4%, 8%) and show Pareto curves (safety vs overhead).
- Run all three causal experiments (targeted, rollback, control) to prove mechanism.
- Compare against baselines: random, magnitude, SNIP, Wanda, CWP-style, layer-uniform.
- Test across model families: Qwen3-8B (primary), Llama-3.1-8B (secondary), Gemma-3-4B (cross-family).
- Report both **safety** (refusal rate, HarmBench ASR, StrongREJECT score) and **utility** (perplexity, MMLU, TruthfulQA).
- Use `--primary-only` flag for faster iteration during development.
- Use `--step N` to resume from a specific step after partial runs.

---

## License

MIT
