from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import torch

from safepress.model.load import load_fp_model
from safepress.model.protect import load_protect_plan, save_protect_plan, select_top_blocks
from safepress.model.score import compute_block_scores
from safepress.model.split_linear import apply_block_splitting
from safepress.model.quantize import quantize_bnb4, quantize_for_bits, save_hf_model
from safepress.utils.io import read_prompts_jsonl, read_yaml
from safepress.utils.logging import init_run_dir, save_json
from safepress.eval.basic import (
    GenConfig,
    generate_completions,
    is_refusal,
    refusal_rate,
    try_strongreject_eval,
)


# ---------------------------------------------------------------------------
# New command handlers (lazy imports to avoid startup overhead)
# ---------------------------------------------------------------------------

def cmd_prepare_data(args: argparse.Namespace) -> None:
    """Download and prepare safety / calibration datasets."""
    from safepress.data.prepare import prepare_all

    prepare_all(
        data_dir=args.data_dir,
        sources=args.sources,
        calib_source=args.calib_source,
        n_calib=args.n_calib,
        cache_dir=getattr(args, "cache_dir", None),
        harmbench_attacks=getattr(args, "harmbench_attacks", None),
        harmbench_attack_target=getattr(args, "harmbench_attack_target", "llama2_7b_chat"),
    )


def cmd_experiment_causal(args: argparse.Namespace) -> None:
    """Run causal experiment: measure per-layer safety drift."""
    from safepress.experiments.causal import targeted_quantize_experiment

    targeted_quantize_experiment(
        model_id=args.model_id,
        scores_csv=args.scores,
        eval_prompts=read_prompts_jsonl(args.eval_prompts, key=getattr(args, "prompt_key", "prompt")),
        out_dir=args.out_dir,
        dtype=args.dtype,
        device_map=args.device_map,
        bits=getattr(args, "bits", 4),
        group_size=getattr(args, "group_size", 128),
        block_size=getattr(args, "block_size", 64),
        budget=getattr(args, "budget", 0.02),
    )


def cmd_experiment_sweep(args: argparse.Namespace) -> None:
    """Run budget-sweep experiment across multiple budgets."""
    from safepress.experiments.sweep import budget_sweep

    budget_sweep(
        model_id=args.model_id,
        scores_csv=args.scores,
        eval_prompts=read_prompts_jsonl(args.eval_prompts, key=getattr(args, "prompt_key", "prompt")),
        budgets=[float(b) for b in args.budgets],
        out_dir=args.out_dir,
        dtype=args.dtype,
        device_map=args.device_map,
        block_size=getattr(args, "block_size", 64),
    )


def cmd_experiment_phase(args: argparse.Namespace) -> None:
    """Run phase-transition experiment across bit-widths.

    Fractional values (e.g. 3.5, 2.5) are supported and trigger per-layer
    mixed-precision quantization that interpolates between adjacent integer
    bit-widths.
    """
    from safepress.experiments.phase_transition import phase_transition_curve

    phase_transition_curve(
        model_id=args.model_id,
        eval_prompts=read_prompts_jsonl(args.eval_prompts, key=getattr(args, "prompt_key", "prompt")),
        out_dir=args.out_dir,
        dtype=args.dtype,
        device_map=args.device_map,
        bit_widths=[float(b) for b in getattr(args, "bit_widths", [8, 5, 4, 3.5, 3, 2.5, 2])],
        group_size=getattr(args, "group_size", 128),
    )


def cmd_analyze_refusal_direction(args: argparse.Namespace) -> None:
    """Compute refusal direction from contrastive prompts."""
    from safepress.analysis.refusal_direction import refusal_signal_profile

    prompt_key = getattr(args, "prompt_key", "prompt")
    harmful = read_prompts_jsonl(args.harmful_prompts, key=prompt_key)
    harmless = read_prompts_jsonl(args.harmless_prompts, key=prompt_key)

    loaded = load_fp_model(
        args.model_id,
        dtype=getattr(args, "dtype", "float16"),
        device_map=getattr(args, "device_map", "auto"),
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = refusal_signal_profile(
        loaded.model, loaded.tokenizer,
        harmful, harmless,
        device=loaded.device,
    )
    df.to_csv(out_dir / "refusal_direction.csv", index=False)
    print(f"[analyze] Refusal direction profile -> {out_dir / 'refusal_direction.csv'}")


def cmd_analyze_layer_error(args: argparse.Namespace) -> None:
    """Analyze per-layer quantization error."""
    from safepress.analysis.layer_analysis import compute_layer_quant_error

    loaded = load_fp_model(
        args.model_id,
        dtype=getattr(args, "dtype", "float16"),
        device_map=getattr(args, "device_map", "auto"),
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = compute_layer_quant_error(
        loaded.model,
        bits=getattr(args, "bits", 4),
        group_size=getattr(args, "group_size", 128),
    )
    df.to_csv(out_dir / "layer_quant_error.csv", index=False)
    print(f"[analyze] Layer quant error -> {out_dir / 'layer_quant_error.csv'}")


def cmd_bounds(args: argparse.Namespace) -> None:
    """Compute per-module Cauchy-Schwarz drift bounds."""
    from safepress.analysis.drift_bound import compute_module_level_bounds

    loaded = load_fp_model(
        args.model_id,
        dtype=args.dtype,
        device_map=args.device_map,
        trust_remote_code=getattr(args, "trust_remote_code", False),
    )

    prompts = read_prompts_jsonl(args.calib_prompts, key=args.prompt_key)
    if args.max_prompts is not None:
        prompts = prompts[: args.max_prompts]

    df = compute_module_level_bounds(
        loaded.model,
        loaded.tokenizer,
        prompts,
        refusal_template=args.refusal_template,
        bits=args.bits,
        group_size=args.group_size,
        max_length=args.max_length,
        device=loaded.device,
    )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"[bounds] wrote {len(df)} rows -> {args.out}")


def cmd_drift_validate(args: argparse.Namespace) -> None:
    """G1 gate: empirically validate the first-order drift bound.

    For each bit-width in ``--bit_widths``, perturb every Linear weight via
    symmetric group-wise quantize-dequantize, compute four predicted
    quantities (signed inner, abs-block sum, abs-module sum, CS-module bound)
    and the measured |ΔL_safe|. Fits two regressions:

    * **signed Taylor**: predicted vs measured signed drift (slope→1 ideal)
    * **upper-bound (THEOREM)**: predicted ``Σ_b |g_b·δw_b|`` vs measured
      ``|ΔL|`` (R² high = magnitude-tight bound; R² low + high Spearman ρ =
      ranking-tight surrogate, recovery path per PLAN §16)

    The G1 gate uses the upper-bound R² because that is the quantity the
    theorem actually bounds.
    """
    from safepress.analysis.drift_bound import (
        fit_drift_scatter_summary,
        validate_drift_bound,
    )

    loaded = load_fp_model(
        args.model_id,
        dtype=args.dtype,
        device_map=args.device_map,
        trust_remote_code=getattr(args, "trust_remote_code", False),
    )

    prompts = read_prompts_jsonl(args.calib_prompts, key=args.prompt_key)
    if args.max_prompts is not None:
        prompts = prompts[: args.max_prompts]

    df = validate_drift_bound(
        loaded.model,
        loaded.tokenizer,
        prompts,
        bit_widths=[int(b) for b in args.bit_widths],
        group_size=args.group_size,
        block_size=getattr(args, "block_size", 64),
        refusal_template=args.refusal_template,
        max_length=args.max_length,
        device=loaded.device,
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    summary = fit_drift_scatter_summary(df)
    save_json(Path(args.out).with_suffix(".fit.json"), summary)

    ub = summary.get("upper_bound", {})
    signed = summary.get("signed_taylor", {})
    spearman = summary.get("upper_bound_spearman_rho", float("nan"))
    r2_ub = float(ub.get("r_squared", float("nan")))
    print(
        f"[drift-validate] wrote {len(df)} rows -> {args.out}"
    )
    print(
        f"[drift-validate] signed Taylor fit (predicted_inner_signed vs measured_dL_signed): "
        f"slope={signed.get('slope', float('nan')):.3f} "
        f"intercept={signed.get('intercept', float('nan')):.3f}  "
        f"R^2={float(signed.get('r_squared', float('nan'))):.3f}"
    )
    print(
        f"[drift-validate] THEOREM upper-bound fit (predicted_abs_block vs |measured_dL|): "
        f"slope={ub.get('slope', float('nan')):.3f} "
        f"intercept={ub.get('intercept', float('nan')):.3f}  "
        f"R^2={r2_ub:.3f}  Spearman_rho={spearman:.3f}"
    )
    print(
        f"[drift-validate] G1 gate (upper-bound R^2 >= 0.85): "
        f"{'PASS' if r2_ub >= 0.85 else 'FAIL'}  "
        f"(if FAIL but Spearman_rho >= 0.85, reframe as ranking-tight per PLAN §16)"
    )


def cmd_viz_heatmap(args: argparse.Namespace) -> None:
    """Generate score heatmap figure from a scores CSV.

    The plotting function takes a DataFrame; we load the CSV here so the CLI
    surface stays "give me a path" while the function stays pure.
    """
    import pandas as pd
    from safepress.viz.plots import plot_block_heatmap

    df = pd.read_csv(args.scores)
    plot_block_heatmap(df, save_path=args.out)


def cmd_viz_phase_transition(args: argparse.Namespace) -> None:
    """Generate phase-transition figure from a results JSON.

    The plotting function takes ``(bit_widths, safety_scores, utility_scores)``;
    we load the JSON written by ``experiment phase`` and extract those three
    sequences.
    """
    import json as _json
    from safepress.viz.plots import plot_phase_transition

    with open(args.results, "r") as f:
        data = _json.load(f)
    bw = data.get("bit_widths") or []
    results = data.get("results") or {}
    # results is keyed by "bits_<b>"; pull refusal_rate (safety proxy) and a
    # utility proxy (avg_response_words / max). MMLU is preferred when present.
    safety_scores: List[float] = []
    utility_scores: List[float] = []
    for b in bw:
        # Match the bit_label format used by phase_transition_curve
        key = f"bits_{int(b)}" if float(b).is_integer() else f"bits_{b}"
        cell = results.get(key, {})
        safety_scores.append(float(cell.get("refusal_rate", 0.0)))
        # Prefer MMLU if present, else fall back to normalised avg_response_words
        if "mmlu" in cell and isinstance(cell["mmlu"], dict):
            utility_scores.append(float(cell["mmlu"].get("accuracy", 0.0)))
        else:
            words = float(cell.get("avg_response_words", 0.0))
            utility_scores.append(min(1.0, words / 200.0))
    plot_phase_transition(bw, safety_scores, utility_scores, save_path=args.out)


def cmd_viz_causal(args: argparse.Namespace) -> None:
    """Generate causal-experiment bar chart from a results JSON."""
    import json as _json
    from safepress.viz.plots import plot_causal_experiment

    with open(args.results, "r") as f:
        data = _json.load(f)
    # Accept either ``{condition_name: value}`` or
    # ``{"results": {condition_name: value}}`` shapes.
    results_dict = data.get("results", data)
    plot_causal_experiment(results_dict, save_path=args.out)


def cmd_score(args: argparse.Namespace) -> None:
    prompts = read_prompts_jsonl(args.calib_prompts, key=args.prompt_key)
    if args.max_prompts is not None:
        prompts = prompts[: args.max_prompts]

    loaded = load_fp_model(
        args.model_id,
        dtype=args.dtype,
        device_map=args.device_map,
        trust_remote_code=args.trust_remote_code,
    )

    df = compute_block_scores(
        loaded.model,
        loaded.tokenizer,
        prompts,
        refusal_template=args.refusal_template,
        bits=args.bits,
        group_size=args.group_size,
        block_size=args.block_size,
        max_length=args.max_length,
        batch_size=args.batch_size,
        device=loaded.device,
        show_progress=True,
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"[score] wrote {len(df)} rows -> {args.out}")


def cmd_build(args: argparse.Namespace) -> None:
    out_dir = init_run_dir(args.out_dir, allow_overwrite=args.overwrite)

    scores = pd.read_csv(args.scores)
    plan = select_top_blocks(scores, budget_ratio=args.budget, block_size=args.block_size)
    save_protect_plan(plan, out_dir / "protect_map.json")

    loaded = load_fp_model(
        args.model_id,
        dtype=args.dtype,
        device_map=args.device_map,
        trust_remote_code=args.trust_remote_code,
    )

    split_report = apply_block_splitting(loaded.model, plan.protect_map, block_size=args.block_size)
    save_json(out_dir / "split_report.json", split_report.__dict__)

    if args.quant_backend == "bnb4":
        qrep = quantize_bnb4(
            loaded.model,
            modules_to_not_convert=split_report.protected_modules_to_skip,
            quant_type=args.bnb_quant_type,
            compute_dtype=args.bnb_compute_dtype,
            use_double_quant=not args.disable_double_quant,
            quant_storage=args.bnb_quant_storage,
        )
        save_json(out_dir / "quantize_report.json", qrep.__dict__)
        model_out = out_dir / "model_quantized"
        save_hf_model(loaded.model, loaded.tokenizer, model_out)
        print(f"[build] wrote quantized model -> {model_out}")
    else:
        raise ValueError(f"Unsupported quant_backend: {args.quant_backend}")


def _load_eval_model(model_path: str, *, dtype: str, device_map: str, trust_remote_code: bool):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=trust_remote_code)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map=device_map,
        torch_dtype={"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[dtype],
        trust_remote_code=trust_remote_code,
    )
    model.eval()
    return model, tok


def _run_single_seed_eval(
    model, tok, prompts: List[str], *,
    seed: int, gen_cfg: GenConfig, max_length: int,
    deterministic: bool,
    try_strongreject: bool, strongreject_evaluator: str,
    llamaguard: bool, harmbench_classifier: bool, harmbench_classifier_id: str,
) -> dict:
    """Run one evaluation pass with a fixed seed and return per-prompt + summary data.
    """
    from safepress.utils.seed import set_seed

    set_seed(seed, deterministic=deterministic)

    responses = generate_completions(
        model, tok, prompts,
        gen=gen_cfg, max_length=max_length, show_progress=True,
    )
    refusals = [int(is_refusal(r)) for r in responses]
    rr = sum(refusals) / max(1, len(refusals))

    result: dict = {
        "seed": seed,
        "refusal_rate": rr,
        "per_prompt_refusal": refusals,
        "responses": responses,
        "n": len(prompts),
    }
    if try_strongreject:
        result["strongreject"] = try_strongreject_eval(
            prompts, responses, evaluator=strongreject_evaluator,
        )
    if llamaguard:
        try:
            from safepress.eval.llamaguard import try_llamaguard_eval
            result["llamaguard"] = try_llamaguard_eval(prompts, responses)
        except Exception as exc:
            result["llamaguard"] = {"ok": False, "error": str(exc)}
    if harmbench_classifier:
        try:
            from safepress.eval.harmbench import harmbench_attack_success_rate
            hb = harmbench_attack_success_rate(
                prompts, responses,
                classifier_id=harmbench_classifier_id,
                show_progress=True,
            )
            result["harmbench"] = {
                "asr": hb["asr"],
                "n_harmful": hb["n_harmful"],
                "n_total": hb["n_total"],
                "per_prompt_harmful": [int(x) for x in hb["per_behavior"]],
            }
        except Exception as exc:
            result["harmbench"] = {"ok": False, "error": str(exc)}
    return result


def cmd_eval(args: argparse.Namespace) -> None:
    model, tok = _load_eval_model(
        args.model_path,
        dtype=args.dtype,
        device_map=args.device_map,
        trust_remote_code=args.trust_remote_code,
    )

    prompts = read_prompts_jsonl(args.eval_prompts, key=args.prompt_key)
    if getattr(args, "n", None) is not None:
        prompts = prompts[: int(args.n)]
    elif args.max_prompts is not None:
        prompts = prompts[: args.max_prompts]

    gen_cfg = GenConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=args.do_sample,
    )

    seeds = getattr(args, "seeds", None) or [0]
    deterministic = bool(getattr(args, "deterministic", False))

    per_seed: List[dict] = []
    for seed in seeds:
        per_seed.append(
            _run_single_seed_eval(
                model, tok, prompts,
                seed=int(seed), gen_cfg=gen_cfg, max_length=args.max_length,
                deterministic=deterministic,
                try_strongreject=bool(getattr(args, "try_strongreject", False)),
                strongreject_evaluator=args.strongreject_evaluator,
                llamaguard=bool(getattr(args, "llamaguard", False)),
                harmbench_classifier=bool(getattr(args, "harmbench_classifier", False)),
                harmbench_classifier_id=getattr(
                    args, "harmbench_classifier_id", "cais/HarmBench-Llama-2-13b-cls",
                ),
            )
        )

    # Aggregate via stats module
    from safepress.eval.stats import (
        aggregate_across_seeds,
        bootstrap_rate_ci,
    )

    refusal_means = [r["refusal_rate"] for r in per_seed]
    seed_agg = aggregate_across_seeds(refusal_means)
    # Per-seed bootstrap CI on the rate -- useful when a single seed was run.
    rate_cis = []
    for r in per_seed:
        ci = bootstrap_rate_ci(r["per_prompt_refusal"], n_bootstrap=10000, seed=int(r["seed"]))
        rate_cis.append({"seed": r["seed"], **ci.to_dict()})

    summary: dict = {
        "model_path": args.model_path,
        "n": len(prompts),
        "n_seeds": len(seeds),
        "seeds": [int(s) for s in seeds],
        "gen": gen_cfg.__dict__,
        "refusal_rate_seed_mean": seed_agg.mean,
        "refusal_rate_seed_std": seed_agg.std,
        "refusal_rate_seed_ci": [seed_agg.ci_low, seed_agg.ci_high],
        "refusal_rate_bootstrap_ci_per_seed": rate_cis,
        "per_seed": per_seed,
    }
    # Back-compat: keep the legacy single-seed top-level "refusal_rate" field.
    summary["refusal_rate"] = seed_agg.mean

    # If StrongREJECT / LlamaGuard / HarmBench were enabled, surface seed means.
    for key in ("strongreject", "llamaguard", "harmbench"):
        per_seed_scores = []
        for r in per_seed:
            if key in r and isinstance(r[key], dict):
                v = r[key]
                if key == "strongreject" and "avg_score" in v:
                    per_seed_scores.append(float(v["avg_score"]))
                if key == "harmbench" and "asr" in v:
                    per_seed_scores.append(float(v["asr"]))
                if key == "llamaguard" and "unsafe_rate" in v:
                    per_seed_scores.append(float(v["unsafe_rate"]))
        if per_seed_scores:
            agg = aggregate_across_seeds(per_seed_scores)
            summary[f"{key}_seed_mean"] = agg.mean
            summary[f"{key}_seed_ci"] = [agg.ci_low, agg.ci_high]

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    save_json(args.out, summary)
    print(
        f"[eval] wrote -> {args.out}  "
        f"refusal={seed_agg.mean:.4f} [{seed_agg.ci_low:.4f}, {seed_agg.ci_high:.4f}]  "
        f"seeds={len(seeds)} n={len(prompts)}"
    )


def cmd_jailbreak(args: argparse.Namespace) -> None:
    """Run an adversarial-attack evaluation (HarmBench GCG/AutoDAN/PAIR)."""
    from safepress.eval.jailbreaks import JailbreakConfig, run_jailbreak_eval

    model, tok = _load_eval_model(
        args.model_path,
        dtype=args.dtype,
        device_map=args.device_map,
        trust_remote_code=args.trust_remote_code,
    )

    cfg = JailbreakConfig(
        attack_prompts_path=args.attack_prompts,
        behaviors_path=args.behaviors,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=args.do_sample,
        max_length=args.max_length,
        max_prompts=getattr(args, "n", None) or args.max_prompts,
        use_harmbench_classifier=bool(args.harmbench_classifier),
        harmbench_classifier_id=args.harmbench_classifier_id,
        harmbench_batch_size=args.harmbench_batch_size,
        save_generations_to=getattr(args, "save_generations_to", None),
    )

    from safepress.utils.seed import set_seed

    set_seed(int(args.seed), deterministic=bool(args.deterministic))
    result = run_jailbreak_eval(model, tok, cfg)

    # Bootstrap CI on the harmful flag vector.
    from safepress.eval.stats import bootstrap_rate_ci

    ci = bootstrap_rate_ci(result["per_prompt_harmful"], n_bootstrap=10000, seed=int(args.seed))
    result["asr_bootstrap_ci"] = ci.to_dict()
    # Avoid dumping the (potentially large) records to the summary JSON unless
    # explicitly requested.
    if not args.include_records:
        result.pop("records", None)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    save_json(args.out, result)
    print(
        f"[jailbreak:{result.get('attack_name','')}] wrote -> {args.out}  "
        f"ASR={result['asr']:.4f} [{ci.ci_low:.4f}, {ci.ci_high:.4f}]  "
        f"classifier={result.get('classifier')}  n={result['n']}"
    )


def cmd_xstest(args: argparse.Namespace) -> None:
    """Run XSTest over-refusal evaluation."""
    from safepress.eval.xstest import XSTestConfig, run_xstest

    model, tok = _load_eval_model(
        args.model_path,
        dtype=args.dtype,
        device_map=args.device_map,
        trust_remote_code=args.trust_remote_code,
    )
    cfg = XSTestConfig(
        prompts_path=args.xstest_prompts,
        label_key=args.label_key,
        prompt_key=args.prompt_key,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=args.do_sample,
        max_length=args.max_length,
        max_prompts_per_label=getattr(args, "n", None) or args.max_prompts,
    )

    from safepress.utils.seed import set_seed

    set_seed(int(args.seed), deterministic=bool(args.deterministic))
    result = run_xstest(model, tok, cfg)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    save_json(args.out, result)
    print(
        f"[xstest] wrote -> {args.out}  "
        f"safe_compliance={result['safe_compliance_rate']:.4f}  "
        f"unsafe_refusal={result['unsafe_refusal_rate']:.4f}  "
        f"score={result['safety_score']:.4f}"
    )


def cmd_pipeline(args: argparse.Namespace) -> None:
    out_dir = init_run_dir(args.out_dir, allow_overwrite=args.overwrite)

    # 1) score
    score_path = out_dir / "scores.csv"
    tmp = argparse.Namespace(**vars(args))
    tmp.out = str(score_path)
    cmd_score(tmp)

    # 2) build
    tmp2 = argparse.Namespace(**vars(args))
    tmp2.scores = str(score_path)
    tmp2.out_dir = str(out_dir)
    cmd_build(tmp2)

    # 3) eval
    tmp3 = argparse.Namespace(**vars(args))
    tmp3.model_path = str(out_dir / "model_quantized")
    tmp3.out = str(out_dir / "eval.json")
    cmd_eval(tmp3)


def cmd_sweep(args: argparse.Namespace) -> None:
    """Run a paper-style sweep from a YAML config.

    The YAML file should have keys:
      model_id, safety_prompts, eval_prompts, out_root,
      methods: [ssmp, random, int4, ...]
      budgets: [0.005, 0.01, 0.02]
    Plus optional: utility_prompts, quant_backend, bits, group_size, block_size, etc.
    """
    from safepress.methods.registry import build_protect_plan, list_methods

    cfg = read_yaml(args.config)
    out_root = Path(cfg.get("out_root", "runs"))
    out_root.mkdir(parents=True, exist_ok=True)

    import gc

    methods = cfg.get("methods") or ["int4", "ssmp"]
    budgets = cfg.get("budgets") or [0.01]
    model_id = cfg["model_id"]
    model_load_kwargs = dict(
        dtype=cfg.get("dtype", "float16"),
        device_map=cfg.get("device_map", "auto"),
        trust_remote_code=bool(cfg.get("trust_remote_code", False)),
    )

    # Load prompts
    prompt_key = cfg.get("prompt_key", "prompt")
    max_prompts = cfg.get("max_prompts")

    safety_prompts = None
    if cfg.get("safety_prompts"):
        safety_prompts = read_prompts_jsonl(cfg["safety_prompts"], key=prompt_key)
        if max_prompts:
            safety_prompts = safety_prompts[:max_prompts]

    utility_prompts = None
    if cfg.get("utility_prompts"):
        utility_prompts = read_prompts_jsonl(cfg["utility_prompts"], key=prompt_key)
        if max_prompts:
            utility_prompts = utility_prompts[:max_prompts]

    eval_prompts_list = None
    if cfg.get("eval_prompts"):
        eval_prompts_list = read_prompts_jsonl(cfg["eval_prompts"], key=prompt_key)
        if max_prompts:
            eval_prompts_list = eval_prompts_list[:max_prompts]

    # Common params
    refusal_template = cfg.get("refusal_template", "I'm sorry, but I can't help with that.")
    bits = int(cfg.get("bits", 4))
    group_size = int(cfg.get("group_size", 128))
    block_size = int(cfg.get("block_size", 64))
    max_length = int(cfg.get("max_length", 2048))
    batch_size = int(cfg.get("batch_size", 1))
    seed = int(cfg.get("seed", 0))
    last_n_layers = int(cfg.get("last_n_layers", 4))
    cwp_beta = float(cfg.get("cwp_beta", 1.0))
    model_tag = cfg.get("model_tag", "model")

    # Gen config for eval
    gen_cfg = GenConfig(
        max_new_tokens=int(cfg.get("max_new_tokens", 256)),
        temperature=float(cfg.get("temperature", 0.0)),
        top_p=float(cfg.get("top_p", 1.0)),
        do_sample=bool(cfg.get("do_sample", False)),
    )

    # Multi-seed support: ``seeds`` overrides the legacy single-``seed`` field.
    seeds: List[int] = [int(s) for s in cfg.get("seeds", [seed])]
    if not seeds:
        seeds = [0]

    use_harmbench_cls = bool(cfg.get("use_harmbench_classifier", False))
    harmbench_cls_id = str(cfg.get("harmbench_classifier_id", "cais/HarmBench-Llama-2-13b-cls"))
    harmbench_cls_batch = int(cfg.get("harmbench_classifier_batch_size", 4))
    deterministic_decode = bool(cfg.get("deterministic", False))

    # Lazy: load the HarmBench classifier once across the entire sweep.
    classifier_model = None
    classifier_tokenizer = None

    def _maybe_load_classifier():
        nonlocal classifier_model, classifier_tokenizer
        if classifier_model is None and use_harmbench_cls:
            try:
                from safepress.eval.harmbench import load_harmbench_classifier
                classifier_model, classifier_tokenizer = load_harmbench_classifier(
                    model_id=harmbench_cls_id,
                )
            except Exception as exc:  # noqa: BLE001
                print(f"[sweep] HarmBench classifier unavailable: {exc}")
                classifier_model, classifier_tokenizer = None, None

    results = []
    for method in methods:
        method_l = str(method).lower()
        # Methods that ignore the YAML ``budgets`` list:
        #   fp16, int4    -- no protection by construction
        #   qresafe       -- empty plan; LoRA-DPO patch handles protection
        #   cwp_published -- registry forces 0.60 internally per paper recipe;
        #                    iterating budgets would log identical results 5x
        #                    with misleading budget labels.
        if method_l in {"fp16", "int4", "qresafe"}:
            eff_budgets = [0.0]
        elif method_l == "cwp_published":
            eff_budgets = [0.60]  # honest label: this baseline IS at 60%
        else:
            eff_budgets = budgets
        for budget in eff_budgets:
            run_name = f"sweep_{model_tag}_{method}_b{budget}"
            run_dir = out_root / run_name
            run_dir.mkdir(parents=True, exist_ok=True)

            print(f"\n{'='*60}")
            print(f"[sweep] method={method}  budget={budget}  -> {run_dir}")
            print(f"{'='*60}")

            per_seed_records: List[Dict[str, object]] = []

            for sd in seeds:
                from safepress.utils.seed import set_seed

                set_seed(sd, deterministic=deterministic_decode)

                # Fresh model per iteration (no in-place mutation leakage).
                loaded = load_fp_model(model_id, **model_load_kwargs)

                # 1) Build protect plan on the clean FP16 model.
                plan = build_protect_plan(
                    method=method,
                    model=loaded.model,
                    tokenizer=loaded.tokenizer,
                    safety_prompts=safety_prompts,
                    utility_prompts=utility_prompts,
                    block_size=block_size,
                    budget_ratio=float(budget),
                    group_size=group_size,
                    bits=bits,
                    refusal_template=refusal_template,
                    max_length=max_length,
                    batch_size=batch_size,
                    seed=sd,
                    last_n_layers=last_n_layers,
                    cwp_beta=cwp_beta,
                    device=loaded.device,
                )
                save_protect_plan(plan, run_dir / f"protect_map_seed{sd}.json")

                # 2) Apply splitting + quantization.
                if method_l != "fp16":
                    skip_modules = []
                    if plan.protect_map:
                        split_report = apply_block_splitting(
                            loaded.model, plan.protect_map, block_size=block_size,
                        )
                        save_json(
                            run_dir / f"split_report_seed{sd}.json",
                            split_report.__dict__,
                        )
                        skip_modules = split_report.protected_modules_to_skip

                    # IMPORTANT: dispatch on the YAML's ``bits`` field, not a
                    # hardcoded bnb4 call. Previously every sweep -- including
                    # bits: 3 -- went through bnb4 (load_in_4bit=True), so
                    # ``configs/paper_emnlp_3bit.yaml`` silently produced 4-bit
                    # results. The G2 scoring ablation at 3-bit must call the
                    # simulated 3-bit path so all scorers see the same δw at
                    # the targeted bit-width.
                    qrep = quantize_for_bits(
                        loaded.model,
                        bits=bits,
                        group_size=group_size,
                        modules_to_not_convert=skip_modules,
                    )
                    print(f"[sweep] quantized ({qrep.backend}): {qrep.note}")

                # 2b) Optional Q-resafe DPO patch (only when method == 'qresafe').
                # Needs a frozen reference and the SNIP scores. Memory layout
                # (fits in 48 GB):
                #   * snip scoring on a separate FP16 copy then immediately
                #     freed (32 GB peak, then back to ~16 GB),
                #   * reference model moved to CPU before the DPO loop (so
                #     it doesn't compete for VRAM with the policy + grads),
                #   * policy is the quantized model + LoRA on cuda:0; LoRA
                #     gradients are tiny.
                # Previous attempt OOM'd because BOTH FP16 reference and
                # quantized policy stayed on the GPU through DPO.
                qresafe_failure: Optional[str] = None
                if method_l == "qresafe":
                    try:
                        from safepress.methods.qresafe import (
                            QResafeConfig,
                            qresafe_patch,
                        )
                        from safepress.model.score import compute_block_scores

                        # ---- score on a fresh FP16 copy then free it ----
                        ref_for_scoring = load_fp_model(model_id, **model_load_kwargs)
                        snip_scores = compute_block_scores(
                            ref_for_scoring.model, ref_for_scoring.tokenizer,
                            list(safety_prompts or []),
                            refusal_template=refusal_template,
                            metric="snip",
                            prompt_mode="refusal",
                            bits=bits, group_size=group_size, block_size=block_size,
                            max_length=max_length, batch_size=batch_size,
                            device=ref_for_scoring.device,
                        )
                        ref_for_scoring.model.zero_grad(set_to_none=True)
                        # Re-purpose this copy as the *frozen reference* but
                        # move it to CPU so the policy + LoRA grads have full
                        # VRAM. The reference is touched only twice per DPO
                        # step (for two log-prob computations); CPU is fine.
                        try:
                            ref_for_scoring.model.to("cpu")
                        except Exception:  # noqa: BLE001
                            pass
                        ref = ref_for_scoring
                        gc.collect()
                        torch.cuda.empty_cache()

                        qcfg = QResafeConfig(
                            snip_budget_ratio=float(cfg.get("qresafe_snip_budget", 0.10)),
                            lora_rank=int(cfg.get("qresafe_lora_rank", 16)),
                            n_steps=int(cfg.get("qresafe_n_steps", 200)),
                            beta=float(cfg.get("qresafe_beta", 0.1)),
                            learning_rate=float(cfg.get("qresafe_lr", 1e-5)),
                            seed=int(sd),
                            refusal_template=refusal_template,
                        )
                        loaded.model, qreport = qresafe_patch(
                            loaded.model, loaded.tokenizer, ref.model,
                            safety_prompts=list(safety_prompts or []),
                            scores_df=snip_scores,
                            config=qcfg,
                            device=loaded.device,
                        )
                        save_json(
                            run_dir / f"qresafe_report_seed{sd}.json",
                            qreport.__dict__,
                        )
                        del ref
                        gc.collect()
                        torch.cuda.empty_cache()
                    except Exception as exc:  # noqa: BLE001
                        # Loud failure: print, mark the row, and SKIP evaluation
                        # for this seed so we don't silently report the
                        # un-patched quantized model under method='qresafe'.
                        qresafe_failure = f"{exc.__class__.__name__}: {exc}"
                        print(
                            f"[sweep] **Q-resafe DPO patch FAILED** "
                            f"(method={method} seed={sd}): {qresafe_failure}"
                        )
                        save_json(
                            run_dir / f"qresafe_failure_seed{sd}.json",
                            {"error": qresafe_failure, "seed": sd, "method": method},
                        )

                # If qresafe patching failed, record the failure for this seed
                # and skip evaluation -- evaluating the un-patched quantized
                # model and labelling it as Q-resafe would be misleading.
                if qresafe_failure is not None:
                    per_seed_records.append({
                        "seed": sd,
                        "qresafe_failed": True,
                        "qresafe_error": qresafe_failure,
                    })
                    del loaded
                    gc.collect()
                    torch.cuda.empty_cache()
                    continue

                # 3) Evaluate
                eval_result: Dict[str, object] = {"seed": sd}
                per_prompt_refusal: List[int] = []
                per_prompt_harmful: List[int] = []
                responses: List[str] = []
                if eval_prompts_list:
                    responses = generate_completions(
                        loaded.model, loaded.tokenizer, eval_prompts_list,
                        gen=gen_cfg, max_length=max_length, show_progress=True,
                    )
                    per_prompt_refusal = [int(is_refusal(r)) for r in responses]
                    rr = sum(per_prompt_refusal) / max(1, len(per_prompt_refusal))
                    eval_result.update({
                        "refusal_rate": rr,
                        "n": len(eval_prompts_list),
                        "gen": gen_cfg.__dict__,
                        "per_prompt_refusal": per_prompt_refusal,
                    })

                    # Per-seed per-prompt records for downstream stats.
                    responses_path = run_dir / f"responses_seed{sd}.jsonl"
                    with responses_path.open("w", encoding="utf-8") as f:
                        for p, r in zip(eval_prompts_list, responses):
                            f.write(
                                json.dumps({"prompt": p, "response": r}, ensure_ascii=False) + "\n"
                            )

                    if cfg.get("try_strongreject") or cfg.get("strongreject_evaluator"):
                        evaluator = cfg.get("strongreject_evaluator", "strongreject_rubric")
                        eval_result["strongreject"] = try_strongreject_eval(
                            eval_prompts_list, responses, evaluator=evaluator,
                        )

                    if use_harmbench_cls:
                        _maybe_load_classifier()
                        if classifier_model is not None:
                            try:
                                from safepress.eval.harmbench import (
                                    harmbench_attack_success_rate,
                                )
                                hb = harmbench_attack_success_rate(
                                    eval_prompts_list, responses,
                                    classifier_model=classifier_model,
                                    classifier_tokenizer=classifier_tokenizer,
                                    batch_size=harmbench_cls_batch,
                                    show_progress=True,
                                )
                                per_prompt_harmful = [int(x) for x in hb["per_behavior"]]
                                eval_result["harmbench"] = {
                                    "asr": hb["asr"],
                                    "n_total": hb["n_total"],
                                    "n_harmful": hb["n_harmful"],
                                    "per_prompt_harmful": per_prompt_harmful,
                                }
                            except Exception as exc:  # noqa: BLE001
                                eval_result["harmbench"] = {"ok": False, "error": str(exc)}

                    save_json(run_dir / f"eval_seed{sd}.json", eval_result)

                per_seed_records.append(eval_result)

                del loaded
                gc.collect()
                torch.cuda.empty_cache()

            # ----------------------------------------------------------------
            # Aggregate across seeds for this (method, budget) cell.
            # ----------------------------------------------------------------
            from safepress.eval.stats import aggregate_across_seeds

            refusal_means = [r.get("refusal_rate") for r in per_seed_records if "refusal_rate" in r]
            agg_refusal = aggregate_across_seeds([v for v in refusal_means if v is not None])

            asr_means = []
            for r in per_seed_records:
                hb = r.get("harmbench") if isinstance(r, dict) else None
                if hb and "asr" in hb:
                    asr_means.append(float(hb["asr"]))
            agg_asr = aggregate_across_seeds(asr_means) if asr_means else None

            sr_means = []
            for r in per_seed_records:
                sr = r.get("strongreject") if isinstance(r, dict) else None
                if sr and sr.get("ok") and "avg_score" in sr:
                    sr_means.append(float(sr["avg_score"]))
            agg_sr = aggregate_across_seeds(sr_means) if sr_means else None

            row = {
                "model_id": model_id,
                "method": method,
                "budget": float(budget),
                "out_dir": str(run_dir),
                "n_seeds": len(seeds),
                "refusal_rate_mean": agg_refusal.mean,
                "refusal_rate_ci_low": agg_refusal.ci_low,
                "refusal_rate_ci_high": agg_refusal.ci_high,
                "refusal_rate": agg_refusal.mean,  # back-compat
            }
            if agg_asr is not None:
                row.update({
                    "harmbench_asr_mean": agg_asr.mean,
                    "harmbench_asr_ci_low": agg_asr.ci_low,
                    "harmbench_asr_ci_high": agg_asr.ci_high,
                })
            if agg_sr is not None:
                row["strongreject_avg"] = agg_sr.mean
            results.append(row)

            print(
                f"[sweep] {method} b={budget}: refusal_rate="
                f"{agg_refusal.mean:.4f} [{agg_refusal.ci_low:.4f}, {agg_refusal.ci_high:.4f}]  "
                f"seeds={len(seeds)}"
            )

    # Free the HarmBench classifier we kept loaded across the sweep.
    if classifier_model is not None:
        del classifier_model
        del classifier_tokenizer
        gc.collect()
        torch.cuda.empty_cache()

    # Write summary CSV
    import pandas as pd
    summary_csv = Path(cfg.get("summary_csv", str(out_root / "sweep_summary.csv")))
    pd.DataFrame(results).to_csv(summary_csv, index=False)
    print(f"\n[sweep] Summary -> {summary_csv}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="safepress")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_common_model(a):
        a.add_argument("--model_id", type=str, help="HF model id or local path")
        a.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
        a.add_argument("--device_map", type=str, default="auto", help="HF device_map (e.g., auto, cuda:0, cpu)")
        a.add_argument("--trust_remote_code", action="store_true")

    def add_common_prompts(a, which: str):
        a.add_argument(f"--{which}_prompts", type=str, required=True, help="JSONL prompts file")
        # Avoid duplicate --prompt_key / --max_prompts when called multiple times on the same parser
        existing = {act.option_strings[0] for act in a._actions if act.option_strings}
        if "--prompt_key" not in existing:
            a.add_argument("--prompt_key", type=str, default="prompt", help="JSONL field name for the prompt string")
        if "--max_prompts" not in existing:
            a.add_argument("--max_prompts", type=int, default=None)

    # score
    ps = sub.add_parser("score", help="Compute safety drift scores per linear out-block")
    add_common_model(ps)
    add_common_prompts(ps, "calib")
    ps.add_argument("--out", type=str, required=True)
    ps.add_argument("--refusal_template", type=str, default="I'm sorry, but I can't help with that.")
    ps.add_argument("--bits", type=int, default=4)
    ps.add_argument("--group_size", type=int, default=128)
    ps.add_argument("--block_size", type=int, default=64)
    ps.add_argument("--max_length", type=int, default=2048)
    ps.add_argument("--batch_size", type=int, default=1)
    ps.set_defaults(func=cmd_score)

    # build
    pb = sub.add_parser("build", help="Select top blocks, split, and quantize")
    add_common_model(pb)
    pb.add_argument("--scores", type=str, required=True, help="CSV from `score` command")
    pb.add_argument("--out_dir", type=str, required=True)
    pb.add_argument("--overwrite", action="store_true")
    pb.add_argument("--budget", type=float, required=True, help="Fraction of Linear params to keep in FP16")
    pb.add_argument("--block_size", type=int, default=64)
    pb.add_argument("--quant_backend", type=str, default="bnb4", choices=["bnb4"])
    pb.add_argument("--bnb_quant_type", type=str, default="nf4", choices=["nf4", "fp4"])
    pb.add_argument("--bnb_compute_dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
    pb.add_argument("--bnb_quant_storage", type=str, default="uint8", choices=["uint8", "int8"])
    pb.add_argument("--disable_double_quant", action="store_true")
    pb.set_defaults(func=cmd_build)

    # eval
    pe = sub.add_parser("eval", help="Generate and compute refusal heuristics (+optional StrongREJECT eval)")
    pe.add_argument("--model_path", type=str, required=True)
    add_common_prompts(pe, "eval")
    pe.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
    pe.add_argument("--device_map", type=str, default="auto")
    pe.add_argument("--trust_remote_code", action="store_true")
    pe.add_argument("--out", type=str, required=True)
    pe.add_argument("--max_new_tokens", type=int, default=256)
    pe.add_argument("--temperature", type=float, default=0.0)
    pe.add_argument("--top_p", type=float, default=1.0)
    pe.add_argument("--do_sample", action="store_true")
    pe.add_argument("--max_length", type=int, default=2048)
    pe.add_argument("--try_strongreject", action="store_true")
    pe.add_argument("--strongreject_evaluator", type=str, default="strongreject_rubric")
    pe.add_argument("--llamaguard", action="store_true", help="Run Llama Guard 3 safety classifier")
    pe.add_argument("--harmbench_classifier", action="store_true",
                    help="Run the HarmBench-Llama2-13B classifier as the primary ASR metric")
    pe.add_argument("--harmbench_classifier_id", type=str,
                    default="cais/HarmBench-Llama-2-13b-cls")
    pe.add_argument("--seeds", type=int, nargs="+", default=None,
                    help="Seeds to run; eval is repeated once per seed and aggregated")
    pe.add_argument("--n", type=int, default=None,
                    help="Truncate the prompt set to the first N items (overrides --max_prompts)")
    pe.add_argument("--deterministic", action="store_true",
                    help="Force deterministic cuDNN; pair with temperature=0 for reproducibility")
    pe.set_defaults(func=cmd_eval)

    # jailbreak  (adversarial attack eval against HarmBench GCG/AutoDAN/PAIR)
    pj = sub.add_parser(
        "jailbreak",
        help="Evaluate the model under pre-generated HarmBench adversarial attacks",
    )
    pj.add_argument("--model_path", type=str, required=True)
    pj.add_argument("--attack_prompts", type=str, required=True,
                    help="JSONL file from `prepare-data --harmbench_attacks ...`")
    pj.add_argument("--behaviors", type=str, default="data/harmbench.jsonl",
                    help="JSONL with canonical HarmBench behaviors")
    pj.add_argument("--dtype", type=str, default="float16",
                    choices=["float16", "bfloat16", "float32"])
    pj.add_argument("--device_map", type=str, default="auto")
    pj.add_argument("--trust_remote_code", action="store_true")
    pj.add_argument("--out", type=str, required=True)
    pj.add_argument("--max_new_tokens", type=int, default=256)
    pj.add_argument("--temperature", type=float, default=0.0)
    pj.add_argument("--top_p", type=float, default=1.0)
    pj.add_argument("--do_sample", action="store_true")
    pj.add_argument("--max_length", type=int, default=2048)
    pj.add_argument("--max_prompts", type=int, default=None)
    pj.add_argument("--n", type=int, default=None)
    pj.add_argument("--seed", type=int, default=0)
    pj.add_argument("--deterministic", action="store_true")
    pj.add_argument("--harmbench_classifier", action="store_true",
                    help="Use the HarmBench-Llama2-13B classifier (heuristic fallback otherwise)")
    pj.add_argument("--harmbench_classifier_id", type=str,
                    default="cais/HarmBench-Llama-2-13b-cls")
    pj.add_argument("--harmbench_batch_size", type=int, default=4)
    pj.add_argument("--save_generations_to", type=str, default=None,
                    help="Optional JSONL path for per-prompt (behavior, response, label) records")
    pj.add_argument("--include_records", action="store_true",
                    help="Embed per-prompt records in the output summary JSON")
    pj.set_defaults(func=cmd_jailbreak)

    # xstest  (over-refusal evaluation)
    px = sub.add_parser("xstest", help="Run XSTest over-refusal evaluation")
    px.add_argument("--model_path", type=str, required=True)
    px.add_argument("--xstest_prompts", type=str, required=True,
                    help="JSONL with prompt + label (safe/unsafe) fields")
    px.add_argument("--label_key", type=str, default="label")
    px.add_argument("--prompt_key", type=str, default="prompt")
    px.add_argument("--dtype", type=str, default="float16",
                    choices=["float16", "bfloat16", "float32"])
    px.add_argument("--device_map", type=str, default="auto")
    px.add_argument("--trust_remote_code", action="store_true")
    px.add_argument("--out", type=str, required=True)
    px.add_argument("--max_new_tokens", type=int, default=256)
    px.add_argument("--temperature", type=float, default=0.0)
    px.add_argument("--top_p", type=float, default=1.0)
    px.add_argument("--do_sample", action="store_true")
    px.add_argument("--max_length", type=int, default=2048)
    px.add_argument("--max_prompts", type=int, default=None)
    px.add_argument("--n", type=int, default=None)
    px.add_argument("--seed", type=int, default=0)
    px.add_argument("--deterministic", action="store_true")
    px.set_defaults(func=cmd_xstest)

    # pipeline
    pp = sub.add_parser("pipeline", help="score -> build -> eval")
    add_common_model(pp)
    add_common_prompts(pp, "calib")
    add_common_prompts(pp, "eval")
    pp.add_argument("--out_dir", type=str, required=True)
    pp.add_argument("--overwrite", action="store_true")
    pp.add_argument("--refusal_template", type=str, default="I'm sorry, but I can't help with that.")
    pp.add_argument("--bits", type=int, default=4)
    pp.add_argument("--group_size", type=int, default=128)
    pp.add_argument("--block_size", type=int, default=64)
    pp.add_argument("--max_length", type=int, default=2048)
    pp.add_argument("--batch_size", type=int, default=1)
    pp.add_argument("--budget", type=float, required=True)
    pp.add_argument("--quant_backend", type=str, default="bnb4", choices=["bnb4"])
    pp.add_argument("--bnb_quant_type", type=str, default="nf4", choices=["nf4", "fp4"])
    pp.add_argument("--bnb_compute_dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
    pp.add_argument("--bnb_quant_storage", type=str, default="uint8", choices=["uint8", "int8"])
    pp.add_argument("--disable_double_quant", action="store_true")
    pp.add_argument("--max_new_tokens", type=int, default=256)
    pp.add_argument("--temperature", type=float, default=0.0)
    pp.add_argument("--top_p", type=float, default=1.0)
    pp.add_argument("--do_sample", action="store_true")
    pp.add_argument("--try_strongreject", action="store_true")
    pp.add_argument("--strongreject_evaluator", type=str, default="strongreject_rubric")
    pp.set_defaults(func=cmd_pipeline)

    # ------------------------------------------------------------------
    # sweep
    # ------------------------------------------------------------------
    psw = sub.add_parser("sweep", help="Run a paper-style method x budget sweep from YAML config")
    psw.add_argument("--config", type=str, required=True, help="YAML config for sweep")
    psw.set_defaults(func=cmd_sweep)

    # ------------------------------------------------------------------
    # prepare-data
    # ------------------------------------------------------------------
    ppd = sub.add_parser("prepare-data", help="Download and prepare safety / calibration datasets")
    ppd.add_argument("--data_dir", type=str, default="data", help="Root directory for JSONL output")
    ppd.add_argument(
        "--sources", type=str, nargs="+", default=None,
        help="Safety-prompt sources (advbench, harmbench, strongreject, xstest, dolly). "
        "Default: all of advbench/harmbench/strongreject/xstest/dolly.",
    )
    ppd.add_argument("--calib_source", type=str, default="c4", choices=["c4", "wikitext"])
    ppd.add_argument("--n_calib", type=int, default=128, help="Number of calibration samples")
    ppd.add_argument("--cache_dir", type=str, default=None, help="HuggingFace cache directory override")
    ppd.add_argument(
        "--harmbench_attacks", type=str, nargs="+", default=None,
        choices=["gcg", "autodan", "pair"],
        help="Optional HarmBench adversarial-attack prompts to download per attack name",
    )
    ppd.add_argument(
        "--harmbench_attack_target", type=str, default="llama2_7b_chat",
        help="Target model identifier used by HarmBench when generating attacks",
    )
    ppd.set_defaults(func=cmd_prepare_data)

    # ------------------------------------------------------------------
    # experiment  (with sub-subcommands: causal, sweep, phase)
    # ------------------------------------------------------------------
    pexp = sub.add_parser("experiment", help="Run research experiments")
    exp_sub = pexp.add_subparsers(dest="experiment_cmd", required=True)

    # experiment causal
    pec = exp_sub.add_parser("causal", help="Per-layer causal safety-drift experiment")
    add_common_model(pec)
    pec.add_argument("--scores", type=str, required=True, help="CSV scores file from `score` command")
    pec.add_argument("--eval_prompts", type=str, required=True, help="JSONL eval prompts")
    pec.add_argument("--out_dir", type=str, required=True)
    pec.add_argument("--bits", type=int, default=4)
    pec.add_argument("--group_size", type=int, default=128)
    pec.add_argument("--block_size", type=int, default=64)
    pec.add_argument("--budget", type=float, default=0.02)
    pec.add_argument("--max_new_tokens", type=int, default=256)
    pec.set_defaults(func=cmd_experiment_causal)

    # experiment sweep
    pes = exp_sub.add_parser("sweep", help="Budget-sweep experiment across multiple protection ratios")
    add_common_model(pes)
    pes.add_argument(
        "--scores", type=str, required=True,
        help="Path to per-block scores CSV from `safepress score`",
    )
    pes.add_argument(
        "--eval_prompts", type=str, required=True,
        help="JSONL file of evaluation prompts",
    )
    pes.add_argument(
        "--prompt_key", type=str, default="prompt",
        help="JSONL field name for the prompt string (default: prompt)",
    )
    pes.add_argument(
        "--budgets", type=float, nargs="+", default=[0.005, 0.01, 0.02, 0.04],
        help="List of FP16-budget ratios to sweep",
    )
    pes.add_argument("--out_dir", type=str, required=True)
    pes.add_argument("--bits", type=int, default=4)
    pes.add_argument("--group_size", type=int, default=128)
    pes.add_argument("--block_size", type=int, default=64)
    pes.add_argument("--max_new_tokens", type=int, default=256)
    pes.set_defaults(func=cmd_experiment_sweep)

    # experiment phase
    pep = exp_sub.add_parser("phase", help="Phase-transition experiment across bit-widths")
    add_common_model(pep)
    pep.add_argument("--eval_prompts", type=str, required=True, help="JSONL eval prompts")
    pep.add_argument("--out_dir", type=str, required=True)
    pep.add_argument(
        "--bit_widths", type=float, nargs="+",
        default=[8, 5, 4, 3.5, 3, 2.5, 2],
        help="Bit-widths to test, integer or fractional (default: 8 5 4 3.5 3 2.5 2). "
        "Fractional values trigger per-layer mixed precision between adjacent integers.",
    )
    pep.add_argument("--group_size", type=int, default=128)
    pep.add_argument("--max_new_tokens", type=int, default=256)
    pep.set_defaults(func=cmd_experiment_phase)

    # ------------------------------------------------------------------
    # analyze  (with sub-subcommands: refusal-direction, layer-error)
    # ------------------------------------------------------------------
    pan = sub.add_parser("analyze", help="Analysis tools (refusal direction, layer error)")
    an_sub = pan.add_subparsers(dest="analyze_cmd", required=True)

    # analyze refusal-direction
    par = an_sub.add_parser("refusal-direction", help="Compute refusal direction from contrastive prompts")
    add_common_model(par)
    par.add_argument("--harmful_prompts", type=str, required=True, help="JSONL with harmful prompts")
    par.add_argument("--harmless_prompts", type=str, required=True, help="JSONL with harmless prompts")
    par.add_argument("--out_dir", type=str, required=True)
    par.set_defaults(func=cmd_analyze_refusal_direction)

    # analyze layer-error
    pal = an_sub.add_parser("layer-error", help="Per-layer quantization error analysis")
    add_common_model(pal)
    pal.add_argument("--out_dir", type=str, required=True)
    pal.set_defaults(func=cmd_analyze_layer_error)

    # analyze bounds
    pab = an_sub.add_parser("bounds", help="Per-module Cauchy-Schwarz drift bounds (theory section)")
    add_common_model(pab)
    add_common_prompts(pab, "calib")
    pab.add_argument("--out", type=str, required=True, help="Output CSV path")
    pab.add_argument("--refusal_template", type=str, default="I'm sorry, but I can't help with that.")
    pab.add_argument("--bits", type=int, default=4)
    pab.add_argument("--group_size", type=int, default=128)
    pab.add_argument("--max_length", type=int, default=2048)
    pab.set_defaults(func=cmd_bounds)

    # analyze drift-validate (G1 gate: predicted-vs-measured drift R^2)
    pdv = an_sub.add_parser(
        "drift-validate",
        help="G1 gate: validate the absolute-drift upper bound (Sum |g.dw| vs |L1-L0|)",
    )
    add_common_model(pdv)
    add_common_prompts(pdv, "calib")
    pdv.add_argument("--out", type=str, required=True, help="Output CSV path (R^2 fit dumped to .fit.json)")
    pdv.add_argument("--refusal_template", type=str, default="I'm sorry, but I can't help with that.")
    pdv.add_argument(
        "--bit_widths", type=int, nargs="+", default=[8, 4, 3, 2],
        help="Bit-widths to test (integer only; fractional values are not used for the bound fit)",
    )
    pdv.add_argument("--group_size", type=int, default=128)
    pdv.add_argument("--block_size", type=int, default=64,
                     help="SSMP block granularity (output rows per block). Must match the scoring block_size for the theorem alignment.")
    pdv.add_argument("--max_length", type=int, default=2048)
    pdv.set_defaults(func=cmd_drift_validate)

    # ------------------------------------------------------------------
    # viz  (with sub-subcommands: heatmap, phase-transition, causal)
    # ------------------------------------------------------------------
    pvz = sub.add_parser("viz", help="Generate publication-quality figures")
    vz_sub = pvz.add_subparsers(dest="viz_cmd", required=True)

    # viz heatmap
    pvh = vz_sub.add_parser("heatmap", help="Score heatmap figure")
    pvh.add_argument("--scores", type=str, required=True, help="CSV scores file")
    pvh.add_argument("--out", type=str, required=True, help="Output figure path (e.g., figure.pdf)")
    pvh.set_defaults(func=cmd_viz_heatmap)

    # viz phase-transition
    pvp = vz_sub.add_parser("phase-transition", help="Phase-transition figure")
    pvp.add_argument("--results", type=str, required=True, help="JSON results file")
    pvp.add_argument("--out", type=str, required=True, help="Output figure path")
    pvp.set_defaults(func=cmd_viz_phase_transition)

    # viz causal
    pvc = vz_sub.add_parser("causal", help="Causal experiment figure")
    pvc.add_argument("--results", type=str, required=True, help="JSON results file")
    pvc.add_argument("--out", type=str, required=True, help="Output figure path")
    pvc.set_defaults(func=cmd_viz_causal)

    return p


def main(argv: Optional[List[str]] = None) -> None:
    p = build_parser()
    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
