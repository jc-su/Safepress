"""Safety phase transition: mapping bit-width vs safety collapse."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from tqdm import tqdm

from safepress.eval.basic import (
    GenConfig,
    generate_completions,
    is_refusal,
    refusal_rate,
    try_strongreject_eval,
)
from safepress.model.load import load_fp_model
from safepress.model.blocks import chunk_indices, iter_linear_modules
from safepress.model.score import _quant_dequant_symmetric_groupwise
from safepress.utils.logging import save_json


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class PhaseTransitionResult:
    """Container for a phase-transition curve result."""
    experiment: str
    model_id: str
    bit_widths: List[float]
    results: Dict[str, Dict[str, Any]]
    summary: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _evaluate_model(
    model: torch.nn.Module,
    tokenizer,
    prompts: List[str],
    *,
    device: torch.device | str | None = None,
) -> Dict[str, Any]:
    """Generate completions and return refusal + utility statistics."""
    gen = GenConfig()
    responses = generate_completions(
        model, tokenizer, prompts, gen=gen, device=device, show_progress=True,
    )
    rr = refusal_rate(responses)
    sr = try_strongreject_eval(prompts, responses)

    # Utility proxy: average response length (longer = more helpful).
    avg_len = sum(len(r.split()) for r in responses) / max(len(responses), 1)

    return {
        "refusal_rate": rr,
        "avg_response_words": avg_len,
        "n": len(prompts),
        "strongreject": sr,
    }


def _guess_layer_index(name: str) -> Optional[int]:
    """Extract the numeric transformer-layer index from a dotted module path."""
    import re

    match = re.search(r"\.(\d+)\.", name)
    return int(match.group(1)) if match else None


@torch.no_grad()
def _simulate_quantize_inplace(
    model: torch.nn.Module,
    *,
    bits: float,
    group_size: int = 128,
    upper_layer_fraction: float = 0.5,
    seed: int = 0,
) -> None:
    """Simulate quantization at *bits* by replacing each Linear weight with its
    quant-then-dequant approximation in place.

    Integer bit-widths (2, 3, 4, 8, 16) quantize every linear at that width.

    **Fractional bit-widths** interpolate between ``floor(bits)`` and
    ``ceil(bits)`` by assigning the higher precision to a fraction of layers
    proportional to ``bits - floor(bits)``. Concretely, for ``bits=3.5`` we
    keep the first half of transformer layers at 4-bit and the rest at 3-bit.
    The split is deterministic given a fixed *seed* (modules without a parseable
    layer index inherit the lower bit-width). This produces a clean
    interpolation between adjacent integer bit-widths for the phase-transition
    curve.

    Parameters
    ----------
    bits:
        Target bit-width. Float values trigger the mixed-precision interpolation.
    upper_layer_fraction:
        Override for the high-bit-width fraction. By default we use
        ``bits - floor(bits)`` -- supply this only to disconnect the mix ratio
        from *bits* itself (e.g. for ablations).
    seed:
        Reserved for future stochastic assignments (currently the assignment
        is deterministic by layer index).
    """
    if bits >= 16:
        return  # FP16 / no quantization

    bits_int = int(bits)
    is_fractional = abs(bits - bits_int) > 1e-9

    if not is_fractional:
        # Uniform integer quantization
        for _name, mod in iter_linear_modules(model):
            w = mod.weight
            w_hat = _quant_dequant_symmetric_groupwise(w, bits=bits_int, group_size=group_size)
            mod.weight.copy_(w_hat)
        return

    # Mixed-precision: high bits on the first ``frac`` of layers, low bits elsewhere
    lower_bits = int(bits)
    upper_bits = int(bits) + 1
    frac_upper = bits - float(lower_bits)
    if upper_layer_fraction != 0.5:
        frac_upper = float(upper_layer_fraction)

    # First pass: collect layer indices that actually occur in the model
    layer_indices: List[int] = []
    name_to_layer: Dict[str, Optional[int]] = {}
    for name, _mod in iter_linear_modules(model):
        idx = _guess_layer_index(name)
        name_to_layer[name] = idx
        if idx is not None:
            layer_indices.append(idx)

    if not layer_indices:
        # No layer indices parsed -- fall back to lower bits everywhere
        for _name, mod in iter_linear_modules(model):
            w = mod.weight
            w_hat = _quant_dequant_symmetric_groupwise(w, bits=lower_bits, group_size=group_size)
            mod.weight.copy_(w_hat)
        return

    min_layer = min(layer_indices)
    max_layer = max(layer_indices)
    span = max(1, max_layer - min_layer + 1)
    threshold = min_layer + frac_upper * span  # layers with idx < threshold get upper bits

    for name, mod in iter_linear_modules(model):
        idx = name_to_layer.get(name)
        if idx is None:
            mod_bits = lower_bits
        elif float(idx) < threshold:
            mod_bits = upper_bits
        else:
            mod_bits = lower_bits
        w = mod.weight
        w_hat = _quant_dequant_symmetric_groupwise(w, bits=mod_bits, group_size=group_size)
        mod.weight.copy_(w_hat)


# ---------------------------------------------------------------------------
# Phase transition curve
# ---------------------------------------------------------------------------

def phase_transition_curve(
    model_id: str,
    eval_prompts: List[str],
    *,
    bit_widths: Optional[List[float]] = None,
    group_size: int = 128,
    out_dir: str | Path,
    device_map: str = "auto",
    dtype: str = "float16",
) -> PhaseTransitionResult:
    """
    Map the safety phase transition as bit-width decreases.

    For each bit-width in *bit_widths*, simulate symmetric groupwise
    quantization and measure both safety (refusal rate) and utility
    (average response length).

    The FP16 baseline (no quantization) is always included as the
    reference point.

    Parameters
    ----------
    model_id : str
        HuggingFace model ID or local path.
    eval_prompts : list[str]
        Harmful prompts for refusal-rate evaluation.
    bit_widths : list[int], optional
        Bit widths to evaluate.  Defaults to ``[8, 4, 3, 2]``.
    group_size : int
        Group size for symmetric quantization simulation.
    out_dir : str | Path
        Directory to write results.
    device_map, dtype
        Forwarded to :func:`load_fp_model`.

    Returns
    -------
    PhaseTransitionResult
        Contains per-bit-width refusal rates and utility metrics suitable
        for plotting the safety phase-transition curve.
    """
    if bit_widths is None:
        # Finer grid around the 3-bit cliff that the paper claims is the
        # interesting regime. Callers can override.
        bit_widths = [8, 5, 4, 3.5, 3, 2.5, 2]

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results: Dict[str, Dict[str, Any]] = {}

    def _bit_label(b: float) -> str:
        # Format integer bit-widths plainly; keep fractional ones precise.
        return f"bits_{int(b)}" if float(b).is_integer() else f"bits_{b}"

    # -- Resume from partial results if available --------------------------
    partial_path = out_dir / "phase_transition_partial.json"
    if partial_path.exists():
        import json
        with open(partial_path) as f:
            saved = json.load(f)
        results = saved.get("results", {})
        print(f"[phase_transition] Resuming from partial results: {list(results.keys())}")

    def _save_partial():
        """Save incremental results so crashes don't lose progress."""
        save_json(partial_path, {"model_id": model_id, "results": results})

    # -- FP16 baseline (16-bit, no quantization) ---------------------------
    if "bits_16" not in results:
        print("[phase_transition] Evaluating FP16 baseline (16-bit) ...")
        loaded = load_fp_model(model_id, dtype=dtype, device_map=device_map)
        fp16_eval = _evaluate_model(
            loaded.model, loaded.tokenizer, eval_prompts, device=loaded.device,
        )
        fp16_eval["bits"] = 16
        results["bits_16"] = fp16_eval
        del loaded
        torch.cuda.empty_cache()
        import gc; gc.collect()
        _save_partial()
    else:
        print("[phase_transition] FP16 baseline already computed, skipping.")

    # -- Each bit-width ----------------------------------------------------
    # Filter out >=16 — already handled as FP16 baseline above
    sub_bits = [b for b in bit_widths if float(b) < 16.0]
    for bits in tqdm(sub_bits, desc="Phase transition"):
        label = _bit_label(float(bits))
        if label in results:
            print(f"[phase_transition] {bits}-bit already computed, skipping.")
            continue
        print(f"[phase_transition] Evaluating {bits}-bit quantization ...")

        loaded = load_fp_model(model_id, dtype=dtype, device_map=device_map)
        _simulate_quantize_inplace(
            loaded.model, bits=float(bits), group_size=group_size,
        )

        eval_out = _evaluate_model(
            loaded.model, loaded.tokenizer, eval_prompts, device=loaded.device,
        )
        eval_out["bits"] = float(bits)
        results[label] = eval_out

        del loaded
        torch.cuda.empty_cache()
        import gc; gc.collect()
        _save_partial()

    # -- Save results ------------------------------------------------------
    summary_parts = [f"16-bit={results['bits_16']['refusal_rate']:.3f}"]
    for bits in sub_bits:
        label = _bit_label(float(bits))
        if label in results:
            summary_parts.append(f"{bits}-bit={results[label]['refusal_rate']:.3f}")

    all_bits: List[float] = [16.0] + [float(b) for b in sub_bits]
    result = PhaseTransitionResult(
        experiment="phase_transition",
        model_id=model_id,
        bit_widths=all_bits,
        results=results,
        summary=", ".join(summary_parts),
    )
    save_json(out_dir / "phase_transition_results.json", result.to_dict())
    print(f"[phase_transition] Done. {result.summary}")
    return result
