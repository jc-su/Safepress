from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import torch
from transformers import BitsAndBytesConfig


@dataclass
class QuantizeReport:
    backend: str
    modules_to_not_convert: List[str]
    note: str



def _should_skip(name: str, skip_list: List[str]) -> bool:
    """Check if a module name (or any of its parents) is in the skip list."""
    for skip in skip_list:
        if name == skip or name.startswith(skip + "."):
            return True
    return False


def quantize_bnb4(
    model: torch.nn.Module,
    *,
    modules_to_not_convert: Optional[List[str]] = None,
    quant_type: str = "nf4",
    compute_dtype: str = "float16",
    use_double_quant: bool = True,
    quant_storage: str = "uint8",
) -> QuantizeReport:
    """
    In-place replace Linear layers with bitsandbytes 4-bit linears,
    except those named in modules_to_not_convert.

    This manually creates Linear4bit modules and transfers weights from
    the already-loaded FP16 model, which triggers actual quantization.
    """
    modules_to_not_convert = modules_to_not_convert or []

    try:
        import bitsandbytes as bnb
    except Exception as e:
        raise ImportError(
            "bitsandbytes is required for bnb4 quantization. Install with `pip install bitsandbytes`."
        ) from e

    torch_compute_dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[compute_dtype]

    torch_storage_dtype = {"uint8": torch.uint8, "int8": torch.int8}[quant_storage]

    qconfig = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=quant_type,
        bnb_4bit_compute_dtype=torch_compute_dtype,
        bnb_4bit_use_double_quant=use_double_quant,
        bnb_4bit_quant_storage=torch_storage_dtype,
        modules_to_not_convert=modules_to_not_convert,
    )

    # Manually replace each Linear with bnb.nn.Linear4bit, copying weights
    replacements = []
    for name, mod in model.named_modules():
        if not isinstance(mod, torch.nn.Linear):
            continue
        if _should_skip(name, modules_to_not_convert):
            continue
        replacements.append(name)

    for name in replacements:
        parts = name.split(".")
        parent = model
        for p in parts[:-1]:
            parent = getattr(parent, p)
        attr = parts[-1]
        old_linear = getattr(parent, attr)

        has_bias = old_linear.bias is not None
        device = old_linear.weight.device
        new_linear = bnb.nn.Linear4bit(
            old_linear.in_features,
            old_linear.out_features,
            bias=has_bias,
            compute_dtype=torch_compute_dtype,
            quant_type=quant_type,
            quant_storage=torch_storage_dtype,
            device=device,
        )
        # Assign weight (triggers quantization on CUDA)
        new_linear.weight = bnb.nn.Params4bit(
            old_linear.weight.data,
            requires_grad=False,
            quant_type=quant_type,
            quant_storage=torch_storage_dtype,
            compress_statistics=use_double_quant,
        ).to(device)
        if has_bias:
            new_linear.bias = torch.nn.Parameter(old_linear.bias.data.clone())

        setattr(parent, attr, new_linear)

    # Attach config so `save_pretrained` preserves it
    if hasattr(model, "config") and getattr(model.config, "quantization_config", None) is None:
        try:
            model.config.quantization_config = qconfig
        except Exception:
            pass

    return QuantizeReport(
        backend="bnb4",
        modules_to_not_convert=modules_to_not_convert,
        note=f"bnb4 conversion complete ({len(replacements)} Linear -> Linear4bit).",
    )


def save_hf_model(model, tokenizer, out_dir: str | Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(out_dir)
    model.save_pretrained(out_dir)


@torch.no_grad()
def quantize_simulated_groupwise(
    model: torch.nn.Module,
    *,
    bits: int,
    group_size: int = 128,
    modules_to_not_convert: Optional[List[str]] = None,
) -> QuantizeReport:
    """In-place simulated symmetric group-wise quantize-dequantize at *bits*.

    The model stays in its original dtype and device; only the values of every
    non-skipped ``nn.Linear.weight`` are replaced with their quant-then-dequant
    approximation. This is the same routine used by
    :func:`safepress.experiments.phase_transition._simulate_quantize_inplace`
    and by :func:`safepress.analysis.drift_bound.validate_drift_bound`, so the
    method-sweep, the phase-transition, and the theory validation all see the
    same δw for a given bit-width. (Critical for the G2 scoring-ablation to
    actually compare scorers at the targeted bit-width.)

    This path does NOT reduce memory -- weights remain FP16 with quantized
    values. For deployable 4-bit results use :func:`quantize_bnb4` instead.

    For ``bits == 4`` this helper is still valid (it produces a *simulated*
    4-bit perturbation, equivalent in distribution to bnb-nf4 quantize-dequant
    up to the choice of nf4 vs symmetric int4 buckets); the sweep should call
    :func:`quantize_bnb4` for 4-bit so that the deployed model is genuinely
    4-bit.
    """
    from safepress.model.blocks import iter_linear_modules
    from safepress.model.score import _quant_dequant_symmetric_groupwise

    modules_to_not_convert = modules_to_not_convert or []
    if bits < 2 or bits > 8:
        raise ValueError(f"quantize_simulated_groupwise: bits must be in [2, 8], got {bits}")

    n_quantized = 0
    for name, mod in iter_linear_modules(model):
        if _should_skip(name, modules_to_not_convert):
            continue
        w = mod.weight
        w_hat = _quant_dequant_symmetric_groupwise(w, bits=int(bits), group_size=int(group_size))
        mod.weight.copy_(w_hat)
        n_quantized += 1

    return QuantizeReport(
        backend=f"simulated_int{int(bits)}",
        modules_to_not_convert=modules_to_not_convert,
        note=(
            f"Simulated {int(bits)}-bit symmetric group-wise quant-dequant "
            f"applied in place to {n_quantized} Linear weights (group_size={int(group_size)})."
        ),
    )


def quantize_for_bits(
    model: torch.nn.Module,
    *,
    bits: int,
    group_size: int = 128,
    modules_to_not_convert: Optional[List[str]] = None,
    quant_type: str = "nf4",
    compute_dtype: str = "float16",
    use_double_quant: bool = True,
    quant_storage: str = "uint8",
) -> QuantizeReport:
    """Dispatch wrapper: bnb4 for ``bits == 4``, simulated groupwise otherwise.

    This is what the sweep should call. The earlier sweep code hard-wired
    ``quantize_bnb4`` regardless of the ``bits`` config; that silently turned
    every ``bits: 3`` sweep into a 4-bit sweep and invalidated the G2 scoring
    ablation at 3-bit.
    """
    if int(bits) == 4 and not os.environ.get("SAFEPRESS_FORCE_SIMULATED"):
        try:
            return quantize_bnb4(
                model,
                modules_to_not_convert=modules_to_not_convert,
                quant_type=quant_type,
                compute_dtype=compute_dtype,
                use_double_quant=use_double_quant,
                quant_storage=quant_storage,
            )
        except (OSError, RuntimeError, ImportError):
            # bitsandbytes can fail to load on systems missing the CUDA
            # toolkit libs (e.g. libnvJitLink.so on CUDA 13 hosts). Fall
            # back to simulated groupwise quant -- same arithmetic, no
            # native dependencies.
            pass
    return quantize_simulated_groupwise(
        model,
        bits=int(bits),
        group_size=int(group_size),
        modules_to_not_convert=modules_to_not_convert,
    )
