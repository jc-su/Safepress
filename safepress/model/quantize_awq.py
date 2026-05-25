"""AWQ quantization backend for SafePress."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from safepress.model.quantize import QuantizeReport


def quantize_awq(
    model_id: str,
    tokenizer,
    *,
    out_dir: str | Path,
    w_bit: int = 4,
    q_group_size: int = 128,
    modules_to_not_convert: Optional[List[str]] = None,
    calib_data: Optional[List[str]] = None,
    max_calib_samples: int = 128,
    max_calib_seq_len: int = 2048,
) -> QuantizeReport:
    """
    Quantize a causal LM using AWQ (Activation-aware Weight Quantization).

    Uses the ``awq`` library (AutoAWQForCausalLM) to perform calibration-based
    weight quantization and saves the result to *out_dir*.

    Parameters
    ----------
    model_id : str
        HuggingFace model identifier or local path to a full-precision model.
    tokenizer
        A pre-loaded HuggingFace tokenizer instance.
    out_dir : str | Path
        Directory where the quantized model and tokenizer will be saved.
    w_bit : int
        Weight bit-width for quantization (default 4).
    q_group_size : int
        Group size for quantization (default 128).
    modules_to_not_convert : list of str, optional
        Module names that should be excluded from quantization.
    calib_data : list of str, optional
        Custom calibration data as a list of text strings.  When ``None`` the
        AWQ library uses its built-in calibration dataset.
    max_calib_samples : int
        Maximum number of calibration samples (default 128).
    max_calib_seq_len : int
        Maximum sequence length for calibration samples (default 2048).

    Returns
    -------
    QuantizeReport
        Dataclass containing backend name, excluded modules, and a note.
    """
    try:
        from awq import AutoAWQForCausalLM  # noqa: F811
    except ImportError as e:
        raise ImportError(
            "The 'autoawq' package is required for AWQ quantization. "
            "Install with `pip install autoawq`."
        ) from e

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    modules_to_not_convert = modules_to_not_convert or []

    # Build quant config understood by AutoAWQ
    quant_config = {
        "w_bit": w_bit,
        "q_group_size": q_group_size,
        "zero_point": True,
        "version": "GEMM",
    }

    if modules_to_not_convert:
        quant_config["modules_to_not_convert"] = modules_to_not_convert

    # Load the model through AutoAWQ's loader (handles device placement)
    model = AutoAWQForCausalLM.from_pretrained(model_id)

    # Quantize -- optionally supply custom calibration data
    quantize_kwargs: dict = {
        "tokenizer": tokenizer,
        "quant_config": quant_config,
        "max_calib_samples": max_calib_samples,
        "max_calib_seq_len": max_calib_seq_len,
    }
    if calib_data is not None:
        quantize_kwargs["calib_data"] = calib_data

    model.quantize(**quantize_kwargs)

    # Persist quantized model and tokenizer
    model.save_quantized(str(out_dir))
    tokenizer.save_pretrained(out_dir)

    return QuantizeReport(
        backend="awq",
        modules_to_not_convert=modules_to_not_convert,
        note=(
            f"AWQ quantization complete (w_bit={w_bit}, "
            f"q_group_size={q_group_size}). Saved to {out_dir}."
        ),
    )


def load_awq_model(path: str | Path):
    """
    Load a previously-saved AWQ quantized model.

    Parameters
    ----------
    path : str | Path
        Directory containing the quantized model files.

    Returns
    -------
    AutoAWQForCausalLM
        The loaded AWQ model ready for inference.
    """
    try:
        from awq import AutoAWQForCausalLM  # noqa: F811
    except ImportError as e:
        raise ImportError(
            "The 'autoawq' package is required to load AWQ models. "
            "Install with `pip install autoawq`."
        ) from e

    path = Path(path)
    model = AutoAWQForCausalLM.from_quantized(str(path), fuse_layers=False)
    return model
