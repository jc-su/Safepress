"""Naive full quantization (no protection) as baseline."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from safepress.model.quantize import QuantizeReport


def quantize_naive_bnb4(
    model_id: str,
    *,
    out_dir: str | Path,
    quant_type: str = "nf4",
    compute_dtype: str = "float16",
) -> QuantizeReport:
    """
    Naive 4-bit quantization with bitsandbytes -- no SSMP protection.

    Every Linear layer is quantized uniformly; this serves as a baseline to
    measure the benefit of selective mixed-precision (SafePress / SSMP).

    Parameters
    ----------
    model_id : str
        HuggingFace model identifier or local path.
    out_dir : str | Path
        Directory where the quantized model will be saved.
    quant_type : str
        BitsAndBytes 4-bit quant type (``"nf4"`` or ``"fp4"``).
    compute_dtype : str
        Compute dtype string (``"float16"``, ``"bfloat16"``, ``"float32"``).

    Returns
    -------
    QuantizeReport
        Dataclass containing backend name, excluded modules, and a note.
    """
    try:
        import bitsandbytes as bnb  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "bitsandbytes is required for naive bnb4 quantization. "
            "Install with `pip install bitsandbytes`."
        ) from e

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch_compute_dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[compute_dtype]

    qconfig = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=quant_type,
        bnb_4bit_compute_dtype=torch_compute_dtype,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=qconfig,
        device_map="auto",
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(out_dir)

    return QuantizeReport(
        backend="naive_bnb4",
        modules_to_not_convert=[],
        note=(
            f"Naive bnb4 quantization complete (quant_type={quant_type}, "
            f"compute_dtype={compute_dtype}). No SSMP protection applied. "
            f"Saved to {out_dir}."
        ),
    )


def quantize_naive_8bit(
    model_id: str,
    *,
    out_dir: str | Path,
) -> QuantizeReport:
    """
    Naive 8-bit quantization with bitsandbytes -- no SSMP protection.

    Every Linear layer is quantized to 8-bit uniformly; this serves as a
    baseline to measure the benefit of selective mixed-precision.

    Parameters
    ----------
    model_id : str
        HuggingFace model identifier or local path.
    out_dir : str | Path
        Directory where the quantized model will be saved.

    Returns
    -------
    QuantizeReport
        Dataclass containing backend name, excluded modules, and a note.
    """
    try:
        import bitsandbytes as bnb  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "bitsandbytes is required for naive 8-bit quantization. "
            "Install with `pip install bitsandbytes`."
        ) from e

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    qconfig = BitsAndBytesConfig(load_in_8bit=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=qconfig,
        device_map="auto",
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(out_dir)

    return QuantizeReport(
        backend="naive_8bit",
        modules_to_not_convert=[],
        note=(
            f"Naive 8-bit quantization complete (load_in_8bit=True). "
            f"No SSMP protection applied. Saved to {out_dir}."
        ),
    )
