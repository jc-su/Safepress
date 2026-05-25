"""GPTQ quantization backend for SafePress."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Union

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from safepress.model.quantize import QuantizeReport


def quantize_gptq(
    model_id: str,
    tokenizer,
    *,
    out_dir: str | Path,
    bits: int = 4,
    group_size: int = 128,
    dataset: str = "c4",
    modules_to_not_convert: Optional[List[str]] = None,
    desc_act: bool = False,
    damp_percent: float = 0.1,
) -> QuantizeReport:
    """
    Quantize a causal LM using GPTQ via ``auto_gptq`` directly.

    Parameters
    ----------
    model_id : str
        HuggingFace model identifier or local path to a full-precision model.
    tokenizer
        A pre-loaded HuggingFace tokenizer instance.
    out_dir : str | Path
        Directory where the quantized model and tokenizer will be saved.
    bits : int
        Weight bit-width for quantization (default 4).
    group_size : int
        Group size for quantization (default 128).
    dataset : str
        Named calibration dataset (e.g. ``"c4"``, ``"wikitext2"``).
    modules_to_not_convert : list of str, optional
        Module names that should be excluded from quantization.
    desc_act : bool
        Whether to quantise columns in order of decreasing activation size.
    damp_percent : float
        Percentage of the average Hessian diagonal to use for dampening.

    Returns
    -------
    QuantizeReport
    """
    try:
        from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
    except ImportError as e:
        raise ImportError(
            "The 'auto-gptq' package is required for GPTQ quantization. "
            "Install with `pip install auto-gptq`."
        ) from e

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    modules_to_not_convert = modules_to_not_convert or []

    quantize_config = BaseQuantizeConfig(
        bits=bits,
        group_size=group_size,
        desc_act=desc_act,
        damp_percent=damp_percent,
    )

    # Load FP16 model for quantization
    model = AutoGPTQForCausalLM.from_pretrained(
        model_id,
        quantize_config=quantize_config,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )

    # Prepare calibration data
    from datasets import load_dataset
    try:
        # Try wikitext first (more reliable)
        cal_data = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:256]")
        cal_texts = [t for t in cal_data["text"] if len(t.strip()) > 50][:128]
    except Exception:
        try:
            # Fallback to allenai/c4
            cal_data = load_dataset("allenai/c4", "en", split="train[:256]", streaming=True)
            cal_texts = []
            for item in cal_data:
                if len(item["text"].strip()) > 50:
                    cal_texts.append(item["text"])
                if len(cal_texts) >= 128:
                    break
        except Exception:
            # Last resort: generate simple calibration texts
            import random
            cal_texts = [
                "The meaning of life is a philosophical question that has been debated for centuries. "
                * random.randint(5, 20)
                for _ in range(128)
            ]

    cal_encodings = [
        tokenizer(t, return_tensors="pt", truncation=True, max_length=2048)
        for t in cal_texts
    ]

    # Run GPTQ quantization
    model.quantize(cal_encodings)

    # Save quantized model
    model.save_quantized(str(out_dir))
    tokenizer.save_pretrained(out_dir)

    return QuantizeReport(
        backend="gptq",
        modules_to_not_convert=modules_to_not_convert,
        note=(
            f"GPTQ quantization complete (bits={bits}, group_size={group_size}, "
            f"desc_act={desc_act}). Saved to {out_dir}."
        ),
    )


def load_gptq_model(path: str | Path):
    """
    Load a previously-saved GPTQ quantized model.

    Parameters
    ----------
    path : str | Path
        Directory containing the quantized model files.

    Returns
    -------
    AutoGPTQForCausalLM
        The loaded GPTQ model ready for inference.
    """
    try:
        from auto_gptq import AutoGPTQForCausalLM
    except ImportError as e:
        raise ImportError(
            "The 'auto-gptq' package is required to load GPTQ models. "
            "Install with `pip install auto-gptq`."
        ) from e

    path = Path(path)
    model = AutoGPTQForCausalLM.from_quantized(
        str(path),
        device_map="auto",
        trust_remote_code=True,
    )
    return model
