from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class LoadedModel:
    model: torch.nn.Module
    tokenizer: any
    device: torch.device


def load_tokenizer(model_id: str, trust_remote_code: bool = False):
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code, use_fast=True)
    # Ensure a pad token for batching
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_fp_model(
    model_id: str,
    dtype: str = "float16",
    device_map: str | None = "auto",
    trust_remote_code: bool = False,
):
    """
    Load a full precision (FP16/BF16/FP32) causal LM.
    """
    torch_dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[dtype]

    tok = load_tokenizer(model_id, trust_remote_code=trust_remote_code)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=trust_remote_code,
    )
    model.eval()

    # Figure out a "primary" device for generation helpers.
    try:
        # If device_map='auto', first param might not be on cuda:0, so pick any param device.
        device = next(p.device for p in model.parameters() if p.is_floating_point())
    except StopIteration:
        device = torch.device("cpu")

    return LoadedModel(model=model, tokenizer=tok, device=device)
