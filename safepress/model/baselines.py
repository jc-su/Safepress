"""
Baseline block-scoring methods for SafePress.

Each function returns a pandas DataFrame with the same columns as
``compute_block_scores`` from ``safepress.model.score``:

    module, block_idx, out_start, out_end, in_features, num_params, score

These baselines are used as experimental controls and comparisons against the
default gradient * quantization_error scoring.
"""
from __future__ import annotations

import math
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from safepress.model.blocks import chunk_indices, iter_linear_modules
from safepress.model.score import (
    ScoreRow,
    build_refusal_supervision,
)

# ---------------------------------------------------------------------------
# Helper: convert accumulated score dicts into the canonical DataFrame
# ---------------------------------------------------------------------------

def _build_score_dataframe(
    meta: Dict[Tuple[str, int], ScoreRow],
    score_accum: Dict[Tuple[str, int], float],
) -> pd.DataFrame:
    """Build the canonical (module, block_idx, ..., score) DataFrame."""
    rows: List[Dict[str, object]] = []
    for key, score in score_accum.items():
        sr = meta[key]
        rows.append(
            dict(
                module=sr.module,
                block_idx=sr.block_idx,
                out_start=sr.out_start,
                out_end=sr.out_end,
                in_features=sr.in_features,
                num_params=sr.num_params,
                score=float(score),
            )
        )
    df = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    return df


def _init_meta_and_accum(
    linear_modules: List[Tuple[str, torch.nn.Linear]],
    block_size: int,
) -> Tuple[Dict[Tuple[str, int], ScoreRow], Dict[Tuple[str, int], float]]:
    """Pre-initialize metadata rows and zero-score accumulators."""
    meta: Dict[Tuple[str, int], ScoreRow] = {}
    score_accum: Dict[Tuple[str, int], float] = {}
    for mod_name, mod in linear_modules:
        out_features, in_features = mod.weight.shape
        blocks = chunk_indices(out_features, block_size)
        for b_idx, (s, e) in enumerate(blocks):
            num_params = (e - s) * in_features
            meta[(mod_name, b_idx)] = ScoreRow(
                module=mod_name,
                block_idx=b_idx,
                out_start=s,
                out_end=e,
                in_features=in_features,
                num_params=num_params,
                score=0.0,
            )
            score_accum[(mod_name, b_idx)] = 0.0
    return meta, score_accum


def _resolve_device(model: torch.nn.Module, device=None) -> torch.device:
    """Resolve the device to use, defaulting to the model's first parameter device."""
    if device is not None:
        return torch.device(device)
    try:
        return next(p.device for p in model.parameters() if p.is_floating_point())
    except StopIteration:
        return torch.device("cpu")


# ===================================================================
# 1. score_random  -- random baseline control
# ===================================================================

@torch.no_grad()
def score_random(
    model: torch.nn.Module,
    block_size: int = 64,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Assign uniformly random scores to each block (random baseline control).

    Parameters
    ----------
    model : torch.nn.Module
        The model whose Linear layers will be enumerated.
    block_size : int
        Number of output rows per block.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Canonical score DataFrame.
    """
    rng = np.random.RandomState(seed)
    linear_modules = list(iter_linear_modules(model))
    if not linear_modules:
        raise RuntimeError("No torch.nn.Linear modules found.")

    meta, score_accum = _init_meta_and_accum(linear_modules, block_size)

    for key in tqdm(list(score_accum.keys()), desc="Random scoring", leave=False):
        score_accum[key] = float(rng.random())

    return _build_score_dataframe(meta, score_accum)


# ===================================================================
# 2. score_magnitude  -- weight magnitude heuristic
# ===================================================================

@torch.no_grad()
def score_magnitude(
    model: torch.nn.Module,
    block_size: int = 64,
) -> pd.DataFrame:
    """
    Score each block by the sum of absolute weight values.

    score(b) = sum(|w_i|) for all weights i in the block.

    Parameters
    ----------
    model : torch.nn.Module
    block_size : int

    Returns
    -------
    pd.DataFrame
    """
    linear_modules = list(iter_linear_modules(model))
    if not linear_modules:
        raise RuntimeError("No torch.nn.Linear modules found.")

    meta, score_accum = _init_meta_and_accum(linear_modules, block_size)

    for mod_name, mod in tqdm(linear_modules, desc="Magnitude scoring", leave=False):
        w = mod.weight.detach()
        out_features = w.shape[0]
        blocks = chunk_indices(out_features, block_size)
        for b_idx, (s, e) in enumerate(blocks):
            score_accum[(mod_name, b_idx)] = w[s:e, :].abs().sum().item()

    return _build_score_dataframe(meta, score_accum)


# ===================================================================
# 3. score_gradient_only  -- diagonal Fisher (squared gradient)
# ===================================================================

def score_gradient_only(
    model: torch.nn.Module,
    tokenizer,
    prompts: List[str],
    *,
    refusal_template: str = "I'm sorry, but I can't help with that.",
    block_size: int = 64,
    max_length: int = 2048,
    batch_size: int = 1,
    device=None,
) -> pd.DataFrame:
    """
    Score each block by the sum of squared gradient magnitudes (diagonal Fisher
    approximation, similar to CWP).

    score(b) = sum(|g_i|^2)  for all weights i in the block.

    No quantization error is included -- only gradient magnitude.

    Parameters
    ----------
    model : torch.nn.Module
    tokenizer : PreTrainedTokenizer
    prompts : list of str
    refusal_template : str
    block_size : int
    max_length : int
    batch_size : int
    device : optional

    Returns
    -------
    pd.DataFrame
    """
    device = _resolve_device(model, device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(True)

    linear_modules = list(iter_linear_modules(model))
    if not linear_modules:
        raise RuntimeError("No torch.nn.Linear modules found.")

    meta, score_accum = _init_meta_and_accum(linear_modules, block_size)

    it = range(0, len(prompts), batch_size)
    it = tqdm(list(it), desc="Gradient-only scoring", leave=False)

    for start in it:
        batch_prompts = prompts[start : start + batch_size]
        batch_inputs = build_refusal_supervision(
            tokenizer,
            batch_prompts,
            refusal_template=refusal_template,
            max_length=max_length,
        )
        batch_inputs = {k: v.to(device) for k, v in batch_inputs.items()}

        model.zero_grad(set_to_none=True)
        out = model(**batch_inputs)
        loss = out.loss
        if loss is None:
            raise RuntimeError("Model did not return a loss. Does it support labels?")
        loss.backward()

        with torch.no_grad():
            for mod_name, mod in linear_modules:
                g = mod.weight.grad
                if g is None:
                    continue
                out_features = g.shape[0]
                blocks = chunk_indices(out_features, block_size)
                for b_idx, (s, e) in enumerate(blocks):
                    g_blk = g[s:e, :].float()
                    contrib = (g_blk ** 2).sum().item()
                    score_accum[(mod_name, b_idx)] += float(contrib)

    return _build_score_dataframe(meta, score_accum)


# ===================================================================
# 4. score_snip  -- SNIP / connection sensitivity (|w * g|)
# ===================================================================

def score_snip(
    model: torch.nn.Module,
    tokenizer,
    prompts: List[str],
    *,
    refusal_template: str = "I'm sorry, but I can't help with that.",
    block_size: int = 64,
    max_length: int = 2048,
    batch_size: int = 1,
    device=None,
) -> pd.DataFrame:
    """
    SNIP-style score: connection sensitivity |w * g| per block
    (similar to the Q-resafe salience criterion).

    score(b) = sum(|w_i * g_i|)  for all weights i in the block.

    Parameters
    ----------
    model : torch.nn.Module
    tokenizer : PreTrainedTokenizer
    prompts : list of str
    refusal_template : str
    block_size : int
    max_length : int
    batch_size : int
    device : optional

    Returns
    -------
    pd.DataFrame
    """
    device = _resolve_device(model, device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(True)

    linear_modules = list(iter_linear_modules(model))
    if not linear_modules:
        raise RuntimeError("No torch.nn.Linear modules found.")

    meta, score_accum = _init_meta_and_accum(linear_modules, block_size)

    it = range(0, len(prompts), batch_size)
    it = tqdm(list(it), desc="SNIP scoring", leave=False)

    for start in it:
        batch_prompts = prompts[start : start + batch_size]
        batch_inputs = build_refusal_supervision(
            tokenizer,
            batch_prompts,
            refusal_template=refusal_template,
            max_length=max_length,
        )
        batch_inputs = {k: v.to(device) for k, v in batch_inputs.items()}

        model.zero_grad(set_to_none=True)
        out = model(**batch_inputs)
        loss = out.loss
        if loss is None:
            raise RuntimeError("Model did not return a loss. Does it support labels?")
        loss.backward()

        with torch.no_grad():
            for mod_name, mod in linear_modules:
                w = mod.weight
                g = mod.weight.grad
                if g is None:
                    continue
                out_features = w.shape[0]
                blocks = chunk_indices(out_features, block_size)
                for b_idx, (s, e) in enumerate(blocks):
                    w_blk = w[s:e, :].float()
                    g_blk = g[s:e, :].float()
                    contrib = (w_blk * g_blk).abs().sum().item()
                    score_accum[(mod_name, b_idx)] += float(contrib)

    return _build_score_dataframe(meta, score_accum)


# ===================================================================
# 5. score_wanda  -- Wanda: |w| * ||X||_2  (activation-aware)
# ===================================================================

def score_wanda(
    model: torch.nn.Module,
    tokenizer,
    prompts: List[str],
    *,
    refusal_template: str = "I'm sorry, but I can't help with that.",
    block_size: int = 64,
    max_length: int = 2048,
    batch_size: int = 1,
    device=None,
) -> pd.DataFrame:
    """
    Wanda-style score: sum(|w_i| * ||X_i||_2) per block, where X_i is the
    L2 norm of the input activation vector corresponding to each input
    feature of the Linear layer.

    Uses forward hooks to capture the input activations to each Linear.

    Parameters
    ----------
    model : torch.nn.Module
    tokenizer : PreTrainedTokenizer
    prompts : list of str
    refusal_template : str
    block_size : int
    max_length : int
    batch_size : int
    device : optional

    Returns
    -------
    pd.DataFrame
    """
    device = _resolve_device(model, device)
    model.eval()

    linear_modules = list(iter_linear_modules(model))
    if not linear_modules:
        raise RuntimeError("No torch.nn.Linear modules found.")

    meta, score_accum = _init_meta_and_accum(linear_modules, block_size)

    # We will accumulate the squared L2 norm of input activations per feature
    # across all tokens and batches.  For a Linear with in_features=d, we
    # accumulate a vector of shape (d,) representing sum_over_tokens ||x_j||^2
    # for each input feature j.  At the end, sqrt gives the L2 norm.
    input_norm_accum: Dict[str, torch.Tensor] = {}
    batch_count: Dict[str, int] = {}
    for mod_name, mod in linear_modules:
        in_features = mod.weight.shape[1]
        input_norm_accum[mod_name] = torch.zeros(in_features, dtype=torch.float64)
        batch_count[mod_name] = 0

    # Register forward hooks
    hooks = []

    def _make_hook(name: str):
        def hook_fn(module, inp, out):
            x = inp[0]  # (batch, seq_len, in_features) or (batch, in_features)
            if x.ndim == 3:
                # Sum over batch and sequence dimensions
                # Result: per-feature squared L2 norm
                x_f = x.float()
                norm_sq = (x_f ** 2).sum(dim=(0, 1))  # (in_features,)
            elif x.ndim == 2:
                x_f = x.float()
                norm_sq = (x_f ** 2).sum(dim=0)  # (in_features,)
            else:
                return
            input_norm_accum[name] += norm_sq.cpu().to(torch.float64)
            batch_count[name] += 1
        return hook_fn

    for mod_name, mod in linear_modules:
        h = mod.register_forward_hook(_make_hook(mod_name))
        hooks.append(h)

    try:
        # Run forward passes (no gradients needed)
        it = range(0, len(prompts), batch_size)
        it = tqdm(list(it), desc="Wanda scoring (forward)", leave=False)

        with torch.no_grad():
            for start in it:
                batch_prompts = prompts[start : start + batch_size]
                batch_inputs = build_refusal_supervision(
                    tokenizer,
                    batch_prompts,
                    refusal_template=refusal_template,
                    max_length=max_length,
                )
                batch_inputs = {k: v.to(device) for k, v in batch_inputs.items()}
                # Remove labels for forward-only pass to avoid computing loss grad
                batch_inputs.pop("labels", None)
                model(**batch_inputs)
    finally:
        for h in hooks:
            h.remove()

    # Compute per-block Wanda scores: sum(|w_ij| * ||X_j||_2)
    # where ||X_j||_2 = sqrt(accumulated squared norms) for feature j
    for mod_name, mod in tqdm(linear_modules, desc="Wanda scoring (blocks)", leave=False):
        w = mod.weight.detach()
        out_features, in_features = w.shape

        # Per-feature input norm: sqrt of accumulated squared norms
        x_norm = input_norm_accum[mod_name].sqrt()  # (in_features,)
        x_norm_device = x_norm.to(dtype=w.dtype, device=w.device)

        blocks = chunk_indices(out_features, block_size)
        for b_idx, (s, e) in enumerate(blocks):
            w_blk = w[s:e, :]  # (block_rows, in_features)
            # score = sum(|w_ij| * ||X_j||_2)
            score_val = (w_blk.abs() * x_norm_device.unsqueeze(0)).sum().item()
            score_accum[(mod_name, b_idx)] = float(score_val)

    return _build_score_dataframe(meta, score_accum)


# ===================================================================
# 6. score_layer_uniform  -- protect entire layers heuristic
# ===================================================================

def _extract_layer_index(module_name: str) -> Optional[int]:
    """
    Attempt to extract a numeric layer index from a module name.

    Common patterns:
        model.layers.12.self_attn.q_proj  -> 12
        transformer.h.0.mlp.dense_4h_to_h -> 0
        model.decoder.layers.5.fc1        -> 5
    """
    # Match the first sequence of ".digits." in the name
    match = re.search(r"\.(\d+)\.", module_name)
    if match:
        return int(match.group(1))
    return None


@torch.no_grad()
def score_layer_uniform(
    model: torch.nn.Module,
    block_size: int = 64,
    protect_layers: Optional[List[int]] = None,
) -> pd.DataFrame:
    """
    Assign high scores to all blocks in specified layers; zero for the rest.

    This is the "just protect certain layers" heuristic baseline.

    Parameters
    ----------
    model : torch.nn.Module
    block_size : int
    protect_layers : list of int or None
        Layer indices to protect.  If None, defaults to the first 2 and last 2
        layers (determined by scanning module names for numeric layer indices).

    Returns
    -------
    pd.DataFrame
    """
    linear_modules = list(iter_linear_modules(model))
    if not linear_modules:
        raise RuntimeError("No torch.nn.Linear modules found.")

    meta, score_accum = _init_meta_and_accum(linear_modules, block_size)

    # Determine layer indices if not provided
    if protect_layers is None:
        all_layer_indices = set()
        for mod_name, _ in linear_modules:
            idx = _extract_layer_index(mod_name)
            if idx is not None:
                all_layer_indices.add(idx)
        if all_layer_indices:
            sorted_layers = sorted(all_layer_indices)
            n_layers = len(sorted_layers)
            # First 2 and last 2
            first_n = min(2, n_layers)
            last_n = min(2, n_layers)
            protect_set = set(sorted_layers[:first_n]) | set(sorted_layers[-last_n:])
            protect_layers = sorted(protect_set)
        else:
            # Cannot determine layers; protect nothing (all scores stay 0)
            protect_layers = []

    protect_set = set(protect_layers)

    for mod_name, mod in tqdm(linear_modules, desc="Layer-uniform scoring", leave=False):
        layer_idx = _extract_layer_index(mod_name)
        if layer_idx is not None and layer_idx in protect_set:
            score_val = 1.0
        else:
            score_val = 0.0
        out_features = mod.weight.shape[0]
        blocks = chunk_indices(out_features, block_size)
        for b_idx, (s, e) in enumerate(blocks):
            score_accum[(mod_name, b_idx)] = score_val

    return _build_score_dataframe(meta, score_accum)


# ===================================================================
# 7. score_cwp_style  -- CWP: I_safe - beta * I_general
# ===================================================================

def _accumulate_fisher_diagonal(
    model: torch.nn.Module,
    tokenizer,
    prompts: List[str],
    *,
    refusal_template: str,
    linear_modules: List[Tuple[str, torch.nn.Linear]],
    block_size: int,
    max_length: int,
    batch_size: int,
    device: torch.device,
    desc: str = "Fisher",
) -> Dict[Tuple[str, int], float]:
    """
    Accumulate diagonal Fisher (squared gradients) per block over a set of
    prompts.  Returns a dict mapping (module_name, block_idx) -> score.
    """
    accum: Dict[Tuple[str, int], float] = {}
    for mod_name, mod in linear_modules:
        out_features = mod.weight.shape[0]
        blocks = chunk_indices(out_features, block_size)
        for b_idx, _ in enumerate(blocks):
            accum[(mod_name, b_idx)] = 0.0

    it = range(0, len(prompts), batch_size)
    it = tqdm(list(it), desc=desc, leave=False)

    for start in it:
        batch_prompts = prompts[start : start + batch_size]
        batch_inputs = build_refusal_supervision(
            tokenizer,
            batch_prompts,
            refusal_template=refusal_template,
            max_length=max_length,
        )
        batch_inputs = {k: v.to(device) for k, v in batch_inputs.items()}

        model.zero_grad(set_to_none=True)
        out = model(**batch_inputs)
        loss = out.loss
        if loss is None:
            raise RuntimeError("Model did not return a loss. Does it support labels?")
        loss.backward()

        with torch.no_grad():
            for mod_name, mod in linear_modules:
                g = mod.weight.grad
                if g is None:
                    continue
                out_features = g.shape[0]
                blocks = chunk_indices(out_features, block_size)
                for b_idx, (s, e) in enumerate(blocks):
                    g_blk = g[s:e, :].float()
                    contrib = (g_blk ** 2).sum().item()
                    accum[(mod_name, b_idx)] += float(contrib)

    return accum


def score_cwp_style(
    model: torch.nn.Module,
    tokenizer,
    safety_prompts: List[str],
    general_prompts: List[str],
    *,
    refusal_template: str = "I'm sorry, but I can't help with that.",
    block_size: int = 64,
    beta: float = 1.0,
    max_length: int = 2048,
    batch_size: int = 1,
    device=None,
) -> pd.DataFrame:
    """
    CWP-style dual-Fisher scoring.

    score(b) = I_safe(b) - beta * I_general(b)

    where I_safe is the diagonal Fisher information computed on safety prompts
    (refusal supervision) and I_general is the diagonal Fisher information
    computed on general prompts (also under refusal supervision to keep the
    supervision mechanism consistent).

    Parameters
    ----------
    model : torch.nn.Module
    tokenizer : PreTrainedTokenizer
    safety_prompts : list of str
        Prompts related to safety/refusal behavior.
    general_prompts : list of str
        Prompts for general-capability text (used to penalize protecting
        weights that are also important for general tasks).
    refusal_template : str
    block_size : int
    beta : float
        Weight for the general Fisher penalty.
    max_length : int
    batch_size : int
    device : optional

    Returns
    -------
    pd.DataFrame
    """
    device = _resolve_device(model, device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(True)

    linear_modules = list(iter_linear_modules(model))
    if not linear_modules:
        raise RuntimeError("No torch.nn.Linear modules found.")

    meta, score_accum = _init_meta_and_accum(linear_modules, block_size)

    # Compute I_safe
    fisher_safe = _accumulate_fisher_diagonal(
        model,
        tokenizer,
        safety_prompts,
        refusal_template=refusal_template,
        linear_modules=linear_modules,
        block_size=block_size,
        max_length=max_length,
        batch_size=batch_size,
        device=device,
        desc="CWP safety Fisher",
    )

    # Compute I_general
    fisher_general = _accumulate_fisher_diagonal(
        model,
        tokenizer,
        general_prompts,
        refusal_template=refusal_template,
        linear_modules=linear_modules,
        block_size=block_size,
        max_length=max_length,
        batch_size=batch_size,
        device=device,
        desc="CWP general Fisher",
    )

    # Combine: score = I_safe - beta * I_general
    for key in score_accum:
        i_safe = fisher_safe.get(key, 0.0)
        i_general = fisher_general.get(key, 0.0)
        score_accum[key] = i_safe - beta * i_general

    return _build_score_dataframe(meta, score_accum)
