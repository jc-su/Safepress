"""
Refusal direction extraction and analysis.

Based on the observation (Arditi et al.) that refusal behavior in LLMs is mediated
by a low-dimensional direction in the residual stream. We extract this direction
and measure how quantization affects the refusal signal.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd
import torch
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Activation extraction
# ---------------------------------------------------------------------------


@torch.no_grad()
def extract_residual_activations(
    model: torch.nn.Module,
    tokenizer,
    prompts: List[str],
    *,
    max_length: int = 512,
    device: Optional[torch.device | str] = None,
) -> Dict[int, torch.Tensor]:
    """
    Hook into each transformer layer's output (residual stream) and collect
    the last-token activation at every layer for each prompt.

    Returns
    -------
    Dict mapping ``layer_index`` -> tensor of shape ``(n_prompts, hidden_dim)``.
    """
    if device is None:
        try:
            device = next(p.device for p in model.parameters() if p.is_floating_point())
        except StopIteration:
            device = torch.device("cpu")
    device = torch.device(device)

    model.eval()

    # Discover transformer layers.  Common attribute names used by HF models:
    #   model.model.layers  (LLaMA / Mistral / Qwen)
    #   model.transformer.h (GPT-2 / GPT-Neo)
    layers = _get_transformer_layers(model)
    n_layers = len(layers)

    # Storage: layer_index -> list of (hidden_dim,) tensors, one per prompt
    collected: Dict[int, List[torch.Tensor]] = {i: [] for i in range(n_layers)}

    # Register forward hooks on every layer
    hooks = []
    _activation_store: Dict[int, torch.Tensor] = {}

    def _make_hook(layer_idx: int):
        def _hook(module, input, output):
            # `output` may be a tuple (hidden_states, ...) or bare tensor.
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output
            # hidden: (batch=1, seq_len, hidden_dim) -- store the full tensor
            _activation_store[layer_idx] = hidden.detach()
        return _hook

    for idx, layer in enumerate(layers):
        hooks.append(layer.register_forward_hook(_make_hook(idx)))

    try:
        for prompt in tqdm(prompts, desc="Extracting activations", leave=False):
            enc = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
            ).to(device)

            seq_len = enc["input_ids"].shape[1]

            _activation_store.clear()
            model(**enc)

            for layer_idx in range(n_layers):
                if layer_idx not in _activation_store:
                    continue
                hidden = _activation_store[layer_idx]  # (1, seq_len, hidden_dim)
                # Take the last *real* token (accounting for potential padding).
                # Since we process one prompt at a time with truncation, the
                # last position is seq_len - 1.
                last_act = hidden[0, seq_len - 1, :].float().cpu()
                collected[layer_idx].append(last_act)
    finally:
        for h in hooks:
            h.remove()

    # Stack into tensors
    result: Dict[int, torch.Tensor] = {}
    for layer_idx, acts in collected.items():
        if acts:
            result[layer_idx] = torch.stack(acts, dim=0)  # (n_prompts, hidden_dim)

    return result


# ---------------------------------------------------------------------------
# Refusal direction computation
# ---------------------------------------------------------------------------


def compute_refusal_direction(
    harmful_acts: Dict[int, torch.Tensor],
    harmless_acts: Dict[int, torch.Tensor],
    method: str = "mean_diff",
) -> Dict[int, torch.Tensor]:
    """
    Compute the *refusal direction* per layer.

    Parameters
    ----------
    harmful_acts : dict
        layer_index -> (n_harmful, hidden_dim) activations from harmful prompts.
    harmless_acts : dict
        layer_index -> (n_harmless, hidden_dim) activations from harmless prompts.
    method : str
        ``"mean_diff"`` -- direction = normalize(mean(harmful) - mean(harmless))
        ``"pca"`` -- PCA on concatenated activations with signed labels, first component.

    Returns
    -------
    Dict mapping layer_index -> unit direction vector of shape ``(hidden_dim,)``.
    """
    if method not in ("mean_diff", "pca"):
        raise ValueError(f"Unknown method: {method!r}. Expected 'mean_diff' or 'pca'.")

    layer_indices = sorted(set(harmful_acts.keys()) & set(harmless_acts.keys()))
    directions: Dict[int, torch.Tensor] = {}

    for layer_idx in layer_indices:
        h = harmful_acts[layer_idx].float()   # (n_harmful, d)
        s = harmless_acts[layer_idx].float()   # (n_harmless, d)

        if method == "mean_diff":
            direction = h.mean(dim=0) - s.mean(dim=0)
        else:
            # PCA approach: center the concatenated data, take first principal
            # component.  The first PC of data with two clusters typically
            # aligns with the separation direction.
            combined = torch.cat([h, s], dim=0)  # (n_total, d)
            mean = combined.mean(dim=0, keepdim=True)
            centered = combined - mean
            # Economy SVD -- we only need the first singular vector.
            # torch.linalg.svd with full_matrices=False is efficient.
            _U, _S, Vt = torch.linalg.svd(centered, full_matrices=False)
            direction = Vt[0]  # first right singular vector (d,)
            # Ensure consistent sign: positive dot product with mean_diff
            mean_diff = h.mean(dim=0) - s.mean(dim=0)
            if torch.dot(direction, mean_diff) < 0:
                direction = -direction

        # Normalize to unit vector
        norm = direction.norm()
        if norm > 0:
            direction = direction / norm

        directions[layer_idx] = direction

    return directions


# ---------------------------------------------------------------------------
# Separation measurement
# ---------------------------------------------------------------------------


def measure_separation(activations: torch.Tensor, direction: torch.Tensor) -> float:
    """
    Project *activations* onto *direction* and return the mean projection value.

    Parameters
    ----------
    activations : Tensor of shape (n, hidden_dim)
    direction : Tensor of shape (hidden_dim,)

    Returns
    -------
    float -- mean scalar projection.
    """
    activations = activations.float()
    direction = direction.float()
    projections = activations @ direction  # (n,)
    return float(projections.mean().item())


# ---------------------------------------------------------------------------
# Full refusal-signal profile
# ---------------------------------------------------------------------------


@torch.no_grad()
def refusal_signal_profile(
    model: torch.nn.Module,
    tokenizer,
    harmful_prompts: List[str],
    harmless_prompts: List[str],
    *,
    max_length: int = 512,
    device: Optional[torch.device | str] = None,
) -> pd.DataFrame:
    """
    Build a per-layer refusal signal profile for a single model.

    For each layer the function:
      1. Extracts last-token residual activations for harmful & harmless prompts.
      2. Computes the refusal direction (mean-diff).
      3. Measures the projection of harmful / harmless activations onto that
         direction.

    Returns
    -------
    DataFrame with columns:
        layer, harmful_projection, harmless_projection, separation, direction_norm
    """
    harmful_acts = extract_residual_activations(
        model, tokenizer, harmful_prompts,
        max_length=max_length, device=device,
    )
    harmless_acts = extract_residual_activations(
        model, tokenizer, harmless_prompts,
        max_length=max_length, device=device,
    )

    # Compute refusal direction using raw (un-normalized) mean diff to also
    # report the norm of the direction vector before normalization.
    layer_indices = sorted(set(harmful_acts.keys()) & set(harmless_acts.keys()))

    rows = []
    for layer_idx in layer_indices:
        h = harmful_acts[layer_idx].float()
        s = harmless_acts[layer_idx].float()

        raw_direction = h.mean(dim=0) - s.mean(dim=0)
        direction_norm = float(raw_direction.norm().item())

        # Normalize
        norm = raw_direction.norm()
        if norm > 0:
            unit_dir = raw_direction / norm
        else:
            unit_dir = raw_direction

        harmful_proj = measure_separation(h, unit_dir)
        harmless_proj = measure_separation(s, unit_dir)

        rows.append(
            dict(
                layer=layer_idx,
                harmful_projection=harmful_proj,
                harmless_projection=harmless_proj,
                separation=harmful_proj - harmless_proj,
                direction_norm=direction_norm,
            )
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Compare two profiles (FP16 vs quantized)
# ---------------------------------------------------------------------------


def compare_refusal_signal(
    profile_fp16: pd.DataFrame,
    profile_quantized: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compare refusal signal profiles from an FP16 model and a quantized model.

    Computes per-layer degradation metrics.

    Returns
    -------
    DataFrame with columns from both profiles plus:
        separation_drop  -- fp16_separation - quant_separation  (positive = degradation)
        relative_drop    -- separation_drop / |fp16_separation|  (if fp16 != 0)
    """
    fp = profile_fp16.rename(
        columns={
            "harmful_projection": "fp16_harmful_projection",
            "harmless_projection": "fp16_harmless_projection",
            "separation": "fp16_separation",
            "direction_norm": "fp16_direction_norm",
        }
    )
    qt = profile_quantized.rename(
        columns={
            "harmful_projection": "quant_harmful_projection",
            "harmless_projection": "quant_harmless_projection",
            "separation": "quant_separation",
            "direction_norm": "quant_direction_norm",
        }
    )

    merged = pd.merge(fp, qt, on="layer", how="inner")

    merged["separation_drop"] = merged["fp16_separation"] - merged["quant_separation"]

    # Relative drop (guard against division by zero)
    merged["relative_drop"] = merged.apply(
        lambda r: (
            r["separation_drop"] / abs(r["fp16_separation"])
            if abs(r["fp16_separation"]) > 1e-12
            else 0.0
        ),
        axis=1,
    )

    return merged


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_transformer_layers(model: torch.nn.Module) -> torch.nn.ModuleList:
    """
    Return the ModuleList of transformer blocks from a HuggingFace causal LM.

    Supports common architectures:
      - LLaMA / Mistral / Qwen:  model.model.layers
      - GPT-2 / GPT-Neo:         model.transformer.h
      - Phi:                      model.model.layers
      - Gemma:                    model.model.layers
      - Falcon:                   model.transformer.h
    """
    # Try common paths
    candidates = [
        ("model", "layers"),
        ("transformer", "h"),
        ("gpt_neox", "layers"),
    ]
    for container_attr, layers_attr in candidates:
        container = getattr(model, container_attr, None)
        if container is not None:
            layers = getattr(container, layers_attr, None)
            if layers is not None and isinstance(layers, torch.nn.ModuleList):
                return layers

    # Fallback: walk the module tree looking for a ModuleList whose children
    # look like transformer blocks (i.e., contain "self_attn" or "attention").
    for _name, mod in model.named_modules():
        if isinstance(mod, torch.nn.ModuleList) and len(mod) > 0:
            child = mod[0]
            child_names = {n for n, _ in child.named_modules()}
            if any(k in child_names for k in ("self_attn", "attention", "attn")):
                return mod

    raise RuntimeError(
        "Could not locate transformer layers. "
        "Ensure the model follows a standard HuggingFace architecture."
    )
