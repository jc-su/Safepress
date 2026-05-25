from __future__ import annotations

import re
from typing import Iterable, List, Optional, Tuple

import pandas as pd
import torch


def chunk_indices(n: int, block_size: int) -> List[Tuple[int, int]]:
    """Split range [0, n) into chunks of block_size."""
    assert block_size > 0
    return [(i, min(i + block_size, n)) for i in range(0, n, block_size)]


def iter_linear_modules(model) -> Iterable[Tuple[str, torch.nn.Linear]]:
    """Yield (module_name, module) for torch.nn.Linear modules."""
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear):
            yield name, mod


def guess_layer_index(module_name: str) -> Optional[int]:
    """Extract numeric layer index from a module name.

    Common patterns:
        model.layers.12.self_attn.q_proj  -> 12
        transformer.h.0.mlp.dense_4h_to_h -> 0
    """
    match = re.search(r"\.(\d+)\.", module_name)
    if match:
        return int(match.group(1))
    return None


def compute_block_metadata(
    model: torch.nn.Module,
    *,
    block_size: int,
) -> pd.DataFrame:
    """Compute metadata (module, block_idx, num_params) for all Linear blocks."""
    rows = []
    for mod_name, mod in iter_linear_modules(model):
        out_features, in_features = mod.weight.shape
        blocks = chunk_indices(out_features, block_size)
        for b_idx, (s, e) in enumerate(blocks):
            num_params = (e - s) * in_features
            rows.append(
                dict(
                    module=mod_name,
                    block_idx=b_idx,
                    out_start=s,
                    out_end=e,
                    in_features=in_features,
                    num_params=num_params,
                )
            )
    return pd.DataFrame(rows)
