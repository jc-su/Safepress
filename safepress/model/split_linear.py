from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import torch


class SplitLinear(torch.nn.Module):
    """
    A Linear layer split into:
      - protected: FP16/FP32 torch.nn.Linear over a subset of output channels
      - quantized: a Linear over the remaining channels (intended to be later converted to 4-bit)

    Output channels are merged back to the original ordering.

    This supports inputs of shape (..., in_features).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        protected_indices: Sequence[int],
        bias: bool = True,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        prot = sorted(set(int(i) for i in protected_indices))
        if any(i < 0 or i >= out_features for i in prot):
            raise ValueError("protected_indices out of range")

        unprot = [i for i in range(out_features) if i not in set(prot)]

        self.in_features = in_features
        self.out_features = out_features

        self.register_buffer("protected_idx", torch.tensor(prot, dtype=torch.long), persistent=False)
        self.register_buffer("unprotected_idx", torch.tensor(unprot, dtype=torch.long), persistent=False)

        self.protected = torch.nn.Linear(in_features, len(prot), bias=bias, dtype=dtype, device=device)
        self.quantized = torch.nn.Linear(in_features, len(unprot), bias=bias, dtype=dtype, device=device)

    @staticmethod
    def from_linear(linear: torch.nn.Linear, protected_indices: Sequence[int]) -> "SplitLinear":
        with torch.no_grad():
            device = linear.weight.device
            dtype = linear.weight.dtype
            split = SplitLinear(
                in_features=linear.in_features,
                out_features=linear.out_features,
                protected_indices=protected_indices,
                bias=linear.bias is not None,
                dtype=dtype,
                device=device,
            )
            # Copy weights
            if split.protected_idx.numel() > 0:
                split.protected.weight.copy_(linear.weight[split.protected_idx])
                if linear.bias is not None:
                    split.protected.bias.copy_(linear.bias[split.protected_idx])
            if split.unprotected_idx.numel() > 0:
                split.quantized.weight.copy_(linear.weight[split.unprotected_idx])
                if linear.bias is not None:
                    split.quantized.bias.copy_(linear.bias[split.unprotected_idx])
        return split

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape[:-1]
        x2 = x.reshape(-1, x.shape[-1])

        y2 = x2.new_zeros((x2.shape[0], self.out_features))
        if self.protected_idx.numel() > 0:
            # Move input to same device as protected weights for multi-GPU support
            x_prot = x2.to(self.protected.weight.device)
            yp = self.protected(x_prot).to(y2.device)
            y2.index_copy_(1, self.protected_idx.to(y2.device), yp)
        if self.unprotected_idx.numel() > 0:
            # Move input to same device as quantized weights for multi-GPU support
            x_quant = x2.to(self.quantized.weight.device)
            yq = self.quantized(x_quant).to(y2.device)
            y2.index_copy_(1, self.unprotected_idx.to(y2.device), yq)

        return y2.reshape(*orig_shape, self.out_features)


def _get_parent_and_attr(root: torch.nn.Module, module_name: str) -> Tuple[torch.nn.Module, str]:
    parts = module_name.split(".")
    parent = root
    for p in parts[:-1]:
        if not hasattr(parent, p):
            raise AttributeError(f"Missing submodule '{p}' in path '{module_name}'")
        parent = getattr(parent, p)
    return parent, parts[-1]


@dataclass
class SplitReport:
    protected_modules_to_skip: List[str]
    split_modules: List[str]
    fully_protected_modules: List[str]


def apply_block_splitting(
    model: torch.nn.Module,
    protect_map: Dict[str, List[int]],
    *,
    block_size: int,
) -> SplitReport:
    """
    Replace specified torch.nn.Linear modules with SplitLinear according to protect_map.

    Returns names of modules to be *excluded* from quantization:
      - fully protected modules: <module_name>
      - protected submodules of SplitLinear: <module_name>.protected
    """
    modules_to_skip: List[str] = []
    split_modules: List[str] = []
    full_modules: List[str] = []

    for mod_name, block_idxs in protect_map.items():
        # Locate module
        try:
            parent, attr = _get_parent_and_attr(model, mod_name)
            mod = getattr(parent, attr)
        except Exception as e:
            # module name mismatch; skip
            continue

        if not isinstance(mod, torch.nn.Linear):
            # We only support splitting torch.nn.Linear in this implementation.
            continue

        out_features, in_features = mod.weight.shape
        protected_indices: List[int] = []
        for b in block_idxs:
            s = int(b) * int(block_size)
            e = min(s + int(block_size), out_features)
            protected_indices.extend(list(range(s, e)))
        protected_indices = sorted(set(protected_indices))

        if len(protected_indices) == 0:
            continue

        if len(protected_indices) >= out_features:
            # Fully protected; just skip quantization for this module.
            modules_to_skip.append(mod_name)
            full_modules.append(mod_name)
            continue

        # Replace module with split
        split = SplitLinear.from_linear(mod, protected_indices)
        setattr(parent, attr, split)
        split_modules.append(mod_name)
        modules_to_skip.append(mod_name + ".protected")

    return SplitReport(
        protected_modules_to_skip=sorted(set(modules_to_skip)),
        split_modules=sorted(split_modules),
        fully_protected_modules=sorted(full_modules),
    )
