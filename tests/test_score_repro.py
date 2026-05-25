"""
Reproducibility tests that do not require loading a real LLM.

These tests cover the deterministic / CPU-side pieces of the scoring +
selection pipeline:

* the symmetric group-wise quantize-dequantize utility is deterministic and
  yields a quantization error that is bounded by the group-wise scale;
* :func:`select_top_blocks` is a pure function of the score DataFrame plus the
  budget;
* the fractional-bit-width mixed assignment in
  :func:`safepress.experiments.phase_transition._simulate_quantize_inplace`
  correctly partitions transformer layers between the two adjacent integer
  bit-widths.

These are intentionally fast (<1 s total) and do not download any models.
"""
from __future__ import annotations

import pandas as pd
import pytest
import torch
import torch.nn as nn

from safepress.model.protect import select_top_blocks
from safepress.model.score import _quant_dequant_symmetric_groupwise


# ---------------------------------------------------------------------------
# Quant / dequant determinism
# ---------------------------------------------------------------------------

def test_quant_dequant_is_deterministic():
    torch.manual_seed(0)
    w = torch.randn(64, 256, dtype=torch.float32)
    a = _quant_dequant_symmetric_groupwise(w, bits=4, group_size=64)
    b = _quant_dequant_symmetric_groupwise(w, bits=4, group_size=64)
    assert torch.equal(a, b)


def test_quant_dequant_error_bounded_by_scale():
    torch.manual_seed(1)
    w = torch.randn(32, 128, dtype=torch.float32) * 2.0
    bits = 4
    group_size = 64
    w_hat = _quant_dequant_symmetric_groupwise(w, bits=bits, group_size=group_size)

    qmax = (2 ** (bits - 1)) - 1
    # The per-group scale is max_abs / qmax; the absolute round-off error per
    # element should therefore be <= scale / 2.
    err = (w_hat - w).abs()
    n_groups = w.shape[1] // group_size
    w_view = w.view(w.shape[0], n_groups, group_size)
    max_abs = w_view.abs().amax(dim=2, keepdim=True)
    scale = max_abs / float(qmax)
    upper = (scale * 0.51).expand_as(w_view).reshape_as(w)  # 0.51 to allow tie-breaks
    assert (err <= upper).all()


def test_quant_dequant_rejects_bad_bits():
    w = torch.randn(4, 8)
    with pytest.raises(ValueError):
        _quant_dequant_symmetric_groupwise(w, bits=1, group_size=8)
    with pytest.raises(ValueError):
        _quant_dequant_symmetric_groupwise(w, bits=9, group_size=8)


# ---------------------------------------------------------------------------
# select_top_blocks is a deterministic pure function
# ---------------------------------------------------------------------------

def _toy_scores() -> pd.DataFrame:
    rows = []
    score = 0.0
    for mod in ("layer0.attn.q_proj", "layer0.attn.k_proj", "layer1.mlp.gate_proj"):
        for b in range(4):
            score += 1.0
            rows.append({"module": mod, "block_idx": b, "num_params": 128, "score": score})
    return pd.DataFrame(rows)


def test_select_top_blocks_is_pure():
    df = _toy_scores()
    plan_a = select_top_blocks(df, budget_ratio=0.25, block_size=64)
    plan_b = select_top_blocks(df.copy(), budget_ratio=0.25, block_size=64)
    assert plan_a.protect_map == plan_b.protect_map
    assert plan_a.protected_params == plan_b.protected_params


def test_select_top_blocks_respects_budget():
    df = _toy_scores()
    total = int(df["num_params"].sum())
    plan = select_top_blocks(df, budget_ratio=0.5, block_size=64)
    assert plan.protected_params <= int(total * 0.5)
    # And we should have selected at least one block.
    assert plan.protected_params > 0


def test_select_top_blocks_picks_highest_score_first():
    df = _toy_scores()
    plan = select_top_blocks(df, budget_ratio=0.10, block_size=64)
    # Highest-score block was layer1.mlp.gate_proj idx=3
    assert "layer1.mlp.gate_proj" in plan.protect_map
    assert 3 in plan.protect_map["layer1.mlp.gate_proj"]


def test_select_top_blocks_rejects_bad_budget():
    df = _toy_scores()
    with pytest.raises(ValueError):
        select_top_blocks(df, budget_ratio=1.5, block_size=64)
    with pytest.raises(ValueError):
        select_top_blocks(df, budget_ratio=0.0, block_size=64)


# ---------------------------------------------------------------------------
# Fractional bit-width assignment (mixed precision via layer index)
# ---------------------------------------------------------------------------

class _ToyTransformer(nn.Module):
    """Minimal stand-in: model.layers.{0..7}.proj. iter_linear_modules walks
    every nn.Linear by attribute name, so this is enough."""

    def __init__(self, n_layers: int = 8, in_f: int = 64, out_f: int = 64):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.ModuleDict({"proj": nn.Linear(in_f, out_f, bias=False)})
                for _ in range(n_layers)
            ]
        )


def test_fractional_bits_partitions_by_layer():
    from safepress.experiments.phase_transition import _simulate_quantize_inplace
    from safepress.model.blocks import iter_linear_modules

    torch.manual_seed(0)
    model = _ToyTransformer(n_layers=8)

    # Snapshot original weights and a 4-bit / 3-bit reference per module.
    originals = {}
    refs_4bit = {}
    refs_3bit = {}
    for name, mod in iter_linear_modules(model):
        originals[name] = mod.weight.detach().clone()
        refs_4bit[name] = _quant_dequant_symmetric_groupwise(
            mod.weight, bits=4, group_size=64,
        ).clone()
        refs_3bit[name] = _quant_dequant_symmetric_groupwise(
            mod.weight, bits=3, group_size=64,
        ).clone()

    # 3.5-bit: first 4 of 8 layers should get 4-bit, last 4 should get 3-bit.
    _simulate_quantize_inplace(model, bits=3.5, group_size=64)

    for name, mod in iter_linear_modules(model):
        # The toy model puts the layer index in ".layers.{i}.proj.weight"
        # iter_linear_modules emits the path; extract the index.
        import re
        m = re.search(r"\.(\d+)\.", name)
        assert m is not None
        idx = int(m.group(1))
        assigned_4bit = torch.equal(mod.weight.detach(), refs_4bit[name])
        assigned_3bit = torch.equal(mod.weight.detach(), refs_3bit[name])
        # Threshold = min_layer + 0.5 * 8 = 0 + 4 = 4, so layers 0..3 get
        # upper (4-bit), layers 4..7 get lower (3-bit).
        if idx < 4:
            assert assigned_4bit, f"layer {idx} should be 4-bit, got mismatch"
        else:
            assert assigned_3bit, f"layer {idx} should be 3-bit, got mismatch"


def test_integer_bits_uniform_assignment():
    from safepress.experiments.phase_transition import _simulate_quantize_inplace
    from safepress.model.blocks import iter_linear_modules

    torch.manual_seed(0)
    model = _ToyTransformer(n_layers=4)
    refs = {}
    for name, mod in iter_linear_modules(model):
        refs[name] = _quant_dequant_symmetric_groupwise(
            mod.weight, bits=4, group_size=64,
        ).clone()

    _simulate_quantize_inplace(model, bits=4, group_size=64)

    for name, mod in iter_linear_modules(model):
        assert torch.equal(mod.weight.detach(), refs[name])


def test_fp16_is_a_noop():
    from safepress.experiments.phase_transition import _simulate_quantize_inplace
    from safepress.model.blocks import iter_linear_modules

    torch.manual_seed(0)
    model = _ToyTransformer(n_layers=2)
    saved = {n: m.weight.detach().clone() for n, m in iter_linear_modules(model)}
    _simulate_quantize_inplace(model, bits=16, group_size=64)
    for name, mod in iter_linear_modules(model):
        assert torch.equal(mod.weight.detach(), saved[name])
