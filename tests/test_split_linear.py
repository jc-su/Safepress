import torch

from safepress.model.split_linear import SplitLinear


def test_split_linear_equivalence():
    torch.manual_seed(0)
    lin = torch.nn.Linear(32, 64, bias=True)
    x = torch.randn(4, 7, 32)

    protected = list(range(0, 64, 3))  # some indices
    split = SplitLinear.from_linear(lin, protected)

    y0 = lin(x)
    y1 = split(x)

    assert torch.allclose(y0, y1, atol=1e-6), (y0 - y1).abs().max().item()
