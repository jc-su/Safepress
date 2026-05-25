"""
Unit tests for safepress.eval.stats.

These tests are CPU-only and run in under a second. They verify that:

* ``bootstrap_rate_ci`` is reproducible under a fixed seed and returns a CI
  containing the empirical mean.
* ``paired_bootstrap_diff`` recovers a known mean difference within tight
  tolerance on a synthetic paired dataset.
* ``mcnemar_paired`` matches a hand-computed exact-binomial p-value on a small
  example, and falls back to the asymptotic test for large disagreement counts.
* ``holm_bonferroni`` is monotone in input rank, identical to Bonferroni when
  all inputs are equal, and matches a worked example.
* ``aggregate_across_seeds`` uses a t-multiplier for small samples and falls
  back to a single-value degenerate case correctly.
* ``compare_methods_paired`` wires the pieces together and produces a stable
  shape.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from safepress.eval.stats import (
    aggregate_across_seeds,
    bootstrap_rate_ci,
    compare_methods_paired,
    holm_bonferroni,
    mcnemar_paired,
    paired_bootstrap_diff,
)


# ---------------------------------------------------------------------------
# bootstrap_rate_ci
# ---------------------------------------------------------------------------

def test_bootstrap_rate_ci_reproducible():
    out = np.array([1, 0, 1, 1, 0, 1, 0, 1, 1, 0], dtype=np.float64)
    a = bootstrap_rate_ci(out, n_bootstrap=2000, seed=42)
    b = bootstrap_rate_ci(out, n_bootstrap=2000, seed=42)
    assert a.to_dict() == b.to_dict()


def test_bootstrap_rate_ci_contains_mean():
    rng = np.random.default_rng(0)
    out = rng.integers(0, 2, size=200).astype(np.float64)
    r = bootstrap_rate_ci(out, n_bootstrap=2000, seed=0)
    assert r.ci_low <= r.rate <= r.ci_high
    assert r.n == 200


def test_bootstrap_rate_ci_empty():
    r = bootstrap_rate_ci([], n_bootstrap=100, seed=0)
    assert r.n == 0
    assert math.isnan(r.rate)


# ---------------------------------------------------------------------------
# paired_bootstrap_diff
# ---------------------------------------------------------------------------

def test_paired_bootstrap_recovers_known_diff():
    rng = np.random.default_rng(0)
    # B is "harder" than A: A succeeds 80%, B 60%, with strong pairing.
    n = 500
    latent = rng.random(n)
    a = (latent > 0.20).astype(np.float64)
    b = (latent > 0.40).astype(np.float64)
    out = paired_bootstrap_diff(a, b, n_bootstrap=2000, seed=0)
    # True diff is ~0.20; bootstrap mean should be within 0.05.
    assert abs(out.diff - 0.20) < 0.05
    # CI should contain the diff.
    assert out.ci_low <= out.diff <= out.ci_high
    # Permutation p should be tiny.
    assert out.p_value_perm is not None
    assert out.p_value_perm < 0.05


def test_paired_bootstrap_zero_diff_has_p_close_to_one():
    rng = np.random.default_rng(1)
    n = 300
    base = rng.integers(0, 2, size=n).astype(np.float64)
    out = paired_bootstrap_diff(base, base.copy(), n_bootstrap=1000, seed=0)
    assert abs(out.diff) < 1e-9
    assert out.p_value_perm is None or out.p_value_perm > 0.5


def test_paired_bootstrap_shape_mismatch():
    with pytest.raises(ValueError):
        paired_bootstrap_diff([1.0, 0.0], [1.0], n_bootstrap=10, seed=0)


# ---------------------------------------------------------------------------
# mcnemar_paired
# ---------------------------------------------------------------------------

def test_mcnemar_exact_small_disagreement():
    # b=8, c=2; under H0 with n_dis=10, P(X<=2) on Bin(10, 0.5) * 2.
    # 2 * (C(10,0)+C(10,1)+C(10,2)) / 1024 = 2 * 56 / 1024 = 0.109375
    a = np.array([1] * 8 + [0] * 2 + [1] * 5 + [0] * 5)
    b = np.array([0] * 8 + [1] * 2 + [1] * 5 + [0] * 5)
    out = mcnemar_paired(a, b)
    assert out.test == "exact_binomial"
    assert out.b == 8 and out.c == 2 and out.n_disagree == 10
    assert abs(out.p_value - 0.109375) < 1e-9


def test_mcnemar_asymptotic_large_disagreement():
    # b=50, c=10, n_dis=60.  Chi^2 = (|50-10|-1)^2 / 60 = 39^2/60 = 25.35.
    # SF of chi^2_1 at 25.35 is on the order of 4.79e-7.
    n = 60
    a = np.concatenate([np.ones(50), np.zeros(10), np.ones(40), np.zeros(40)])
    b = np.concatenate([np.zeros(50), np.ones(10), np.ones(40), np.zeros(40)])
    out = mcnemar_paired(a, b, use_exact_threshold=25)
    assert out.test == "asymptotic_chi2"
    assert out.b == 50 and out.c == 10 and out.n_disagree == n
    assert out.p_value < 1e-5


def test_mcnemar_no_disagreement():
    a = np.array([1, 0, 1, 1, 0])
    b = np.array([1, 0, 1, 1, 0])
    out = mcnemar_paired(a, b)
    assert out.n_disagree == 0
    assert out.p_value == 1.0


def test_mcnemar_rejects_non_binary():
    with pytest.raises(ValueError):
        mcnemar_paired([0, 1, 2], [0, 0, 1])


# ---------------------------------------------------------------------------
# holm_bonferroni
# ---------------------------------------------------------------------------

def test_holm_basic_example():
    # 4 tests with p = [0.01, 0.04, 0.03, 0.005].
    # Ranks (ascending): 0.005 -> w=4 -> 0.02
    #                    0.01  -> w=3 -> 0.03 (max with 0.02 = 0.03)
    #                    0.03  -> w=2 -> 0.06
    #                    0.04  -> w=1 -> 0.04 (max with 0.06 = 0.06)
    out = holm_bonferroni([0.01, 0.04, 0.03, 0.005], alpha=0.05)
    adj = out["adjusted_p"]
    assert abs(adj[3] - 0.02) < 1e-9   # smallest input -> 4 * 0.005
    assert abs(adj[0] - 0.03) < 1e-9
    assert abs(adj[2] - 0.06) < 1e-9
    assert abs(adj[1] - 0.06) < 1e-9   # capped to running max


def test_holm_all_equal_matches_bonferroni():
    out = holm_bonferroni([0.02, 0.02, 0.02, 0.02], alpha=0.05)
    for a in out["adjusted_p"]:
        # n * 0.02 = 0.08, capped at 1
        assert abs(a - 0.08) < 1e-9


def test_holm_caps_at_one():
    out = holm_bonferroni([0.9, 0.9, 0.9], alpha=0.05)
    for a in out["adjusted_p"]:
        assert a <= 1.0


def test_holm_rejects_invalid_p():
    with pytest.raises(ValueError):
        holm_bonferroni([0.5, 1.2], alpha=0.05)


# ---------------------------------------------------------------------------
# aggregate_across_seeds
# ---------------------------------------------------------------------------

def test_aggregate_single_seed_is_degenerate():
    out = aggregate_across_seeds([0.7])
    assert out.n_seeds == 1
    assert out.std == 0.0
    assert out.ci_low == out.ci_high == out.mean == 0.7


def test_aggregate_three_seeds_uses_t():
    out = aggregate_across_seeds([0.65, 0.70, 0.75])
    assert out.n_seeds == 3
    # std = 0.05, se = 0.05/sqrt(3) ~= 0.02887
    # t_0.975, df=2 = 4.303 -> half-width ~= 0.1242
    assert abs(out.mean - 0.70) < 1e-9
    assert abs(out.std - 0.05) < 1e-6
    assert abs((out.ci_high - out.ci_low) / 2.0 - 4.303 * out.se) < 1e-3


def test_aggregate_drops_nan():
    out = aggregate_across_seeds([0.5, float("nan"), 0.6])
    assert out.n_seeds == 2


# ---------------------------------------------------------------------------
# compare_methods_paired
# ---------------------------------------------------------------------------

def test_compare_methods_paired_shape_and_reference():
    rng = np.random.default_rng(0)
    n = 200
    latent = rng.random(n)
    methods = {
        "ssmp": (latent > 0.10).astype(np.float64),
        "fisher": (latent > 0.30).astype(np.float64),
        "magnitude": (latent > 0.50).astype(np.float64),
        "ref_quant": (latent > 0.70).astype(np.float64),
    }
    out = compare_methods_paired(
        methods, reference="ref_quant", n_bootstrap=500, seed=0,
    )
    assert set(out.keys()) == set(methods.keys())
    assert out["ref_quant"]["reference"] is True
    assert out["ssmp"]["diff_ci"]["diff"] > 0
    # ssmp should clearly beat ref_quant after Holm correction.
    assert out["ssmp"]["reject"] is True


def test_compare_methods_paired_unknown_reference_raises():
    with pytest.raises(ValueError):
        compare_methods_paired({"a": [0, 1], "b": [0, 0]}, reference="zzz")
