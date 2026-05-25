"""
Statistical comparison helpers for safety / utility evaluations.

The legacy ``confidence_intervals.json`` workflow reported one bootstrap CI per
condition. That is conservative and frequently misleads readers into thinking
two conditions differ when their per-prompt outcomes are highly correlated, and
vice-versa. The paper instead needs:

* paired bootstrap on the per-prompt difference (the same prompts are scored
  under every condition);
* McNemar's test for paired binary outcomes (refusal yes/no);
* Holm--Bonferroni correction for multi-method comparison families;
* effect-size CIs reported alongside p-values.

All routines here are NumPy-only and have no torch / GPU dependency.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from math import isfinite
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class RateCI:
    """Bootstrap CI for a single rate (mean of a 0/1 vector)."""

    rate: float
    ci_low: float
    ci_high: float
    se: float
    n: int
    n_bootstrap: int
    confidence: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PairedDiffCI:
    """Bootstrap CI for the paired difference ``mean(A) - mean(B)``."""

    diff: float
    ci_low: float
    ci_high: float
    se: float
    n: int
    n_bootstrap: int
    confidence: float
    p_value_perm: Optional[float] = None  # two-sided permutation p-value

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class McNemarResult:
    """Result of an exact McNemar's test on paired binary outcomes."""

    b: int  # A=1, B=0  (A "wins")
    c: int  # A=0, B=1  (B "wins")
    n_disagree: int
    statistic: float
    p_value: float
    test: str  # "exact_binomial" or "asymptotic_chi2"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Single-rate bootstrap CI (replaces the legacy unpaired CI)
# ---------------------------------------------------------------------------

def bootstrap_rate_ci(
    outcomes: Sequence[float],
    *,
    n_bootstrap: int = 10000,
    confidence: float = 0.95,
    seed: int = 0,
) -> RateCI:
    """Percentile bootstrap CI for ``mean(outcomes)``.

    ``outcomes`` is typically a 0/1 vector (refusal yes/no), one entry per
    prompt. Works for any bounded numeric vector.
    """
    arr = np.asarray(outcomes, dtype=np.float64)
    if arr.size == 0:
        return RateCI(
            rate=float("nan"),
            ci_low=float("nan"),
            ci_high=float("nan"),
            se=float("nan"),
            n=0,
            n_bootstrap=n_bootstrap,
            confidence=confidence,
        )

    rng = np.random.default_rng(seed)
    n = arr.shape[0]
    idx = rng.integers(0, n, size=(n_bootstrap, n))
    means = arr[idx].mean(axis=1)

    alpha = 1.0 - confidence
    low, high = np.quantile(means, [alpha / 2.0, 1.0 - alpha / 2.0])
    return RateCI(
        rate=float(arr.mean()),
        ci_low=float(low),
        ci_high=float(high),
        se=float(means.std(ddof=1)) if n_bootstrap > 1 else 0.0,
        n=int(n),
        n_bootstrap=int(n_bootstrap),
        confidence=float(confidence),
    )


# ---------------------------------------------------------------------------
# Paired-bootstrap CI on the difference of means
# ---------------------------------------------------------------------------

def paired_bootstrap_diff(
    outcomes_a: Sequence[float],
    outcomes_b: Sequence[float],
    *,
    n_bootstrap: int = 10000,
    confidence: float = 0.95,
    seed: int = 0,
    permutation_p: bool = True,
) -> PairedDiffCI:
    """Paired percentile bootstrap CI on ``mean(A) - mean(B)``.

    Both inputs must be aligned by prompt: ``outcomes_a[i]`` and
    ``outcomes_b[i]`` are the outcomes for the same prompt under conditions A
    and B respectively. This is much more powerful than CI overlap on the raw
    rates when per-prompt outcomes are correlated across conditions, which is
    almost always the case when comparing different protection methods applied
    to the same model on the same eval set.

    When ``permutation_p`` is True, also compute a two-sided sign-flip
    permutation p-value on the paired difference.
    """
    a = np.asarray(outcomes_a, dtype=np.float64)
    b = np.asarray(outcomes_b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(
            f"paired_bootstrap_diff: shapes mismatch {a.shape} vs {b.shape}"
        )
    if a.size == 0:
        return PairedDiffCI(
            diff=float("nan"),
            ci_low=float("nan"),
            ci_high=float("nan"),
            se=float("nan"),
            n=0,
            n_bootstrap=n_bootstrap,
            confidence=confidence,
            p_value_perm=None,
        )

    d = a - b
    n = d.shape[0]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_bootstrap, n))
    boot = d[idx].mean(axis=1)
    alpha = 1.0 - confidence
    low, high = np.quantile(boot, [alpha / 2.0, 1.0 - alpha / 2.0])

    p_perm: Optional[float] = None
    if permutation_p:
        signs = rng.choice([-1.0, 1.0], size=(n_bootstrap, n))
        perm_means = (signs * d).mean(axis=1)
        observed = float(d.mean())
        # Two-sided
        p_perm = float((np.abs(perm_means) >= abs(observed)).mean())

    return PairedDiffCI(
        diff=float(d.mean()),
        ci_low=float(low),
        ci_high=float(high),
        se=float(boot.std(ddof=1)) if n_bootstrap > 1 else 0.0,
        n=int(n),
        n_bootstrap=int(n_bootstrap),
        confidence=float(confidence),
        p_value_perm=p_perm,
    )


# ---------------------------------------------------------------------------
# McNemar's test for paired binary outcomes
# ---------------------------------------------------------------------------

def mcnemar_paired(
    outcomes_a: Sequence[float],
    outcomes_b: Sequence[float],
    *,
    use_exact_threshold: int = 25,
) -> McNemarResult:
    """McNemar's test on paired 0/1 outcomes.

    Tests H0: P(A=1, B=0) = P(A=0, B=1). Useful for refusal-yes / refusal-no
    comparisons across two conditions on the same prompt set.

    For ``n_disagree <= use_exact_threshold`` we use the exact two-sided
    binomial test; otherwise we use the asymptotic chi-square approximation
    with continuity correction.
    """
    a = np.asarray(outcomes_a, dtype=np.int64)
    b = np.asarray(outcomes_b, dtype=np.int64)
    if a.shape != b.shape:
        raise ValueError(f"mcnemar_paired: shapes mismatch {a.shape} vs {b.shape}")
    if not (np.isin(a, [0, 1]).all() and np.isin(b, [0, 1]).all()):
        raise ValueError("mcnemar_paired requires 0/1 outcome vectors")

    b_only_a = int(((a == 1) & (b == 0)).sum())
    c_only_b = int(((a == 0) & (b == 1)).sum())
    n_dis = b_only_a + c_only_b

    if n_dis == 0:
        return McNemarResult(
            b=b_only_a,
            c=c_only_b,
            n_disagree=0,
            statistic=0.0,
            p_value=1.0,
            test="exact_binomial",
        )

    if n_dis <= use_exact_threshold:
        # Two-sided exact binomial p-value at theta=0.5
        k = min(b_only_a, c_only_b)
        # P(X <= k) under Binomial(n_dis, 0.5)
        from math import comb

        tail = sum(comb(n_dis, i) for i in range(k + 1)) / (2 ** n_dis)
        p = min(1.0, 2.0 * tail)
        stat = float(k)
        test = "exact_binomial"
    else:
        # Chi-square with continuity correction
        stat = (abs(b_only_a - c_only_b) - 1.0) ** 2 / float(n_dis)
        # Upper tail of chi2_1
        # Use survival function via erfc to avoid a scipy dependency.
        # chi2_1 SF(x) = erfc(sqrt(x/2))
        from math import erfc, sqrt

        p = erfc(sqrt(stat / 2.0))
        test = "asymptotic_chi2"

    return McNemarResult(
        b=b_only_a,
        c=c_only_b,
        n_disagree=n_dis,
        statistic=float(stat),
        p_value=float(p),
        test=test,
    )


# ---------------------------------------------------------------------------
# Multiple-comparison correction
# ---------------------------------------------------------------------------

def holm_bonferroni(
    p_values: Sequence[float],
    *,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Holm--Bonferroni step-down correction.

    Returns a dict with ``adjusted_p`` (same order as inputs), the per-test
    reject decision at level *alpha*, and the family-wise alpha.

    Adjusted p-values clip to [0, 1] and are monotone in input rank.
    """
    p_arr = np.asarray(p_values, dtype=np.float64)
    if p_arr.size == 0:
        return {"adjusted_p": [], "reject": [], "alpha": float(alpha)}

    if not np.all((p_arr >= 0.0) & (p_arr <= 1.0)):
        raise ValueError("holm_bonferroni: p-values must be in [0, 1]")

    n = p_arr.size
    order = np.argsort(p_arr, kind="stable")
    ranks = np.empty_like(order)
    ranks[order] = np.arange(n)

    # Sorted adjusted = max running of (n - rank) * p
    sorted_p = p_arr[order]
    weights = (n - np.arange(n)).astype(np.float64)
    sorted_adj = np.maximum.accumulate(np.minimum(sorted_p * weights, 1.0))

    adj = sorted_adj[ranks]
    reject = adj < alpha
    return {
        "adjusted_p": adj.tolist(),
        "reject": reject.tolist(),
        "alpha": float(alpha),
    }


# ---------------------------------------------------------------------------
# Aggregation helpers for multi-seed runs
# ---------------------------------------------------------------------------

@dataclass
class SeedAggregate:
    """Mean / std / SE / 95% CI across seeds for a scalar metric."""

    mean: float
    std: float
    se: float
    ci_low: float
    ci_high: float
    n_seeds: int
    values: List[float]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def aggregate_across_seeds(values: Sequence[float], *, confidence: float = 0.95) -> SeedAggregate:
    """Mean +/- t-CI across seeds. Uses a small-sample t multiplier when n<30.

    With 3-5 seeds we use the t distribution explicitly to avoid claiming
    spuriously tight intervals.
    """
    vals = [float(v) for v in values if v is not None and isfinite(float(v))]
    n = len(vals)
    if n == 0:
        return SeedAggregate(
            mean=float("nan"),
            std=float("nan"),
            se=float("nan"),
            ci_low=float("nan"),
            ci_high=float("nan"),
            n_seeds=0,
            values=[],
        )
    arr = np.asarray(vals, dtype=np.float64)
    mean = float(arr.mean())
    if n == 1:
        return SeedAggregate(
            mean=mean,
            std=0.0,
            se=0.0,
            ci_low=mean,
            ci_high=mean,
            n_seeds=1,
            values=vals,
        )
    std = float(arr.std(ddof=1))
    se = std / (n ** 0.5)
    # Two-sided t critical for confidence level, df = n-1.
    # Hard-coded common values to avoid a scipy dependency.
    _T_CRIT_95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
                  7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 15: 2.131, 20: 2.086,
                  30: 2.042, 60: 2.000, 120: 1.980}
    if abs(confidence - 0.95) > 1e-6:
        # Fall back to z=1.96 for non-95% confidence; documents the assumption.
        t_crit = 1.959963984540054
    else:
        df = n - 1
        t_crit = _T_CRIT_95.get(df, 1.96 + 2.5 / max(df, 1))
    half = t_crit * se
    return SeedAggregate(
        mean=mean,
        std=std,
        se=se,
        ci_low=mean - half,
        ci_high=mean + half,
        n_seeds=n,
        values=vals,
    )


# ---------------------------------------------------------------------------
# Convenience: run a family of paired comparisons with Holm correction
# ---------------------------------------------------------------------------

def compare_methods_paired(
    outcomes_by_method: Mapping[str, Sequence[float]],
    *,
    reference: str,
    n_bootstrap: int = 10000,
    confidence: float = 0.95,
    seed: int = 0,
    alpha: float = 0.05,
) -> Dict[str, Dict[str, Any]]:
    """Compare every method against *reference* with paired bootstrap +
    Holm-adjusted McNemar p-values.

    Inputs are aligned per-prompt outcome vectors per method (0/1 typically).
    Returns ``{method: {diff_ci, mcnemar, holm_p, reject}}``.

    The reference itself is included in the result with ``diff=0`` for symmetry,
    but excluded from the multiple-testing family.
    """
    if reference not in outcomes_by_method:
        raise ValueError(f"reference '{reference}' not in methods: {list(outcomes_by_method)}")

    ref = outcomes_by_method[reference]
    comparisons = [m for m in outcomes_by_method if m != reference]

    raw_p: List[float] = []
    diff_cis: Dict[str, PairedDiffCI] = {}
    mcnemars: Dict[str, McNemarResult] = {}
    for m in comparisons:
        d = paired_bootstrap_diff(
            outcomes_by_method[m], ref,
            n_bootstrap=n_bootstrap, confidence=confidence, seed=seed,
        )
        diff_cis[m] = d
        mn = mcnemar_paired(outcomes_by_method[m], ref)
        mcnemars[m] = mn
        raw_p.append(mn.p_value)

    correction = holm_bonferroni(raw_p, alpha=alpha)

    out: Dict[str, Dict[str, Any]] = {
        reference: {
            "reference": True,
            "diff_ci": PairedDiffCI(
                diff=0.0, ci_low=0.0, ci_high=0.0, se=0.0,
                n=len(ref), n_bootstrap=n_bootstrap, confidence=confidence,
                p_value_perm=None,
            ).to_dict(),
            "mcnemar": None,
            "holm_p": None,
            "reject": False,
        }
    }
    for i, m in enumerate(comparisons):
        out[m] = {
            "reference": False,
            "diff_ci": diff_cis[m].to_dict(),
            "mcnemar": mcnemars[m].to_dict(),
            "holm_p": float(correction["adjusted_p"][i]),
            "reject": bool(correction["reject"][i]),
        }
    return out
