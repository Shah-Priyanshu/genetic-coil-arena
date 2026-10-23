"""Statistics for comparing conditions across independent runs.

Every experiment reports a mean with a Student-t confidence interval, plus the
interquartile mean (IQM) with a stratified bootstrap confidence interval, following
Agarwal et al. 2021 ("Deep RL at the Edge of the Statistical Precipice"). Conditions
are compared with Mann-Whitney U tests, Holm-corrected for multiple comparisons, and
the Vargha-Delaney A12 effect size.
"""

import itertools
from collections.abc import Iterable, Mapping, Sequence
from typing import (
    Any,
    Callable,
    Optional,
)

import numpy as np
from scipy import stats as scipy_stats


def mean_ci(values: Iterable[float], confidence: float = 0.95) -> tuple[float, float]:
    """Mean and half-width of the Student-t confidence interval for the mean.

    The single implementation; ``analysis.benchmark`` re-exports it.

    Args:
        values: Sample values
        confidence: Confidence level

    Returns:
        (mean, half_width); half_width is NaN for fewer than 2 values
    """
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        raise ValueError("mean_ci needs at least one value")

    mean = float(arr.mean())
    if arr.size < 2:
        return mean, float("nan")

    sem = arr.std(ddof=1) / np.sqrt(arr.size)
    t_crit = scipy_stats.t.ppf((1 + confidence) / 2, df=arr.size - 1)
    return mean, float(t_crit * sem)


def iqm(values: Any) -> float:
    """Interquartile mean: the mean of the middle 50% of all values (any shape)."""
    arr = np.asarray(values, dtype=float).ravel()
    if arr.size == 0:
        raise ValueError("iqm needs at least one value")
    return float(scipy_stats.trim_mean(arr, 0.25))


def stratified_bootstrap_ci(
    scores: Any,
    statistic: Callable[[np.ndarray], float] = iqm,
    n_boot: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Point estimate and percentile bootstrap CI, resampling runs within each task.

    Args:
        scores: Shape (n_runs,) or (n_runs, n_tasks); one value per independent run
        statistic: Aggregate applied to the whole (n_runs, n_tasks) matrix
        n_boot: Number of bootstrap resamples
        confidence: Confidence level
        seed: Seed for the resampling generator

    Returns:
        (point, low, high)
    """
    arr = np.asarray(scores, dtype=float)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2 or arr.shape[0] == 0:
        raise ValueError(
            f"scores must have shape (n_runs,) or (n_runs, n_tasks), got {arr.shape}"
        )

    point = float(statistic(arr))
    n_runs, n_tasks = arr.shape
    if n_runs < 2:
        return point, float("nan"), float("nan")

    rng = np.random.default_rng(seed)
    run_idx = rng.integers(0, n_runs, size=(n_boot, n_runs, n_tasks))
    samples = arr[
        run_idx, np.arange(n_tasks)
    ]  # Each task column resampled independently
    if statistic is iqm:  # Vectorized fast path for the default statistic
        boot = scipy_stats.trim_mean(samples.reshape(n_boot, -1), 0.25, axis=1)
    else:
        boot = np.array([statistic(sample) for sample in samples], dtype=float)

    alpha = (1 - confidence) / 2
    low, high = np.quantile(boot, [alpha, 1 - alpha])
    return point, float(low), float(high)


def vargha_delaney_a12(a: Sequence[float], b: Sequence[float]) -> float:
    """Probability that a value from ``a`` beats one from ``b`` (ties count half).

    0.5 means no effect; 0.56 / 0.64 / 0.71 are the conventional small / medium / large
    thresholds.
    """
    x = np.asarray(a, dtype=float).ravel()
    y = np.asarray(b, dtype=float).ravel()
    if x.size == 0 or y.size == 0:
        raise ValueError("A12 needs at least one value in each sample")

    greater = np.sum(x[:, None] > y[None, :])
    equal = np.sum(x[:, None] == y[None, :])
    return float((greater + 0.5 * equal) / (x.size * y.size))


def mann_whitney(
    a: Sequence[float], b: Sequence[float], alternative: str = "two-sided"
) -> tuple[float, float]:
    """Mann-Whitney U statistic and p-value; identical constant samples give p = 1."""
    x = np.asarray(a, dtype=float).ravel()
    y = np.asarray(b, dtype=float).ravel()
    if x.size == 0 or y.size == 0:
        raise ValueError("Mann-Whitney needs at least one value in each sample")

    if np.unique(np.concatenate([x, y])).size == 1:
        return x.size * y.size / 2, 1.0  # All values tied: no evidence of a difference

    result = scipy_stats.mannwhitneyu(x, y, alternative=alternative)
    return float(result.statistic), float(result.pvalue)


def holm_correction(pvalues: Sequence[float]) -> np.ndarray:
    """Holm-Bonferroni step-down adjusted p-values (monotone, capped at 1)."""
    p = np.asarray(pvalues, dtype=float).ravel()
    adjusted = np.empty_like(p)
    running = 0.0
    for rank, idx in enumerate(np.argsort(p, kind="stable")):
        running = max(running, min(1.0, (p.size - rank) * p[idx]))
        adjusted[idx] = running
    return adjusted


def compare_conditions(
    samples: Mapping[str, Sequence[float]],
    pairs: Optional[Sequence[tuple[str, str]]] = None,
    reference: Optional[str] = None,
) -> list[dict[str, Any]]:
    """Pairwise Mann-Whitney comparisons with Holm correction across the family.

    Args:
        samples: Condition name -> one value per independent run
        pairs: Explicit (a, b) pairs; default compares every condition to ``reference``,
            or all pairs when no reference is given
        reference: Condition every other one is compared against

    Returns:
        One row per pair with u, p, p_holm, a12 (P(a > b)), mean_diff (a - b) and sizes
    """
    if pairs is None:
        names = list(samples)
        if reference is not None:
            if reference not in samples:
                raise KeyError(f"Reference condition {reference!r} not in samples")
            pairs = [(name, reference) for name in names if name != reference]
        else:
            pairs = list(itertools.combinations(names, 2))

    rows = []
    for a, b in pairs:
        u, p = mann_whitney(samples[a], samples[b])
        rows.append(
            {
                "a": a,
                "b": b,
                "u": u,
                "p": p,
                "a12": vargha_delaney_a12(samples[a], samples[b]),
                "mean_diff": float(np.mean(samples[a]) - np.mean(samples[b])),
                "n_a": len(samples[a]),
                "n_b": len(samples[b]),
            }
        )

    for row, adjusted in zip(rows, holm_correction([row["p"] for row in rows])):
        row["p_holm"] = float(adjusted)
    return rows


def paired_difference(
    a: Sequence[float], b: Sequence[float], confidence: float = 0.95
) -> dict[str, float]:
    """Mean and t-CI of the paired differences a - b (e.g. intact vs intervened, per seed)."""
    x = np.asarray(a, dtype=float).ravel()
    y = np.asarray(b, dtype=float).ravel()
    if x.shape != y.shape:
        raise ValueError(
            f"Paired samples must have equal length, got {x.size} and {y.size}"
        )

    mean, half = mean_ci(x - y, confidence)
    return {"mean": mean, "ci": half, "low": mean - half, "high": mean + half}
