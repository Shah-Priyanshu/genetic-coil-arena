"""Test the evaluation statistics: CIs, IQM bootstrap, Mann-Whitney + Holm, A12."""

import numpy as np
import pytest

from genetic_coil_arena.evaluation.stats import (
    compare_conditions,
    holm_correction,
    iqm,
    mann_whitney,
    mean_ci,
    paired_difference,
    stratified_bootstrap_ci,
    vargha_delaney_a12,
)


class TestMeanCI:
    """Student-t confidence interval (the one implementation in the repo)."""

    def test_known_values(self):
        mean, half = mean_ci([1, 2, 3, 4, 5])
        assert mean == 3.0
        assert half == pytest.approx(2.776445 * np.sqrt(2.5) / np.sqrt(5), rel=1e-5)

    def test_benchmark_harness_reexports_it(self):
        from genetic_coil_arena.analysis.benchmark import mean_ci as benchmark_mean_ci

        assert benchmark_mean_ci is mean_ci

    def test_confidence_level_widens_interval(self):
        values = np.random.default_rng(3).normal(size=17)
        halves = [mean_ci(values, c)[1] for c in (0.9, 0.95, 0.99)]
        assert halves == sorted(halves)

    def test_single_value_and_empty(self):
        assert np.isnan(mean_ci([2.0])[1])
        with pytest.raises(ValueError):
            mean_ci([])


class TestIQM:
    """Interquartile mean and the stratified bootstrap."""

    def test_iqm_ignores_outer_quartiles(self):
        assert iqm([1, 2, 3, 4, 5, 6, 7, 1000]) == pytest.approx(4.5)
        assert iqm([-1000, 2, 3, 4, 5, 6, 7, 8]) == pytest.approx(4.5)

    def test_iqm_of_constant(self):
        assert iqm(np.full((4, 3), 2.5)) == 2.5

    def test_bootstrap_ci_contains_point_and_is_deterministic(self):
        scores = np.random.default_rng(0).normal(10, 2, size=20)
        point, low, high = stratified_bootstrap_ci(scores, n_boot=500, seed=1)
        assert low <= point <= high
        assert point == pytest.approx(iqm(scores))
        assert (point, low, high) == stratified_bootstrap_ci(scores, n_boot=500, seed=1)

    def test_vectorized_iqm_matches_generic_path(self):
        scores = np.random.default_rng(2).normal(size=(12, 3))
        fast = stratified_bootstrap_ci(scores, n_boot=300, seed=4)
        slow = stratified_bootstrap_ci(scores, lambda s: iqm(s), n_boot=300, seed=4)
        assert fast == pytest.approx(slow)

    def test_bootstrap_ci_narrows_with_more_runs(self):
        rng = np.random.default_rng(1)
        _, lo_small, hi_small = stratified_bootstrap_ci(rng.normal(size=10), n_boot=500)
        _, lo_big, hi_big = stratified_bootstrap_ci(rng.normal(size=1000), n_boot=500)
        assert hi_big - lo_big < hi_small - lo_small

    def test_bootstrap_is_stratified_by_task(self):
        # Task columns with very different scales: each resample keeps one value per
        # (run, task) from the same task column, so the column means are preserved.
        scores = np.stack([np.arange(10.0), 100 + np.arange(10.0)], axis=1)
        _, low, high = stratified_bootstrap_ci(
            scores, statistic=lambda s: s[:, 1].min(), n_boot=200
        )
        assert low >= 100

    def test_bootstrap_single_run_has_nan_ci(self):
        point, low, high = stratified_bootstrap_ci([3.0])
        assert point == 3.0 and np.isnan(low) and np.isnan(high)

    def test_bootstrap_rejects_bad_shapes(self):
        with pytest.raises(ValueError):
            stratified_bootstrap_ci(np.zeros((2, 2, 2)))


class TestComparisons:
    """Mann-Whitney U, Holm correction and Vargha-Delaney A12."""

    def test_a12_extremes(self):
        assert vargha_delaney_a12([5, 6, 7], [1, 2, 3]) == 1.0
        assert vargha_delaney_a12([1, 2, 3], [5, 6, 7]) == 0.0
        assert vargha_delaney_a12([1, 2, 3], [1, 2, 3]) == 0.5

    def test_a12_counts_ties_half(self):
        assert vargha_delaney_a12([1], [1]) == 0.5
        assert vargha_delaney_a12([2, 1], [1]) == 0.75

    def test_mann_whitney_detects_shift(self):
        rng = np.random.default_rng(0)
        _, p = mann_whitney(rng.normal(3, 1, 30), rng.normal(0, 1, 30))
        assert p < 1e-6

    def test_mann_whitney_identical_constants(self):
        u, p = mann_whitney([1, 1, 1], [1, 1])
        assert p == 1.0 and u == 3.0

    def test_holm_known_values(self):
        adjusted = holm_correction([0.01, 0.04, 0.03, 0.005])
        np.testing.assert_allclose(adjusted, [0.03, 0.06, 0.06, 0.02])

    def test_holm_is_capped_and_monotone(self):
        p = np.array([0.5, 0.4, 0.001, 0.3])
        adjusted = holm_correction(p)
        assert np.all(adjusted <= 1.0)
        assert np.all(adjusted >= p)
        order = np.argsort(p)
        assert np.all(np.diff(adjusted[order]) >= 0)

    def test_holm_empty(self):
        assert holm_correction([]).size == 0

    def test_compare_conditions_against_reference(self):
        samples = {"a": [5, 6, 7, 8], "b": [1, 2, 3, 4], "ref": [1, 2, 3, 4]}
        rows = compare_conditions(samples, reference="ref")
        assert [(r["a"], r["b"]) for r in rows] == [("a", "ref"), ("b", "ref")]
        assert rows[0]["a12"] == 1.0 and rows[1]["a12"] == 0.5
        assert rows[0]["mean_diff"] == 4.0
        assert all(r["p_holm"] >= r["p"] for r in rows)

    def test_compare_conditions_all_pairs_and_unknown_reference(self):
        rows = compare_conditions({"a": [1, 2], "b": [3, 4], "c": [5, 6]})
        assert len(rows) == 3
        with pytest.raises(KeyError):
            compare_conditions({"a": [1]}, reference="z")

    def test_paired_difference(self):
        out = paired_difference([3, 4, 5], [1, 2, 3])
        assert out["mean"] == 2.0 and out["ci"] == 0.0
        with pytest.raises(ValueError):
            paired_difference([1, 2], [1])
