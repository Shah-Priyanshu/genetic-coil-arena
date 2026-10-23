"""Test the benchmark harness: CI math, curve aggregation and agent loading."""

import gzip
import pickle

import numpy as np
import pandas as pd
import pytest

from genetic_coil_arena.analysis.benchmark import (
    aggregate_curves,
    baseline_policies,
    benchmark_policies,
    load_agent_file,
    mean_ci,
)
from genetic_coil_arena.config import Config
from genetic_coil_arena.evo.genome import Genome
from genetic_coil_arena.snake_core.agent import SnakeAgent, get_observation_size


class TestMeanCI:
    """Student-t confidence intervals."""

    def test_known_values(self):
        mean, half_width = mean_ci([1, 2, 3, 4, 5])
        # t(0.975, df=4) = 2.776445, sample std = sqrt(2.5)
        assert mean == 3.0
        assert half_width == pytest.approx(2.776445 * np.sqrt(2.5) / np.sqrt(5), rel=1e-5)

    def test_constant_values_have_zero_width(self):
        assert mean_ci([4.0, 4.0, 4.0]) == (4.0, 0.0)

    def test_single_value_has_undefined_width(self):
        mean, half_width = mean_ci([7.0])
        assert mean == 7.0
        assert np.isnan(half_width)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            mean_ci([])

    def test_coverage_is_about_95_percent(self):
        rng = np.random.default_rng(0)
        trials = 2000
        covered = 0
        for _ in range(trials):
            mean, half_width = mean_ci(rng.normal(10.0, 3.0, size=8))
            covered += abs(mean - 10.0) <= half_width
        assert covered / trials == pytest.approx(0.95, abs=0.015)


class TestAggregateCurves:
    """Per-generation aggregation across seeds."""

    def test_mean_ci_and_uneven_run_lengths(self):
        rows = []
        for seed, values in ((0, [1.0, 2.0, 3.0]), (1, [3.0, 4.0])):
            for generation, value in enumerate(values):
                rows.append({"seed": seed, "generation": generation, "best_fitness": value})
        curves = aggregate_curves(pd.DataFrame(rows), metrics=["best_fitness"])

        assert list(curves["n_seeds"]) == [2, 2, 1]
        assert list(curves["best_fitness_mean"]) == [2.0, 3.0, 3.0]
        expected_width = mean_ci([1.0, 3.0])[1]
        assert curves.loc[0, "best_fitness_ci95"] == pytest.approx(expected_width)
        assert np.isnan(curves.loc[2, "best_fitness_ci95"])


class TestAgentLoading:
    """load_agent_file accepts every format the project saves."""

    @staticmethod
    def agent(config: Config) -> SnakeAgent:
        obs_size = get_observation_size(config.game.board_size, config.game.observation_radius)
        return SnakeAgent(obs_size, config.agent.hidden_size, config.agent.second_hidden_size, seed=3)

    def test_gzip_agent_and_plain_hall_of_fame(self, tmp_path):
        config = Config()
        agent = self.agent(config)

        agent_path = tmp_path / "best_agent.pkl"
        agent_path.write_bytes(gzip.compress(pickle.dumps(agent)))
        hof_path = tmp_path / "hall_of_fame.pkl"
        hof_path.write_bytes(pickle.dumps([Genome(weights=agent.get_genome())]))

        for path in (agent_path, hof_path):
            loaded = load_agent_file(path, config)
            assert np.allclose(loaded.get_genome(), agent.get_genome())

    def test_mismatched_config_raises(self, tmp_path):
        config = Config()
        path = tmp_path / "genome.pkl"
        path.write_bytes(pickle.dumps(Genome(weights=self.agent(config).get_genome())))

        config.agent.hidden_size = 8
        with pytest.raises(ValueError, match="trained with"):
            load_agent_file(path, config)


def test_benchmark_policies_reports_every_metric():
    config = Config()
    df = benchmark_policies(config, baseline_policies(), episodes=3)

    assert list(df["policy"]) == ["random", "straight", "greedy"]
    for metric in ("food", "fitness", "steps", "starvation"):
        assert {f"{metric}_mean", f"{metric}_ci95"} <= set(df.columns)
    assert (df["episodes"] == 3).all()
