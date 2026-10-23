"""End-to-end tests: evolution, persistent experiments with resume, and the CLI."""

import json

import pandas as pd
import pytest
from typer.testing import CliRunner

from genetic_coil_arena.cli.main import app
from genetic_coil_arena.config import Config
from genetic_coil_arena.evo.arena import EvolutionArena
from genetic_coil_arena.persistence.arena import PersistentEvolutionArena


def tiny_config(tmp_path) -> Config:
    """A config small enough to evolve in-process in a few seconds."""
    config = Config()
    config.game.starvation_steps = 20
    config.evolution.population_size = 6
    config.evolution.generations = 2
    config.evolution.elite_size = 1
    config.evaluation.episodes_per_agent = 1
    config.evaluation.parallel_processes = 1
    config.evaluation.deterministic_seed = 0
    config.logging.output_dir = tmp_path / "outputs"
    return config


def test_evolution_arena_writes_stats_and_baseline(tmp_path):
    config = tiny_config(tmp_path)
    result = EvolutionArena(config).evolve()

    assert len(result.generation_stats) == 2
    assert result.random_baseline_fitness is not None

    stats = pd.read_csv(config.logging.output_dir / "generation_stats.csv")
    assert {"best_score", "mean_score", "starvation_rate", "mean_steps"} <= set(stats.columns)
    assert "relative_to_baseline" not in stats.columns

    summary = json.loads((config.logging.output_dir / "summary.json").read_text())
    assert summary["fitness_function"] == "food"
    assert summary["random_baseline_fitness"] == pytest.approx(result.random_baseline_fitness)


def test_persistent_arena_evolves_and_resumes(tmp_path):
    config = tiny_config(tmp_path)
    base_path = str(tmp_path / "experiments")

    arena = PersistentEvolutionArena(config, base_path=base_path)
    results = arena.evolve(generations=2, verbose=False)
    assert arena.generation == 2
    assert len(results["fitness_history"]) == 2

    exp_path = arena.experiment_path
    for filename in ("results.json", "best_agent.pkl", "generation_stats.csv", "latest_checkpoint.pkl"):
        assert (exp_path / filename).exists(), filename

    resumed_config = Config.from_yaml(exp_path / "config.yaml")
    assert resumed_config.game.starvation_steps == 20

    resumed = PersistentEvolutionArena(resumed_config, experiment_id=arena.experiment_id,
                                       base_path=base_path)
    assert resumed.generation == 2
    assert len(resumed.population) == config.evolution.population_size
    assert len(resumed.generation_stats) == 2

    resumed.evolve(generations=1, verbose=False)
    assert resumed.generation == 3
    assert len(resumed.generation_stats) == 3
    assert len(pd.read_csv(exp_path / "generation_stats.csv")) == 3


def test_cli_evolve_resume_and_benchmark(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()

    config_path = tmp_path / "config.yaml"
    config = tiny_config(tmp_path)
    config.to_yaml(config_path)

    result = runner.invoke(app, ["evolve", "--config", str(config_path), "--base-path", "experiments"])
    assert result.exit_code == 0, result.output

    (experiment_id,) = [p.name for p in (tmp_path / "experiments").iterdir()]
    result = runner.invoke(app, ["resume", experiment_id, "-g", "1", "--base-path", "experiments"])
    assert result.exit_code == 0, result.output
    assert "Final generation: 3" in result.output

    agent_file = tmp_path / "experiments" / experiment_id / "best_agent.pkl"
    result = runner.invoke(app, ["benchmark", "--config", str(config_path), "-n", "3",
                                 "--agent", str(agent_file), "-o", "bench.csv"])
    assert result.exit_code == 0, result.output
    bench = pd.read_csv(tmp_path / "bench.csv")
    assert list(bench["policy"]) == ["random", "straight", "greedy", str(agent_file)]


def test_cli_multiseed_writes_curves_and_benchmark(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config_path = tmp_path / "config.yaml"
    tiny_config(tmp_path).to_yaml(config_path)

    result = CliRunner().invoke(app, ["multiseed", "--config", str(config_path), "-k", "2",
                                      "-n", "3", "-o", "multi"])
    assert result.exit_code == 0, result.output

    curves = pd.read_csv(tmp_path / "multi" / "curves_ci.csv")
    assert list(curves["n_seeds"]) == [2, 2]
    bench = pd.read_csv(tmp_path / "multi" / "final_benchmark.csv")
    assert bench["policy"].iloc[-1] == "evolved (mean of 2 seeds)"
    assert bench["ci_over"].iloc[-1] == "seeds"
