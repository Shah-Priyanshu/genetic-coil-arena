"""Multi-seed benchmarking: baselines and saved agents with 95% confidence intervals."""

import copy
import gzip
import pickle
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from ..config import Config
from ..evo.arena import EvolutionArena
from ..evo.baselines import (
    BASELINE_POLICIES,
    BENCHMARK_SEED_OFFSET,
    PolicyFactory,
    baseline_factory,
    benchmark_seeds,
    evaluate_policy,
)
from ..evo.fitness import EvaluationResult, agent_policy, build_agent, compute_fitness
from ..evo.genome import Genome
from ..snake_core.agent import SnakeAgent, get_observation_size

# Per-generation columns aggregated across seeds by aggregate_curves
CURVE_METRICS = ("best_fitness", "mean_fitness", "best_score", "mean_score",
                 "mean_steps", "starvation_rate")

# Validation seeds pick the best hall-of-fame genome; test seeds then score it
VALIDATION_SEED_OFFSET = BENCHMARK_SEED_OFFSET + 1_000_000


def mean_ci(values: Iterable[float], confidence: float = 0.95) -> Tuple[float, float]:
    """Mean and half-width of the Student-t confidence interval for the mean.

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


def summarize_policy(name: str, results: List[EvaluationResult], fitness_function: str,
                     game_config: Dict[str, Any]) -> Dict[str, Any]:
    """One benchmark row: mean ± 95% CI over episodes for each metric."""
    metrics = {
        "food": [r.score for r in results],
        "fitness": [compute_fitness(r, fitness_function, game_config) for r in results],
        "steps": [r.steps for r in results],
        "starvation": [float(r.end_reason == "starvation") for r in results],
    }

    row: Dict[str, Any] = {"policy": name, "episodes": len(results), "ci_over": "episodes"}
    for metric, values in metrics.items():
        row[f"{metric}_mean"], row[f"{metric}_ci95"] = mean_ci(values)
    return row


def load_agent_file(path: Path, config: Config) -> SnakeAgent:
    """Load a saved agent for the given config.

    Accepts a pickled SnakeAgent, Genome, weight array, or a list of these
    (e.g. hall_of_fame.pkl, best first), plain or gzip-compressed.
    """
    data = Path(path).read_bytes()
    if data[:2] == b"\x1f\x8b":
        data = gzip.decompress(data)
    obj = pickle.loads(data)

    if isinstance(obj, (list, tuple)):
        if not obj:
            raise ValueError(f"{path} contains an empty list")
        obj = obj[0]

    game_config = asdict(config.game)
    obs_size = get_observation_size(config.game.board_size, config.game.observation_radius)

    if isinstance(obj, SnakeAgent):
        if obj.input_size != obs_size:
            raise ValueError(f"Agent in {path} expects {obj.input_size} inputs but the config "
                             f"produces {obs_size}; pass the config it was trained with")
        return obj

    weights = obj.weights if isinstance(obj, Genome) else np.asarray(obj, dtype=np.float32)
    expected = SnakeAgent(obs_size, config.agent.hidden_size,
                          config.agent.second_hidden_size, config.agent.activation).genome_size
    if weights.size != expected:
        raise ValueError(f"Genome in {path} has {weights.size} weights but the config's network "
                         f"has {expected}; pass the config it was trained with")
    return build_agent(weights, game_config, asdict(config.agent))


def saved_agent_factory(agent: SnakeAgent) -> PolicyFactory:
    """Policy factory for a fixed agent (the seed only affects the game)."""
    return lambda seed: agent_policy(agent)


def benchmark_policies(config: Config, policies: Dict[str, PolicyFactory], episodes: int = 100,
                       seed_offset: int = BENCHMARK_SEED_OFFSET) -> pd.DataFrame:
    """Evaluate each policy on the same held-out seeds.

    Args:
        config: Configuration (game and fitness function)
        policies: Policy factories by display name
        episodes: Episodes per policy
        seed_offset: First episode seed

    Returns:
        One row per policy with mean and 95% CI half-width for each metric
    """
    game_config = asdict(config.game)
    seeds = benchmark_seeds(episodes, seed_offset)
    rows = []
    for name, factory in policies.items():
        results = evaluate_policy(factory, game_config, seeds)
        rows.append(summarize_policy(name, results, config.evaluation.fitness_function, game_config))
    return pd.DataFrame(rows)


def baseline_policies() -> Dict[str, PolicyFactory]:
    """All baseline policy factories by name."""
    return {name: baseline_factory(name) for name in BASELINE_POLICIES}


def format_benchmark_table(df: pd.DataFrame) -> str:
    """Render a benchmark DataFrame as aligned text."""
    def cell(row: pd.Series, metric: str, fmt: str) -> str:
        mean, half_width = row[f"{metric}_mean"], row[f"{metric}_ci95"]
        if np.isnan(half_width):
            return f"{mean:{fmt}}"
        return f"{mean:{fmt}} ± {half_width:{fmt}}"

    lines = [f"{'policy':<28} {'n':>5}  {'food':>17}  {'fitness':>17}  {'steps':>19}  {'starved':>7}"]
    for _, row in df.iterrows():
        n_label = f"{row['episodes']}" + ("s" if row.get("ci_over") == "seeds" else "")
        lines.append(
            f"{row['policy']:<28} {n_label:>5}  {cell(row, 'food', '.2f'):>17}  "
            f"{cell(row, 'fitness', '.3f'):>17}  {cell(row, 'steps', '.1f'):>19}  "
            f"{row['starvation_mean']:>7.0%}"
        )
    return "\n".join(lines)


def aggregate_curves(curves: pd.DataFrame, metrics: Sequence[str] = CURVE_METRICS) -> pd.DataFrame:
    """Per-generation mean and 95% CI across seeds.

    Args:
        curves: Long table with columns seed, generation and the metrics
        metrics: Columns to aggregate

    Returns:
        One row per generation: n_seeds, <metric>_mean, <metric>_ci95
    """
    rows = []
    for generation, group in curves.groupby("generation", sort=True):
        row: Dict[str, Any] = {"generation": int(generation), "n_seeds": int(group["seed"].nunique())}
        for metric in metrics:
            row[f"{metric}_mean"], row[f"{metric}_ci95"] = mean_ci(group[metric])
        rows.append(row)
    return pd.DataFrame(rows)


def select_best_genome(candidates: List[Genome], config: Config, episodes: int = 30) -> Genome:
    """Pick the candidate with the highest mean fitness on validation seeds.

    Hall-of-fame order comes from a few noisy training episodes, so re-ranking on
    separate seeds avoids reporting the luckiest genome instead of the best one.
    """
    if len(candidates) == 1:
        return candidates[0]

    game_config = asdict(config.game)
    agent_config = asdict(config.agent)
    seeds = benchmark_seeds(episodes, VALIDATION_SEED_OFFSET)

    def validation_fitness(genome: Genome) -> float:
        agent = build_agent(genome.weights, game_config, agent_config)
        results = evaluate_policy(saved_agent_factory(agent), game_config, seeds)
        return float(np.mean([compute_fitness(r, config.evaluation.fitness_function, game_config)
                              for r in results]))

    return max(candidates, key=validation_fitness)


def run_multi_seed_evolution(config: Config, seeds: Sequence[int], output_dir: Path,
                             benchmark_episodes: int = 100, top_k: int = 5,
                             validation_episodes: int = 30) -> Dict[str, pd.DataFrame]:
    """Evolve once per seed, then aggregate curves and benchmark the results.

    Writes to output_dir:
        seed_<s>/            each run's usual outputs
        curves_by_seed.csv   per-generation stats, one row per (seed, generation)
        curves_ci.csv        per-generation mean and 95% CI across seeds
        final_benchmark.csv  baselines and each seed's best genome on held-out seeds,
                             plus the across-seed mean with a CI over seeds

    Args:
        config: Base configuration; deterministic_seed and output_dir are set per run
        seeds: Seeds to run
        output_dir: Directory for all outputs
        benchmark_episodes: Held-out test episodes per policy
        top_k: Hall-of-fame genomes re-ranked on validation seeds
        validation_episodes: Episodes per candidate for re-ranking

    Returns:
        {"curves": per-generation aggregate, "benchmark": final benchmark table}
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    game_config = asdict(config.game)

    curve_rows = []
    evolved_rows = []
    for seed in seeds:
        run_config = copy.deepcopy(config)
        run_config.evaluation.deterministic_seed = seed
        run_config.logging.output_dir = output_dir / f"seed_{seed}"

        start_time = time.time()
        result = EvolutionArena(run_config).evolve()
        elapsed = time.time() - start_time

        for stats in result.generation_stats:
            curve_rows.append({"seed": seed, **asdict(stats)})

        best = select_best_genome(result.hall_of_fame[:top_k], run_config, validation_episodes)
        agent = build_agent(best.weights, game_config, asdict(config.agent))
        results = evaluate_policy(saved_agent_factory(agent), game_config,
                                  benchmark_seeds(benchmark_episodes))
        row = summarize_policy(f"evolved (seed {seed})", results,
                               config.evaluation.fitness_function, game_config)
        row["seed"] = seed
        row["generations"] = len(result.generation_stats)
        row["wall_clock_s"] = elapsed
        evolved_rows.append(row)

    curves = pd.DataFrame(curve_rows)
    curves.to_csv(output_dir / "curves_by_seed.csv", index=False)
    aggregated = aggregate_curves(curves)
    aggregated.to_csv(output_dir / "curves_ci.csv", index=False)

    baselines = benchmark_policies(config, baseline_policies(), benchmark_episodes)
    evolved = pd.DataFrame(evolved_rows)

    # Across-seed summary: each seed's mean is one sample, so the CI covers run-to-run variance
    across: Dict[str, Any] = {"policy": f"evolved (mean of {len(evolved)} seeds)",
                              "episodes": len(evolved), "ci_over": "seeds"}
    for metric in ("food", "fitness", "steps", "starvation"):
        across[f"{metric}_mean"], across[f"{metric}_ci95"] = mean_ci(evolved[f"{metric}_mean"])

    benchmark = pd.concat([baselines, evolved, pd.DataFrame([across])], ignore_index=True)
    benchmark.to_csv(output_dir / "final_benchmark.csv", index=False)

    return {"curves": aggregated, "benchmark": benchmark}
