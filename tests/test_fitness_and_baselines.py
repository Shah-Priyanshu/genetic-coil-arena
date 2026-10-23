"""Test fitness functions, failure handling and the baseline policies."""

from concurrent.futures import Executor, Future
from dataclasses import asdict

import numpy as np
import pytest

from genetic_coil_arena.config import Config, GameConfig
from genetic_coil_arena.evo.baselines import (
    BASELINE_POLICIES,
    baseline_factory,
    benchmark_seeds,
    evaluate_policy,
    greedy_policy,
    make_policy,
    safe_moves,
)
from genetic_coil_arena.evo.fitness import (
    EvaluationResult,
    FitnessEvaluator,
    agent_policy,
    build_agent,
    compute_fitness,
    evaluate_agent_episode,
    max_episode_steps,
    run_episode,
)
from genetic_coil_arena.evo.genome import Genome
from genetic_coil_arena.snake_core.agent import SnakeAgent, get_observation_size
from genetic_coil_arena.snake_core.game import Action, Direction, Position, SnakeGame

GAME_CONFIG = asdict(GameConfig())


def result(score: int, steps: int, total_reward: float = 0.0) -> EvaluationResult:
    return EvaluationResult(score=score, steps=steps, final_length=score + 1,
                            success=score > 0, total_reward=total_reward)


class TestFitnessFunctions:
    """compute_fitness modes."""

    @pytest.mark.parametrize("score, steps, expected", [
        (0, 30, 1.0),    # 0 + 3 - 2 early death penalty
        (2, 30, 5.0),    # No penalty once food was eaten
        (0, 500, 50.0),  # The survival trap: no food, high fitness
    ])
    def test_legacy_matches_original_formula(self, score, steps, expected):
        assert compute_fitness(result(score, steps), "legacy", GAME_CONFIG) == pytest.approx(expected)

    def test_food_counts_food_and_breaks_ties_by_survival(self):
        longest = max_episode_steps(GAME_CONFIG)
        assert 3 < compute_fitness(result(3, 100), "food", GAME_CONFIG) < 4
        # One food beats surviving as long as possible without eating
        assert (compute_fitness(result(1, 1), "food", GAME_CONFIG)
                > compute_fitness(result(0, longest), "food", GAME_CONFIG))
        assert (compute_fitness(result(2, 200), "food", GAME_CONFIG)
                > compute_fitness(result(2, 100), "food", GAME_CONFIG))

    def test_return_is_total_reward(self):
        assert compute_fitness(result(1, 10, total_reward=2.5), "return", GAME_CONFIG) == 2.5

    def test_unknown_function_raises(self):
        with pytest.raises(ValueError):
            compute_fitness(result(1, 10), "nope", GAME_CONFIG)
        with pytest.raises(ValueError):
            FitnessEvaluator(GAME_CONFIG, asdict(Config().agent), fitness_function="nope")


class TestSharedEpisodePath:
    """Evolved agents and baselines run through the same episode code."""

    def test_agent_episode_equals_run_episode_with_agent_policy(self):
        config = Config()
        agent_config = asdict(config.agent)
        genome = Genome(genome_size=SnakeAgent(
            get_observation_size(), agent_config["hidden_size"], agent_config["second_hidden_size"]
        ).genome_size)

        via_worker = evaluate_agent_episode(genome.weights, GAME_CONFIG, agent_config, 11)
        agent = build_agent(genome.weights, GAME_CONFIG, agent_config)
        direct = run_episode(SnakeGame.from_config(GAME_CONFIG, seed=11), agent_policy(agent))
        assert via_worker == direct


class FakeExecutor(Executor):
    """Runs tasks synchronously; tasks whose index is in fail_on raise instead."""

    def __init__(self, fail_on):
        self.fail_on = fail_on
        self.calls = 0

    def submit(self, fn, *args, **kwargs):
        future = Future()
        if self.fail_on == "all" or self.calls in self.fail_on:
            future.set_exception(RuntimeError(f"worker {self.calls} crashed"))
        else:
            future.set_result(fn(*args, **kwargs))
        self.calls += 1
        return future

    def shutdown(self, wait=True, **kwargs):
        pass


class TestFailureHandling:
    """Worker failures are retried or raised, never scored as fitness."""

    def make_evaluator(self) -> FitnessEvaluator:
        config = Config()
        config.game.starvation_steps = 20
        return FitnessEvaluator(asdict(config.game), asdict(config.agent),
                                episodes_per_agent=2, parallel_processes=2,
                                deterministic_seed=0)

    def genomes(self, n: int = 2):
        agent_config = asdict(Config().agent)
        size = SnakeAgent(get_observation_size(), agent_config["hidden_size"],
                          agent_config["second_hidden_size"]).genome_size
        return [Genome(genome_size=size) for _ in range(n)]

    def test_partial_failures_are_retried_in_process(self):
        evaluator = self.make_evaluator()
        evaluator._executor = FakeExecutor(fail_on={0, 3})
        stats = evaluator.evaluate_genomes(self.genomes())
        assert all(s.episodes_played == 2 for s in stats)

    def test_all_failures_raise(self):
        evaluator = self.make_evaluator()
        evaluator._executor = FakeExecutor(fail_on="all")
        with pytest.raises(RuntimeError, match="All 4 evaluation episodes failed"):
            evaluator.evaluate_genomes(self.genomes())

    def test_in_process_errors_propagate(self):
        evaluator = self.make_evaluator()
        evaluator.parallel_processes = 1
        bad = Genome(weights=np.zeros(5, dtype=np.float32))  # Wrong size for the network
        with pytest.raises(ValueError):
            evaluator.evaluate_genomes([bad])


class TestBaselines:
    """Reference policies."""

    def test_greedy_avoids_immediate_death(self):
        game = SnakeGame(board_size=16, seed=0)
        game.snake_positions = np.array([[8, 8], [7, 8], [6, 8]], dtype=np.int32)
        game.snake_direction = Direction.WEST
        game.food_position = Position(2, 8)  # Straight ahead, behind the body

        action = greedy_policy(game)
        assert action != Action.STRAIGHT
        assert action in [move[0] for move in safe_moves(game)]

    @pytest.mark.parametrize("food, expected", [
        (Position(8, 4), Action.STRAIGHT),
        (Position(12, 8), Action.RIGHT),
        (Position(4, 8), Action.LEFT),
    ])
    def test_greedy_heads_for_food(self, food, expected):
        game = SnakeGame(board_size=16, seed=0)  # Head at (8, 8) facing north
        game.food_position = food
        assert greedy_policy(game) == expected

    def test_safe_moves_respect_walls(self):
        game = SnakeGame(board_size=8, walls=True, seed=0)
        game.snake_positions = np.array([[0, 0]], dtype=np.int32)
        game.snake_direction = Direction.NORTH
        assert [move[0] for move in safe_moves(game)] == [Action.RIGHT]

    def test_random_policy_is_reproducible(self):
        game = SnakeGame(seed=0)
        policy_a, policy_b = make_policy("random", 7), make_policy("random", 7)
        actions = [policy_a(game) for _ in range(20)]
        assert actions == [policy_b(game) for _ in range(20)]
        assert set(actions) <= set(Action)

    def test_unknown_policy_raises(self):
        with pytest.raises(ValueError):
            make_policy("clairvoyant")

    def test_default_game_separates_baselines(self):
        """Under the default game, only a food-seeking policy eats much."""
        seeds = benchmark_seeds(20)
        food = {}
        for name in BASELINE_POLICIES:
            results = evaluate_policy(baseline_factory(name), GAME_CONFIG, seeds)
            food[name] = np.mean([r.score for r in results])
            if name != "greedy":
                assert all(r.end_reason == "starvation" for r in results)

        assert food["random"] < 0.5
        assert food["straight"] < 0.5
        assert food["greedy"] > 10
