"""Fitness evaluation for snake agents."""

import numpy as np
from concurrent.futures import Executor, ProcessPoolExecutor, as_completed
from typing import Callable, List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import logging
import time
import traceback

from ..snake_core.game import SnakeGame, Action
from ..snake_core.agent import SnakeAgent, get_observation_size
from .genome import Genome, GenomeStats

logger = logging.getLogger(__name__)

FITNESS_FUNCTIONS = ("food", "legacy", "return")


@dataclass
class EvaluationResult:
    """Result of evaluating a single episode."""
    score: int
    steps: int
    final_length: int
    success: bool  # Whether agent reached food before timeout
    total_reward: float = 0.0  # Sum of the game's per-step rewards
    end_reason: str = "unknown"  # "collision", "wall", "starvation" or "timeout"


def run_episode(game: SnakeGame, choose_action: Callable[[SnakeGame], Action]) -> EvaluationResult:
    """Play one episode from reset; shared by evolved agents and baseline policies.

    Args:
        game: Game to play (reset before the first move)
        choose_action: Policy mapping the current game to an action

    Returns:
        Evaluation result for this episode
    """
    state = game.reset()
    total_reward = 0.0

    while state.is_alive:
        action = choose_action(game)

        state, reward, done = game.step(action)
        total_reward += reward

        if done:
            break

    return EvaluationResult(
        score=state.score,
        steps=state.steps,
        final_length=state.snake_length,
        success=state.score > 0,
        total_reward=total_reward,
        end_reason=state.end_reason or "unknown"
    )


def build_agent(genome_weights: np.ndarray, game_config: Dict[str, Any],
                agent_config: Dict[str, Any]) -> SnakeAgent:
    """Build an agent whose network weights are the given genome."""
    obs_size = get_observation_size(
        game_config["board_size"],
        game_config["observation_radius"]
    )

    agent = SnakeAgent(
        input_size=obs_size,
        hidden_size=agent_config["hidden_size"],
        second_hidden_size=agent_config["second_hidden_size"],
        activation=agent_config["activation"]
    )
    agent.set_genome(genome_weights)
    return agent


def agent_policy(agent: SnakeAgent) -> Callable[[SnakeGame], Action]:
    """Wrap an agent as a policy for run_episode."""
    return lambda game: agent.predict(game.get_observation())


def evaluate_agent_episode(genome_weights: np.ndarray, game_config: Dict[str, Any],
                          agent_config: Dict[str, Any], episode_seed: Optional[int]) -> EvaluationResult:
    """Evaluate a single episode for an agent.

    This function is designed to be called in a separate process.

    Args:
        genome_weights: Flattened weight vector for the agent
        game_config: Game configuration parameters
        agent_config: Agent configuration parameters
        episode_seed: Random seed for this episode

    Returns:
        Evaluation result for this episode
    """
    game = SnakeGame.from_config(game_config, seed=episode_seed)
    agent = build_agent(genome_weights, game_config, agent_config)
    return run_episode(game, agent_policy(agent))


def max_episode_steps(game_config: Dict[str, Any]) -> int:
    """Upper bound on the steps any episode can last."""
    board_size = game_config["board_size"]
    # Timeout is max_steps + 2 * length, and length can't exceed the board area
    return game_config["max_steps"] + 2 * board_size * board_size


def compute_fitness(result: EvaluationResult, fitness_function: str,
                    game_config: Dict[str, Any]) -> float:
    """Fitness of one episode.

    Args:
        result: Episode result
        fitness_function: "food", "legacy" or "return"
        game_config: Game configuration parameters

    Returns:
        Episode fitness
    """
    if fitness_function == "food":
        # Food eaten; the survival term is < 1 so it only breaks ties between equal food counts
        return result.score + result.steps / (max_episode_steps(game_config) + 1)
    if fitness_function == "legacy":
        # Original formula: food_eaten + 0.1 * steps_survived, with an early death penalty
        fitness = result.score + 0.1 * result.steps
        if result.steps < 50 and result.score == 0:
            fitness -= 2.0
        return fitness
    if fitness_function == "return":
        return result.total_reward
    raise ValueError(f"Unknown fitness function {fitness_function!r}; "
                     f"expected one of {FITNESS_FUNCTIONS}")


def summarize_episodes(episode_results: List[EvaluationResult], fitness_function: str,
                       game_config: Dict[str, Any]) -> GenomeStats:
    """Aggregate episode results into genome statistics."""
    fitness_values = [compute_fitness(r, fitness_function, game_config) for r in episode_results]

    return GenomeStats(
        fitness=float(np.mean(fitness_values)),
        episodes_played=len(episode_results),
        avg_score=float(np.mean([r.score for r in episode_results])),
        avg_length=float(np.mean([r.final_length for r in episode_results])),
        avg_steps=float(np.mean([r.steps for r in episode_results])),
        success_rate=float(np.mean([r.success for r in episode_results])),
        avg_return=float(np.mean([r.total_reward for r in episode_results])),
        starvation_rate=float(np.mean([r.end_reason == "starvation" for r in episode_results]))
    )


class FitnessEvaluator:
    """Parallel fitness evaluator for snake agents.

    The worker pool is created on first use and reused; call close() when done.
    """

    def __init__(self, game_config: Dict[str, Any], agent_config: Dict[str, Any],
                 episodes_per_agent: int = 3, parallel_processes: int = 8,
                 deterministic_seed: Optional[int] = None,
                 fitness_function: str = "food"):
        """Initialize evaluator.

        Args:
            game_config: Game configuration parameters
            agent_config: Agent configuration parameters
            episodes_per_agent: Number of episodes to evaluate per agent
            parallel_processes: Number of parallel processes to use (<= 1 runs in-process)
            deterministic_seed: Base seed for deterministic evaluation
            fitness_function: "food", "legacy" or "return"
        """
        if fitness_function not in FITNESS_FUNCTIONS:
            raise ValueError(f"Unknown fitness function {fitness_function!r}; "
                             f"expected one of {FITNESS_FUNCTIONS}")

        self.game_config = game_config
        self.agent_config = agent_config
        self.episodes_per_agent = episodes_per_agent
        self.parallel_processes = parallel_processes
        self.deterministic_seed = deterministic_seed
        self.fitness_function = fitness_function
        self._executor: Optional[Executor] = None

    def close(self) -> None:
        """Shut down the worker pool."""
        if self._executor is not None:
            self._executor.shutdown()
            self._executor = None

    def __enter__(self) -> "FitnessEvaluator":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()

    def _get_executor(self) -> Executor:
        if self._executor is None:
            self._executor = ProcessPoolExecutor(max_workers=self.parallel_processes)
        return self._executor

    def evaluate_genome(self, genome: Genome) -> GenomeStats:
        """Evaluate a single genome across multiple episodes.

        Args:
            genome: Genome to evaluate

        Returns:
            Genome statistics
        """
        results = self.evaluate_genomes([genome])
        return results[0]

    def evaluate_genomes(self, genomes: List[Genome]) -> List[GenomeStats]:
        """Evaluate multiple genomes in parallel.

        Failed episodes are logged and retried once in-process; if every episode
        fails, or a retry fails, the error is raised instead of scored.

        Args:
            genomes: List of genomes to evaluate

        Returns:
            List of genome statistics
        """
        start_time = time.time()

        # Prepare evaluation tasks
        tasks = []
        for genome_idx, genome in enumerate(genomes):
            for episode in range(self.episodes_per_agent):
                if self.deterministic_seed is not None:
                    # Use deterministic seed based on genome ID and episode
                    episode_seed = self.deterministic_seed + genome.id * 1000 + episode
                else:
                    episode_seed = None

                tasks.append((genome_idx, genome.weights, episode_seed))

        results: Dict[int, List[EvaluationResult]] = {idx: [] for idx in range(len(genomes))}

        if self.parallel_processes <= 1:
            for genome_idx, weights, episode_seed in tasks:
                results[genome_idx].append(evaluate_agent_episode(
                    weights, self.game_config, self.agent_config, episode_seed
                ))
        else:
            self._evaluate_in_pool(tasks, results)

        stats_list = []
        for genome_idx, genome in enumerate(genomes):
            stats = summarize_episodes(results[genome_idx], self.fitness_function, self.game_config)

            # Update genome with stats
            genome.stats = stats
            stats_list.append(stats)

        evaluation_time = time.time() - start_time
        games_per_second = len(tasks) / evaluation_time if evaluation_time > 0 else 0.0

        print(f"Evaluated {len(genomes)} genomes ({len(tasks)} episodes) "
              f"in {evaluation_time:.2f}s ({games_per_second:.0f} games/s)")

        return stats_list

    def _evaluate_in_pool(self, tasks: List[Tuple[int, np.ndarray, Optional[int]]],
                          results: Dict[int, List[EvaluationResult]]) -> None:
        """Run tasks in the worker pool, appending results per genome."""
        executor = self._get_executor()
        future_to_task = {}
        for task in tasks:
            genome_idx, weights, episode_seed = task
            future = executor.submit(
                evaluate_agent_episode,
                weights,
                self.game_config,
                self.agent_config,
                episode_seed
            )
            future_to_task[future] = task

        failures: List[Tuple[Tuple[int, np.ndarray, Optional[int]], BaseException]] = []
        for future in as_completed(future_to_task):
            task = future_to_task[future]
            try:
                results[task[0]].append(future.result())
            except Exception as e:
                failures.append((task, e))

        if not failures:
            return

        for (genome_idx, _, episode_seed), error in failures:
            logger.error(f"Evaluation failed for genome {genome_idx} (seed {episode_seed}): {error!r}")
        first_error = failures[0][1]
        logger.error("First evaluation failure:\n" + "".join(
            traceback.format_exception(type(first_error), first_error, first_error.__traceback__)
        ))

        if len(failures) == len(tasks):
            # Most likely a broken pool or an environment problem; scoring it would hide that
            self.close()
            raise RuntimeError(
                f"All {len(tasks)} evaluation episodes failed; first error: {first_error!r}"
            ) from first_error

        logger.error(f"{len(failures)}/{len(tasks)} episodes failed; retrying them in-process")
        for genome_idx, weights, episode_seed in (task for task, _ in failures):
            results[genome_idx].append(evaluate_agent_episode(
                weights, self.game_config, self.agent_config, episode_seed
            ))


class BatchEvaluator:
    """Batch evaluator that groups agents for more efficient evaluation."""

    def __init__(self, fitness_evaluator: FitnessEvaluator, batch_size: int = 16):
        """Initialize batch evaluator.

        Args:
            fitness_evaluator: Base fitness evaluator
            batch_size: Number of agents to evaluate in each batch
        """
        self.fitness_evaluator = fitness_evaluator
        self.batch_size = batch_size

    def evaluate_population(self, population: List[Genome]) -> List[GenomeStats]:
        """Evaluate entire population in batches.

        Args:
            population: Population to evaluate

        Returns:
            List of genome statistics
        """
        all_stats = []

        # Process population in batches
        for i in range(0, len(population), self.batch_size):
            batch = population[i:i + self.batch_size]
            batch_stats = self.fitness_evaluator.evaluate_genomes(batch)
            all_stats.extend(batch_stats)

        return all_stats
