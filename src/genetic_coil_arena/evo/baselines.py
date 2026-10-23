"""Reference policies, played through the same game and fitness code as evolved agents."""

from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np

from ..snake_core.game import (
    Action,
    SnakeGame,
    _check_self_collision,
    _get_new_direction,
    _next_position,
)
from .fitness import EvaluationResult, run_episode

Policy = Callable[[SnakeGame], Action]
PolicyFactory = Callable[[Optional[int]], Policy]

BASELINE_POLICIES = ("random", "straight", "greedy")

# Benchmark episodes use seeds from here up; training seeds (base + genome_id * 1000
# + episode, genome ids < 1e6) stay below it, so benchmarks are held out.
BENCHMARK_SEED_OFFSET = 2_000_000_000


def straight_policy(game: SnakeGame) -> Action:
    """Always keep going straight."""
    return Action.STRAIGHT


def make_random_policy(seed: Optional[int] = None) -> Policy:
    """Uniformly random actions, from an RNG independent of the game's food RNG."""
    rng = np.random.default_rng(None if seed is None else [seed, 1])
    return lambda game: Action(int(rng.integers(3)))


def safe_moves(game: SnakeGame) -> List[Tuple[Action, int, int]]:
    """Actions that don't die on the next step, with the head position they lead to."""
    head = game.snake_positions[0]
    moves = []
    for action in (Action.STRAIGHT, Action.LEFT, Action.RIGHT):
        direction = _get_new_direction(int(game.snake_direction), int(action))
        x, y, hit_wall = _next_position(
            int(head[0]), int(head[1]), direction, game.board_size, not game.walls
        )
        if not hit_wall and not _check_self_collision(game.snake_positions, x, y):
            moves.append((action, x, y))
    return moves


def greedy_policy(game: SnakeGame) -> Action:
    """Move toward the food, never into an immediate death.

    Ties prefer STRAIGHT, then LEFT, then RIGHT. No lookahead, so it can still trap itself.
    """
    moves = safe_moves(game)
    if not moves:
        return Action.STRAIGHT  # Every move dies
    return min(moves, key=lambda move: game.food_distance(move[1], move[2]))[0]


def make_policy(name: str, seed: Optional[int] = None) -> Policy:
    """Create a baseline policy by name."""
    if name == "random":
        return make_random_policy(seed)
    if name == "straight":
        return straight_policy
    if name == "greedy":
        return greedy_policy
    raise ValueError(f"Unknown baseline policy {name!r}; expected one of {BASELINE_POLICIES}")


def baseline_factory(name: str) -> PolicyFactory:
    """Factory that builds a fresh baseline policy for each episode seed."""
    make_policy(name)  # Validate the name up front
    return lambda seed: make_policy(name, seed)


def evaluate_policy(policy_factory: PolicyFactory, game_config: Dict[str, Any],
                    seeds: Iterable[int]) -> List[EvaluationResult]:
    """Play one episode per seed.

    Args:
        policy_factory: Builds the policy for an episode from its seed
        game_config: Game configuration parameters
        seeds: Episode seeds (food placement, and the random policy's RNG)

    Returns:
        One result per seed
    """
    results = []
    for seed in seeds:
        game = SnakeGame.from_config(game_config, seed=seed)
        results.append(run_episode(game, policy_factory(seed)))
    return results


def benchmark_seeds(episodes: int, offset: int = BENCHMARK_SEED_OFFSET) -> range:
    """Held-out episode seeds for benchmarking."""
    return range(offset, offset + episodes)
