"""Test starvation, walls, configurable rewards and seeding in the game engine."""

from dataclasses import asdict

import numpy as np
import pytest

from genetic_coil_arena.config import GameConfig
from genetic_coil_arena.snake_core.game import (
    GAME_CONFIG_PARAMS,
    Action,
    Direction,
    Position,
    SnakeGame,
)


def far_food(game: SnakeGame) -> Position:
    """A food position off the snake's current column, so going straight north never eats it."""
    head = game.snake_positions[0]
    return Position((int(head[0]) + game.board_size // 2) % game.board_size, 0)


class TestConfigWiring:
    """SnakeGame defaults and GameConfig must not drift apart."""

    def test_defaults_match_game_config(self):
        game = SnakeGame(seed=1)
        config = GameConfig()
        for param in GAME_CONFIG_PARAMS:
            assert getattr(game, param) == getattr(config, param), param

    def test_from_config_dataclass_and_dict(self):
        config = GameConfig(board_size=12, starvation_steps=7, walls=True, food_reward=3.0)
        for source in (config, asdict(config)):
            game = SnakeGame.from_config(source, seed=3)
            assert game.board_size == 12
            assert game.starvation_steps == 7
            assert game.walls is True
            assert game.food_reward == 3.0
            assert game.seed == 3

    def test_from_config_missing_keys_use_defaults(self):
        # Old checkpoints only stored the original fields
        old = {"board_size": 16, "observation_radius": 3, "max_steps": 500}
        game = SnakeGame.from_config(old)
        assert game.board_size == 16
        assert game.starvation_steps == GameConfig().starvation_steps


class TestStarvation:
    """The hunger timer."""

    def test_dies_after_starvation_steps_without_food(self):
        game = SnakeGame(board_size=16, starvation_steps=5, seed=0)
        game.food_position = far_food(game)

        for _ in range(4):
            state, _, done = game.step(Action.STRAIGHT)
            assert not done and state.is_alive

        state, _, done = game.step(Action.STRAIGHT)
        assert done
        assert not state.is_alive
        assert state.end_reason == "starvation"
        assert state.steps == 5

    def test_eating_resets_the_timer(self):
        game = SnakeGame(board_size=16, starvation_steps=3, seed=0)
        head = game.snake_positions[0]
        game.food_position = Position(int(head[0]), int(head[1]) - 2)

        game.step(Action.STRAIGHT)
        state, _, _ = game.step(Action.STRAIGHT)  # Eats
        assert state.score == 1
        game.food_position = far_food(game)

        for _ in range(2):
            state, _, done = game.step(Action.STRAIGHT)
            assert not done
        state, _, done = game.step(Action.STRAIGHT)
        assert done and state.end_reason == "starvation"
        assert state.steps == 5

    def test_zero_disables_starvation(self):
        game = SnakeGame(board_size=16, max_steps=50, starvation_steps=0, seed=0)
        game.food_position = far_food(game)

        done, state = False, None
        while not done:
            state, _, done = game.step(Action.STRAIGHT)
        assert state.end_reason == "timeout"
        assert state.is_alive


class TestWalls:
    """The optional walls mode."""

    def test_leaving_the_board_kills(self):
        game = SnakeGame(board_size=8, walls=True, seed=0)
        game.snake_positions = np.array([[4, 0]], dtype=np.int32)
        game.snake_direction = Direction.NORTH

        state, reward, done = game.step(Action.STRAIGHT)
        assert done
        assert not state.is_alive
        assert state.end_reason == "wall"
        assert reward == game.death_penalty

    def test_observation_shows_outside_as_deadly(self):
        game = SnakeGame(board_size=8, observation_radius=2, walls=True, seed=0)
        game.snake_positions = np.array([[0, 0]], dtype=np.int32)
        game.snake_direction = Direction.NORTH
        game.food_position = Position(6, 6)

        grid = game.get_observation()[:25].reshape(5, 5)
        assert np.all(grid[:2, :] == 1)  # Rows above the board
        assert np.all(grid[:, :2] == 1)  # Columns left of the board
        assert grid[2, 2] == 1  # The head itself
        assert np.all(grid[3:, 3:] == 0)  # Empty board cells

    def test_wrap_observation_sees_across_the_edge(self):
        game = SnakeGame(board_size=8, observation_radius=2, walls=False, seed=0)
        game.snake_positions = np.array([[0, 0]], dtype=np.int32)
        game.snake_direction = Direction.NORTH
        game.food_position = Position(6, 6)

        grid = game.get_observation()[:25].reshape(5, 5)
        assert grid[0, 0] == 2  # Food at (6, 6) is two cells up-left across the wrap


class TestRewards:
    """GameConfig reward fields drive the per-step rewards."""

    def make_game(self) -> SnakeGame:
        return SnakeGame(board_size=16, seed=0, food_reward=5.0, step_reward=0.0,
                         death_penalty=-7.0, early_death_threshold=10)

    def test_food_and_step_rewards(self):
        game = self.make_game()
        head = game.snake_positions[0]
        game.food_position = Position(int(head[0]), int(head[1]) - 2)

        _, reward, _ = game.step(Action.STRAIGHT)
        assert reward == 0.0
        _, reward, _ = game.step(Action.STRAIGHT)
        assert reward == 5.0

    @pytest.mark.parametrize("steps_before, expected", [(0, -7.0), (10, 0.0)])
    def test_death_penalty_only_before_threshold(self, steps_before, expected):
        game = self.make_game()
        game.steps = steps_before
        game.snake_positions = np.array([[8, 8], [7, 8], [6, 8]], dtype=np.int32)
        game.snake_direction = Direction.WEST

        state, reward, done = game.step(Action.STRAIGHT)
        assert done and state.end_reason == "collision"
        assert reward == expected


class TestSeeding:
    """Games use a private RNG."""

    def test_does_not_touch_global_rng(self):
        np.random.seed(123)
        expected = np.random.rand()

        np.random.seed(123)
        game = SnakeGame(seed=5)
        game.step(Action.STRAIGHT)
        game.reset()
        assert np.random.rand() == expected

    def test_food_sequence_matches_legacy_global_seeding(self):
        # The original engine called np.random.seed(seed) then np.random.randint for food
        rng = np.random.RandomState(42)
        expected = Position(rng.randint(0, 32), rng.randint(0, 32))
        assert SnakeGame(board_size=32, seed=42).food_position == expected
