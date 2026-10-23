"""Test the core snake game engine."""

import pytest
import numpy as np

from genetic_coil_arena.snake_core.game import (
    SnakeGame, Direction, Action, Position, GameState
)


class TestSnakeGame:
    """Test the SnakeGame class."""
    
    def test_initialization(self):
        """Test game initialization."""
        game = SnakeGame(board_size=16, seed=42)
        
        assert game.board_size == 16
        assert game.observation_radius == 5
        assert game.max_steps == 2000
        assert game.is_alive is True
        assert game.score == 0
        assert game.steps == 0
    
    def test_reset(self):
        """Test game reset functionality."""
        game = SnakeGame(seed=42)
        
        # Play a few moves
        game.step(Action.STRAIGHT)
        game.step(Action.LEFT)
        
        # Reset and check initial state
        state = game.reset()
        
        assert state.is_alive is True
        assert state.score == 0
        assert state.steps == 0
        assert state.snake_length == 1
        assert state.snake_direction == Direction.NORTH
    
    def test_movement(self):
        """Test snake movement."""
        game = SnakeGame(seed=42)
        initial_state = game.reset()
        initial_head = initial_state.head_position
        
        # Move straight (north)
        state, reward, done = game.step(Action.STRAIGHT)
        new_head = state.head_position
        
        # Check head moved north (y decreased)
        expected_y = (initial_head.y - 1) % game.board_size
        assert new_head.y == expected_y
        assert new_head.x == initial_head.x
        assert not done
    
    def test_direction_changes(self):
        """Test relative direction changes."""
        game = SnakeGame(seed=42)
        game.reset()
        
        # Initial direction is NORTH
        assert game.snake_direction == Direction.NORTH
        
        # Turn left (should become WEST)
        game.step(Action.LEFT)
        assert game.snake_direction == Direction.WEST
        
        # Turn right (should become NORTH again)
        game.step(Action.RIGHT)
        assert game.snake_direction == Direction.NORTH
        
        # Turn right (should become EAST)
        game.step(Action.RIGHT)
        assert game.snake_direction == Direction.EAST
    
    def test_food_eating(self):
        """Test food eating and growth."""
        game = SnakeGame(seed=42)
        initial_state = game.reset()
        initial_length = initial_state.snake_length
        
        # Manually place food at snake's next position
        head = initial_state.head_position
        next_pos = Position(head.x, (head.y - 1) % game.board_size)
        game.food_position = next_pos
          # Move to food
        state, reward, done = game.step(Action.STRAIGHT)
        
        assert state.score == 1
        assert state.snake_length == initial_length + 1
        assert reward == 1.0  # Food reward
        assert not done
    
    def test_self_collision(self):
        """Test self collision detection."""
        game = SnakeGame(seed=42)
        game.reset()
        
        # Create a setup that will actually cause collision
        # Snake facing left toward its own body
        game.snake_positions = np.array([
            [16, 16],  # Head
            [15, 16],  # Body (to the left)
            [14, 16],  # Body (further left)
        ], dtype=np.int32)
        game.snake_direction = Direction.WEST
        
        # Move straight into body - should collide
        state, reward, done = game.step(Action.STRAIGHT)
        
        assert not state.is_alive
        assert done
        assert reward <= 0  # Death penalty or no reward
    
    def test_toroidal_wrapping(self):
        """Test that the board wraps around at edges."""
        game = SnakeGame(board_size=8, seed=42)
        game.reset()
        
        # Place snake at top edge
        game.snake_positions = np.array([[4, 0]], dtype=np.int32)
        game.snake_direction = Direction.NORTH
        
        # Move north (should wrap to bottom)
        state, _, _ = game.step(Action.STRAIGHT)
        new_head = state.head_position
        
        assert new_head.y == 7  # Wrapped to bottom of 8x8 board
        assert new_head.x == 4  # X should remain same
    
    def test_observation_size(self):
        """Test observation vector size."""
        game = SnakeGame(board_size=32, observation_radius=5, seed=42)
        game.reset()
        
        observation = game.get_observation()
        
        # Grid (11x11) + length (1) + food direction (2) = 123 elements
        expected_size = 11 * 11 + 1 + 2
        assert len(observation) == expected_size
        assert observation.dtype == np.float32
    
    def test_deterministic_behavior(self):
        """Test that same seed produces same results."""
        game1 = SnakeGame(seed=42)
        game2 = SnakeGame(seed=42)
        
        state1 = game1.reset()
        state2 = game2.reset()
        
        # Initial food positions should be same
        assert state1.food_position == state2.food_position
        
        # Same sequence of moves should produce same results
        for _ in range(10):
            state1, _, _ = game1.step(Action.STRAIGHT)
            state2, _, _ = game2.step(Action.STRAIGHT)
            
            assert state1.head_position == state2.head_position
            assert state1.score == state2.score
    
    def test_timeout(self):
        """Test episode timeout."""
        game = SnakeGame(max_steps=10, seed=42)
        game.reset()
        
        # Play until timeout
        done = False
        steps = 0
        while not done and steps < 20:  # Safety limit
            _, _, done = game.step(Action.STRAIGHT)
            steps += 1
        
        assert done
        assert steps >= 10  # Should timeout around max_steps


class TestGameComponents:
    """Test individual game components."""
    
    def test_position_namedtuple(self):
        """Test Position namedtuple."""
        pos = Position(5, 10)
        assert pos.x == 5
        assert pos.y == 10
        assert pos[0] == 5  # Tuple access
        assert pos[1] == 10
    
    def test_direction_enum(self):
        """Test Direction enum values."""
        assert Direction.NORTH == 0
        assert Direction.EAST == 1
        assert Direction.SOUTH == 2
        assert Direction.WEST == 3
    
    def test_action_enum(self):
        """Test Action enum values."""
        assert Action.STRAIGHT == 0
        assert Action.LEFT == 1
        assert Action.RIGHT == 2
    
    def test_game_state_properties(self):
        """Test GameState properties."""
        # Create mock game state
        snake_positions = np.array([[5, 5], [5, 6]], dtype=np.int32)
        board = np.zeros((10, 10), dtype=np.int32)
        
        state = GameState(
            board=board,
            snake_positions=snake_positions,
            snake_direction=Direction.NORTH,
            food_position=Position(3, 3),
            score=1,
            steps=50,
            is_alive=True
        )
        
        assert state.snake_length == 2
        assert state.head_position == Position(5, 5)


if __name__ == "__main__":
    pytest.main([__file__])
