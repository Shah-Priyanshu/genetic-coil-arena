"""Core snake game engine - renderer agnostic."""

import numpy as np
import numba  # type: ignore
from dataclasses import dataclass
from enum import IntEnum
from typing import Tuple, Optional, NamedTuple


class Direction(IntEnum):
    """Snake movement directions."""
    NORTH = 0
    EAST = 1
    SOUTH = 2
    WEST = 3


class Action(IntEnum):
    """Relative actions the snake can take."""
    STRAIGHT = 0
    LEFT = 1
    RIGHT = 2


class Position(NamedTuple):
    """2D position on the game board."""
    x: int
    y: int


@dataclass
class GameState:
    """Immutable game state snapshot."""
    
    board: np.ndarray  # Shape: (height, width) 
    snake_positions: np.ndarray  # Shape: (length, 2)
    snake_direction: Direction
    food_position: Position
    score: int
    steps: int
    is_alive: bool
    
    @property
    def snake_length(self) -> int:
        """Current snake length."""
        return len(self.snake_positions)
    
    @property
    def head_position(self) -> Position:
        """Current head position."""
        head = self.snake_positions[0]
        return Position(int(head[0]), int(head[1]))


@numba.njit(cache=True)  # type: ignore
def _move_position(pos_x: int, pos_y: int, direction: int, board_size: int) -> Tuple[int, int]:
    """Move position in given direction with toroidal wrapping."""
    if direction == Direction.NORTH:
        return pos_x, (pos_y - 1) % board_size
    elif direction == Direction.EAST:
        return (pos_x + 1) % board_size, pos_y
    elif direction == Direction.SOUTH:
        return pos_x, (pos_y + 1) % board_size
    elif direction == Direction.WEST:
        return (pos_x - 1) % board_size, pos_y
    else:
        return pos_x, pos_y


@numba.njit(cache=True)  # type: ignore
def _get_new_direction(current_direction: int, action: int) -> int:
    """Get new direction based on current direction and relative action."""
    if action == Action.STRAIGHT:
        return current_direction
    elif action == Action.LEFT:
        return (current_direction - 1) % 4
    elif action == Action.RIGHT:
        return (current_direction + 1) % 4
    else:
        return current_direction


@numba.njit(cache=True)  # type: ignore
def _check_self_collision(snake_positions: np.ndarray, new_head_x: int, new_head_y: int) -> bool:
    """Check if new head position collides with snake body."""
    for i in range(len(snake_positions)):
        if snake_positions[i, 0] == new_head_x and snake_positions[i, 1] == new_head_y:
            return True
    return False


@numba.njit(cache=True)  # type: ignore
def _update_board(board: np.ndarray, snake_positions: np.ndarray, food_x: int, food_y: int) -> None:
    """Update board representation with current snake and food positions."""
    board.fill(0)  # Clear board
    
    # Place snake
    for i in range(len(snake_positions)):
        x, y = int(snake_positions[i, 0]), int(snake_positions[i, 1])
        board[y, x] = 1  # Snake body
    
    # Place food
    board[food_y, food_x] = 2


@numba.njit(cache=True)  # type: ignore
def _get_observation(board: np.ndarray, head_x: int, head_y: int, 
                    direction: int, radius: int) -> np.ndarray:
    """Get egocentric observation window around snake head."""
    board_size = board.shape[0]
    window_size = 2 * radius + 1
    observation = np.zeros((window_size, window_size), dtype=np.float32)
    
    # Get observation in world coordinates
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            world_x = (head_x + dx) % board_size
            world_y = (head_y + dy) % board_size
            obs_x = dx + radius
            obs_y = dy + radius
            observation[obs_y, obs_x] = board[world_y, world_x]
    
    # Rotate observation based on snake direction to make it egocentric
    if direction == Direction.NORTH:
        pass  # No rotation needed
    elif direction == Direction.EAST:
        observation = np.rot90(observation, k=-1)
    elif direction == Direction.SOUTH:
        observation = np.rot90(observation, k=2)
    elif direction == Direction.WEST:
        observation = np.rot90(observation, k=1)
    
    return observation


class SnakeGame:
    """High-performance snake game implementation."""
    
    def __init__(self, board_size: int = 32, observation_radius: int = 5, 
                 max_steps: int = 500, seed: Optional[int] = None):
        """Initialize game.
        
        Args:
            board_size: Size of the square game board
            observation_radius: Radius of observation window (creates 2*radius+1 square)
            max_steps: Maximum steps before timeout            seed: Random seed for reproducibility
        """
        self.board_size = board_size
        self.observation_radius = observation_radius
        self.max_steps = max_steps
        self.seed = seed
        
        if seed is not None:
            np.random.seed(seed)
        
        self.reset()
    
    def reset(self) -> GameState:
        """Reset game to initial state."""
        # Reseed for deterministic behavior
        if self.seed is not None:
            np.random.seed(self.seed)
            
        # Initialize snake in center of board
        center = self.board_size // 2
        self.snake_positions = np.array([[center, center]], dtype=np.int32)
        self.snake_direction = Direction.NORTH
        
        # Place food randomly (avoid snake)
        self._place_food()
        
        # Initialize other state
        self.score = 0
        self.steps = 0
        self.is_alive = True
        
        # Create board representation
        self.board = np.zeros((self.board_size, self.board_size), dtype=np.int32)
        
        return self._get_current_state()
    
    def step(self, action: Action) -> Tuple[GameState, float, bool]:
        """Execute one game step.
        
        Args:
            action: Action to take (STRAIGHT, LEFT, RIGHT)
            
        Returns:
            Tuple of (new_state, reward, done)
        """
        if not self.is_alive:
            return self._get_current_state(), 0.0, True
        
        # Update direction based on action
        self.snake_direction = Direction(_get_new_direction(self.snake_direction, action))
        
        # Move snake head
        head = self.snake_positions[0]
        new_head_x, new_head_y = _move_position(
            int(head[0]), int(head[1]), self.snake_direction, self.board_size
        )
        
        # Check self collision
        if _check_self_collision(self.snake_positions, new_head_x, new_head_y):
            self.is_alive = False
            reward = -2.0 if self.steps < 50 else 0.0  # Early death penalty
            return self._get_current_state(), reward, True
        
        # Check food collision
        ate_food = (new_head_x == self.food_position.x and 
                   new_head_y == self.food_position.y)
        
        if ate_food:
            # Grow snake
            new_positions = np.zeros((len(self.snake_positions) + 1, 2), dtype=np.int32)
            new_positions[0] = [new_head_x, new_head_y]
            new_positions[1:] = self.snake_positions
            self.snake_positions = new_positions
            
            self.score += 1
            self._place_food()
            reward = 1.0
        else:
            # Move snake (replace tail with new head)
            new_positions = self.snake_positions.copy()
            new_positions[1:] = self.snake_positions[:-1]
            new_positions[0] = [new_head_x, new_head_y]
            self.snake_positions = new_positions
            reward = 0.1  # Small reward for staying alive
        
        self.steps += 1
        
        # Check timeout
        max_steps_for_length = self.max_steps + 2 * len(self.snake_positions)
        done = self.steps >= max_steps_for_length
        
        return self._get_current_state(), reward, done
    
    def get_observation(self) -> np.ndarray:
        """Get current observation for the agent.
        
        Returns:
            Flattened observation vector containing:
            - Egocentric grid view (flattened)
            - Snake length (normalized)
            - Food direction vector (x, y)
        """
        state = self._get_current_state()
        
        # Get egocentric grid observation
        grid_obs = _get_observation(
            state.board, 
            state.head_position.x, 
            state.head_position.y,
            state.snake_direction,
            self.observation_radius
        )
        
        # Snake length (normalized by max reasonable length)
        length_obs = np.array([state.snake_length / (self.board_size * 2)], dtype=np.float32)
        
        # Food direction (normalized)
        head_pos = state.head_position
        food_dx = state.food_position.x - head_pos.x
        food_dy = state.food_position.y - head_pos.y
        
        # Handle toroidal distance
        if abs(food_dx) > self.board_size // 2:
            food_dx = food_dx - np.sign(food_dx) * self.board_size
        if abs(food_dy) > self.board_size // 2:
            food_dy = food_dy - np.sign(food_dy) * self.board_size
            
        food_direction = np.array([
            food_dx / self.board_size,
            food_dy / self.board_size
        ], dtype=np.float32)
        
        # Combine all observations
        return np.concatenate([
            grid_obs.flatten(),
            length_obs,
            food_direction
        ])
    
    def get_current_state(self) -> GameState:
        """Get current game state for visualization purposes.
        
        Returns:
            Current game state
        """
        return self._get_current_state()

    def _place_food(self) -> None:
        """Place food at random empty position."""
        while True:
            x = np.random.randint(0, self.board_size)
            y = np.random.randint(0, self.board_size)
            
            # Check if position is occupied by snake
            occupied = False
            for i in range(len(self.snake_positions)):
                if self.snake_positions[i, 0] == x and self.snake_positions[i, 1] == y:
                    occupied = True
                    break
            
            if not occupied:
                self.food_position = Position(x, y)
                break
    
    def _get_current_state(self) -> GameState:
        """Get current game state."""
        # Update board representation
        _update_board(self.board, self.snake_positions, 
                     self.food_position.x, self.food_position.y)
        
        return GameState(
            board=self.board.copy(),
            snake_positions=self.snake_positions.copy(),
            snake_direction=self.snake_direction,
            food_position=self.food_position,
            score=self.score,
            steps=self.steps,
            is_alive=self.is_alive
        )
