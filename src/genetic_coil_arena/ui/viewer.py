"""Real-time game visualization using Pygame."""

import pygame
import numpy as np
from typing import Optional, Tuple, List, Dict, Any, Union
import time
from dataclasses import dataclass
from enum import Enum
import threading
import queue

from ..snake_core.game import SnakeGame, GameState, Direction, Action
from ..snake_core.agent import SnakeAgent
from ..config import Config


class ViewMode(Enum):
    """Different visualization modes."""
    GAME_ONLY = "game_only"
    WITH_OBSERVATION = "with_observation"
    AGENT_ANALYSIS = "agent_analysis"
    EVOLUTION_PROGRESS = "evolution_progress"


@dataclass
class VisualizationConfig:
    """Configuration for the visualization system."""
    window_width: int = 1400
    window_height: int = 900
    cell_size: int = 20
    fps: int = 10
    colors: Optional[Dict[str, Tuple[int, int, int]]] = None
    show_grid: bool = True
    show_observation_overlay: bool = True
    show_neural_activity: bool = True
    
    def __post_init__(self):
        if self.colors is None:
            self.colors = {
                'background': (20, 20, 20),      # Dark gray
                'grid': (60, 60, 60),            # Medium gray
                'snake_head': (0, 255, 100),     # Bright green
                'snake_body': (0, 200, 80),      # Green
                'food': (255, 100, 100),         # Red
                'observation': (100, 100, 255),  # Blue
                'text': (255, 255, 255),         # White
                'info_bg': (40, 40, 40),         # Dark background for info
                'neural_positive': (0, 255, 0),  # Green for positive activations
                'neural_negative': (255, 0, 0),  # Red for negative activations
                'progress_bar': (100, 255, 100), # Light green
                'elite': (255, 215, 0),          # Gold for elite agents
            }


@dataclass
class GameFrame:
    """A single frame of game state for visualization."""
    game_state: GameState
    observation: np.ndarray
    action: Optional[Action] = None
    neural_outputs: Optional[np.ndarray] = None
    hidden_activations: Optional[np.ndarray] = None
    fitness_score: Optional[float] = None
    generation: Optional[int] = None
    agent_id: Optional[int] = None

class GameViewer:
    """Real-time visualization of Snake game with agent behavior."""    def __init__(self, config: Optional[VisualizationConfig] = None):
        """Initialize the game viewer.
        
        Args:
            config: Visualization configuration
        """
        self.config = config if config is not None else VisualizationConfig()
        
        # Initialize Pygame
        pygame.init()
        pygame.font.init()
        
        # Create display
        self.screen = pygame.display.set_mode(
            (self.config.window_width, self.config.window_height)
        )
        pygame.display.set_caption("Genetic Coil Arena - Game Viewer")
        
        # Fonts
        self.font_large = pygame.font.Font(None, 24)
        self.font_medium = pygame.font.Font(None, 18)
        self.font_small = pygame.font.Font(None, 14)
        
        # Clock for FPS control
        self.clock = pygame.time.Clock()
        
        # Game area calculations
        self.game_area_width = 600
        self.game_area_height = 600
        self.game_area_x = 50
        self.game_area_y = 50
        
        # Info panel area
        self.info_panel_x = self.game_area_x + self.game_area_width + 20
        self.info_panel_width = self.config.window_width - self.info_panel_x - 20
        
        # Running state
        self.running = False
        
    def run_game_session(self, game: SnakeGame, agent: Optional[SnakeAgent] = None,
                        max_steps: int = 1000, mode: ViewMode = ViewMode.GAME_ONLY) -> dict:
        """Run a single game session with visualization."""
        self.running = True
        game.reset()
        
        step_count = 0
        total_reward = 0
        start_time = time.time()
        
        stats = {
            'final_score': 0,
            'steps_taken': 0,
            'total_reward': 0,
            'survival_time': 0,
            'final_length': 0,
            'reason_ended': 'unknown'
        }
        
        while self.running and step_count < max_steps:
            # Handle events
            action = self._handle_events(agent is None)
            
            # Get agent action if agent is provided
            if agent is not None:
                observation = game.get_observation()
                action = agent.predict(observation)
            
            # Step the game if we have an action
            if action is not None:
                state, reward, done = game.step(action)
                total_reward += reward
                step_count += 1
                
                if done:
                    stats['reason_ended'] = 'game_over'
                    break
            
            # Render the current state
            self._render_frame(game, agent, mode, step_count, total_reward)
            
            # Control frame rate
            self.clock.tick(self.config.fps)
        
        # Update final statistics
        current_state = game.get_state()
        stats.update({
            'final_score': current_state.score,
            'steps_taken': step_count,
            'total_reward': total_reward,
            'survival_time': time.time() - start_time,
            'final_length': current_state.snake_length,
        })
        
        if step_count >= max_steps:
            stats['reason_ended'] = 'timeout'
            
        return stats
    
    def _handle_events(self, manual_control: bool = False) -> Optional[Action]:
        """Handle pygame events and return action if manual control."""
        action = None
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif manual_control:
                    if event.key == pygame.K_UP or event.key == pygame.K_w:
                        action = Action.STRAIGHT
                    elif event.key == pygame.K_LEFT or event.key == pygame.K_a:
                        action = Action.LEFT
                    elif event.key == pygame.K_RIGHT or event.key == pygame.K_d:
                        action = Action.RIGHT
        
        return action
    
    def _render_frame(self, game: SnakeGame, agent: Optional[SnakeAgent],
                     mode: ViewMode, step_count: int, total_reward: float):
        """Render a single frame."""
        # Clear screen
        self.screen.fill(self.config.colors['background'])        # Get current game state
        state = game.get_current_state()
        
        # Render game board
        self._render_game_board(state)
        
        # Render info panel
        self._render_info_panel(state, agent, step_count, total_reward, mode)
        
        # Update display
        pygame.display.flip()
    
    def _render_game_board(self, state: GameState):
        """Render the main game board."""
        board_height, board_width = state.board.shape
        
        # Calculate cell size to fit in game area
        cell_width = self.game_area_width // board_width
        cell_height = self.game_area_height // board_height
        cell_size = min(cell_width, cell_height)
        
        # Center the board in the game area
        board_pixel_width = board_width * cell_size
        board_pixel_height = board_height * cell_size
        board_x = self.game_area_x + (self.game_area_width - board_pixel_width) // 2
        board_y = self.game_area_y + (self.game_area_height - board_pixel_height) // 2
        
        # Draw grid
        for x in range(board_width + 1):
            pygame.draw.line(
                self.screen,
                self.config.colors['grid'],
                (board_x + x * cell_size, board_y),
                (board_x + x * cell_size, board_y + board_pixel_height)
            )
        
        for y in range(board_height + 1):
            pygame.draw.line(
                self.screen,
                self.config.colors['grid'],
                (board_x, board_y + y * cell_size),
                (board_x + board_pixel_width, board_y + y * cell_size)
            )
        
        # Draw food
        food_x, food_y = state.food_position
        food_rect = pygame.Rect(
            board_x + food_x * cell_size + 2,
            board_y + food_y * cell_size + 2,
            cell_size - 4,
            cell_size - 4
        )
        pygame.draw.ellipse(self.screen, self.config.colors['food'], food_rect)
        
        # Draw snake
        for i, (x, y) in enumerate(state.snake_positions):
            x, y = int(x), int(y)
            snake_rect = pygame.Rect(
                board_x + x * cell_size + 1,
                board_y + y * cell_size + 1,
                cell_size - 2,
                cell_size - 2
            )
            
            color = self.config.colors['snake_head'] if i == 0 else self.config.colors['snake_body']
            pygame.draw.rect(self.screen, color, snake_rect)
            
            # Draw direction indicator on head
            if i == 0:
                center_x = board_x + x * cell_size + cell_size // 2
                center_y = board_y + y * cell_size + cell_size // 2
                
                # Draw direction arrow based on snake direction
                if state.snake_direction == Direction.NORTH:
                    points = [(center_x, center_y - 6), (center_x - 4, center_y + 2), (center_x + 4, center_y + 2)]
                elif state.snake_direction == Direction.SOUTH:
                    points = [(center_x, center_y + 6), (center_x - 4, center_y - 2), (center_x + 4, center_y - 2)]
                elif state.snake_direction == Direction.EAST:
                    points = [(center_x + 6, center_y), (center_x - 2, center_y - 4), (center_x - 2, center_y + 4)]
                elif state.snake_direction == Direction.WEST:
                    points = [(center_x - 6, center_y), (center_x + 2, center_y - 4), (center_x + 2, center_y + 4)]
                
                pygame.draw.polygon(self.screen, self.config.colors['background'], points)
    
    def _render_info_panel(self, state: GameState, agent: Optional[SnakeAgent],
                          step_count: int, total_reward: float, mode: ViewMode):
        """Render the information panel."""
        # Info panel background
        info_rect = pygame.Rect(
            self.info_panel_x, self.game_area_y,
            self.info_panel_width, self.game_area_height
        )
        pygame.draw.rect(self.screen, self.config.colors['info_bg'], info_rect)
        
        # Title
        title_text = self.font_large.render("Game Information", True, self.config.colors['text'])
        self.screen.blit(title_text, (self.info_panel_x + 10, self.game_area_y + 10))
        
        y_offset = 50
        line_height = 25
        
        # Game statistics
        info_lines = [
            f"Score: {state.score}",
            f"Length: {state.snake_length}",
            f"Steps: {step_count}",
            f"Reward: {total_reward:.2f}",
            f"Direction: {state.snake_direction.name}",
            "",
            f"Mode: {mode.value}",
        ]
        
        if agent is not None:
            info_lines.extend([
                "",
                "Agent Info:",
                f"Input size: {agent.input_size}",
                f"Hidden: {agent.hidden_size}→{agent.second_hidden_size}→3",
                f"Parameters: {self._count_agent_parameters(agent)}",
            ])
        
        # Controls
        info_lines.extend([
            "",
            "Controls:",
            "ESC - Quit",
            "WASD/Arrows - Manual control",
        ])
        
        for i, line in enumerate(info_lines):
            if line:  # Skip empty lines
                text = self.font_medium.render(line, True, self.config.colors['text'])
                self.screen.blit(text, (self.info_panel_x + 10, self.game_area_y + y_offset + i * line_height))
    
    def _count_agent_parameters(self, agent: SnakeAgent) -> int:
        """Count total parameters in agent."""
        total = 0
        total += agent.weights1.size + agent.bias1.size
        total += agent.weights2.size + agent.bias2.size
        total += agent.weights3.size + agent.bias3.size
        return total
    
    def close(self):
        """Clean up and close the viewer."""
        pygame.quit()


def demo_viewer():
    """Demo function to test the viewer."""
    from ..config import Config
    
    # Create configuration
    config = Config()
    config.game.board_size = 16
    config.game.observation_radius = 3
    
    # Create game and agent
    game = SnakeGame(
        board_size=config.game.board_size,
        observation_radius=config.game.observation_radius,
        seed=42
    )
    
    obs_size = game.get_observation().shape[0]
    agent = SnakeAgent(input_size=obs_size, seed=42)
    
    # Create viewer
    viewer = GameViewer()
    
    print("Starting game viewer demo...")
    print("Controls: WASD or arrow keys for manual control, ESC to quit")
    
    try:
        # Run a game with the agent
        stats = viewer.run_game_session(
            game=game,
            agent=agent,
            max_steps=1000,
            mode=ViewMode.WITH_OBSERVATION
        )
        
        print(f"Game completed!")
        print(f"Final score: {stats['final_score']}")
        print(f"Steps taken: {stats['steps_taken']}")
        print(f"Total reward: {stats['total_reward']:.2f}")
        print(f"Survival time: {stats['survival_time']:.2f}s")
        print(f"Reason ended: {stats['reason_ended']}")
        
    finally:
        viewer.close()


if __name__ == "__main__":
    demo_viewer()
