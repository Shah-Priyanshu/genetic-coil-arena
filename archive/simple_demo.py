"""Simplified demo without Numba compilation."""

import sys
import time
from pathlib import Path
import numpy as np

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent / "src"))

from genetic_coil_arena.snake_core.game import SnakeGame, Action


def simple_agent_demo():
    """Simple demo showing the basic system working."""
    print("🚀 Genetic Coil Arena - Simple Demo\n")
    print("=" * 50)
    
    # Test basic game functionality
    print("🎮 Testing basic game functionality...")
    
    game = SnakeGame(board_size=16, seed=42)
    state = game.reset()
    
    print(f"  Initial state: score={state.score}, length={state.snake_length}")
    
    # Create a simple hardcoded "agent" that just goes straight
    total_reward = 0
    steps = 0
    
    while state.is_alive and steps < 100:
        # Simple strategy: mostly go straight, occasionally turn
        if steps % 20 == 0:
            action = Action.LEFT
        elif steps % 17 == 0:
            action = Action.RIGHT
        else:
            action = Action.STRAIGHT
            
        state, reward, done = game.step(action)
        total_reward += reward
        steps += 1
        
        if done:
            break
    
    print(f"  Final result: {steps} steps, score={state.score}, total_reward={total_reward:.1f}")
    
    # Test observation functionality
    print("\n🔍 Testing observation system...")
    
    game.reset()
    observation = game.get_observation()
    
    print(f"  Observation shape: {observation.shape}")
    print(f"  Observation type: {observation.dtype}")
    print(f"  Sample values: {observation[:5]}")
    
    print("\n✅ Basic demo completed successfully!")
    print("\nNext steps:")
    print("  • Install pygame for visualization: pip install pygame")
    print("  • Install pytest for testing: pip install pytest")
    print("  • Try running: python -m genetic_coil_arena.cli.main info")
    print("  • Check out the notebooks/ directory (coming soon)")


if __name__ == "__main__":
    try:
        simple_agent_demo()
    except Exception as e:
        print(f"❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
