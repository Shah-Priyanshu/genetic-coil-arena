"""Create a simple demo script to verify the installation."""

import sys
import time
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent / "src"))

from genetic_coil_arena import Config, EvolutionArena
from genetic_coil_arena.snake_core.game import SnakeGame, Action
from genetic_coil_arena.snake_core.agent import SnakeAgent, get_observation_size


def test_game_basic():
    """Test basic game functionality."""
    print("🎮 Testing basic game functionality...")
    
    game = SnakeGame(board_size=16, seed=42)
    state = game.reset()
    
    print(f"  Initial state: score={state.score}, length={state.snake_length}")
    
    # Play a few moves
    for i in range(10):
        action = Action(i % 3)  # Cycle through actions
        state, reward, done = game.step(action)
        print(f"  Step {i+1}: action={action.name}, reward={reward:.1f}, score={state.score}")
        
        if done:
            print(f"  Game ended at step {i+1}")
            break
    
    print("✅ Game test completed\n")


def test_agent_basic():
    """Test basic agent functionality."""
    print("🧠 Testing neural network agent...")
    
    obs_size = get_observation_size(board_size=16, observation_radius=3)
    agent = SnakeAgent(input_size=obs_size, hidden_size=16, seed=42)
    
    print(f"  Agent created with {agent.genome_size} parameters")
      # Test with random observation
    import numpy as np
    observation = np.array([0.5] * obs_size, dtype=np.float32)
    action = agent.predict(observation)
    probs = agent.predict_probabilities(observation)
    
    print(f"  Prediction: {action.name}")
    print(f"  Probabilities: {[f'{p:.3f}' for p in probs]}")
    
    print("✅ Agent test completed\n")


def test_integration():
    """Test game and agent integration."""
    print("🔗 Testing game-agent integration...")
    
    game = SnakeGame(board_size=16, observation_radius=3, seed=42)
    obs_size = get_observation_size(16, 3)
    agent = SnakeAgent(input_size=obs_size, seed=42)
    
    state = game.reset()
    total_reward = 0
    step = 0
    
    for step in range(20):
        observation = game.get_observation()
        action = agent.predict(observation)
        state, reward, done = game.step(action)
        total_reward += reward
        
        if done:
            break
    
    print(f"  Episode completed: {step+1} steps, score={state.score}, total_reward={total_reward:.1f}")
    print("✅ Integration test completed\n")


def test_mini_evolution():
    """Test a mini evolution run."""
    print("🧬 Testing mini evolution (this may take a minute)...")
    
    config = Config()
    config.evolution.population_size = 8
    config.evolution.generations = 3
    config.evaluation.episodes_per_agent = 1
    config.evaluation.parallel_processes = 2
    config.game.board_size = 16
    config.logging.output_dir = Path("demo_output")
    
    arena = EvolutionArena(config)
    
    start_time = time.time()
    result = arena.evolve()
    elapsed = time.time() - start_time
    
    print(f"  Completed {len(result.generation_stats)} generations in {elapsed:.1f}s")
    if result.hall_of_fame:
        print(f"  Best fitness: {result.hall_of_fame[0].fitness:.3f}")
    else:
        print("  No valid genomes found")
    print(f"  Results saved to: {config.logging.output_dir}")
    
    print("✅ Mini evolution completed\n")


def main():
    """Run all demo tests."""
    print("🚀 Genetic Coil Arena - Demo & Verification\n")
    print("=" * 60)
    
    try:
        test_game_basic()
        test_agent_basic()
        test_integration()
        test_mini_evolution()
        
        print("🎉 All tests passed! Your installation is working correctly.")
        print("\nNext steps:")
        print("  • Run: genetic-coil evolve --generations 10 --population 32")
        print("  • Edit config.yaml to customize parameters")
        print("  • Check out the notebooks/ directory for analysis examples")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        print("Please check your installation and try again.")
        sys.exit(1)


if __name__ == "__main__":
    main()
