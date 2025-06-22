"""Performance optimization and benchmarking demo."""

import time
import numpy as np
from src.genetic_coil_arena.config import load_config
from src.genetic_coil_arena.snake_core.game import SnakeGame
from src.genetic_coil_arena.snake_core.agent import SnakeAgent, get_observation_size
from src.genetic_coil_arena.evo.genome import Genome


def benchmark_game_performance():
    """Benchmark game performance."""
    print("🎮 Benchmarking Game Performance")
    print("=" * 50)
    
    # Test different board sizes
    board_sizes = [16, 32, 64]
    episodes = 100
    
    for board_size in board_sizes:
        print(f"\nBoard size: {board_size}x{board_size}")
        
        game = SnakeGame(board_size=board_size, observation_radius=5, seed=42)
        
        start_time = time.time()
        for _ in range(episodes):
            game.reset()
            for _ in range(50):  # 50 steps per episode
                game.step(np.random.choice([0, 1, 2]))
                
        elapsed = time.time() - start_time
        steps_per_sec = (episodes * 50) / elapsed
        
        print(f"  Episodes: {episodes}")
        print(f"  Steps per episode: 50")
        print(f"  Total time: {elapsed:.2f}s")
        print(f"  Performance: {steps_per_sec:.0f} steps/sec")


def benchmark_agent_performance():
    """Benchmark agent performance."""
    print("\n🤖 Benchmarking Agent Performance")
    print("=" * 50)
    
    # Test different network sizes
    hidden_sizes = [16, 32, 64]
    predictions = 1000
    
    obs_size = get_observation_size(board_size=32, observation_radius=5)
    dummy_observation = np.random.random(obs_size).astype(np.float32)
    
    for hidden_size in hidden_sizes:
        print(f"\nHidden size: {hidden_size}")
        
        agent = SnakeAgent(input_size=obs_size, hidden_size=hidden_size, seed=42)
        
        # Warm up
        for _ in range(10):
            agent.predict(dummy_observation)
        
        start_time = time.time()
        for _ in range(predictions):
            agent.predict(dummy_observation)
            
        elapsed = time.time() - start_time
        predictions_per_sec = predictions / elapsed
        
        print(f"  Predictions: {predictions}")
        print(f"  Total time: {elapsed:.3f}s")
        print(f"  Performance: {predictions_per_sec:.0f} predictions/sec")
        print(f"  Genome size: {agent.genome_size} parameters")


def benchmark_genome_operations():
    """Benchmark genome operations."""
    print("\n🧬 Benchmarking Genome Operations")
    print("=" * 50)
    
    genome_sizes = [100, 1000, 5000]
    operations = 100
    
    for genome_size in genome_sizes:
        print(f"\nGenome size: {genome_size}")
        
        # Create test genomes
        genome1 = Genome(genome_size=genome_size)
        genome2 = Genome(genome_size=genome_size)
        
        # Benchmark mutation
        start_time = time.time()
        for _ in range(operations):
            genome1.mutate(sigma=0.1, mutation_prob=0.1)
        mutation_time = time.time() - start_time
        
        # Benchmark crossover
        start_time = time.time()
        for _ in range(operations):
            genome1.crossover(genome2, alpha=0.3)
        crossover_time = time.time() - start_time
        
        print(f"  Mutation: {operations} ops in {mutation_time:.3f}s ({operations/mutation_time:.0f} ops/sec)")
        print(f"  Crossover: {operations} ops in {crossover_time:.3f}s ({operations/crossover_time:.0f} ops/sec)")


def test_memory_efficiency():
    """Test memory efficiency."""
    print("\n💾 Testing Memory Efficiency")
    print("=" * 50)
    
    # Test creating many agents
    num_agents = 100
    obs_size = get_observation_size(board_size=32, observation_radius=5)
    
    print(f"Creating {num_agents} agents...")
    start_time = time.time()
    
    agents = []
    for i in range(num_agents):
        agent = SnakeAgent(input_size=obs_size, hidden_size=32, seed=i)
        agents.append(agent)
    
    creation_time = time.time() - start_time
    
    # Test prediction with all agents
    dummy_observation = np.random.random(obs_size).astype(np.float32)
    
    start_time = time.time()
    for agent in agents:
        agent.predict(dummy_observation)
    prediction_time = time.time() - start_time
    
    print(f"  Agent creation: {creation_time:.2f}s ({num_agents/creation_time:.0f} agents/sec)")
    print(f"  Batch prediction: {prediction_time:.3f}s ({num_agents/prediction_time:.0f} predictions/sec)")
    print(f"  Memory per agent: ~{agents[0].genome_size * 4 / 1024:.1f} KB (genome only)")


def test_deterministic_performance():
    """Test that performance is consistent with deterministic seeds."""
    print("\n🎯 Testing Deterministic Performance")
    print("=" * 50)
    
    config = load_config()
    
    # Test multiple runs with same seed
    times = []
    results = []
    
    for run in range(3):
        print(f"  Run {run + 1}/3...")
        
        game = SnakeGame(
            board_size=config.game.board_size,
            observation_radius=config.game.observation_radius,
            seed=42  # Same seed for all runs
        )
        
        obs_size = get_observation_size(
            board_size=config.game.board_size,
            observation_radius=config.game.observation_radius
        )
        agent = SnakeAgent(input_size=obs_size, hidden_size=16, seed=42)
        
        start_time = time.time()
        
        # Run deterministic episode
        total_reward = 0
        state = game.reset()
        
        for step in range(100):
            observation = game.get_observation()
            action = agent.predict(observation)
            state, reward, done = game.step(action)
            total_reward += reward
            
            if done:
                break
        
        elapsed = time.time() - start_time
        times.append(elapsed)
        results.append((total_reward, state.score, step + 1))
    
    # Check consistency
    print(f"\n  Performance consistency:")
    print(f"    Times: {[f'{t:.3f}s' for t in times]}")
    print(f"    Results: {results}")
    
    # All results should be identical with same seed
    if len(set(results)) == 1:
        print(f"    ✅ Deterministic: All runs produced identical results")
    else:
        print(f"    ❌ Non-deterministic: Results varied between runs")


def main():
    """Run all benchmarks."""
    print("🚀 Genetic Coil Arena - Performance Benchmarks")
    print("=" * 60)
    
    benchmark_game_performance()
    benchmark_agent_performance()
    benchmark_genome_operations()
    test_memory_efficiency()
    test_deterministic_performance()
    
    print("\n🎉 All benchmarks completed!")
    print("\n💡 Performance Tips:")
    print("  - Use smaller board sizes for faster training")
    print("  - Reduce observation radius to decrease input size")
    print("  - Use parallel evaluation for population fitness")
    print("  - Consider smaller hidden layers for faster inference")
    print("  - Enable Numba compilation for neural network speedup")


if __name__ == "__main__":
    main()
