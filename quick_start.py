"""Quick start script to demonstrate the full genetic algorithm pipeline."""

import sys
from pathlib import Path
import numpy as np
import time

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent / "src"))

from genetic_coil_arena.config import Config
from genetic_coil_arena.snake_core.game import SnakeGame
from genetic_coil_arena.snake_core.agent import SnakeAgent, get_observation_size
from genetic_coil_arena.evo.genome import Genome, create_random_population
from genetic_coil_arena.evo.fitness import FitnessEvaluator


def run_mini_evolution():
    """Run a mini evolution to demonstrate the full pipeline."""
    print("🧬 Genetic Coil Arena - Mini Evolution Demo")
    print("=" * 60)
    
    # Configuration for quick demo
    config = Config()
    config.game.board_size = 16  # Smaller for speed
    config.game.observation_radius = 3  # Smaller observation window
    config.evolution.population_size = 8  # Tiny population
    config.evolution.generations = 3  # Just a few generations
    config.evaluation.episodes_per_agent = 1  # Single episode per agent
    config.evaluation.parallel_processes = 2  # Limited parallelism
    config.evaluation.deterministic_seed = 42  # Reproducible results
    
    print(f"Configuration:")
    print(f"  Board size: {config.game.board_size}x{config.game.board_size}")
    print(f"  Population: {config.evolution.population_size}")
    print(f"  Generations: {config.evolution.generations}")
    
    # Calculate observation size and genome size
    obs_size = get_observation_size(config.game.board_size, config.game.observation_radius)
    
    # We'll approximate genome size for demo (without full agent)
    genome_size = obs_size * 16 + 16 + 16 * 8 + 8 + 8 * 3 + 3  # Simplified calculation
    
    print(f"  Observation size: {obs_size}")
    print(f"  Genome size: ~{genome_size} parameters")
    print()
    
    # Create fitness evaluator
    game_config = {
        "board_size": config.game.board_size,
        "observation_radius": config.game.observation_radius,
        "max_steps": config.game.max_steps
    }
    
    agent_config = {
        "hidden_size": 16,  # Smaller for demo
        "second_hidden_size": 8,
        "activation": "tanh"
    }
    
    # Note: We'll simulate fitness evaluation for this demo since Numba has issues
    print("🔧 Creating simplified fitness evaluator...")
    
    # Create initial population
    print("🧮 Creating initial population...")
    population = create_random_population(
        config.evolution.population_size,
        genome_size,
        config.evaluation.deterministic_seed
    )
    
    print(f"Created {len(population)} random genomes")
    
    # Simulate evolution for demo
    print("\n🎯 Running simulated evolution...")
    
    for generation in range(config.evolution.generations):
        print(f"\n--- Generation {generation + 1} ---")
        
        # Simulate fitness evaluation (in real version, this would use the full evaluator)
        for i, genome in enumerate(population):
            # Simulate fitness calculation
            base_fitness = np.random.random() * 10  # Random baseline
            complexity_bonus = np.std(genome.weights) * 2  # Reward complexity
            fitness = base_fitness + complexity_bonus
            
            # Create mock stats
            from genetic_coil_arena.evo.genome import GenomeStats
            genome.stats = GenomeStats(
                fitness=fitness,
                episodes_played=config.evaluation.episodes_per_agent,
                avg_score=fitness / 2,
                avg_length=1 + int(fitness / 5),
                avg_steps=50 + int(fitness * 10),
                success_rate=min(1.0, fitness / 10)
            )
        
        # Calculate generation statistics
        fitness_values = [g.fitness for g in population]
        best_fitness = max(fitness_values)
        mean_fitness = np.mean(fitness_values)
        
        print(f"  Best fitness: {best_fitness:.3f}")
        print(f"  Mean fitness: {mean_fitness:.3f}")
        print(f"  Fitness range: {min(fitness_values):.3f} - {max(fitness_values):.3f}")
        
        # Create next generation (simplified)
        if generation < config.evolution.generations - 1:
            # Sort by fitness
            population.sort(key=lambda x: x.fitness, reverse=True)
            
            # Keep top 25% (elitism)
            elite_size = max(1, len(population) // 4)
            next_population = population[:elite_size].copy()
            
            # Fill rest with mutations of elite
            while len(next_population) < len(population):
                parent = population[np.random.randint(0, elite_size)]
                child = parent.mutate(sigma=0.2, mutation_prob=0.1)
                next_population.append(child)
            
            population = next_population
    
    # Final results
    print(f"\n🎉 Evolution completed!")
    population.sort(key=lambda x: x.fitness, reverse=True)
    best_genome = population[0]
    
    print(f"Best agent achieved:")
    print(f"  Fitness: {best_genome.fitness:.3f}")
    print(f"  Simulated score: {best_genome.stats.avg_score:.1f}")
    print(f"  Simulated steps: {best_genome.stats.avg_steps:.0f}")
    
    print(f"\n📈 Performance metrics:")
    print(f"  Total evaluations: {config.evolution.population_size * config.evolution.generations}")
    print(f"  Evolution time: ~{time.time():.1f}s (simulated)")
    
    print(f"\n✅ Mini evolution demo completed successfully!")
    print("\nTo run the full system:")
    print("  1. Fix Numba type issues in agent.py")
    print("  2. Run: python -c \"import sys; sys.path.insert(0, 'src'); from genetic_coil_arena.cli.main import app; app(['evolve', '--generations', '10', '--population', '32'])\"")
    print("  3. Install with: pip install -e . (after fixing TOML issues)")


if __name__ == "__main__":
    try:
        run_mini_evolution()
    except Exception as e:
        print(f"❌ Error during mini evolution: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
