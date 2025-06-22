#!/usr/bin/env python3
"""
Phase 3 Persistence Demo - Simplified Version

This demo showcases the basic persistence features without complex evolution:
- Model persistence and loading
- Experiment directory management
- Configuration handling
- Basic agent creation and serialization

Usage:
    python phase3_simple_demo.py
"""

import logging
import time
from pathlib import Path
import numpy as np
import shutil

from genetic_coil_arena.config import Config
from genetic_coil_arena.persistence import ModelPersistence, ExperimentMetadata
from genetic_coil_arena.snake_core.agent import SnakeAgent, get_observation_size, create_random_agent
from genetic_coil_arena.snake_core.game import SnakeGame
from genetic_coil_arena.evo.genome import create_random_population

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def demo_model_persistence():
    """Demonstrate saving and loading individual models."""
    print("🧬 Phase 3 Demo: Model Persistence")
    print("=" * 50)
    
    # Create configuration
    config = Config()
    
    print("1. Creating and testing agent...")
    
    # Create a random agent
    input_size = get_observation_size(config.game.board_size, config.game.observation_radius)
    agent = create_random_agent(
        input_size=input_size,
        hidden_size=config.agent.hidden_size,
        second_hidden_size=config.agent.second_hidden_size,
        activation=config.agent.activation
    )
    
    print(f"✅ Created agent with {len(agent.get_genome())} parameters")
    
    # Test the agent
    print("🎮 Testing agent performance...")
    game = SnakeGame(
        board_size=config.game.board_size,
        observation_radius=config.game.observation_radius
    )
    
    scores = []
    for episode in range(5):
        game.reset()
        steps = 0
        while game.is_alive and steps < 100:  # Short episodes for demo
            obs = game.get_observation()
            action = agent.predict(obs)
            game.step(action)
            steps += 1
        scores.append(game.score)
    
    print(f"🏆 Agent Performance:")
    print(f"   Average Score: {np.mean(scores):.1f}")
    print(f"   Best Score: {max(scores)}")
    print(f"   Episodes: {len(scores)}")
    
    return agent, config


def demo_persistence_system(agent, config):
    """Demonstrate the persistence system."""
    print("\n2. Persistence System Demo")
    print("-" * 30)
    
    # Create persistence system
    persistence = ModelPersistence("demo_experiments")
    
    # Create an experiment
    exp_id, exp_path = persistence.create_experiment(
        config=config,
        description="Phase 3 persistence demonstration",
        tags=["demo", "phase3", "persistence"]
    )
    
    print(f"✅ Created experiment: {exp_id}")
    print(f"📁 Experiment path: {exp_path}")
    
    # Save the agent
    print("💾 Saving agent...")
    agent_file = persistence.save_agent(exp_path, agent, "test_agent.pkl")
    print(f"✅ Agent saved to: {agent_file}")
    
    # Save some population data
    print("👥 Creating and saving population...")
    population = create_random_population(
        population_size=10,
        genome_size=len(agent.get_genome()),
        seed=42
    )
    pop_file = persistence.save_population(exp_path, population, "test_population.pkl")
    print(f"✅ Population saved to: {pop_file}")
    
    # Save experiment results
    print("📊 Saving experiment results...")
    results = {
        "total_generations": 50,
        "best_fitness": 12.5,
        "final_population_size": 10,
        "convergence_achieved": True,
        "execution_time": 120.5
    }
    results_file = persistence.save_results(exp_path, results)
    print(f"✅ Results saved to: {results_file}")
    
    return exp_id, exp_path, persistence


def demo_loading_and_verification(exp_id, exp_path, persistence):
    """Demonstrate loading and verifying saved data."""
    print("\n3. Loading and Verification Demo")
    print("-" * 35)
    
    # Load metadata
    print("📋 Loading experiment metadata...")
    metadata = persistence.load_metadata(exp_path / "metadata.json")
    print(f"✅ Experiment ID: {metadata.experiment_id}")
    print(f"   Created: {metadata.created_at}")
    print(f"   Description: {metadata.description}")
    print(f"   Tags: {', '.join(metadata.tags)}")
    
    # Load configuration
    print("⚙️  Loading configuration...")
    config = persistence.load_config(exp_path / "config.yaml")
    print(f"✅ Config loaded - Population size: {config.evolution.population_size}")
    
    # Load agent
    print("🤖 Loading agent...")
    loaded_agent = persistence.load_agent(exp_path / "test_agent.pkl")
    print(f"✅ Agent loaded with {len(loaded_agent.get_genome())} parameters")
    
    # Load population
    print("👥 Loading population...")
    loaded_population = persistence.load_population(exp_path / "test_population.pkl")
    print(f"✅ Population loaded with {len(loaded_population)} genomes")
    
    # Load results
    print("📊 Loading results...")
    loaded_results = persistence.load_results(exp_path / "results.json")
    print(f"✅ Results loaded - Best fitness: {loaded_results['best_fitness']}")
    
    # Test loaded agent
    print("🎮 Testing loaded agent...")
    game = SnakeGame(board_size=config.game.board_size, observation_radius=config.game.observation_radius)
    game.reset()
    obs = game.get_observation()
    action = loaded_agent.predict(obs)
    print(f"✅ Loaded agent prediction: {action}")
    
    return loaded_agent, loaded_population, loaded_results


def demo_experiment_management(persistence):
    """Demonstrate experiment management features."""
    print("\n4. Experiment Management Demo")
    print("-" * 35)
    
    # List experiments
    print("📁 Listing experiments...")
    experiments = []
    for exp_id, metadata in persistence.list_experiments():
        experiments.append({
            'experiment_id': exp_id,
            'created_at': metadata.created_at.isoformat(),
            'description': metadata.description,
            'tags': metadata.tags,
            'total_generations': metadata.total_generations,
            'best_fitness': metadata.best_fitness,
        })
    
    print(f"✅ Found {len(experiments)} experiments:")
    for exp in experiments:
        print(f"   - {exp['experiment_id']}: {exp['description']}")
    
    # Export experiment
    if experiments:
        exp_id = experiments[0]['experiment_id']
        export_path = Path("demo_export.tar.gz")
        print(f"📦 Exporting experiment {exp_id}...")
        persistence.export_experiment(exp_id, export_path, include_checkpoints=False)
        export_size = export_path.stat().st_size / 1024
        print(f"✅ Exported to {export_path} ({export_size:.1f} KB)")
        
        # Clean up export
        export_path.unlink()
        print("🗑️  Cleaned up export file")


def demo_advanced_features():
    """Demonstrate advanced persistence features."""
    print("\n5. Advanced Features Demo")
    print("-" * 30)
    
    # Create multiple agents with different architectures
    config = Config()
    input_size = get_observation_size(config.game.board_size, config.game.observation_radius)
    
    agents = []
    for i, activation in enumerate(['tanh', 'relu', 'sigmoid']):
        agent = create_random_agent(
            input_size=input_size,
            hidden_size=16 + i * 8,  # Different sizes
            activation=activation,
            seed=i * 100
        )
        agents.append((f"agent_{activation}", agent))
    
    print(f"🔬 Created {len(agents)} agents with different architectures")
    
    # Save all agents
    persistence = ModelPersistence("demo_experiments")
    
    for name, agent in agents:
        # Create individual experiment for each agent
        exp_id, exp_path = persistence.create_experiment(
            config=config,
            description=f"Agent architecture test - {name}",
            tags=["architecture", "comparison", name.split('_')[1]]
        )
        
        persistence.save_agent(exp_path, agent, f"{name}.pkl")
        print(f"💾 Saved {name} to experiment {exp_id}")
    
    print("✅ Advanced demo completed")


def performance_benchmark():
    """Benchmark persistence operations."""
    print("\n6. Performance Benchmark")
    print("-" * 25)
    
    config = Config()
    input_size = get_observation_size(config.game.board_size, config.game.observation_radius)
    persistence = ModelPersistence("demo_experiments")
    
    # Benchmark agent saving/loading
    print("⏱️  Benchmarking agent operations...")
    agent = create_random_agent(input_size=input_size)
    
    # Create temporary experiment
    exp_id, exp_path = persistence.create_experiment(
        config=config,
        description="Performance benchmark",
        tags=["benchmark"]
    )
    
    # Time save operation
    start_time = time.time()
    for i in range(10):
        persistence.save_agent(exp_path, agent, f"agent_{i}.pkl")
    save_time = (time.time() - start_time) / 10
    
    # Time load operation
    start_time = time.time()
    for i in range(10):
        persistence.load_agent(exp_path / f"agent_{i}.pkl")
    load_time = (time.time() - start_time) / 10
    
    print(f"📊 Performance Results:")
    print(f"   Agent Save: {save_time*1000:.1f} ms")
    print(f"   Agent Load: {load_time*1000:.1f} ms")
    print(f"   Parameters: {len(agent.get_genome()):,}")
    
    # File size analysis
    agent_file = exp_path / "agent_0.pkl"
    file_size = agent_file.stat().st_size
    print(f"   File Size: {file_size:,} bytes ({file_size/1024:.1f} KB)")


def cleanup_demo():
    """Clean up demo files."""
    print("\n7. Cleanup")
    print("-" * 15)
    
    demo_path = Path("demo_experiments")
    cleanup_size = 0
    
    if demo_path.exists():
        cleanup_size = sum(f.stat().st_size for f in demo_path.rglob('*') if f.is_file())
        shutil.rmtree(demo_path)
        print(f"🗑️  Removed demo experiments directory")
        print(f"💾 Freed {cleanup_size / 1024:.1f} KB")
    else:
        print("🧹 Nothing to clean up")


def main():
    """Run the complete Phase 3 persistence demo."""
    print("🎯 Phase 3 Persistence Features Demo (Simplified)")
    print("=" * 55)
    print("This demo showcases the core persistence capabilities")
    print()
    
    start_time = time.time()
    
    try:
        # Basic model persistence
        agent, config = demo_model_persistence()
        
        # Persistence system features
        exp_id, exp_path, persistence = demo_persistence_system(agent, config)
        
        # Loading and verification
        loaded_agent, loaded_population, loaded_results = demo_loading_and_verification(
            exp_id, exp_path, persistence
        )
        
        # Experiment management
        demo_experiment_management(persistence)
        
        # Advanced features
        demo_advanced_features()
        
        # Performance benchmark
        performance_benchmark()
        
        # Summary
        elapsed_time = time.time() - start_time
        print(f"\n⏱️  Demo completed in {elapsed_time:.1f} seconds")
        
        print("\n🎉 Phase 3 Persistence Demo Summary:")
        print("✅ Model serialization and loading")
        print("✅ Experiment metadata tracking")
        print("✅ File-based persistence system")
        print("✅ Configuration management")
        print("✅ Population data handling")
        print("✅ Results archival")
        print("✅ Export functionality")
        print("✅ Performance benchmarking")
        print("\n🔬 Core persistence infrastructure is ready!")
        print("   Next: Web dashboard and REST API integration")
        
    except KeyboardInterrupt:
        print("\n⚠️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Always cleanup
        cleanup_demo()


if __name__ == "__main__":
    main()
