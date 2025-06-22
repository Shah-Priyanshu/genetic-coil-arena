#!/usr/bin/env python3
"""
Phase 3 Persistence Demo - Genetic Coil Arena

This demo showcases the new persistence and experiment management features:
- Resumable evolution with automatic checkpointing
- Experiment metadata tracking
- Complete state management
- Model persistence and loading
- Experiment comparison and analysis

Usage:
    python phase3_persistence_demo.py
"""

import logging
import time
from pathlib import Path
import numpy as np

from genetic_coil_arena.config import Config
from genetic_coil_arena.persistence.arena import PersistentEvolutionArena, list_experiments
from genetic_coil_arena.persistence import ModelPersistence, quick_load_agent
from genetic_coil_arena.snake_core.game import SnakeGame
from genetic_coil_arena.snake_core.agent import SnakeAgent

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def demo_experiment_creation_and_resumption():
    """Demonstrate creating experiments and resuming them."""
    print("🧬 Phase 3 Demo: Persistence and Experiment Management")
    print("=" * 60)
    
    # Create configuration for a quick demo
    config = Config()
    config.evolution.generations = 20  # Small for demo
    config.evolution.population_size = 50
    config.evaluation.episodes_per_agent = 2
    
    print("1. Creating new experiment...")
    
    # Create persistent arena with experiment tracking
    arena = PersistentEvolutionArena(
        config=config,
        base_path="demo_experiments",
        checkpoint_frequency=5  # Checkpoint every 5 generations
    )
    
    # Set experiment metadata
    if arena.metadata:
        arena.metadata.description = "Phase 3 persistence demonstration"
        arena.metadata.tags = ["demo", "phase3", "persistence"]
    
    print(f"✅ Created experiment: {arena.experiment_id}")
    print(f"📁 Experiment path: {arena.experiment_path}")
    
    # Run partial evolution
    print("\n2. Running initial evolution (10 generations)...")
    try:
        result = arena.evolve(generations=10, verbose=True)
        print(f"✅ Completed 10 generations")
        print(f"Current generation: {arena.generation}")
        print(f"Best fitness: {result.get('final_best_fitness', 'N/A'):.3f}")
    except Exception as e:
        print(f"❌ Error during evolution: {e}")
        return None
    
    # Show experiment summary
    summary = arena.get_experiment_summary()
    print(f"\n📊 Experiment Summary:")
    print(f"ID: {summary['experiment_id']}")
    print(f"Current Generation: {summary['current_generation']}")
    print(f"Population Size: {summary['population_size']}")
    print(f"Best Fitness: {summary.get('current_best_fitness', 'N/A'):.4f}")
    
    return arena.experiment_id


def demo_experiment_resumption(experiment_id: str):
    """Demonstrate resuming an experiment from checkpoint."""
    print(f"\n3. Resuming experiment: {experiment_id}")
    
    # Load existing experiment
    config = Config()  # Will be overridden by saved config
    resumed_arena = PersistentEvolutionArena(
        config=config,
        experiment_id=experiment_id,
        base_path="demo_experiments"
    )
    
    print(f"✅ Resumed from generation {resumed_arena.generation}")
    
    # Continue evolution
    print("Running additional 10 generations...")
    try:
        result = resumed_arena.evolve(generations=10, verbose=True)
        print(f"✅ Completed additional evolution")
        print(f"Final generation: {resumed_arena.generation}")
        print(f"Final best fitness: {result.get('final_best_fitness', 'N/A'):.3f}")
    except Exception as e:
        print(f"❌ Error during resumed evolution: {e}")
        return
    
    # Export experiment
    export_path = Path("demo_export.tar.gz")
    print(f"\n4. Exporting experiment to {export_path}...")
    try:
        resumed_arena.export_experiment(export_path, include_checkpoints=True)
        print(f"✅ Exported to {export_path} ({export_path.stat().st_size / 1024:.1f} KB)")
    except Exception as e:
        print(f"❌ Error during export: {e}")


def demo_model_persistence():
    """Demonstrate saving and loading individual models."""
    print("\n5. Model Persistence Demo")
    print("-" * 30)
    
    persistence = ModelPersistence("demo_experiments")
    experiments = list_experiments("demo_experiments")
    
    if not experiments:
        print("No experiments found for model persistence demo")
        return
    
    # Use the first experiment
    exp_id = experiments[0]['experiment_id']
    exp_path = persistence.base_path / exp_id
    
    print(f"Using experiment: {exp_id}")
    
    # Check for best agent
    best_agent_file = exp_path / "best_agent.pkl"
    if best_agent_file.exists():
        print("📦 Loading best agent...")
        try:
            agent = persistence.load_agent(best_agent_file)
            print(f"✅ Loaded agent with {len(agent.weights)} parameters")
            
            # Test the agent
            print("🎮 Testing agent performance...")
            config = Config()
            game = SnakeGame(
                board_size=config.game.board_size,
                observation_radius=config.game.observation_radius
            )
            
            scores = []
            for episode in range(5):
                game.reset()
                steps = 0
                while not game.done and steps < 500:
                    obs = game.get_observation()
                    action = agent.predict(obs)
                    game.step(action)
                    steps += 1
                scores.append(game.score)
            
            print(f"🏆 Agent Performance:")
            print(f"   Average Score: {np.mean(scores):.1f}")
            print(f"   Best Score: {max(scores)}")
            print(f"   Episodes: {len(scores)}")
            
        except Exception as e:
            print(f"❌ Error loading/testing agent: {e}")
    else:
        print("❌ No best agent found")


def demo_experiment_management():
    """Demonstrate experiment listing and management."""
    print("\n6. Experiment Management Demo")
    print("-" * 35)
    
    experiments = list_experiments("demo_experiments")
    
    if not experiments:
        print("No experiments found")
        return
    
    print(f"📁 Found {len(experiments)} experiments:")
    
    for i, exp in enumerate(experiments, 1):
        print(f"\n{i}. {exp['experiment_id']}")
        print(f"   Created: {exp['created_at']}")
        print(f"   Generations: {exp['total_generations']}")
        print(f"   Best Fitness: {exp['best_fitness']:.4f}")
        print(f"   Description: {exp['description']}")
        print(f"   Tags: {', '.join(exp['tags']) if exp['tags'] else 'None'}")
    
    # Show detailed status for first experiment
    if experiments:
        exp_id = experiments[0]['experiment_id']
        print(f"\n📊 Detailed status for {exp_id}:")
        
        persistence = ModelPersistence("demo_experiments")
        exp_path = persistence.base_path / exp_id
        
        # Check files
        files = {
            "metadata.json": "Metadata",
            "config.yaml": "Configuration",
            "results.json": "Results",
            "best_agent.pkl": "Best Agent",
            "hall_of_fame.pkl": "Hall of Fame",
            "latest_checkpoint.pkl": "Latest Checkpoint"
        }
        
        for filename, description in files.items():
            file_path = exp_path / filename
            if file_path.exists():
                size = file_path.stat().st_size
                print(f"   ✅ {description}: {size} bytes")
            else:
                print(f"   ❌ {description}: Not found")
        
        # Count checkpoints
        checkpoints = list(exp_path.glob("checkpoint_gen_*.pkl"))
        print(f"   📁 Checkpoints: {len(checkpoints)} files")
        
        # Total directory size
        total_size = sum(f.stat().st_size for f in exp_path.rglob('*') if f.is_file())
        print(f"   💾 Total Size: {total_size / 1024:.1f} KB")


def demo_advanced_features():
    """Demonstrate advanced persistence features."""
    print("\n7. Advanced Features Demo")
    print("-" * 30)
    
    # Demonstrate generation callbacks for custom monitoring
    config = Config()
    config.evolution.generations = 5
    config.evolution.population_size = 20
    
    print("🔄 Creating arena with custom monitoring...")
    arena = PersistentEvolutionArena(
        config=config,
        base_path="demo_experiments",
        checkpoint_frequency=2
    )
    
    # Add custom callback for monitoring
    def monitoring_callback(arena):
        """Custom callback to track additional metrics."""
        if arena.fitness_history:
            latest_fitness = arena.fitness_history[-1]
            diversity = arena.diversity_history[-1] if arena.diversity_history else 0
            print(f"   [Callback] Gen {arena.generation}: "
                  f"Best={max(latest_fitness):.3f}, "
                  f"Diversity={diversity:.3f}")
    
    arena.add_generation_callback(monitoring_callback)
    
    # Set experiment metadata
    if arena.metadata:
        arena.metadata.description = "Advanced features demonstration"
        arena.metadata.tags = ["advanced", "callbacks", "monitoring"]
    
    print("🚀 Running evolution with custom monitoring...")
    try:
        result = arena.evolve(generations=5, verbose=False)  # Callbacks provide output
        print(f"✅ Advanced demo completed")
    except Exception as e:
        print(f"❌ Error in advanced demo: {e}")


def cleanup_demo():
    """Clean up demo files."""
    print("\n8. Cleanup")
    print("-" * 15)
    
    demo_path = Path("demo_experiments")
    export_file = Path("demo_export.tar.gz")
    
    cleanup_size = 0
    
    if demo_path.exists():
        cleanup_size += sum(f.stat().st_size for f in demo_path.rglob('*') if f.is_file())
        import shutil
        shutil.rmtree(demo_path)
        print(f"🗑️  Removed demo experiments directory")
    
    if export_file.exists():
        cleanup_size += export_file.stat().st_size
        export_file.unlink()
        print(f"🗑️  Removed export file")
    
    if cleanup_size > 0:
        print(f"💾 Freed {cleanup_size / 1024:.1f} KB")
    else:
        print("🧹 Nothing to clean up")


def main():
    """Run the complete Phase 3 persistence demo."""
    print("🎯 Starting Phase 3 Persistence Features Demo")
    print("This demo showcases experiment management and persistence capabilities")
    print()
    
    start_time = time.time()
    
    try:
        # Create and run initial experiment
        experiment_id = demo_experiment_creation_and_resumption()
        
        if experiment_id:
            # Resume and continue the experiment
            demo_experiment_resumption(experiment_id)
            
            # Demonstrate model persistence
            demo_model_persistence()
            
            # Show experiment management
            demo_experiment_management()
            
            # Advanced features
            demo_advanced_features()
        
        # Performance summary
        elapsed_time = time.time() - start_time
        print(f"\n⏱️  Demo completed in {elapsed_time:.1f} seconds")
        
        print("\n🎉 Phase 3 Persistence Demo Summary:")
        print("✅ Experiment creation and metadata tracking")
        print("✅ Automatic checkpointing and resumption")
        print("✅ Model persistence and loading")
        print("✅ Experiment export and management")
        print("✅ Advanced monitoring with callbacks")
        print("\n🔬 The genetic-coil-arena now supports production-grade")
        print("   experiment management for long-running research!")
        
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
