"""Main CLI application using Typer."""

import typer
from pathlib import Path
from typing import Optional, List
import sys
import yaml
from datetime import datetime

from ..config import Config
from ..evo.arena import EvolutionArena
from ..evo.baselines import BENCHMARK_SEED_OFFSET
from ..snake_core.agent import SnakeAgent, get_observation_size
# Add persistence imports
from ..persistence.arena import PersistentEvolutionArena, list_experiments
from ..persistence import ModelPersistence

app = typer.Typer(
    name="genetic-coil",
    help="Genetic Coil Arena - Evolve Snake AI with Genetic Algorithms",
    add_completion=False
)


@app.command()
def evolve(
    config_file: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Configuration YAML file"
    ),
    generations: Optional[int] = typer.Option(
        None, "--generations", "-g", help="Number of generations to run"
    ),
    population: Optional[int] = typer.Option(
        None, "--population", "-p", help="Population size"
    ),
    output_dir: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Output directory for results"
    ),
    parallel_processes: Optional[int] = typer.Option(
        None, "--processes", help="Number of parallel processes"
    ),
    seed: Optional[int] = typer.Option(
        None, "--seed", help="Random seed for reproducibility"
    ),
    experiment_id: Optional[str] = typer.Option(
        None, "--experiment", "-e", help="Resume existing experiment ID"
    ),
    description: str = typer.Option(
        "", "--description", "-d", help="Experiment description"
    ),
    tags: List[str] = typer.Option(
        [], "--tag", help="Experiment tags (can be used multiple times)"
    ),
    checkpoint_freq: int = typer.Option(
        10, "--checkpoint-freq", help="Checkpoint frequency (generations)"
    ),
    base_path: str = typer.Option(
        "experiments", "--base-path", help="Base directory for experiments"
    ),
) -> None:
    """Run genetic algorithm evolution to train snake agents with persistence."""
    
    # Load configuration
    experiment_config = Path(base_path) / experiment_id / "config.yaml" if experiment_id else None
    if config_file and config_file.exists():
        config = Config.from_yaml(config_file)
        typer.echo(f"Loaded configuration from {config_file}")
    elif experiment_config and experiment_config.exists():
        # Resuming: keep the configuration the experiment was started with
        config = Config.from_yaml(experiment_config)
        typer.echo(f"Loaded configuration from {experiment_config}")
    else:
        config = Config()
        typer.echo("Using default configuration")

    # Override config with CLI arguments
    if generations is not None:
        config.evolution.generations = generations
    if population is not None:
        config.evolution.population_size = population
    if output_dir is not None:
        config.logging.output_dir = output_dir
    if parallel_processes is not None:
        config.evaluation.parallel_processes = parallel_processes
    if seed is not None:
        config.evaluation.deterministic_seed = seed
    
    # Create persistent arena
    typer.echo("Initializing persistent evolution arena...")
    arena = PersistentEvolutionArena(
        config=config,
        experiment_id=experiment_id,
        base_path=base_path,
        checkpoint_frequency=checkpoint_freq
    )
    
    if experiment_id:
        typer.echo(f"Resumed experiment: {experiment_id}")
    else:
        # Set description and tags for new experiment
        if hasattr(arena, 'metadata') and arena.metadata:
            arena.metadata.description = description
            arena.metadata.tags = tags
        typer.echo(f"Created new experiment: {arena.experiment_id}")
    
    summary = arena.get_experiment_summary()
    typer.echo(f"Current generation: {summary['current_generation']}")
    typer.echo(f"Population size: {summary['population_size']}")
    
    # Run evolution
    typer.echo("Starting evolution...")
    
    try:
        result = arena.evolve(generations=generations or config.evolution.generations, verbose=True)
        
        typer.echo(f"\n✅ Evolution completed!")
        typer.echo(f"Experiment ID: {arena.experiment_id}")
        typer.echo(f"Final generation: {arena.generation}")
        typer.echo(f"Results saved to: {arena.experiment_path}")
        
        if 'final_best_fitness' in result:
            typer.echo(f"Best fitness: {result['final_best_fitness']:.3f} "
                       f"(random policy: {result['random_baseline_fitness']:.3f})")
        typer.echo(f"Benchmark the best agent with: genetic-coil benchmark "
                   f"--config {arena.experiment_path / 'config.yaml'} "
                   f"--agent {arena.experiment_path / 'best_agent.pkl'}")

    except KeyboardInterrupt:
        typer.echo("\n⚠️  Evolution interrupted by user")
        typer.echo(f"Progress saved for experiment: {arena.experiment_id}")
        sys.exit(1)
    except Exception as e:
        typer.echo(f"\n❌ Error during evolution: {e}")
        typer.echo(f"Emergency checkpoint saved for experiment: {arena.experiment_id}")
        sys.exit(1)


@app.command()
def replay(
    agent_file: Path = typer.Argument(..., help="Path to saved agent (pickle file)"),
    episodes: int = typer.Option(5, "--episodes", "-e", help="Number of episodes to replay"),
    render: bool = typer.Option(False, "--render", help="Show visual replay"),
    config_file: Optional[Path] = typer.Option(None, "--config", "-c", help="Configuration file"),
) -> None:
    """Replay a trained agent's performance."""
    
    typer.echo("🎮 Replay functionality not yet implemented")
    typer.echo("Coming in the next version with Pygame visualization!")


@app.command()
def analyze(
    results_dir: Path = typer.Argument(..., help="Directory containing evolution results"),
    output_file: Optional[Path] = typer.Option(None, "--output", "-o", help="Output file for analysis report"),
    plot: bool = typer.Option(False, "--plot", help="Generate and save plots"),
    interactive: bool = typer.Option(False, "--interactive", help="Show interactive plots"),
) -> None:
    """Analyze evolution results and generate reports."""
    
    # Check if results directory exists
    if not results_dir.exists():
        typer.echo(f"❌ Results directory not found: {results_dir}")
        raise typer.Exit(1)
    
    # Look for generation stats file
    stats_file = results_dir / "generation_stats.csv"
    if not stats_file.exists():
        typer.echo(f"❌ Generation stats file not found: {stats_file}")
        raise typer.Exit(1)
    
    typer.echo(f"📊 Analyzing results from: {results_dir}")
    
    try:
        import pandas as pd
        import matplotlib.pyplot as plt
        
        # Load generation statistics
        df = pd.read_csv(stats_file)
        
        # Basic analysis
        total_generations = len(df)
        best_fitness = df['best_fitness'].max()
        final_fitness = df['best_fitness'].iloc[-1]
        improvement = final_fitness - df['best_fitness'].iloc[0]
        
        # Create report
        report = f"""
EVOLUTION ANALYSIS REPORT
========================

OVERVIEW
--------
Total Generations: {total_generations}
Best Fitness Achieved: {best_fitness:.4f}
Final Fitness: {final_fitness:.4f}
Total Improvement: {improvement:.4f}

PERFORMANCE METRICS
-------------------
Final Mean Fitness: {df['mean_fitness'].iloc[-1]:.4f}
Final Best / Mean Food: {df['best_score'].iloc[-1]:.2f} / {df['mean_score'].iloc[-1]:.2f}
Final Success Rate: {df['success_rate'].iloc[-1]:.1%}
Final Diversity: {df['diversity'].iloc[-1]:.4f}

TRENDS
------
Fitness Trend: {"Improving" if df['best_fitness'].diff().tail(5).mean() > 0 else "Plateaued"}
Diversity Trend: {"Decreasing" if df['diversity'].diff().tail(5).mean() < 0 else "Stable"}
"""
        
        # Output report
        if output_file:
            output_file.write_text(report)
            typer.echo(f"📄 Report saved to: {output_file}")
        else:
            typer.echo(report)
        
        # Generate plots if requested
        if plot or interactive:
            plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'default')
            
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            fig.suptitle('Evolution Analysis Results')
            
            # Fitness over time
            axes[0, 0].plot(df['generation'], df['best_fitness'], 'b-', label='Best', linewidth=2)
            axes[0, 0].plot(df['generation'], df['mean_fitness'], 'g-', label='Mean', alpha=0.7)
            axes[0, 0].set_title('Fitness Evolution')
            axes[0, 0].set_xlabel('Generation')
            axes[0, 0].set_ylabel('Fitness')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # Success rate
            axes[0, 1].plot(df['generation'], df['success_rate'] * 100, 'r-', linewidth=2)
            axes[0, 1].set_title('Success Rate')
            axes[0, 1].set_xlabel('Generation')
            axes[0, 1].set_ylabel('Success Rate (%)')
            axes[0, 1].grid(True, alpha=0.3)
            
            # Diversity
            axes[1, 0].plot(df['generation'], df['diversity'], 'm-', linewidth=2)
            axes[1, 0].set_title('Population Diversity')
            axes[1, 0].set_xlabel('Generation')
            axes[1, 0].set_ylabel('Diversity')
            axes[1, 0].grid(True, alpha=0.3)
            
            # Food eaten per episode
            axes[1, 1].plot(df['generation'], df['best_score'], 'c-', label='Best', linewidth=2)
            axes[1, 1].plot(df['generation'], df['mean_score'], 'c--', label='Mean', alpha=0.7)
            axes[1, 1].set_title('Food Eaten per Episode')
            axes[1, 1].set_xlabel('Generation')
            axes[1, 1].set_ylabel('Food')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if plot:
                plot_file = results_dir / "evolution_analysis.png"
                plt.savefig(plot_file, dpi=300, bbox_inches='tight')
                typer.echo(f"📈 Plots saved to: {plot_file}")
            
            if interactive:
                typer.echo("📊 Showing interactive plots...")
                plt.show()
            
            if not interactive:
                plt.close()
        
        typer.echo("✅ Analysis complete!")
        
    except ImportError as e:
        typer.echo(f"❌ Missing required packages for analysis: {e}")
        typer.echo("Install with: pip install pandas matplotlib seaborn")
        raise typer.Exit(1)
    except Exception as e:
        typer.echo(f"❌ Error during analysis: {e}")
        raise typer.Exit(1)


@app.command()
def visualize(
    agent_file: Optional[Path] = typer.Option(None, "--agent", "-a", help="Path to saved agent to visualize"),
    config_file: Optional[Path] = typer.Option(None, "--config", "-c", help="Configuration file"),
    episodes: int = typer.Option(5, "--episodes", "-e", help="Number of episodes to run"),
    fps: int = typer.Option(10, "--fps", help="Frames per second for visualization"),
) -> None:
    """Launch game visualization with or without an agent."""
    
    try:
        from ..ui.viewer import GameViewer, VisualizationConfig
        from ..snake_core.game import SnakeGame
        from ..config import load_config
        from ..analysis.benchmark import load_agent_file

        # Load configuration
        config = load_config(config_file) if config_file else load_config()

        # Create game
        game = SnakeGame.from_config(config.game)

        # Load agent if provided
        agent = None
        if agent_file and agent_file.exists():
            agent = load_agent_file(agent_file, config)
            typer.echo(f"🤖 Loaded agent from: {agent_file}")
        else:
            typer.echo("🎮 Manual control mode (use arrow keys or WASD)")
        
        # Create viewer
        viewer_config = VisualizationConfig(fps=fps)
        viewer = GameViewer(viewer_config)
        
        typer.echo("🎬 Starting visualization...")
        typer.echo("Press ESC to quit, SPACE to pause")
        
        # Run visualization
        stats = viewer.run_game_session(game, agent, max_steps=1000)
        
        # Show results
        if stats:
            typer.echo(f"\n📊 Session Results:")
            typer.echo(f"Episodes: {stats.get('episodes', 0)}")
            typer.echo(f"Average Score: {stats.get('avg_score', 0):.1f}")
            typer.echo(f"Best Score: {stats.get('best_score', 0)}")
            typer.echo(f"Success Rate: {stats.get('success_rate', 0):.1%}")
        
    except ImportError as e:
        typer.echo(f"❌ Missing pygame for visualization: {e}")
        typer.echo("Install with: pip install pygame")
        raise typer.Exit(1)
    except Exception as e:
        typer.echo(f"❌ Error during visualization: {e}")
        raise typer.Exit(1)


@app.command()
def init_config(
    output: Path = typer.Option(
        Path("config.yaml"), "--output", "-o", help="Output configuration file"
    ),
) -> None:
    """Generate a default configuration file."""
    
    config = Config()
    config.to_yaml(output)
    
    typer.echo(f"✅ Default configuration saved to {output}")
    typer.echo("You can edit this file and use it with --config flag")


@app.command()
def info() -> None:
    """Show system information and configuration."""
    
    config = Config()
    
    typer.echo("🧬 Genetic Coil Arena - System Information")
    typer.echo("=" * 50)
    
    # Game configuration
    typer.echo(f"Board size: {config.game.board_size}x{config.game.board_size}")
    typer.echo(f"Observation window: {2*config.game.observation_radius+1}x{2*config.game.observation_radius+1}")
    typer.echo(f"Max steps per episode: {config.game.max_steps}")
    typer.echo(f"Starvation after: {config.game.starvation_steps or 'disabled'} steps without food")
    typer.echo(f"Board edges: {'walls' if config.game.walls else 'wrap (torus)'}")
    typer.echo(f"Fitness function: {config.evaluation.fitness_function}")

    # Evolution configuration  
    typer.echo(f"Population size: {config.evolution.population_size}")
    typer.echo(f"Tournament size: {config.evolution.tournament_size}")
    typer.echo(f"Elite size: {config.evolution.elite_size}")
    
    # Agent configuration
    obs_size = get_observation_size(config.game.board_size, config.game.observation_radius)
    typer.echo(f"Neural network input size: {obs_size}")
    typer.echo(f"Hidden layers: {config.agent.hidden_size} → {config.agent.second_hidden_size} → 3")
    
    # Calculate total parameters
    total_params = (obs_size * config.agent.hidden_size + config.agent.hidden_size +
                   config.agent.hidden_size * config.agent.second_hidden_size + config.agent.second_hidden_size +
                   config.agent.second_hidden_size * 3 + 3)
    typer.echo(f"Total parameters per agent: {total_params:,}")
    
    # Performance
    typer.echo(f"Parallel processes: {config.evaluation.parallel_processes}")
    typer.echo(f"Episodes per agent: {config.evaluation.episodes_per_agent}")


@app.command()
def experiments(
    base_path: str = typer.Option("experiments", "--base-path", help="Base directory for experiments"),
    show_details: bool = typer.Option(False, "--details", help="Show detailed information"),
) -> None:
    """List all available experiments."""
    
    try:
        experiments = list_experiments(base_path)
        
        if not experiments:
            typer.echo("No experiments found.")
            return
        
        typer.echo(f"📁 Found {len(experiments)} experiments in {base_path}:")
        typer.echo("=" * 80)
        
        for exp in experiments:
            typer.echo(f"ID: {exp['experiment_id']}")
            typer.echo(f"Created: {exp['created_at']}")
            typer.echo(f"Generations: {exp['total_generations']}")
            typer.echo(f"Best Fitness: {exp['best_fitness']:.4f}")
            
            if exp['description']:
                typer.echo(f"Description: {exp['description']}")
            if exp['tags']:
                typer.echo(f"Tags: {', '.join(exp['tags'])}")
            
            if show_details:
                typer.echo(f"Total Evaluations: {exp['total_evaluations']}")
            
            typer.echo("-" * 40)
        
    except Exception as e:
        typer.echo(f"❌ Error listing experiments: {e}")
        raise typer.Exit(1)


@app.command()
def resume(
    experiment_id: str = typer.Argument(..., help="Experiment ID to resume"),
    generations: Optional[int] = typer.Option(None, "--generations", "-g", help="Additional generations to run"),
    base_path: str = typer.Option("experiments", "--base-path", help="Base directory for experiments"),
) -> None:
    """Resume an existing experiment."""
    
    # Load existing experiment configuration
    exp_path = Path(base_path) / experiment_id
    if not exp_path.exists():
        typer.echo(f"❌ Experiment {experiment_id} not found in {base_path}")
        raise typer.Exit(1)
    
    config_path = exp_path / "config.yaml"
    if not config_path.exists():
        typer.echo(f"❌ Configuration not found for experiment {experiment_id}")
        raise typer.Exit(1)
    
    config = Config.from_yaml(config_path)
    
    # Create persistent arena and resume
    arena = PersistentEvolutionArena(
        config=config,
        experiment_id=experiment_id,
        base_path=base_path
    )
    
    summary = arena.get_experiment_summary()
    typer.echo(f"📊 Experiment Summary:")
    typer.echo(f"ID: {summary['experiment_id']}")
    typer.echo(f"Current Generation: {summary['current_generation']}")
    typer.echo(f"Population Size: {summary['population_size']}")
    
    if 'current_best_fitness' in summary:
        typer.echo(f"Current Best Fitness: {summary['current_best_fitness']:.4f}")
    
    # Run additional generations
    if generations:
        typer.echo(f"Running {generations} additional generations...")
        try:
            result = arena.evolve(generations=generations, verbose=True)
            typer.echo(f"✅ Evolution completed!")
            typer.echo(f"Final generation: {arena.generation}")
        except KeyboardInterrupt:
            typer.echo("\n⚠️  Evolution interrupted by user")
            typer.echo(f"Progress saved for experiment: {arena.experiment_id}")
        except Exception as e:
            typer.echo(f"\n❌ Error during evolution: {e}")
            raise typer.Exit(1)
    else:
        typer.echo("Use --generations to continue evolution")


@app.command()
def export(
    experiment_id: str = typer.Argument(..., help="Experiment ID to export"),
    output_file: Path = typer.Option(None, "--output", "-o", help="Output archive file (.tar.gz)"),
    include_checkpoints: bool = typer.Option(False, "--checkpoints", help="Include checkpoint files"),
    base_path: str = typer.Option("experiments", "--base-path", help="Base directory for experiments"),
) -> None:
    """Export an experiment to a portable archive."""
    
    if output_file is None:
        output_file = Path(f"{experiment_id}_export.tar.gz")
    
    try:
        persistence = ModelPersistence(base_path)
        persistence.export_experiment(experiment_id, output_file, include_checkpoints)
        
        typer.echo(f"✅ Exported experiment {experiment_id} to {output_file}")
        typer.echo(f"Archive size: {output_file.stat().st_size / 1024 / 1024:.1f} MB")
        
    except Exception as e:
        typer.echo(f"❌ Error exporting experiment: {e}")
        raise typer.Exit(1)


@app.command()
def status(
    experiment_id: str = typer.Argument(..., help="Experiment ID to check"),
    base_path: str = typer.Option("experiments", "--base-path", help="Base directory for experiments"),
) -> None:
    """Show detailed status of an experiment."""
    
    try:
        persistence = ModelPersistence(base_path)
        exp_path = persistence.base_path / experiment_id
        
        if not exp_path.exists():
            typer.echo(f"❌ Experiment {experiment_id} not found")
            raise typer.Exit(1)
        
        # Load metadata
        metadata = persistence.load_metadata(exp_path / "metadata.json")
        
        typer.echo(f"📊 Experiment Status: {experiment_id}")
        typer.echo("=" * 60)
        typer.echo(f"Created: {metadata.created_at}")
        typer.echo(f"Description: {metadata.description}")
        typer.echo(f"Tags: {', '.join(metadata.tags) if metadata.tags else 'None'}")
        typer.echo(f"Total Generations: {metadata.total_generations}")
        typer.echo(f"Total Evaluations: {metadata.total_evaluations}")
        typer.echo(f"Best Fitness: {metadata.best_fitness:.4f}")
        
        # Check for checkpoints
        latest_checkpoint = persistence.find_latest_checkpoint(exp_path)
        if latest_checkpoint:
            typer.echo(f"Latest Checkpoint: {latest_checkpoint.name}")
        else:
            typer.echo("Latest Checkpoint: None")
        
        # Check for results
        results_file = exp_path / "results.json"
        if results_file.exists():
            typer.echo("Results: Available")
        else:
            typer.echo("Results: Not available")
        
        # Check for saved models
        best_agent_file = exp_path / "best_agent.pkl"
        if best_agent_file.exists():
            typer.echo("Best Agent: Saved")
        else:
            typer.echo("Best Agent: Not saved")
        
        # Directory size
        total_size = sum(f.stat().st_size for f in exp_path.rglob('*') if f.is_file())
        typer.echo(f"Disk Usage: {total_size / 1024 / 1024:.1f} MB")
        
    except Exception as e:
        typer.echo(f"❌ Error checking experiment status: {e}")
        raise typer.Exit(1)


@app.command()
def cleanup(
    experiment_id: str = typer.Argument(..., help="Experiment ID to clean up"),
    keep_every: int = typer.Option(10, "--keep-every", help="Keep every Nth checkpoint"),
    base_path: str = typer.Option("experiments", "--base-path", help="Base directory for experiments"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be deleted without actually deleting"),
) -> None:
    """Clean up old checkpoints for an experiment."""
    
    try:
        persistence = ModelPersistence(base_path)
        exp_path = persistence.base_path / experiment_id
        
        if not exp_path.exists():
            typer.echo(f"❌ Experiment {experiment_id} not found")
            raise typer.Exit(1)
        
        # Find checkpoints to delete
        checkpoints = list(exp_path.glob("checkpoint_gen_*.pkl"))
        checkpoints.sort(key=lambda p: int(p.stem.split('_')[-1]))
        
        to_delete = []
        for i, checkpoint in enumerate(checkpoints):
            generation = int(checkpoint.stem.split('_')[-1])
            if i < len(checkpoints) - 1 and generation % keep_every != 0:
                to_delete.append(checkpoint)
        
        if not to_delete:
            typer.echo("No checkpoints to clean up")
            return
        
        total_size = sum(f.stat().st_size for f in to_delete)
        typer.echo(f"Found {len(to_delete)} checkpoints to clean up")
        typer.echo(f"Would free: {total_size / 1024 / 1024:.1f} MB")
        
        if dry_run:
            typer.echo("Checkpoints that would be deleted:")
            for checkpoint in to_delete:
                typer.echo(f"  {checkpoint.name}")
        else:
            for checkpoint in to_delete:
                checkpoint.unlink()
            typer.echo(f"✅ Cleaned up {len(to_delete)} checkpoints")
        
    except Exception as e:
        typer.echo(f"❌ Error during cleanup: {e}")
        raise typer.Exit(1)


@app.command()
def best(
    experiment_id: str = typer.Argument(..., help="Experiment ID"),
    base_path: str = typer.Option("experiments", "--base-path", help="Base directory for experiments"),
    save_agent: Optional[Path] = typer.Option(None, "--save", help="Save best agent to file"),
) -> None:
    """Show and optionally save the best agent from an experiment."""
    
    try:
        persistence = ModelPersistence(base_path)
        exp_path = persistence.base_path / experiment_id
        
        if not exp_path.exists():
            typer.echo(f"❌ Experiment {experiment_id} not found")
            raise typer.Exit(1)
        
        # Load best agent
        best_agent_file = exp_path / "best_agent.pkl"
        if not best_agent_file.exists():
            typer.echo(f"❌ Best agent not found for experiment {experiment_id}")
            raise typer.Exit(1)
        
        agent = persistence.load_agent(best_agent_file)
        
        # Load metadata for fitness info
        metadata = persistence.load_metadata(exp_path / "metadata.json")
        
        typer.echo(f"🏆 Best Agent from {experiment_id}")
        typer.echo("=" * 40)
        typer.echo(f"Best Fitness: {metadata.best_fitness:.4f}")
        typer.echo(f"Architecture: {agent.architecture}")
        typer.echo(f"Total Parameters: {len(agent.weights):,}")
        
        # Save agent if requested
        if save_agent:
            persistence.save_agent(save_agent.parent, agent, save_agent.name)
            typer.echo(f"✅ Best agent saved to {save_agent}")
        
    except Exception as e:
        typer.echo(f"❌ Error accessing best agent: {e}")
        raise typer.Exit(1)


@app.command()
def benchmark(
    config_file: Optional[Path] = typer.Option(None, "--config", "-c", help="Configuration YAML file"),
    episodes: int = typer.Option(100, "--episodes", "-n", help="Episodes per policy (held-out seeds)"),
    agent_files: List[Path] = typer.Option(
        [], "--agent", "-a",
        help="Saved agent, genome or hall-of-fame pickle (can be used multiple times)"
    ),
    baselines: bool = typer.Option(True, "--baselines/--no-baselines", help="Include random/straight/greedy"),
    output_file: Optional[Path] = typer.Option(None, "--output", "-o", help="Write results to CSV"),
    seed_offset: int = typer.Option(BENCHMARK_SEED_OFFSET, "--seed-offset", help="First episode seed"),
) -> None:
    """Score baselines and saved agents on the same held-out seeds (mean ± 95% CI)."""

    from ..config import load_config
    from ..analysis.benchmark import (
        baseline_policies, benchmark_policies, format_benchmark_table,
        load_agent_file, saved_agent_factory,
    )

    config = load_config(config_file)
    policies = baseline_policies() if baselines else {}
    for agent_file in agent_files:
        try:
            agent = load_agent_file(agent_file, config)
        except (OSError, ValueError) as e:
            typer.echo(f"❌ Could not load {agent_file}: {e}")
            raise typer.Exit(1)
        policies[str(agent_file)] = saved_agent_factory(agent)

    if not policies:
        typer.echo("Nothing to benchmark: pass --agent or keep --baselines")
        raise typer.Exit(1)

    typer.echo(f"Fitness function: {config.evaluation.fitness_function}, "
               f"starvation: {config.game.starvation_steps or 'off'}, "
               f"{'walls' if config.game.walls else 'wrap'}, {episodes} episodes per policy")
    results = benchmark_policies(config, policies, episodes, seed_offset)
    typer.echo(format_benchmark_table(results))

    if output_file:
        results.to_csv(output_file, index=False)
        typer.echo(f"📄 Results saved to: {output_file}")


@app.command()
def multiseed(
    config_file: Optional[Path] = typer.Option(None, "--config", "-c", help="Configuration YAML file"),
    seeds: int = typer.Option(3, "--seeds", "-k", help="Number of independent runs"),
    first_seed: int = typer.Option(0, "--first-seed", help="Seed of the first run"),
    generations: Optional[int] = typer.Option(None, "--generations", "-g", help="Generations per run"),
    population: Optional[int] = typer.Option(None, "--population", "-p", help="Population size"),
    parallel_processes: Optional[int] = typer.Option(None, "--processes", help="Number of parallel processes"),
    episodes: int = typer.Option(100, "--episodes", "-n", help="Held-out episodes for the final benchmark"),
    early_stop: bool = typer.Option(True, "--early-stop/--no-early-stop", help="Allow early stopping"),
    output_dir: Path = typer.Option(Path("outputs/multiseed"), "--output", "-o", help="Output directory"),
) -> None:
    """Evolve with K seeds; write per-generation curves with 95% CIs and a final benchmark."""

    from ..config import load_config
    from ..analysis.benchmark import format_benchmark_table, run_multi_seed_evolution

    config = load_config(config_file)
    if generations is not None:
        config.evolution.generations = generations
    if population is not None:
        config.evolution.population_size = population
    if parallel_processes is not None:
        config.evaluation.parallel_processes = parallel_processes
    if not early_stop:
        config.evolution.early_stop_patience = config.evolution.generations + 1

    run_seeds = list(range(first_seed, first_seed + seeds))
    typer.echo(f"Running {len(run_seeds)} seeds {run_seeds} → {output_dir}")
    outputs = run_multi_seed_evolution(config, run_seeds, output_dir, benchmark_episodes=episodes)

    typer.echo("\nFinal benchmark (held-out seeds):")
    typer.echo(format_benchmark_table(outputs["benchmark"]))
    typer.echo(f"\n📄 Curves: {output_dir / 'curves_ci.csv'}, benchmark: {output_dir / 'final_benchmark.csv'}")


@app.command("protocol-demo")
def protocol_demo(
    trace: Optional[Path] = typer.Option(
        None, "--trace", help="Write the JSONL message trace to this file"
    ),
) -> None:
    """Run the protocol/runtime demo: command centre, specialists, signed envelopes."""
    from ..runtime.demo import main as demo_main

    code = demo_main([str(trace)] if trace is not None else [])
    if code != 0:
        raise typer.Exit(code)


@app.command("squad-benchmark")
def squad_benchmark(
    seeds: int = typer.Option(
        100, "--seeds", "-n", help="Held-out episodes per setting"
    ),
    radii: list[int] = typer.Option(
        [], "--radius", "-r", help="Window radius (repeatable; default 3, 5, 7)"
    ),
    batch: int = typer.Option(256, "--batch", help="Throughput batch size"),
    throughput: bool = typer.Option(
        True, "--throughput/--no-throughput", help="Measure agent-steps per second"
    ),
    output_file: Optional[Path] = typer.Option(
        None, "--out", "-o", help="Write the Markdown report here"
    ),
) -> None:
    """Squad need-to-talk baselines: team food with 95% CIs per radius and centre."""
    from ..analysis.need_to_talk import RADII, run_need_to_talk

    typer.echo(
        run_need_to_talk(
            seeds,
            radii or RADII,
            batch_size=batch,
            throughput=throughput,
            out=output_file,
        )
    )


@app.command(
    "learned-protocol",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
    add_help_option=False,
)
def learned_protocol(ctx: typer.Context) -> None:
    """Squad R6: learned vs hand-coded protocol (ES training + held-out evaluation).

    Takes the flags of scripts/learned_protocol.py, e.g. `--scale smoke`,
    `--radii 3 --runs 4 --generations 1000 --out docs/results/learned_protocol.md`,
    or the code-formation aids `--entropy-bonus 1 --centre-temperature 1`;
    `--help` lists them all.
    """
    from ..experiments.learned_protocol_report import main as report_main

    report_main(list(ctx.args))


if __name__ == "__main__":
    app()
