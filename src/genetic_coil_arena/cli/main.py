"""Main CLI application using Typer."""

import typer
from pathlib import Path
from typing import Optional
import sys
import yaml

from ..config import Config
from ..evo.arena import EvolutionArena
from ..snake_core.agent import SnakeAgent, get_observation_size

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
) -> None:
    """Run genetic algorithm evolution to train snake agents."""
    
    # Load configuration
    if config_file and config_file.exists():
        config = Config.from_yaml(config_file)
        typer.echo(f"Loaded configuration from {config_file}")
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
    
    # Create output directory
    config.logging.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run evolution
    typer.echo("Starting evolution...")
    arena = EvolutionArena(config)
    
    try:
        result = arena.evolve()
        
        typer.echo(f"\n✅ Evolution completed!")
        typer.echo(f"Best fitness: {result.hall_of_fame[0].fitness:.3f}")
        typer.echo(f"Total time: {result.total_time:.2f} seconds")
        typer.echo(f"Results saved to: {config.logging.output_dir}")
        
        if result.converged:
            typer.echo(f"Converged at generation {result.convergence_generation}")
        
    except KeyboardInterrupt:
        typer.echo("\n⚠️  Evolution interrupted by user")
        sys.exit(1)
    except Exception as e:
        typer.echo(f"\n❌ Error during evolution: {e}")
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
Final Success Rate: {df['success_rate'].iloc[-1]:.1%}
Final Diversity: {df['diversity'].iloc[-1]:.4f}
Relative to Baseline: {df['relative_to_baseline'].iloc[-1]:.2f}x

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
            
            # Performance relative to baseline
            axes[1, 1].plot(df['generation'], df['relative_to_baseline'], 'c-', linewidth=2)
            axes[1, 1].set_title('Performance vs Baseline')
            axes[1, 1].set_xlabel('Generation')
            axes[1, 1].set_ylabel('Relative Performance')
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
        from ..snake_core.agent import SnakeAgent
        from ..config import load_config
        
        # Load configuration
        config = load_config(config_file) if config_file else load_config()
        
        # Create game
        game = SnakeGame(
            board_size=config.game.board_size,
            observation_radius=config.game.observation_radius
        )
        
        # Load agent if provided
        agent = None
        if agent_file and agent_file.exists():
            import pickle
            with open(agent_file, 'rb') as f:
                genome = pickle.load(f)
            agent = SnakeAgent.from_genome(genome, config.agent)
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


if __name__ == "__main__":
    app()
