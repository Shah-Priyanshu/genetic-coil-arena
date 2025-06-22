#!/usr/bin/env python3
"""
Demonstration of Phase 2 advanced features.

This script shows off the new analysis, visualization, and advanced evolution capabilities
that have been added to the genetic-coil-arena project.
"""

import numpy as np
from pathlib import Path
import tempfile
import os

from genetic_coil_arena.config import Config
from genetic_coil_arena.evo.arena import EvolutionArena
from genetic_coil_arena.analysis.evolution_monitor import EvolutionAnalyzer


def main():
    print("🎉 Genetic Coil Arena - Phase 2 Features Demo")
    print("=" * 50)
    
    # Create a small test configuration
    config = Config()
    config.evolution.population_size = 20
    config.evolution.generations = 10
    config.evaluation.episodes_per_agent = 1
    config.evaluation.parallel_processes = 2
    config.game.board_size = 12
    config.game.max_steps = 100
    
    print(f"🧪 Running quick evolution demo...")
    print(f"Population: {config.evolution.population_size}")
    print(f"Generations: {config.evolution.generations}")
    print(f"Board size: {config.game.board_size}x{config.game.board_size}")
    
    # Create temporary output directory
    with tempfile.TemporaryDirectory() as temp_dir:
        config.logging.output_dir = Path(temp_dir)
        
        # Run evolution
        arena = EvolutionArena(config)
        result = arena.evolve()
        
        print(f"\n✅ Evolution completed!")
        print(f"Best fitness: {result.hall_of_fame[0].fitness:.4f}")
        print(f"Total time: {result.total_time:.1f} seconds")
        
        # Demonstrate analysis capabilities
        print(f"\n📊 Running analysis...")
        analyzer = EvolutionAnalyzer()
        
        # Convert result to updates format for analysis
        from genetic_coil_arena.analysis.evolution_monitor import EvolutionUpdate
        updates = []
        for i, gen_stats in enumerate(result.generation_stats):
            update = EvolutionUpdate(
                generation=i,
                generation_stats=gen_stats,
                best_genome=result.hall_of_fame[0] if result.hall_of_fame else None,
                population_sample=[],
                elapsed_time=(i + 1) * (result.total_time / len(result.generation_stats))
            )
            updates.append(update)
        
        # Generate analysis report
        if updates:
            report = analyzer.generate_report(updates)
            print(report)
            
            # Convergence analysis
            convergence = analyzer.analyze_convergence(updates)
            print(f"🔍 Convergence Analysis:")
            print(f"Total improvement: {convergence.get('total_improvement', 0):.4f}")
            print(f"Trend: {convergence.get('improvement_trend', 'Unknown')}")
            
            # Diversity analysis  
            diversity = analyzer.analyze_diversity(updates)
            print(f"\n🌱 Diversity Analysis:")
            print(f"Final diversity: {diversity.get('final_diversity', 0):.4f}")
            print(f"Trend: {diversity.get('diversity_trend', 'Unknown')}")
        
        # Show available files
        print(f"\n📁 Generated files:")
        output_files = list(config.logging.output_dir.glob("*"))
        for file in output_files:
            print(f"  - {file.name}")
    
    print(f"\n🎯 Advanced Features Available:")
    print(f"✓ Real-time evolution monitoring")
    print(f"✓ Comprehensive analysis and reporting")  
    print(f"✓ Novelty search and diversity preservation")
    print(f"✓ Multi-objective optimization")
    print(f"✓ Speciation-based evolution")
    print(f"✓ Adaptive parameter tuning")
    print(f"✓ Interactive Jupyter notebooks")
    print(f"✓ Pygame-based game visualization")
    
    print(f"\n🚀 Try these commands:")
    print(f"  genetic-coil evolve --generations 50")
    print(f"  genetic-coil analyze outputs/ --plot")
    print(f"  genetic-coil visualize --agent outputs/hall_of_fame.pkl")
    
    print(f"\n📚 Check out the notebooks:")
    print(f"  notebooks/01_quick_start.ipynb")
    print(f"  notebooks/02_advanced_techniques.ipynb")
    
    print(f"\n🎉 Phase 2 demonstration complete!")


if __name__ == "__main__":
    main()
