#!/usr/bin/env python3
"""Debug version of demo to find the exact error location."""

import sys
import traceback
sys.path.insert(0, 'src')

def debug_mini_evolution():
    """Test a mini evolution run with detailed error reporting."""
    try:
        from genetic_coil_arena.config import Config
        from genetic_coil_arena.evo.arena import EvolutionArena
        from pathlib import Path
        import time
        
        print("🧬 Testing mini evolution with debug info...")
        
        config = Config()
        config.evolution.population_size = 8
        config.evolution.generations = 3
        config.evaluation.episodes_per_agent = 1
        config.evaluation.parallel_processes = 2
        config.game.board_size = 16
        config.logging.output_dir = Path("demo_output")
        
        print("Creating EvolutionArena...")
        arena = EvolutionArena(config)
        
        print("Starting evolution...")
        start_time = time.time()
        result = arena.evolve()
        elapsed = time.time() - start_time
        
        print(f"✅ Evolution completed in {elapsed:.1f}s")
        print(f"   Generations: {len(result.generation_stats)}")
        print(f"   Hall of fame size: {len(result.hall_of_fame)}")
        
        if result.hall_of_fame:
            print(f"   Best fitness: {result.hall_of_fame[0].fitness:.3f}")
        else:
            print("   No valid genomes found")
            
    except Exception as e:
        print(f"❌ Error during evolution: {e}")
        print("\nFull traceback:")
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = debug_mini_evolution()
    sys.exit(0 if success else 1)
