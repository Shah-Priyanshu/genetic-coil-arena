"""Evolution progress monitoring and real-time visualization."""

import threading
import queue
import time
from typing import List, Optional, Dict, Any, Callable
from dataclasses import dataclass
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.figure import Figure
import pandas as pd
import numpy as np

from ..evo.genome import Genome, GenomeStats
from ..evo.arena import GenerationStats, EvolutionResult


@dataclass
class EvolutionUpdate:
    """Update from evolution process."""
    generation: int
    generation_stats: GenerationStats
    best_genome: Genome
    population_sample: List[Genome]  # Sample of current population
    elapsed_time: float
    
    
class EvolutionMonitor:
    """Monitor evolution progress in real-time."""
    
    def __init__(self, update_interval: float = 1.0):
        """Initialize the monitor.
        
        Args:
            update_interval: How often to update display (seconds)
        """
        self.update_interval = update_interval
        self.update_queue: queue.Queue = queue.Queue()
        self.updates: List[EvolutionUpdate] = []
        self.callbacks: List[Callable[[EvolutionUpdate], None]] = []
        self.running = False
        self._monitor_thread: Optional[threading.Thread] = None
        
    def add_callback(self, callback: Callable[[EvolutionUpdate], None]) -> None:
        """Add a callback for evolution updates.
        
        Args:
            callback: Function to call with each update
        """
        self.callbacks.append(callback)
        
    def start_monitoring(self) -> None:
        """Start the monitoring thread."""
        if self.running:
            return
            
        self.running = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop)
        self._monitor_thread.daemon = True
        self._monitor_thread.start()
        
    def stop_monitoring(self) -> None:
        """Stop the monitoring thread."""
        self.running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2.0)
            
    def add_update(self, update: EvolutionUpdate) -> None:
        """Add an evolution update.
        
        Args:
            update: The evolution update to process
        """
        try:
            self.update_queue.put_nowait(update)
        except queue.Full:
            pass  # Skip if queue is full
            
    def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        while self.running:
            try:
                # Process all available updates
                while not self.update_queue.empty():
                    update = self.update_queue.get_nowait()
                    self.updates.append(update)
                    
                    # Call all callbacks
                    for callback in self.callbacks:
                        try:
                            callback(update)
                        except Exception as e:
                            print(f"Error in evolution monitor callback: {e}")
                            
                time.sleep(self.update_interval)
                
            except Exception as e:
                print(f"Error in evolution monitor loop: {e}")
                
    def get_progress_summary(self) -> Dict[str, Any]:
        """Get a summary of evolution progress.
        
        Returns:
            Summary statistics
        """
        if not self.updates:
            return {}
            
        latest = self.updates[-1]
        generations_completed = len(self.updates)
        
        # Calculate improvement rate
        if len(self.updates) >= 2:
            first_fitness = self.updates[0].generation_stats.best_fitness
            latest_fitness = latest.generation_stats.best_fitness
            improvement = latest_fitness - first_fitness
            improvement_rate = improvement / (latest.elapsed_time / 60.0)  # per minute
        else:
            improvement_rate = 0.0
            
        return {
            'generations_completed': generations_completed,
            'current_generation': latest.generation,
            'best_fitness': latest.generation_stats.best_fitness,
            'mean_fitness': latest.generation_stats.mean_fitness,
            'fitness_std': latest.generation_stats.std_fitness,
            'success_rate': latest.generation_stats.success_rate,
            'diversity': latest.generation_stats.diversity,
            'improvement_rate_per_minute': improvement_rate,
            'elapsed_time_minutes': latest.elapsed_time / 60.0,
            'relative_to_baseline': latest.generation_stats.relative_to_baseline,
        }


class LivePlotter:
    """Real-time plotting of evolution metrics."""
    
    def __init__(self, metrics: List[str] = None):
        """Initialize the live plotter.
        
        Args:
            metrics: List of metrics to plot
        """
        if metrics is None:
            metrics = ['best_fitness', 'mean_fitness', 'success_rate', 'diversity']
        
        self.metrics = metrics
        self.fig: Optional[Figure] = None
        self.axes: Dict[str, Any] = {}
        self.data: Dict[str, List[float]] = {metric: [] for metric in metrics}
        self.generations: List[int] = []
        self.animation: Optional[animation.FuncAnimation] = None
        
    def setup_plots(self) -> None:
        """Set up the matplotlib figure and axes."""
        plt.ion()  # Enable interactive mode
        
        num_metrics = len(self.metrics)
        rows = (num_metrics + 1) // 2
        cols = 2 if num_metrics > 1 else 1
        
        self.fig, axes = plt.subplots(rows, cols, figsize=(12, 8))
        if num_metrics == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes if hasattr(axes, '__len__') else [axes]
        else:
            axes = axes.flatten()
            
        self.fig.suptitle('Evolution Progress - Real Time')
        
        for i, metric in enumerate(self.metrics):
            ax = axes[i] if i < len(axes) else None
            if ax is not None:
                ax.set_title(metric.replace('_', ' ').title())
                ax.set_xlabel('Generation')
                ax.set_ylabel(metric)
                ax.grid(True, alpha=0.3)
                self.axes[metric] = ax
                
        # Hide unused subplots
        for i in range(len(self.metrics), len(axes)):
            axes[i].set_visible(False)
            
        plt.tight_layout()
        plt.show()
        
    def update_plots(self, update: EvolutionUpdate) -> None:
        """Update plots with new data.
        
        Args:
            update: Evolution update with new data
        """
        if self.fig is None:
            self.setup_plots()
            
        # Add new data
        self.generations.append(update.generation)
        
        # Extract metrics from generation stats
        stats = update.generation_stats
        metric_values = {
            'best_fitness': stats.best_fitness,
            'mean_fitness': stats.mean_fitness,
            'success_rate': stats.success_rate * 100,  # Convert to percentage
            'diversity': stats.diversity,
            'relative_to_baseline': stats.relative_to_baseline,
            'evaluation_time': stats.evaluation_time,
        }
        
        for metric in self.metrics:
            if metric in metric_values:
                self.data[metric].append(metric_values[metric])
                
                # Update plot
                ax = self.axes.get(metric)
                if ax is not None:
                    ax.clear()
                    ax.plot(self.generations, self.data[metric], 'b-', linewidth=2)
                    ax.set_title(metric.replace('_', ' ').title())
                    ax.set_xlabel('Generation')
                    ax.set_ylabel(metric)
                    ax.grid(True, alpha=0.3)
                    
                    # Add trend line for fitness metrics
                    if 'fitness' in metric and len(self.generations) > 5:
                        z = np.polyfit(self.generations, self.data[metric], 1)
                        p = np.poly1d(z)
                        ax.plot(self.generations, p(self.generations), 'r--', alpha=0.7, linewidth=1)
        
        # Refresh the display
        if self.fig:
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            
    def save_plots(self, filename: str) -> None:
        """Save current plots to file.
        
        Args:
            filename: Output filename
        """
        if self.fig:
            self.fig.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Evolution plots saved to: {filename}")


class EvolutionAnalyzer:
    """Analyze evolution results and generate reports."""
    
    def __init__(self):
        """Initialize the analyzer."""
        pass
        
    def analyze_convergence(self, updates: List[EvolutionUpdate]) -> Dict[str, Any]:
        """Analyze convergence behavior.
        
        Args:
            updates: List of evolution updates
            
        Returns:
            Convergence analysis results
        """
        if len(updates) < 5:
            return {'error': 'Not enough data for convergence analysis'}
            
        # Extract fitness data
        generations = [u.generation for u in updates]
        best_fitness = [u.generation_stats.best_fitness for u in updates]
        mean_fitness = [u.generation_stats.mean_fitness for u in updates]
        
        # Calculate improvement rates
        fitness_improvements = np.diff(best_fitness)
        
        # Find potential convergence point
        window_size = min(10, len(fitness_improvements) // 3)
        if window_size < 3:
            convergence_gen = None
        else:
            # Look for plateau (small improvements over window)
            plateau_threshold = np.std(best_fitness) * 0.01  # 1% of total std
            
            convergence_gen = None
            for i in range(window_size, len(fitness_improvements)):
                window_improvements = fitness_improvements[i-window_size:i]
                if np.mean(np.abs(window_improvements)) < plateau_threshold:
                    convergence_gen = generations[i]
                    break
        
        return {
            'total_generations': len(updates),
            'final_fitness': best_fitness[-1],
            'total_improvement': best_fitness[-1] - best_fitness[0],
            'convergence_generation': convergence_gen,
            'plateau_threshold': plateau_threshold if window_size >= 3 else None,
            'improvement_trend': 'increasing' if fitness_improvements[-5:].mean() > 0 else 'plateau',
            'fitness_variance': np.var(best_fitness),
        }
        
    def analyze_diversity(self, updates: List[EvolutionUpdate]) -> Dict[str, Any]:
        """Analyze population diversity over time.
        
        Args:
            updates: List of evolution updates
            
        Returns:
            Diversity analysis results
        """
        if not updates:
            return {}
            
        diversity_values = [u.generation_stats.diversity for u in updates]
        generations = [u.generation for u in updates]
        
        # Calculate diversity trend
        if len(diversity_values) >= 5:
            z = np.polyfit(generations, diversity_values, 1)
            diversity_trend = 'increasing' if z[0] > 0 else 'decreasing'
            diversity_slope = z[0]
        else:
            diversity_trend = 'unknown'
            diversity_slope = 0.0
            
        return {
            'initial_diversity': diversity_values[0],
            'final_diversity': diversity_values[-1],
            'diversity_trend': diversity_trend,
            'diversity_slope': diversity_slope,
            'max_diversity': max(diversity_values),
            'min_diversity': min(diversity_values),
            'diversity_variance': np.var(diversity_values),
        }
        
    def generate_report(self, updates: List[EvolutionUpdate]) -> str:
        """Generate a comprehensive evolution report.
        
        Args:
            updates: List of evolution updates
            
        Returns:
            Formatted report string
        """
        if not updates:
            return "No evolution data available."
            
        latest = updates[-1]
        convergence = self.analyze_convergence(updates)
        diversity = self.analyze_diversity(updates)
        
        report = f"""
EVOLUTION ANALYSIS REPORT
========================

OVERVIEW
--------
Total Generations: {len(updates)}
Final Generation: {latest.generation}
Elapsed Time: {latest.elapsed_time:.1f} seconds ({latest.elapsed_time/60:.1f} minutes)

PERFORMANCE
-----------
Best Fitness: {latest.generation_stats.best_fitness:.4f}
Mean Fitness: {latest.generation_stats.mean_fitness:.4f}
Success Rate: {latest.generation_stats.success_rate:.1%}
Relative to Baseline: {latest.generation_stats.relative_to_baseline:.2f}x

CONVERGENCE ANALYSIS
--------------------
Total Improvement: {convergence.get('total_improvement', 0):.4f}
Convergence Generation: {convergence.get('convergence_generation', 'Not detected')}
Trend: {convergence.get('improvement_trend', 'Unknown')}

DIVERSITY ANALYSIS
------------------
Final Diversity: {diversity.get('final_diversity', 0):.4f}
Diversity Trend: {diversity.get('diversity_trend', 'Unknown')}
Max Diversity: {diversity.get('max_diversity', 0):.4f}

RECOMMENDATIONS
---------------
"""
        
        # Add recommendations based on analysis
        if convergence.get('improvement_trend') == 'plateau':
            report += "- Evolution appears to have converged. Consider:\n"
            report += "  * Increasing mutation rate\n"
            report += "  * Adding diversity-promoting mechanisms\n"
            report += "  * Adjusting selection pressure\n"
            
        if diversity.get('diversity_trend') == 'decreasing':
            report += "- Population diversity is decreasing. Consider:\n"
            report += "  * Implementing niching or speciation\n"
            report += "  * Reducing selection pressure\n"
            report += "  * Adding novelty search\n"
            
        if latest.generation_stats.success_rate < 0.1:
            report += "- Low success rate detected. Consider:\n"
            report += "  * Adjusting reward function\n"
            report += "  * Increasing network capacity\n"
            report += "  * Tuning hyperparameters\n"
            
        return report
