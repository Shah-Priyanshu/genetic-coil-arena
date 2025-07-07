"""
Phase 4A: Experiment Comparison Tools
Advanced side-by-side experiment analysis and comparison.
"""

import json
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import logging

from .models import ExperimentStatus
from .database import DatabaseManager
from ..persistence.arena import list_experiments as list_stored_experiments

logger = logging.getLogger(__name__)


class ComparisonMetric(Enum):
    """Metrics for experiment comparison."""
    BEST_FITNESS = "best_fitness"
    AVERAGE_FITNESS = "average_fitness"
    CONVERGENCE_RATE = "convergence_rate"
    DIVERSITY = "diversity"
    GENERATION_TIME = "generation_time"
    TOTAL_TIME = "total_time"
    EFFICIENCY = "efficiency"


@dataclass
class ComparisonData:
    """Data structure for experiment comparison."""
    experiment_id: str
    name: str
    status: str
    metrics: Dict[str, float]
    fitness_history: List[float]
    diversity_history: List[float]
    generation_times: List[float]
    config: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ComparisonResult:
    """Result of experiment comparison analysis."""
    experiments: List[ComparisonData]
    summary: Dict[str, Any]
    rankings: Dict[ComparisonMetric, List[Tuple[str, float]]]  # (experiment_id, score)
    statistical_tests: Dict[str, Any]
    recommendations: List[str]


class ExperimentComparison:
    """Advanced experiment comparison and analysis tools."""
    
    def __init__(self, experiment_manager, db_manager: DatabaseManager):
        self.experiment_manager = experiment_manager
        self.db_manager = db_manager
    
    async def compare_experiments(self, experiment_ids: List[str]) -> ComparisonResult:
        """Compare multiple experiments across various metrics."""
        if len(experiment_ids) < 2:
            raise ValueError("At least 2 experiments required for comparison")
        
        # Collect experiment data
        comparison_data = []
        for exp_id in experiment_ids:
            data = await self._collect_experiment_data(exp_id)
            if data:
                comparison_data.append(data)
        
        if len(comparison_data) < 2:
            raise ValueError("Could not load data for sufficient experiments")
        
        # Perform comparison analysis
        summary = self._calculate_summary_stats(comparison_data)
        rankings = self._calculate_rankings(comparison_data)
        statistical_tests = self._perform_statistical_tests(comparison_data)
        recommendations = self._generate_recommendations(comparison_data, rankings)
        
        return ComparisonResult(
            experiments=comparison_data,
            summary=summary,
            rankings=rankings,
            statistical_tests=statistical_tests,
            recommendations=recommendations
        )
    
    async def _collect_experiment_data(self, experiment_id: str) -> Optional[ComparisonData]:
        """Collect comprehensive data for a single experiment."""
        try:
            # Try to get from active experiments first
            status = self.experiment_manager.get_experiment_status(experiment_id)
            if status and experiment_id in self.experiment_manager.active_arenas:
                exp = self.experiment_manager.active_arenas[experiment_id]
                arena = exp['arena']
                
                # Get fitness and diversity history
                fitness_history = [max(gen) for gen in arena.fitness_history] if arena.fitness_history else []
                diversity_history = arena.diversity_history if arena.diversity_history else []
                
                # Calculate metrics
                metrics = self._calculate_metrics(fitness_history, diversity_history, exp)
                
                return ComparisonData(
                    experiment_id=experiment_id,
                    name=exp['config'].name,
                    status=exp['status'],
                    metrics=metrics,
                    fitness_history=fitness_history,
                    diversity_history=diversity_history,
                    generation_times=[],  # Would need to track this
                    config=exp['config'].model_dump(),
                    metadata={
                        'start_time': exp['start_time'].isoformat(),
                        'generations': arena.generation
                    }
                )
            
            # Try to get from persistent storage
            stored_experiments = list_stored_experiments(self.experiment_manager.base_path)
            for exp_info in stored_experiments:
                if exp_info['experiment_id'] == experiment_id:
                    # Load experiment data from storage
                    metrics = {
                        'best_fitness': exp_info.get('best_fitness', 0.0),
                        'total_generations': exp_info.get('total_generations', 0),
                        'convergence_rate': 0.0,  # Would need to calculate from data
                        'efficiency': 0.0
                    }
                    
                    return ComparisonData(
                        experiment_id=experiment_id,
                        name=exp_info.get('description', 'Stored Experiment'),
                        status='completed',
                        metrics=metrics,
                        fitness_history=[],  # Would need to load from files
                        diversity_history=[],
                        generation_times=[],
                        config={},  # Would need to load from files
                        metadata=exp_info
                    )
            
            logger.warning(f"Experiment {experiment_id} not found")
            return None
            
        except Exception as e:
            logger.error(f"Failed to collect data for experiment {experiment_id}: {e}")
            return None
    
    def _calculate_metrics(self, fitness_history: List[float], diversity_history: List[float], 
                          experiment: Dict[str, Any]) -> Dict[str, float]:
        """Calculate comparison metrics for an experiment."""
        metrics = {}
        
        # Basic fitness metrics
        if fitness_history:
            metrics['best_fitness'] = max(fitness_history)
            metrics['average_fitness'] = sum(fitness_history) / len(fitness_history)
            metrics['final_fitness'] = fitness_history[-1]
            
            # Convergence rate (improvement per generation)
            if len(fitness_history) > 1:
                improvements = [fitness_history[i] - fitness_history[i-1] 
                               for i in range(1, len(fitness_history)) if fitness_history[i] > fitness_history[i-1]]
                metrics['convergence_rate'] = sum(improvements) / len(fitness_history) if improvements else 0.0
            else:
                metrics['convergence_rate'] = 0.0
        else:
            metrics['best_fitness'] = 0.0
            metrics['average_fitness'] = 0.0
            metrics['final_fitness'] = 0.0
            metrics['convergence_rate'] = 0.0
        
        # Diversity metrics
        if diversity_history:
            metrics['average_diversity'] = sum(diversity_history) / len(diversity_history)
            metrics['final_diversity'] = diversity_history[-1]
        else:
            metrics['average_diversity'] = 0.0
            metrics['final_diversity'] = 0.0
        
        # Time metrics
        if 'start_time' in experiment:
            elapsed = (datetime.now() - experiment['start_time']).total_seconds()
            metrics['total_time'] = elapsed
            
            # Efficiency (fitness improvement per second)
            if elapsed > 0 and fitness_history:
                metrics['efficiency'] = metrics['best_fitness'] / elapsed
            else:
                metrics['efficiency'] = 0.0
        else:
            metrics['total_time'] = 0.0
            metrics['efficiency'] = 0.0
        
        # Configuration-based metrics
        config = experiment.get('config', {})
        if hasattr(config, 'model_dump'):
            config = config.model_dump()
        
        metrics['population_size'] = config.get('population_size', 0)
        metrics['generations'] = config.get('generations', 0)
        metrics['mutation_rate'] = config.get('mutation_rate', 0.0)
        metrics['crossover_rate'] = config.get('crossover_rate', 0.0)
        
        return metrics
    
    def _calculate_summary_stats(self, experiments: List[ComparisonData]) -> Dict[str, Any]:
        """Calculate summary statistics across all experiments."""
        summary = {
            'total_experiments': len(experiments),
            'completed_experiments': len([e for e in experiments if e.status == 'completed']),
            'running_experiments': len([e for e in experiments if e.status == 'running']),
        }
        
        # Aggregate metrics
        all_metrics = {}
        for metric_name in ['best_fitness', 'average_fitness', 'convergence_rate', 'efficiency']:
            values = [exp.metrics.get(metric_name, 0.0) for exp in experiments]
            if values:
                all_metrics[metric_name] = {
                    'min': min(values),
                    'max': max(values),
                    'mean': sum(values) / len(values),
                    'std': float(np.std(values)) if len(values) > 1 else 0.0
                }
        
        summary['metrics'] = all_metrics
        return summary
    
    def _calculate_rankings(self, experiments: List[ComparisonData]) -> Dict[ComparisonMetric, List[Tuple[str, float]]]:
        """Calculate rankings for each comparison metric."""
        rankings = {}
        
        for metric in ComparisonMetric:
            metric_name = metric.value
            
            # Get values for this metric
            values = []
            for exp in experiments:
                value = exp.metrics.get(metric_name, 0.0)
                values.append((exp.experiment_id, value))
            
            # Sort by value (descending for most metrics)
            reverse_sort = metric not in [ComparisonMetric.GENERATION_TIME, ComparisonMetric.TOTAL_TIME]
            values.sort(key=lambda x: x[1], reverse=reverse_sort)
            
            rankings[metric] = values
        
        return rankings
    
    def _perform_statistical_tests(self, experiments: List[ComparisonData]) -> Dict[str, Any]:
        """Perform statistical tests on experiment results."""
        tests = {}
        
        # Basic statistical comparisons
        fitness_values = [exp.metrics.get('best_fitness', 0.0) for exp in experiments]
        
        if len(fitness_values) > 1:
            tests['fitness_variance'] = float(np.var(fitness_values))
            tests['fitness_range'] = max(fitness_values) - min(fitness_values)
            tests['fitness_coefficient_of_variation'] = float(np.std(fitness_values)) / np.mean(fitness_values) if np.mean(fitness_values) > 0 else 0.0
        
        # Configuration analysis
        config_analysis = {}
        for config_param in ['population_size', 'mutation_rate', 'crossover_rate']:
            values = [exp.metrics.get(config_param, 0.0) for exp in experiments]
            fitness_values = [exp.metrics.get('best_fitness', 0.0) for exp in experiments]
            
            if len(values) > 1 and len(fitness_values) > 1:
                # Simple correlation analysis
                correlation = float(np.corrcoef(values, fitness_values)[0, 1]) if not np.isnan(np.corrcoef(values, fitness_values)[0, 1]) else 0.0
                config_analysis[config_param] = {
                    'correlation_with_fitness': correlation,
                    'values_range': [min(values), max(values)]
                }
        
        tests['configuration_analysis'] = config_analysis
        return tests
    
    def _generate_recommendations(self, experiments: List[ComparisonData], 
                                rankings: Dict[ComparisonMetric, List[Tuple[str, float]]]) -> List[str]:
        """Generate recommendations based on comparison analysis."""
        recommendations = []
        
        # Best performing experiment
        best_fitness_ranking = rankings.get(ComparisonMetric.BEST_FITNESS, [])
        if best_fitness_ranking:
            best_exp_id = best_fitness_ranking[0][0]
            best_exp = next((e for e in experiments if e.experiment_id == best_exp_id), None)
            if best_exp:
                recommendations.append(
                    f"🏆 Best performing experiment: '{best_exp.name}' "
                    f"(fitness: {best_exp.metrics.get('best_fitness', 0.0):.3f})"
                )
        
        # Efficiency analysis
        efficiency_ranking = rankings.get(ComparisonMetric.EFFICIENCY, [])
        if efficiency_ranking:
            most_efficient_id = efficiency_ranking[0][0]
            most_efficient = next((e for e in experiments if e.experiment_id == most_efficient_id), None)
            if most_efficient:
                recommendations.append(
                    f"⚡ Most efficient experiment: '{most_efficient.name}' "
                    f"(efficiency: {most_efficient.metrics.get('efficiency', 0.0):.6f})"
                )
        
        # Configuration recommendations
        population_sizes = [exp.metrics.get('population_size', 0) for exp in experiments]
        fitness_values = [exp.metrics.get('best_fitness', 0.0) for exp in experiments]
        
        if len(population_sizes) > 1 and len(fitness_values) > 1:
            # Find optimal population size
            pop_fitness_pairs = list(zip(population_sizes, fitness_values))
            pop_fitness_pairs.sort(key=lambda x: x[1], reverse=True)
            best_pop_size = pop_fitness_pairs[0][0]
            
            recommendations.append(
                f"📊 Optimal population size appears to be: {best_pop_size}"
            )
        
        # Diversity recommendations
        diversity_values = [exp.metrics.get('average_diversity', 0.0) for exp in experiments]
        if diversity_values:
            avg_diversity = sum(diversity_values) / len(diversity_values)
            if avg_diversity < 0.1:
                recommendations.append(
                    "🔄 Consider increasing mutation rate or population size to improve diversity"
                )
            elif avg_diversity > 0.8:
                recommendations.append(
                    "🎯 Consider decreasing mutation rate or adding more selection pressure"
                )
        
        # Convergence recommendations
        convergence_rates = [exp.metrics.get('convergence_rate', 0.0) for exp in experiments]
        if convergence_rates:
            avg_convergence = sum(convergence_rates) / len(convergence_rates)
            if avg_convergence < 0.01:
                recommendations.append(
                    "🐌 Slow convergence detected. Try increasing selection pressure or adjusting parameters"
                )
        
        return recommendations
    
    async def generate_comparison_report(self, experiment_ids: List[str]) -> Dict[str, Any]:
        """Generate a comprehensive comparison report."""
        comparison_result = await self.compare_experiments(experiment_ids)
        
        # Create detailed report
        report = {
            'timestamp': datetime.now().isoformat(),
            'experiment_count': len(comparison_result.experiments),
            'summary': comparison_result.summary,
            'rankings': {
                metric.value: [
                    {'experiment_id': exp_id, 'score': score, 'experiment_name': 
                     next((e.name for e in comparison_result.experiments if e.experiment_id == exp_id), 'Unknown')}
                    for exp_id, score in ranking
                ]
                for metric, ranking in comparison_result.rankings.items()
            },
            'statistical_analysis': comparison_result.statistical_tests,
            'recommendations': comparison_result.recommendations,
            'experiment_details': [
                {
                    'id': exp.experiment_id,
                    'name': exp.name,
                    'status': exp.status,
                    'metrics': exp.metrics,
                    'config': exp.config
                }
                for exp in comparison_result.experiments
            ]
        }
        
        return report
    
    def export_comparison_data(self, comparison_result: ComparisonResult, format: str = 'json') -> str:
        """Export comparison data in various formats."""
        if format == 'json':
            data = {
                'experiments': [
                    {
                        'id': exp.experiment_id,
                        'name': exp.name,
                        'status': exp.status,
                        'metrics': exp.metrics,
                        'fitness_history': exp.fitness_history,
                        'diversity_history': exp.diversity_history,
                        'config': exp.config,
                        'metadata': exp.metadata
                    }
                    for exp in comparison_result.experiments
                ],
                'summary': comparison_result.summary,
                'rankings': {
                    metric.value: ranking 
                    for metric, ranking in comparison_result.rankings.items()
                },
                'statistical_tests': comparison_result.statistical_tests,
                'recommendations': comparison_result.recommendations
            }
            return json.dumps(data, indent=2, default=str)
        
        elif format == 'csv':
            # Generate CSV format for metrics
            import io
            output = io.StringIO()
            
            # Header
            metrics = comparison_result.experiments[0].metrics.keys() if comparison_result.experiments else []
            output.write('experiment_id,name,status,' + ','.join(metrics) + '\n')
            
            # Data rows
            for exp in comparison_result.experiments:
                row = [exp.experiment_id, exp.name, exp.status]
                row.extend([str(exp.metrics.get(metric, '')) for metric in metrics])
                output.write(','.join(row) + '\n')
            
            return output.getvalue()
        
        else:
            raise ValueError(f"Unsupported export format: {format}")
