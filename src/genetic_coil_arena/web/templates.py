"""
Phase 4A: Experiment Templates System
Provides pre-configured experiment templates for quick start and best practices.
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
import json
from dataclasses import dataclass, asdict
from enum import Enum


class ExperimentType(Enum):
    """Types of pre-configured experiments."""
    QUICK_START = "quick_start"
    PERFORMANCE_TEST = "performance_test"
    LONG_EVOLUTION = "long_evolution"
    COMPARISON_STUDY = "comparison_study"
    PARAMETER_SWEEP = "parameter_sweep"
    BASELINE_BENCHMARK = "baseline_benchmark"


@dataclass
class ExperimentTemplate:
    """Template for creating experiments with pre-configured settings."""
    id: str
    name: str
    description: str
    category: ExperimentType
    config: Dict[str, Any]
    estimated_duration: str
    difficulty_level: str  # "beginner", "intermediate", "advanced"
    tags: List[str]
    created_by: str = "system"
    version: str = "1.0"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert template to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExperimentTemplate':
        """Create template from dictionary."""
        return cls(**data)


class ExperimentTemplateManager:
    """Manages experiment templates and template-based experiment creation."""
    
    def __init__(self):
        self.templates: Dict[str, ExperimentTemplate] = {}
        self._initialize_default_templates()
    
    def _initialize_default_templates(self):
        """Initialize the default set of experiment templates."""
        
        # Quick Start Template
        quick_start = ExperimentTemplate(
            id="quick_start_basic",
            name="🚀 Quick Start Evolution",
            description="Perfect for beginners! A fast 10-generation evolution to see the system in action.",
            category=ExperimentType.QUICK_START,
            config={
                "population_size": 50,
                "generations": 10,
                "mutation_rate": 0.1,
                "crossover_rate": 0.8,
                "selection_pressure": 1.5,
                "elite_count": 5,
                "neural_network": {
                    "hidden_layers": [16, 8],
                    "activation": "tanh"
                },
                "environment": {
                    "grid_size": 15,
                    "food_count": 3,
                    "max_steps": 200
                }
            },
            estimated_duration="2-5 minutes",
            difficulty_level="beginner",
            tags=["quick", "demo", "beginner", "tutorial"]
        )
        
        # Performance Test Template
        performance_test = ExperimentTemplate(
            id="performance_benchmark",
            name="⚡ Performance Benchmark",
            description="High-performance test with large population to benchmark system capabilities.",
            category=ExperimentType.PERFORMANCE_TEST,
            config={
                "population_size": 500,
                "generations": 25,
                "mutation_rate": 0.08,
                "crossover_rate": 0.9,
                "selection_pressure": 2.0,
                "elite_count": 25,
                "neural_network": {
                    "hidden_layers": [32, 16],
                    "activation": "relu"
                },
                "environment": {
                    "grid_size": 20,
                    "food_count": 5,
                    "max_steps": 300
                },
                "parallel_processing": True
            },
            estimated_duration="10-20 minutes",
            difficulty_level="intermediate",
            tags=["performance", "benchmark", "large-scale"]
        )
        
        # Long Evolution Template
        long_evolution = ExperimentTemplate(
            id="long_evolution_study",
            name="🧬 Long Evolution Study",
            description="Extended evolution over 100 generations to observe long-term adaptation patterns.",
            category=ExperimentType.LONG_EVOLUTION,
            config={
                "population_size": 200,
                "generations": 100,
                "mutation_rate": 0.05,
                "crossover_rate": 0.85,
                "selection_pressure": 1.8,
                "elite_count": 10,
                "neural_network": {
                    "hidden_layers": [24, 12, 6],
                    "activation": "tanh"
                },
                "environment": {
                    "grid_size": 18,
                    "food_count": 4,
                    "max_steps": 400
                },
                "adaptive_mutation": True,
                "diversity_preservation": True
            },
            estimated_duration="45-90 minutes",
            difficulty_level="advanced",
            tags=["long-term", "research", "adaptation", "advanced"]
        )
        
        # Comparison Study Template
        comparison_study = ExperimentTemplate(
            id="algorithm_comparison",
            name="🔬 Algorithm Comparison",
            description="Compare different evolutionary algorithms side by side.",
            category=ExperimentType.COMPARISON_STUDY,
            config={
                "population_size": 100,
                "generations": 50,
                "algorithms": [
                    {
                        "name": "Standard GA",
                        "mutation_rate": 0.1,
                        "crossover_rate": 0.8,
                        "selection": "tournament"
                    },
                    {
                        "name": "Enhanced GA",
                        "mutation_rate": 0.08,
                        "crossover_rate": 0.9,
                        "selection": "rank_based"
                    }
                ],
                "neural_network": {
                    "hidden_layers": [20, 10],
                    "activation": "sigmoid"
                },
                "environment": {
                    "grid_size": 16,
                    "food_count": 3,
                    "max_steps": 250
                }
            },
            estimated_duration="30-60 minutes",
            difficulty_level="intermediate",
            tags=["comparison", "research", "algorithms"]
        )
        
        # Parameter Sweep Template
        parameter_sweep = ExperimentTemplate(
            id="parameter_sweep_study",
            name="📊 Parameter Sweep Analysis",
            description="Systematic exploration of parameter space to find optimal configurations.",
            category=ExperimentType.PARAMETER_SWEEP,
            config={
                "base_config": {
                    "population_size": 100,
                    "generations": 30,
                    "neural_network": {
                        "hidden_layers": [16, 8],
                        "activation": "tanh"
                    },
                    "environment": {
                        "grid_size": 15,
                        "food_count": 3,
                        "max_steps": 200
                    }
                },
                "sweep_parameters": {
                    "mutation_rate": [0.05, 0.1, 0.15, 0.2],
                    "crossover_rate": [0.7, 0.8, 0.9],
                    "selection_pressure": [1.2, 1.5, 2.0]
                },
                "runs_per_config": 3
            },
            estimated_duration="2-4 hours",
            difficulty_level="advanced",
            tags=["parameter-sweep", "optimization", "research", "systematic"]
        )
        
        # Baseline Benchmark Template
        baseline_benchmark = ExperimentTemplate(
            id="baseline_benchmark",
            name="📈 Baseline Benchmark",
            description="Standard benchmark configuration for comparing against published results.",
            category=ExperimentType.BASELINE_BENCHMARK,
            config={
                "population_size": 100,
                "generations": 50,
                "mutation_rate": 0.1,
                "crossover_rate": 0.8,
                "selection_pressure": 1.5,
                "elite_count": 5,
                "neural_network": {
                    "hidden_layers": [20, 10],
                    "activation": "tanh",
                    "bias": True
                },
                "environment": {
                    "grid_size": 16,
                    "food_count": 3,
                    "max_steps": 200,
                    "wall_penalty": -10
                },
                "random_seed": 42,  # Reproducible results
                "recording_enabled": True
            },
            estimated_duration="20-40 minutes",
            difficulty_level="intermediate",
            tags=["benchmark", "standard", "reproducible", "baseline"]
        )
        
        # Store all templates
        templates = [
            quick_start, performance_test, long_evolution, 
            comparison_study, parameter_sweep, baseline_benchmark
        ]
        
        for template in templates:
            self.templates[template.id] = template
    
    def get_template(self, template_id: str) -> Optional[ExperimentTemplate]:
        """Get a specific template by ID."""
        return self.templates.get(template_id)
    
    def list_templates(self, category: Optional[ExperimentType] = None) -> List[ExperimentTemplate]:
        """List all templates, optionally filtered by category."""
        templates = list(self.templates.values())
        
        if category:
            templates = [t for t in templates if t.category == category]
        
        return sorted(templates, key=lambda t: (t.difficulty_level, t.name))
    
    def get_templates_by_difficulty(self, difficulty: str) -> List[ExperimentTemplate]:
        """Get templates filtered by difficulty level."""
        return [t for t in self.templates.values() if t.difficulty_level == difficulty]
    
    def get_templates_by_tags(self, tags: List[str]) -> List[ExperimentTemplate]:
        """Get templates that match any of the provided tags."""
        matching_templates = []
        for template in self.templates.values():
            if any(tag in template.tags for tag in tags):
                matching_templates.append(template)
        return matching_templates
    
    def create_experiment_from_template(self, template_id: str, 
                                      experiment_name: str,
                                      user_id: Optional[str] = None,
                                      custom_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Create a new experiment configuration from a template."""
        template = self.get_template(template_id)
        if not template:
            raise ValueError(f"Template {template_id} not found")
        
        # Start with template config
        experiment_config = template.config.copy()
        
        # Apply custom overrides if provided
        if custom_config:
            experiment_config.update(custom_config)
        
        # Create experiment metadata
        experiment = {
            "name": experiment_name,
            "description": f"Created from template: {template.name}",
            "template_id": template_id,
            "config": experiment_config,
            "created_at": datetime.now().isoformat(),
            "created_by": user_id or "anonymous",
            "status": "created",
            "tags": template.tags + ["template-based"],
            "estimated_duration": template.estimated_duration
        }
        
        return experiment
    
    def add_custom_template(self, template: ExperimentTemplate) -> bool:
        """Add a custom user-created template."""
        if template.id in self.templates:
            return False  # Template ID already exists
        
        self.templates[template.id] = template
        return True
    
    def export_template(self, template_id: str) -> Optional[str]:
        """Export a template as JSON string."""
        template = self.get_template(template_id)
        if not template:
            return None
        
        return json.dumps(template.to_dict(), indent=2)
    
    def import_template(self, template_json: str) -> bool:
        """Import a template from JSON string."""
        try:
            template_data = json.loads(template_json)
            template = ExperimentTemplate.from_dict(template_data)
            return self.add_custom_template(template)
        except (json.JSONDecodeError, TypeError, ValueError):
            return False


# Global template manager instance
template_manager = ExperimentTemplateManager()
