"""
Model persistence and state management for genetic-coil-arena.

This module provides comprehensive save/load functionality for:
- Individual genomes and agents
- Complete populations and evolution state
- Experiment metadata and configuration
- Results and analysis data

Features:
- Version-compatible serialization
- Incremental checkpointing
- Metadata tracking
- Compression and optimization
- Cross-platform compatibility
"""

import json
import pickle
import gzip
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple
from datetime import datetime
import hashlib
import logging

import numpy as np
from dataclasses import dataclass, asdict
from genetic_coil_arena.config import Config
from genetic_coil_arena.snake_core.agent import SnakeAgent
from genetic_coil_arena.evo.genome import Genome

logger = logging.getLogger(__name__)

# Version for backwards compatibility
PERSISTENCE_VERSION = "1.0.0"


@dataclass
class ExperimentMetadata:
    """Metadata for evolution experiments."""
    experiment_id: str
    created_at: datetime
    config_hash: str
    description: str
    tags: List[str]
    total_generations: int
    total_evaluations: int
    best_fitness: float
    persistence_version: str = PERSISTENCE_VERSION
    
    @classmethod
    def create(cls, config: Config, description: str = "", tags: List[str] = None) -> 'ExperimentMetadata':
        """Create metadata from config."""
        # Convert config to dict and handle Path objects
        config_dict = asdict(config)
        config_dict = cls._convert_paths_to_strings(config_dict)
        config_str = json.dumps(config_dict, sort_keys=True)
        config_hash = hashlib.md5(config_str.encode()).hexdigest()
        
        return cls(
            experiment_id=f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            created_at=datetime.now(),
            config_hash=config_hash,
            description=description,
            tags=tags or [],
            total_generations=0,
            total_evaluations=0,
            best_fitness=0.0,
        )
    
    @staticmethod
    def _convert_paths_to_strings(obj):
        """Recursively convert Path objects to strings for JSON serialization."""
        if isinstance(obj, dict):
            return {key: ExperimentMetadata._convert_paths_to_strings(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [ExperimentMetadata._convert_paths_to_strings(item) for item in obj]
        elif isinstance(obj, Path):
            return str(obj)
        else:
            return obj


@dataclass
class CheckpointState:
    """Complete evolution state for checkpointing."""
    generation: int
    population: List[Genome]
    hall_of_fame: List[Genome]
    fitness_history: List[List[float]]
    diversity_history: List[float]
    metadata: ExperimentMetadata
    config: Config
    random_state: Dict[str, Any]
    evolution_state: Dict[str, Any]  # Additional evolution algorithm state


class ModelPersistence:
    """Comprehensive model persistence system."""
    
    def __init__(self, base_path: Union[str, Path] = "experiments"):
        """Initialize persistence system.
        
        Args:
            base_path: Base directory for experiments
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(exist_ok=True)
        
    def create_experiment(self, config: Config, description: str = "", 
                         tags: List[str] = None) -> Tuple[str, Path]:
        """Create new experiment directory with metadata.
        
        Args:
            config: Experiment configuration
            description: Human-readable description
            tags: Tags for categorization
            
        Returns:
            (experiment_id, experiment_path)
        """
        metadata = ExperimentMetadata.create(config, description, tags)
        exp_path = self.base_path / metadata.experiment_id
        exp_path.mkdir(exist_ok=True)
        
        # Save initial metadata and config
        self.save_metadata(exp_path, metadata)
        self.save_config(exp_path, config)
        
        logger.info(f"Created experiment {metadata.experiment_id} at {exp_path}")
        return metadata.experiment_id, exp_path
    
    def save_genome(self, path: Path, genome: Genome, 
                   filename: str = "genome.pkl") -> Path:
        """Save individual genome.
        
        Args:
            path: Directory path
            genome: Genome to save
            filename: Output filename
            
        Returns:
            Full path to saved file
        """
        file_path = path / filename
        with gzip.open(file_path, 'wb') as f:
            pickle.dump(genome, f, protocol=pickle.HIGHEST_PROTOCOL)
        return file_path
    
    def load_genome(self, file_path: Path) -> Genome:
        """Load individual genome.
        
        Args:
            file_path: Path to genome file
            
        Returns:
            Loaded genome
        """
        with gzip.open(file_path, 'rb') as f:
            return pickle.load(f)
    
    def save_population(self, path: Path, population: List[Genome],
                       filename: str = "population.pkl") -> Path:
        """Save complete population.
        
        Args:
            path: Directory path
            population: Population to save
            filename: Output filename
            
        Returns:
            Full path to saved file
        """
        file_path = path / filename
        with gzip.open(file_path, 'wb') as f:
            pickle.dump(population, f, protocol=pickle.HIGHEST_PROTOCOL)
        return file_path
    
    def load_population(self, file_path: Path) -> List[Genome]:
        """Load complete population.
        
        Args:
            file_path: Path to population file
            
        Returns:
            Loaded population
        """
        with gzip.open(file_path, 'rb') as f:
            return pickle.load(f)
    
    def save_agent(self, path: Path, agent: SnakeAgent,
                  filename: str = "agent.pkl") -> Path:
        """Save neural agent.
        
        Args:
            path: Directory path
            agent: Agent to save
            filename: Output filename
            
        Returns:
            Full path to saved file
        """
        file_path = path / filename
        with gzip.open(file_path, 'wb') as f:
            pickle.dump(agent, f, protocol=pickle.HIGHEST_PROTOCOL)
        return file_path
    
    def load_agent(self, file_path: Path) -> SnakeAgent:
        """Load neural agent.
        
        Args:
            file_path: Path to agent file
            
        Returns:
            Loaded agent
        """
        with gzip.open(file_path, 'rb') as f:
            return pickle.load(f)
    
    def save_checkpoint(self, path: Path, state: CheckpointState,
                       filename: str = None) -> Path:
        """Save complete evolution checkpoint.
        
        Args:
            path: Directory path
            state: Checkpoint state
            filename: Output filename (auto-generated if None)
            
        Returns:
            Full path to saved file
        """
        if filename is None:
            filename = f"checkpoint_gen_{state.generation:06d}.pkl"
        
        file_path = path / filename
        with gzip.open(file_path, 'wb') as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        # Also save as "latest_checkpoint.pkl" for easy resumption
        latest_path = path / "latest_checkpoint.pkl"
        shutil.copy2(file_path, latest_path)
        
        logger.info(f"Saved checkpoint for generation {state.generation}")
        return file_path
    
    def load_checkpoint(self, file_path: Path) -> CheckpointState:
        """Load evolution checkpoint.
        
        Args:
            file_path: Path to checkpoint file
            
        Returns:
            Loaded checkpoint state
        """
        with gzip.open(file_path, 'rb') as f:
            state = pickle.load(f)
        
        # Validate version compatibility
        if hasattr(state.metadata, 'persistence_version'):
            if state.metadata.persistence_version != PERSISTENCE_VERSION:
                logger.warning(f"Version mismatch: file={state.metadata.persistence_version}, "
                             f"current={PERSISTENCE_VERSION}")
        
        return state
    
    def find_latest_checkpoint(self, exp_path: Path) -> Optional[Path]:
        """Find the most recent checkpoint in experiment directory.
        
        Args:
            exp_path: Experiment directory path
            
        Returns:
            Path to latest checkpoint or None if not found
        """
        latest_path = exp_path / "latest_checkpoint.pkl"
        if latest_path.exists():
            return latest_path
        
        # Fallback: search for numbered checkpoints
        checkpoints = list(exp_path.glob("checkpoint_gen_*.pkl"))
        if checkpoints:
            # Sort by generation number
            checkpoints.sort(key=lambda p: int(p.stem.split('_')[-1]))
            return checkpoints[-1]
        
        return None
    
    def save_metadata(self, path: Path, metadata: ExperimentMetadata,
                     filename: str = "metadata.json") -> Path:
        """Save experiment metadata.
        
        Args:
            path: Directory path
            metadata: Metadata to save
            filename: Output filename
            
        Returns:
            Full path to saved file
        """
        file_path = path / filename
        
        # Convert datetime to ISO format for JSON serialization
        metadata_dict = asdict(metadata)
        metadata_dict['created_at'] = metadata.created_at.isoformat()
        
        with open(file_path, 'w') as f:
            json.dump(metadata_dict, f, indent=2)
        
        return file_path
    
    def load_metadata(self, file_path: Path) -> ExperimentMetadata:
        """Load experiment metadata.
        
        Args:
            file_path: Path to metadata file
            
        Returns:
            Loaded metadata
        """
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Convert ISO format back to datetime
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        
        return ExperimentMetadata(**data)
    
    def save_config(self, path: Path, config: Config,
                   filename: str = "config.yaml") -> Path:
        """Save experiment configuration.
        
        Args:
            path: Directory path
            config: Configuration to save
            filename: Output filename
            
        Returns:
            Full path to saved file
        """
        import yaml
        
        file_path = path / filename
        with open(file_path, 'w') as f:
            yaml.dump(asdict(config), f, default_flow_style=False)
        
        return file_path
    
    def load_config(self, file_path: Path) -> Config:
        """Load experiment configuration.
        
        Args:
            file_path: Path to config file
            
        Returns:
            Loaded configuration
        """
        import yaml
        
        with open(file_path, 'r') as f:
            data = yaml.safe_load(f)
        
        return Config(**data)
    
    def save_results(self, path: Path, results: Dict[str, Any],
                    filename: str = "results.json") -> Path:
        """Save experiment results and analysis.
        
        Args:
            path: Directory path
            results: Results dictionary
            filename: Output filename
            
        Returns:
            Full path to saved file
        """
        file_path = path / filename
        
        # Convert numpy arrays to lists for JSON serialization
        json_results = self._serialize_results(results)
        
        with open(file_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        return file_path
    
    def load_results(self, file_path: Path) -> Dict[str, Any]:
        """Load experiment results.
        
        Args:
            file_path: Path to results file
            
        Returns:
            Loaded results dictionary
        """
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        return self._deserialize_results(data)
    
    def _serialize_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Convert numpy arrays and other non-JSON types for serialization."""
        serialized = {}
        for key, value in results.items():
            if isinstance(value, np.ndarray):
                serialized[key] = value.tolist()
            elif isinstance(value, dict):
                serialized[key] = self._serialize_results(value)
            elif isinstance(value, list) and value and isinstance(value[0], np.ndarray):
                serialized[key] = [v.tolist() for v in value]
            else:
                serialized[key] = value
        return serialized
    
    def _deserialize_results(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert serialized data back to appropriate types."""
        # For now, just return as-is since we're primarily saving for analysis
        # In the future, could add type hints to recreate numpy arrays
        return data
    
    def list_experiments(self) -> List[Tuple[str, ExperimentMetadata]]:
        """List all experiments with their metadata.
        
        Returns:
            List of (experiment_id, metadata) tuples
        """
        experiments = []
        
        for exp_dir in self.base_path.iterdir():
            if exp_dir.is_dir():
                metadata_path = exp_dir / "metadata.json"
                if metadata_path.exists():
                    try:
                        metadata = self.load_metadata(metadata_path)
                        experiments.append((exp_dir.name, metadata))
                    except Exception as e:
                        logger.warning(f"Failed to load metadata for {exp_dir.name}: {e}")
        
        # Sort by creation time, newest first
        experiments.sort(key=lambda x: x[1].created_at, reverse=True)
        return experiments
    
    def cleanup_old_checkpoints(self, exp_path: Path, keep_every: int = 10):
        """Clean up old checkpoints, keeping every Nth checkpoint.
        
        Args:
            exp_path: Experiment directory path
            keep_every: Keep every Nth checkpoint (delete others)
        """
        checkpoints = list(exp_path.glob("checkpoint_gen_*.pkl"))
        checkpoints.sort(key=lambda p: int(p.stem.split('_')[-1]))
        
        for i, checkpoint in enumerate(checkpoints):
            generation = int(checkpoint.stem.split('_')[-1])
            if i < len(checkpoints) - 1 and generation % keep_every != 0:
                checkpoint.unlink()
                logger.debug(f"Removed checkpoint {checkpoint.name}")
    
    def export_experiment(self, exp_id: str, output_path: Path, 
                         include_checkpoints: bool = False):
        """Export experiment to a portable archive.
        
        Args:
            exp_id: Experiment ID to export
            output_path: Output file path
            include_checkpoints: Whether to include checkpoint files
        """
        exp_path = self.base_path / exp_id
        
        if not exp_path.exists():
            raise ValueError(f"Experiment {exp_id} not found")
        
        # Create tar.gz archive
        import tarfile
        
        with tarfile.open(output_path, 'w:gz') as tar:
            # Always include metadata, config, results
            for filename in ["metadata.json", "config.yaml", "results.json"]:
                file_path = exp_path / filename
                if file_path.exists():
                    tar.add(file_path, arcname=filename)
            
            # Include best models
            for pattern in ["hall_of_fame.pkl", "best_agent.pkl"]:
                for file_path in exp_path.glob(pattern):
                    tar.add(file_path, arcname=file_path.name)
            
            # Optionally include checkpoints
            if include_checkpoints:
                for file_path in exp_path.glob("checkpoint_*.pkl"):
                    tar.add(file_path, arcname=file_path.name)
        
        logger.info(f"Exported experiment {exp_id} to {output_path}")


# Convenience functions for common operations
def quick_save_agent(agent: SnakeAgent, filename: str) -> Path:
    """Quick save for an agent."""
    persistence = ModelPersistence()
    output_path = Path(filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return persistence.save_agent(output_path.parent, agent, output_path.name)


def quick_load_agent(filename: str) -> SnakeAgent:
    """Quick load for an agent."""
    persistence = ModelPersistence()
    return persistence.load_agent(Path(filename))


def quick_save_population(population: List[Genome], filename: str) -> Path:
    """Quick save for a population."""
    persistence = ModelPersistence()
    output_path = Path(filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return persistence.save_population(output_path.parent, population, output_path.name)


def quick_load_population(filename: str) -> List[Genome]:
    """Quick load for a population."""
    persistence = ModelPersistence()
    return persistence.load_population(Path(filename))
