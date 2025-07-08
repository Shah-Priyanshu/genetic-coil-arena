"""
Phase 3D: Database Integration & Data Persistence
File-based storage for development and production experiment storage
"""

import json
from datetime import datetime
from typing import Optional, List, Dict, Any
from pathlib import Path
import logging

from .file_storage import file_storage

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Database manager using file-based storage for experiment persistence and user data."""
    
    def __init__(self, storage_path: str = "data"):
        self.storage = file_storage
        self.storage_path = storage_path
    
    def create_user(self, username: str, email: str, hashed_password: str, role: str = "researcher") -> str:
        """Create a new user."""
        try:
            user_id = self.storage.create_user(username, email, hashed_password, role)
            logger.info(f"Created user: {username} with ID: {user_id}")
            return user_id
        except Exception as e:
            logger.error(f"Error creating user {username}: {e}")
            raise
    
    def get_user(self, username: str) -> Optional[Dict[str, Any]]:
        """Get user by username."""
        return self.storage.get_user(username)
    
    def update_last_login(self, username: str):
        """Update user's last login timestamp."""
        self.storage.update_last_login(username)
    
    def create_experiment(self, experiment_data: Dict[str, Any]) -> str:
        """Create a new experiment."""
        try:
            experiment_id = self.storage.create_experiment(experiment_data)
            logger.info(f"Created experiment: {experiment_id}")
            return experiment_id
        except Exception as e:
            logger.error(f"Error creating experiment: {e}")
            raise
    
    def update_experiment(self, experiment_id: str, updates: Dict[str, Any]):
        """Update experiment data."""
        try:
            self.storage.update_experiment(experiment_id, updates)
            logger.info(f"Updated experiment: {experiment_id}")
        except Exception as e:
            logger.error(f"Error updating experiment {experiment_id}: {e}")
            raise
    
    def get_experiment(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Get experiment by ID."""
        return self.storage.get_experiment(experiment_id)
    
    def get_user_experiments(self, user_id: str) -> List[Dict[str, Any]]:
        """Get all experiments for a user."""
        return self.storage.get_user_experiments(user_id)
    
    def record_generation(self, experiment_id: str, generation_data: Dict[str, Any]):
        """Record generation data for an experiment."""
        try:
            # Add generation data to experiment
            experiment = self.get_experiment(experiment_id)
            if experiment:
                if "generations" not in experiment:
                    experiment["generations"] = []
                
                generation_entry = {
                    "generation": generation_data.get("generation", len(experiment["generations"])),
                    "timestamp": datetime.now().isoformat(),
                    "best_fitness": generation_data.get("best_fitness"),
                    "avg_fitness": generation_data.get("avg_fitness"),
                    "metrics": generation_data.get("metrics", {}),
                    "population_size": generation_data.get("population_size")
                }
                
                experiment["generations"].append(generation_entry)
                self.storage.update_experiment(experiment_id, experiment)
                logger.info(f"Recorded generation {generation_entry['generation']} for experiment {experiment_id}")
        except Exception as e:
            logger.error(f"Error recording generation for {experiment_id}: {e}")
            raise
    
    def get_experiment_generations(self, experiment_id: str) -> List[Dict[str, Any]]:
        """Get all generation data for an experiment."""
        experiment = self.get_experiment(experiment_id)
        if experiment:
            return experiment.get("generations", [])
        return []
    
    def record_system_metrics(self, metrics: Dict[str, Any]):
        """Record system performance metrics."""
        try:
            # Save metrics to a special system metrics file
            metrics_data = {
                "timestamp": datetime.now().isoformat(),
                "metrics": metrics
            }
            
            # Read existing metrics
            metrics_file = Path(self.storage.storage_dir) / "system_metrics.json"
            all_metrics = []
            
            if metrics_file.exists():
                try:
                    with open(metrics_file, 'r') as f:
                        all_metrics = json.load(f)
                except:
                    all_metrics = []
            
            all_metrics.append(metrics_data)
            
            # Keep only last 1000 entries
            if len(all_metrics) > 1000:
                all_metrics = all_metrics[-1000:]
            
            with open(metrics_file, 'w') as f:
                json.dump(all_metrics, f, indent=2, default=str)
            
            logger.info("Recorded system metrics")
        except Exception as e:
            logger.error(f"Error recording system metrics: {e}")
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get system statistics."""
        return self.storage.get_system_stats()
    
    def cleanup_old_data(self, days: int = 30):
        """Clean up old data (placeholder for file-based system)."""
        logger.info(f"Cleanup requested for data older than {days} days")
        # Implementation depends on requirements
        pass
    
    async def initialize(self):
        """Initialize storage (async compatibility)."""
        await self.storage.initialize()
    
    def is_connected(self) -> bool:
        """Check if storage is accessible."""
        return self.storage.is_connected()
    
    async def close(self):
        """Close storage connections."""
        await self.storage.close()


# Global database instance
db = DatabaseManager()


def get_db_connection():
    """Get storage connection for compatibility with SQLite-based code."""
    from .file_storage import get_db_connection as get_file_connection
    return get_file_connection()
