"""
JSON File-based Storage System
A lightweight alternative to database storage for development and testing.
"""

import json
import os
from datetime import datetime
from typing import Optional, List, Dict, Any
from pathlib import Path
import logging
import threading
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class FileStorageManager:
    """File-based storage manager using JSON files."""
    
    def __init__(self, storage_dir: str = "data"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        self._lock = threading.Lock()
        
        # Create subdirectories for different data types
        self.users_dir = self.storage_dir / "users"
        self.experiments_dir = self.storage_dir / "experiments"
        self.teams_dir = self.storage_dir / "teams"
        self.notifications_dir = self.storage_dir / "notifications"
        self.versions_dir = self.storage_dir / "versions"
        
        for dir_path in [self.users_dir, self.experiments_dir, self.teams_dir, 
                        self.notifications_dir, self.versions_dir]:
            dir_path.mkdir(exist_ok=True)
        
        self.init_storage()
    
    def init_storage(self):
        """Initialize storage with default files."""
        # Initialize index files if they don't exist
        index_files = {
            "users_index.json": {},
            "experiments_index.json": {},
            "teams_index.json": {},
            "notifications_index.json": {},
            "versions_index.json": {}
        }
        
        for filename, default_data in index_files.items():
            file_path = self.storage_dir / filename
            if not file_path.exists():
                self._write_json(file_path, default_data)
    
    @contextmanager
    def _file_lock(self):
        """Context manager for thread-safe file operations."""
        self._lock.acquire()
        try:
            yield
        finally:
            self._lock.release()
    
    def _read_json(self, file_path: Path) -> Dict[str, Any]:
        """Read JSON data from file."""
        try:
            if file_path.exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            logger.error(f"Error reading {file_path}: {e}")
            return {}
    
    def _write_json(self, file_path: Path, data: Dict[str, Any]):
        """Write JSON data to file."""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, default=str, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error writing {file_path}: {e}")
    
    # User Management
    def create_user(self, username: str, email: str, hashed_password: str, role: str = "researcher") -> str:
        """Create a new user."""
        with self._file_lock():
            user_id = f"user_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(self.get_all_users())}"
            
            user_data = {
                "id": user_id,
                "username": username,
                "email": email,
                "hashed_password": hashed_password,
                "role": role,
                "created_at": datetime.now().isoformat(),
                "last_login": None,
                "is_active": True
            }
            
            # Save user data
            user_file = self.users_dir / f"{user_id}.json"
            self._write_json(user_file, user_data)
            
            # Update users index
            index_file = self.storage_dir / "users_index.json"
            index = self._read_json(index_file)
            index[username] = user_id
            index[email] = user_id
            self._write_json(index_file, index)
            
            return user_id
    
    def get_user(self, username: str) -> Optional[Dict[str, Any]]:
        """Get user by username."""
        with self._file_lock():
            index = self._read_json(self.storage_dir / "users_index.json")
            user_id = index.get(username)
            
            if user_id:
                user_file = self.users_dir / f"{user_id}.json"
                return self._read_json(user_file)
            return None
    
    def get_all_users(self) -> List[Dict[str, Any]]:
        """Get all users."""
        users = []
        for user_file in self.users_dir.glob("*.json"):
            user_data = self._read_json(user_file)
            if user_data:
                users.append(user_data)
        return users
    
    def update_last_login(self, username: str):
        """Update user's last login time."""
        with self._file_lock():
            user = self.get_user(username)
            if user:
                user["last_login"] = datetime.now().isoformat()
                user_file = self.users_dir / f"{user['id']}.json"
                self._write_json(user_file, user)
    
    # Experiment Management
    def create_experiment(self, experiment_data: Dict[str, Any]) -> str:
        """Create a new experiment."""
        with self._file_lock():
            experiment_id = experiment_data.get("id") or f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            experiment_data.update({
                "id": experiment_id,
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat()
            })
            
            # Save experiment data
            exp_file = self.experiments_dir / f"{experiment_id}.json"
            self._write_json(exp_file, experiment_data)
            
            # Update experiments index
            index_file = self.storage_dir / "experiments_index.json"
            index = self._read_json(index_file)
            index[experiment_id] = {
                "user_id": experiment_data.get("user_id"),
                "name": experiment_data.get("name"),
                "created_at": experiment_data["created_at"],
                "status": experiment_data.get("status", "created")
            }
            self._write_json(index_file, index)
            
            return experiment_id
    
    def get_experiment(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Get experiment by ID."""
        exp_file = self.experiments_dir / f"{experiment_id}.json"
        return self._read_json(exp_file)
    
    def update_experiment(self, experiment_id: str, updates: Dict[str, Any]):
        """Update experiment data."""
        with self._file_lock():
            experiment = self.get_experiment(experiment_id)
            if experiment:
                experiment.update(updates)
                experiment["updated_at"] = datetime.now().isoformat()
                
                exp_file = self.experiments_dir / f"{experiment_id}.json"
                self._write_json(exp_file, experiment)
    
    def get_user_experiments(self, user_id: str) -> List[Dict[str, Any]]:
        """Get all experiments for a user."""
        experiments = []
        index = self._read_json(self.storage_dir / "experiments_index.json")
        
        for exp_id, exp_info in index.items():
            if exp_info.get("user_id") == user_id:
                exp_data = self.get_experiment(exp_id)
                if exp_data:
                    experiments.append(exp_data)
        
        return experiments
    
    # Team Management
    def save_team(self, team_data: Dict[str, Any]):
        """Save team data."""
        with self._file_lock():
            team_id = team_data["id"]
            team_file = self.teams_dir / f"{team_id}.json"
            self._write_json(team_file, team_data)
            
            # Update teams index
            index_file = self.storage_dir / "teams_index.json"
            index = self._read_json(index_file)
            index[team_id] = {
                "name": team_data["name"],
                "owner_id": team_data["owner_id"],
                "created_at": team_data["created_at"],
                "member_count": len(team_data.get("members", []))
            }
            self._write_json(index_file, index)
    
    def get_team(self, team_id: str) -> Optional[Dict[str, Any]]:
        """Get team by ID."""
        team_file = self.teams_dir / f"{team_id}.json"
        return self._read_json(team_file)
    
    def get_user_teams(self, user_id: str) -> List[Dict[str, Any]]:
        """Get all teams for a user."""
        teams = []
        for team_file in self.teams_dir.glob("*.json"):
            team_data = self._read_json(team_file)
            if team_data:
                # Check if user is a member
                members = team_data.get("members", [])
                if any(member.get("user_id") == user_id for member in members):
                    teams.append(team_data)
        return teams
    
    def delete_team(self, team_id: str):
        """Delete a team."""
        with self._file_lock():
            team_file = self.teams_dir / f"{team_id}.json"
            if team_file.exists():
                team_file.unlink()
            
            # Update index
            index_file = self.storage_dir / "teams_index.json"
            index = self._read_json(index_file)
            index.pop(team_id, None)
            self._write_json(index_file, index)
    
    # Notification Management
    def save_notification(self, notification_data: Dict[str, Any]):
        """Save notification data."""
        with self._file_lock():
            notif_id = notification_data["id"]
            notif_file = self.notifications_dir / f"{notif_id}.json"
            self._write_json(notif_file, notification_data)
    
    def get_user_notifications(self, user_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Get notifications for a user."""
        notifications = []
        for notif_file in self.notifications_dir.glob("*.json"):
            notif_data = self._read_json(notif_file)
            if notif_data and notif_data.get("user_id") == user_id:
                notifications.append(notif_data)
        
        # Sort by created_at descending and limit
        notifications.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        return notifications[:limit]
    
    # Version Control
    def save_version(self, version_data: Dict[str, Any]):
        """Save version data."""
        with self._file_lock():
            version_id = version_data["id"]
            version_file = self.versions_dir / f"{version_id}.json"
            self._write_json(version_file, version_data)
    
    def get_experiment_versions(self, experiment_id: str) -> List[Dict[str, Any]]:
        """Get all versions for an experiment."""
        versions = []
        for version_file in self.versions_dir.glob("*.json"):
            version_data = self._read_json(version_file)
            if version_data and version_data.get("experiment_id") == experiment_id:
                versions.append(version_data)
        
        # Sort by created_at
        versions.sort(key=lambda x: x.get("created_at", ""))
        return versions
    
    # System Stats
    def get_system_stats(self) -> Dict[str, Any]:
        """Get system statistics."""
        users_count = len(list(self.users_dir.glob("*.json")))
        experiments_count = len(list(self.experiments_dir.glob("*.json")))
        teams_count = len(list(self.teams_dir.glob("*.json")))
        notifications_count = len(list(self.notifications_dir.glob("*.json")))
        
        return {
            "users_count": users_count,
            "experiments_count": experiments_count,
            "teams_count": teams_count,
            "notifications_count": notifications_count,
            "storage_type": "file_based",
            "storage_location": str(self.storage_dir.absolute()),
            "last_updated": datetime.now().isoformat()
        }
    
    # Compatibility methods for existing database interface
    async def initialize(self):
        """Initialize storage (async compatibility)."""
        logger.info("✅ File storage initialized successfully")
    
    def is_connected(self) -> bool:
        """Check if storage is accessible."""
        return self.storage_dir.exists() and os.access(self.storage_dir, os.R_OK | os.W_OK)
    
    async def close(self):
        """Close storage (async compatibility)."""
        logger.info("📁 File storage closed")


# Create a function that works like get_db_connection for compatibility
def get_file_storage():
    """Get file storage instance for compatibility with existing code."""
    return file_storage


# Global file storage instance
file_storage = FileStorageManager()

# Alias for backward compatibility
db = file_storage


def get_db_connection():
    """Get storage connection for compatibility."""
    class MockConnection:
        def __init__(self, storage):
            self.storage = storage
        
        def __enter__(self):
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            pass
        
        def execute(self, query, params=None):
            # This is a mock for compatibility - in practice you'd implement
            # translation from SQL-like operations to file operations
            pass
        
        def fetchall(self):
            return []
        
        def fetchone(self):
            return None
        
        def commit(self):
            pass
    
    return MockConnection(file_storage)
