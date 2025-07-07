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
    """Database manager for experiment persistence and user data."""
    
    def __init__(self, db_path: str = "genetic_coil_arena.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database tables."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Users table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    email TEXT UNIQUE NOT NULL,
                    hashed_password TEXT NOT NULL,
                    role TEXT DEFAULT 'researcher',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_login TIMESTAMP,
                    is_active BOOLEAN DEFAULT 1
                )
            """)
            
            # Experiments table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS experiments (
                    id TEXT PRIMARY KEY,
                    user_id INTEGER,
                    name TEXT NOT NULL,
                    description TEXT,
                    config_json TEXT NOT NULL,
                    status TEXT DEFAULT 'created',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    started_at TIMESTAMP,
                    completed_at TIMESTAMP,
                    current_generation INTEGER DEFAULT 0,
                    target_generations INTEGER,
                    best_fitness REAL DEFAULT 0.0,
                    avg_fitness REAL DEFAULT 0.0,
                    diversity REAL DEFAULT 1.0,
                    results_json TEXT,
                    error_message TEXT,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            """)
            
            # Generations table for detailed tracking
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS generations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id TEXT,
                    generation_number INTEGER,
                    best_fitness REAL,
                    avg_fitness REAL,
                    worst_fitness REAL,
                    diversity REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    generation_data_json TEXT,
                    FOREIGN KEY (experiment_id) REFERENCES experiments (id)
                )
            """)
            
            # System metrics table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS system_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    active_experiments INTEGER,
                    total_generations INTEGER,
                    cpu_usage REAL,
                    memory_usage REAL,
                    disk_usage REAL
                )
            """)
            
            conn.commit()
            logger.info("Database initialized successfully")
    
    def create_user(self, username: str, email: str, hashed_password: str, role: str = "researcher") -> int:
        """Create a new user and return user ID."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO users (username, email, hashed_password, role)
                VALUES (?, ?, ?, ?)
            """, (username, email, hashed_password, role))
            conn.commit()
            return cursor.lastrowid
    
    def get_user(self, username: str) -> Optional[Dict[str, Any]]:
        """Get user by username."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM users WHERE username = ? AND is_active = 1
            """, (username,))
            row = cursor.fetchone()
            return dict(row) if row else None
    
    def update_last_login(self, username: str):
        """Update user's last login timestamp."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE users SET last_login = CURRENT_TIMESTAMP
                WHERE username = ?
            """, (username,))
            conn.commit()
    
    def create_experiment(self, experiment_data: Dict[str, Any]) -> str:
        """Create a new experiment record."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO experiments (
                    id, user_id, name, description, config_json, status,
                    target_generations, started_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                experiment_data['id'],
                experiment_data.get('user_id'),
                experiment_data['name'],
                experiment_data.get('description', ''),
                json.dumps(experiment_data['config']),
                experiment_data.get('status', 'created'),
                experiment_data.get('target_generations', 100),
                datetime.now() if experiment_data.get('status') == 'running' else None
            ))
            conn.commit()
            return experiment_data['id']
    
    def update_experiment(self, experiment_id: str, updates: Dict[str, Any]):
        """Update experiment with new data."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Build dynamic update query
            set_clauses = []
            values = []
            
            for key, value in updates.items():
                if key in ['status', 'current_generation', 'best_fitness', 'avg_fitness', 'diversity', 'error_message']:
                    set_clauses.append(f"{key} = ?")
                    values.append(value)
                elif key == 'results':
                    set_clauses.append("results_json = ?")
                    values.append(json.dumps(value))
                elif key == 'completed':
                    set_clauses.append("completed_at = CURRENT_TIMESTAMP")
            
            if set_clauses:
                query = f"UPDATE experiments SET {', '.join(set_clauses)} WHERE id = ?"
                values.append(experiment_id)
                cursor.execute(query, values)
                conn.commit()
    
    def get_experiment(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Get experiment by ID."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("""
                SELECT e.*, u.username as owner_username
                FROM experiments e
                LEFT JOIN users u ON e.user_id = u.id
                WHERE e.id = ?
            """, (experiment_id,))
            row = cursor.fetchone()
            if row:
                experiment = dict(row)
                if experiment['config_json']:
                    experiment['config'] = json.loads(experiment['config_json'])
                if experiment['results_json']:
                    experiment['results'] = json.loads(experiment['results_json'])
                return experiment
            return None
    
    def get_user_experiments(self, user_id: int) -> List[Dict[str, Any]]:
        """Get all experiments for a user."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM experiments
                WHERE user_id = ?
                ORDER BY created_at DESC
            """, (user_id,))
            rows = cursor.fetchall()
            
            experiments = []
            for row in rows:
                experiment = dict(row)
                if experiment['config_json']:
                    experiment['config'] = json.loads(experiment['config_json'])
                experiments.append(experiment)
            return experiments
    
    def record_generation(self, experiment_id: str, generation_data: Dict[str, Any]):
        """Record generation data for detailed tracking."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO generations (
                    experiment_id, generation_number, best_fitness, avg_fitness,
                    worst_fitness, diversity, generation_data_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                experiment_id,
                generation_data['generation'],
                generation_data.get('best_fitness'),
                generation_data.get('avg_fitness'),
                generation_data.get('worst_fitness'),
                generation_data.get('diversity'),
                json.dumps(generation_data)
            ))
            conn.commit()
    
    def get_experiment_generations(self, experiment_id: str) -> List[Dict[str, Any]]:
        """Get all generation data for an experiment."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM generations
                WHERE experiment_id = ?
                ORDER BY generation_number
            """, (experiment_id,))
            rows = cursor.fetchall()
            
            generations = []
            for row in rows:
                generation = dict(row)
                if generation['generation_data_json']:
                    generation['data'] = json.loads(generation['generation_data_json'])
                generations.append(generation)
            return generations
    
    def record_system_metrics(self, metrics: Dict[str, Any]):
        """Record system performance metrics."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO system_metrics (
                    active_experiments, total_generations, cpu_usage, memory_usage, disk_usage
                ) VALUES (?, ?, ?, ?, ?)
            """, (
                metrics.get('active_experiments', 0),
                metrics.get('total_generations', 0),
                metrics.get('cpu_usage', 0.0),
                metrics.get('memory_usage', 0.0),
                metrics.get('disk_usage', 0.0)
            ))
            conn.commit()
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Get experiment counts
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_experiments,
                    COUNT(CASE WHEN status = 'running' THEN 1 END) as active_experiments,
                    COUNT(CASE WHEN status = 'completed' THEN 1 END) as completed_experiments,
                    COUNT(CASE WHEN status = 'failed' THEN 1 END) as failed_experiments
                FROM experiments
            """)
            exp_stats = dict(cursor.fetchone())
            
            # Get user count
            cursor.execute("SELECT COUNT(*) as total_users FROM users WHERE is_active = 1")
            user_stats = dict(cursor.fetchone())
            
            # Get generation count
            cursor.execute("SELECT COUNT(*) as total_generations FROM generations")
            gen_stats = dict(cursor.fetchone())
            
            return {**exp_stats, **user_stats, **gen_stats}
    
    def cleanup_old_data(self, days: int = 30):
        """Clean up old system metrics and completed experiments."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Clean old system metrics
            cursor.execute("""
                DELETE FROM system_metrics 
                WHERE timestamp < datetime('now', '-{} days')
            """.format(days))
            
            # Optionally clean very old completed experiments
            cursor.execute("""
                DELETE FROM experiments 
                WHERE status = 'completed' 
                AND completed_at < datetime('now', '-{} days')
            """.format(days * 2))
            
            conn.commit()
            logger.info(f"Cleaned up data older than {days} days")
    
    async def initialize(self):
        """Async wrapper for database initialization."""
        try:
            self.init_database()
            logger.info("Database initialization completed successfully")
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            raise
    
    def is_connected(self) -> bool:
        """Check if database connection is working."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT 1")
                return True
        except Exception:
            return False
    
    async def close(self):
        """Close database connections (placeholder for async compatibility)."""
        logger.info("Database connections closed")


# Global database instance
db = DatabaseManager()


def get_db_connection():
    """Get a database connection for direct SQLite operations."""
    return sqlite3.connect(db.db_path)
