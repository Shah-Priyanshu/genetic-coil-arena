"""
Production deployment configuration for Genetic Coil Arena.
"""

import os
from typing import Dict, Any

# Environment-based configuration
class DeploymentConfig:
    """Configuration for different deployment environments."""
    
    # Development configuration
    DEVELOPMENT = {
        "host": "localhost",
        "port": 8000,
        "debug": True,
        "reload": True,
        "workers": 1,
        "log_level": "debug",
        "cors_origins": ["*"],
        "database_url": "sqlite:///./genetic_coil_arena_dev.db",
        "secret_key": "dev-secret-key-change-in-production",
        "access_token_expire_minutes": 1440,  # 24 hours
    }
    
    # Production configuration
    PRODUCTION = {
        "host": "0.0.0.0",
        "port": int(os.getenv("PORT", 8000)),
        "debug": False,
        "reload": False,
        "workers": int(os.getenv("WORKERS", 4)),
        "log_level": "info",
        "cors_origins": [
            "https://your-domain.com",
            "https://www.your-domain.com",
            # Add your production domains
        ],
        "database_url": os.getenv("DATABASE_URL", "sqlite:///./genetic_coil_arena.db"),
        "secret_key": os.getenv("SECRET_KEY", "change-this-in-production"),
        "access_token_expire_minutes": int(os.getenv("TOKEN_EXPIRE_MINUTES", 30)),
    }
    
    # Testing configuration
    TESTING = {
        "host": "localhost",
        "port": 8001,
        "debug": True,
        "reload": False,
        "workers": 1,
        "log_level": "warning",
        "cors_origins": ["*"],
        "database_url": "sqlite:///:memory:",
        "secret_key": "test-secret-key",
        "access_token_expire_minutes": 5,
    }
    
    @classmethod
    def get_config(cls, environment: str = None) -> Dict[str, Any]:
        """Get configuration for specified environment."""
        env = environment or os.getenv("ENVIRONMENT", "development").lower()
        
        if env == "production":
            return cls.PRODUCTION
        elif env == "testing":
            return cls.TESTING
        else:
            return cls.DEVELOPMENT
    
    @classmethod
    def get_uvicorn_config(cls, environment: str = None) -> Dict[str, Any]:
        """Get uvicorn-specific configuration."""
        config = cls.get_config(environment)
        
        # Extract uvicorn-specific settings
        uvicorn_config = {
            "host": config["host"],
            "port": config["port"],
            "log_level": config["log_level"],
            "reload": config["reload"],
        }
        
        # Add workers only in production
        if environment == "production":
            uvicorn_config["workers"] = config["workers"]
        
        return uvicorn_config


# Environment variables documentation
ENV_VARS = {
    "ENVIRONMENT": "Deployment environment (development, production, testing)",
    "PORT": "Server port (default: 8000)",
    "WORKERS": "Number of worker processes for production (default: 4)",
    "DATABASE_URL": "Database connection URL",
    "SECRET_KEY": "JWT secret key for authentication",
    "TOKEN_EXPIRE_MINUTES": "JWT token expiration time in minutes",
}


def print_env_info():
    """Print current environment configuration."""
    env = os.getenv("ENVIRONMENT", "development")
    config = DeploymentConfig.get_config(env)
    
    print(f"🌍 Environment: {env}")
    print(f"🌐 Host: {config['host']}:{config['port']}")
    print(f"🔒 Database: {config['database_url']}")
    print(f"👥 Workers: {config.get('workers', 1)}")
    print(f"📊 Debug: {config['debug']}")
    print(f"🔄 Reload: {config['reload']}")


if __name__ == "__main__":
    print_env_info()
