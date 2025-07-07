#!/usr/bin/env python3
"""
Production deployment script for Genetic Coil Arena.
This script sets up and runs the complete production server.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path


def install_dependencies():
    """Install all required dependencies."""
    print("📦 Installing dependencies...")
    
    dependencies = [
        "fastapi",
        "uvicorn[standard]",
        "websockets",
        "numpy",
        "python-multipart",
        "passlib[bcrypt]",
        "python-jose[cryptography]",
        "aiosqlite",  # For async SQLite support
    ]
    
    for dep in dependencies:
        print(f"   Installing {dep}...")
        subprocess.run([sys.executable, "-m", "pip", "install", dep], check=True)
    
    print("✅ Dependencies installed successfully")


def setup_environment():
    """Set up environment variables for production."""
    print("🌍 Setting up environment...")
    
    # Set environment variables
    env_vars = {
        "ENVIRONMENT": "production",
        "SECRET_KEY": "your-super-secret-key-change-this-in-production",
        "DATABASE_URL": "sqlite:///./genetic_coil_arena_production.db",
        "TOKEN_EXPIRE_MINUTES": "30"
    }
    
    for key, value in env_vars.items():
        if key not in os.environ:
            os.environ[key] = value
            print(f"   Set {key}={value}")
    
    print("✅ Environment configured")


def run_production_server(host="0.0.0.0", port=8000, workers=1):
    """Run the production server."""
    print(f"🚀 Starting Genetic Coil Arena production server...")
    print(f"   Host: {host}")
    print(f"   Port: {port}")
    print(f"   Workers: {workers}")
    
    try:
        from src.genetic_coil_arena.web.server_production import run_production_server
        run_production_server(host=host, port=port)
    except ImportError:
        print("❌ Could not import production server. Please check installation.")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Server error: {e}")
        sys.exit(1)


def main():
    """Main deployment function."""
    parser = argparse.ArgumentParser(description="Deploy Genetic Coil Arena")
    parser.add_argument("--host", default="localhost", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--workers", type=int, default=1, help="Number of workers")
    parser.add_argument("--install", action="store_true", help="Install dependencies first")
    parser.add_argument("--setup-env", action="store_true", help="Setup environment variables")
    
    args = parser.parse_args()
    
    print("🧬 Genetic Coil Arena - Production Deployment")
    print("=" * 50)
    
    if args.install:
        install_dependencies()
    
    if args.setup_env:
        setup_environment()
    
    # Run the production server
    run_production_server(host=args.host, port=args.port, workers=args.workers)


if __name__ == "__main__":
    main()
