#!/usr/bin/env python3
"""
Simple server start script for Genetic Coil Arena
"""

import sys
import os
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

def main():
    """Start the Genetic Coil Arena server."""
    try:
        from genetic_coil_arena.web.server import main as server_main
        print("🚀 Starting Genetic Coil Arena...")
        server_main()
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure all dependencies are installed:")
        print("pip install -e .")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
