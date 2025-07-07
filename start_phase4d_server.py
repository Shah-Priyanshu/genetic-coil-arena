#!/usr/bin/env python3
"""
Start the Phase 4D-enabled production server
"""

from src.genetic_coil_arena.web.server_production import ProductionServer

if __name__ == "__main__":
    server = ProductionServer(port=8001)
    print("🚀 Starting Genetic Coil Arena with Phase 4D Performance & Scalability...")
    server.run()
