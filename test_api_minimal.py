"""Test version of api.py with debug prints"""

print("DEBUG: Starting api.py import...")

# Test basic imports first
try:
    import asyncio
    import uuid
    import logging
    print("DEBUG: Basic imports OK")
except Exception as e:
    print(f"DEBUG: Basic imports failed: {e}")
    exit(1)

try:
    from fastapi import APIRouter
    print("DEBUG: FastAPI imports OK")
except Exception as e:
    print(f"DEBUG: FastAPI imports failed: {e}")
    exit(1)

try:
    from genetic_coil_arena.web.models import ExperimentConfig
    print("DEBUG: Models imports OK")
except Exception as e:
    print(f"DEBUG: Models imports failed: {e}")
    exit(1)

try:
    from genetic_coil_arena.persistence.arena import PersistentEvolutionArena
    print("DEBUG: Persistence imports OK")
except Exception as e:
    print(f"DEBUG: Persistence imports failed: {e}")
    exit(1)

# Create router
try:
    router = APIRouter()
    print(f"DEBUG: Router created: {router}")
except Exception as e:
    print(f"DEBUG: Router creation failed: {e}")
    exit(1)

print("DEBUG: Test completed successfully!")
