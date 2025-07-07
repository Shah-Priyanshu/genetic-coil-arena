#!/usr/bin/env python3
"""Debug script to test API module import."""

import sys
import traceback

print("=== DEBUG: Testing API module import ===")

try:
    print("1. Importing basic dependencies...")
    import asyncio
    import uuid
    import logging
    import numpy as np
    from threading import Thread
    from typing import List, Dict, Any, Optional
    from datetime import datetime
    print("   ✓ Basic imports successful")
    
    print("2. Importing FastAPI components...")
    from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
    from fastapi.responses import HTMLResponse
    print("   ✓ FastAPI imports successful")
    
    print("3. Importing project models...")
    from genetic_coil_arena.web.models import (
        ExperimentConfig, ExperimentStatus, GenerationStats, 
        ExperimentListItem, WebSocketMessage
    )
    print("   ✓ Models imports successful")
    
    print("4. Importing persistence components...")
    from genetic_coil_arena.persistence.arena import PersistentEvolutionArena, list_experiments as list_stored_experiments
    from genetic_coil_arena.config import Config
    print("   ✓ Persistence imports successful")
    
    print("5. Creating router...")
    router = APIRouter()
    print(f"   ✓ Router created: {router}")
    
    print("6. Testing full API module import...")
    import genetic_coil_arena.web.api as api_module
    print(f"   ✓ API module imported")
    print(f"   Available attributes: {[attr for attr in dir(api_module) if not attr.startswith('_')]}")
    
    if hasattr(api_module, 'router'):
        print(f"   ✓ Router found in module: {api_module.router}")
    else:
        print("   ✗ Router NOT found in module")
        
except Exception as e:
    print(f"✗ Error during import: {e}")
    traceback.print_exc()

print("\n=== DEBUG COMPLETE ===")
