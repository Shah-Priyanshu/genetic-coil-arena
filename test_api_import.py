#!/usr/bin/env python3
"""Test script to debug API import issues."""

import sys
import traceback

def test_import_step_by_step():
    """Test importing the API module step by step."""
    try:
        print("Step 1: Importing basic modules...")
        import asyncio
        import uuid
        import logging
        import numpy as np
        from threading import Thread
        from typing import List, Dict, Any, Optional
        from datetime import datetime
        print("✓ Basic modules imported")
        
        print("Step 2: Importing FastAPI...")
        from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
        from fastapi.responses import HTMLResponse
        print("✓ FastAPI imported")
        
        print("Step 3: Importing project modules...")
        from src.genetic_coil_arena.web.models import (
            ExperimentConfig, ExperimentStatus, GenerationStats, 
            ExperimentListItem, WebSocketMessage
        )
        print("✓ Models imported")
        
        from src.genetic_coil_arena.persistence.arena import PersistentEvolutionArena, list_experiments as list_stored_experiments
        print("✓ Arena imported")
        
        from src.genetic_coil_arena.config import Config
        print("✓ Config imported")
        
        print("Step 4: Creating logger and router...")
        logger = logging.getLogger(__name__)
        router = APIRouter()
        print("✓ Logger and router created")
        
        print("Step 5: Importing entire module...")
        import src.genetic_coil_arena.web.api as api_module
        print("✓ Module imported")
        
        print("Step 6: Checking router attribute...")
        if hasattr(api_module, 'router'):
            print("✓ Router found in module")
        else:
            print("✗ Router NOT found in module")
            print(f"Available attributes: {dir(api_module)}")
        
    except Exception as e:
        print(f"✗ Error at step: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    test_import_step_by_step()
