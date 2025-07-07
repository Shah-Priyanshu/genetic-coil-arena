#!/usr/bin/env python3
"""Test script to execute API file imports line by line."""

def test_api_imports():
    """Test each import in the API file."""
    
    print("Testing imports from API file...")
    
    try:
        print("1. Testing docstring...")
        # Just the docstring should work
        exec('"""Integrated FastAPI routes for the web dashboard with persistent experiment management."""')
        print("✓ Docstring OK")
        
        print("2. Testing basic imports...")
        exec("import asyncio")
        exec("import uuid") 
        exec("import logging")
        exec("import numpy as np")
        exec("from threading import Thread")
        exec("from typing import List, Dict, Any, Optional")
        exec("from datetime import datetime")
        print("✓ Basic imports OK")
        
        print("3. Testing FastAPI imports...")
        exec("from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect")
        exec("from fastapi.responses import HTMLResponse")
        print("✓ FastAPI imports OK")
        
        print("4. Testing relative imports...")
        # These are the problematic ones
        try:
            exec("from .models import ExperimentConfig, ExperimentStatus, GenerationStats, ExperimentListItem, WebSocketMessage")
            print("✓ Models import OK")
        except Exception as e:
            print(f"✗ Models import failed: {e}")
            
        try:
            exec("from ..persistence.arena import PersistentEvolutionArena, list_experiments as list_stored_experiments")
            print("✓ Arena import OK")
        except Exception as e:
            print(f"✗ Arena import failed: {e}")
            
        try:
            exec("from ..config import Config")
            print("✓ Config import OK")
        except Exception as e:
            print(f"✗ Config import failed: {e}")
            
        print("5. Testing logger and router creation...")
        exec("logger = logging.getLogger(__name__)")
        exec("router = APIRouter()")
        print("✓ Logger and router creation OK")
        
    except Exception as e:
        print(f"Error during import testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_api_imports()
