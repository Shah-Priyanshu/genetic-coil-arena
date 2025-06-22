"""
Web interface module for genetic-coil-arena.

This module provides a FastAPI-based web dashboard for real-time monitoring,
control, and analysis of genetic algorithm experiments.
"""

try:
    from .server import WebServer
    from .api import router as api_router
    from .models import *
    
    __all__ = [
        'WebServer',
        'api_router'
    ]
except ImportError:
    # Graceful fallback if dependencies are missing
    WebServer = None
    api_router = None
    __all__ = []
