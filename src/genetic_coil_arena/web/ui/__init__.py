"""
Web UI Module

This module contains UI components and routers:
- batch_ui.py - Batch processing UI
- team_management_ui.py - Team management interface
- templates_ui.py - Template management UI
- visualization_3d_ui.py - 3D visualization UI
- *_ui_router.py - UI-specific routers
"""

# Import UI routers
try:
    from .templates_ui import templates_ui_router
    from .batch_ui import batch_ui_router
    from .team_management_ui_router import router as team_ui_router
    from .visualization_3d_ui_router import router as viz_3d_ui_router
    
    __all__ = [
        'templates_ui_router',
        'batch_ui_router', 
        'team_ui_router',
        'viz_3d_ui_router',
    ]
except ImportError as e:
    # Handle missing UI components gracefully
    __all__ = []
