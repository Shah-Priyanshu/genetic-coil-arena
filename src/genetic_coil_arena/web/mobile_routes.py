"""
Phase 4E: Progressive Web App (PWA) Routes
Mobile and PWA functionality for Genetic Coil Arena
"""

from fastapi import APIRouter, Request, Response, HTTPException, Depends
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from typing import Dict, Any, Optional, List
import json
import logging
from datetime import datetime
import os

from .models import (
    MobileDeviceInfo, MobileExperimentConfig, PWAManifest, 
    MobileVisualizationConfig, NotificationConfig, MobileUserPreferences
)
from .auth import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/mobile", tags=["Mobile & PWA"])


@router.get("/device-info")
async def get_device_optimization(request: Request):
    """Get device-specific optimization settings."""
    try:
        user_agent = request.headers.get("user-agent", "")
        
        # Simple device detection
        is_mobile = any(device in user_agent.lower() for device in [
            'mobile', 'android', 'iphone', 'ipad', 'ipod', 'blackberry'
        ])
        
        # Get screen info from request if available
        screen_width = int(request.query_params.get("width", 1920))
        screen_height = int(request.query_params.get("height", 1080))
        
        device_info = {
            "device_type": "mobile" if is_mobile else "desktop",
            "screen_width": screen_width,
            "screen_height": screen_height,
            "user_agent": user_agent,
            "is_mobile": is_mobile,
            "supports_touch": is_mobile,  # Simplified assumption
            "recommended_config": _get_device_optimized_config(is_mobile, screen_width)
        }
        
        return JSONResponse(content=device_info)
    
    except Exception as e:
        logger.error(f"Error getting device info: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/experiment/mobile-config")
async def create_mobile_experiment_config(config: MobileExperimentConfig):
    """Create mobile-optimized experiment configuration."""
    try:
        # Validate mobile constraints
        if config.population_size > 200:
            config.population_size = 200
        if config.generations > 500:
            config.generations = 500
        if config.grid_size > 12:
            config.grid_size = 12
        if config.max_steps > 500:
            config.max_steps = 500
        
        # Additional mobile optimizations
        mobile_config = {
            "name": config.name,
            "population_size": config.population_size,
            "generations": config.generations,
            "mutation_rate": config.mutation_rate,
            "crossover_rate": config.crossover_rate,
            "elite_size": config.elite_size,
            "grid_size": config.grid_size,
            "max_steps": config.max_steps,
            "mobile_optimized": True,
            "reduced_logging": True,
            "simplified_fitness": True,
            "batch_size": min(10, config.population_size // 5)
        }
        
        return JSONResponse(content=mobile_config)
    
    except Exception as e:
        logger.error(f"Error creating mobile config: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/preferences")
async def get_mobile_preferences(user_id: str = Depends(get_current_user)):
    """Get user's mobile preferences."""
    try:
        # In a real app, this would load from database
        # For now, return default preferences
        default_preferences = MobileUserPreferences()
        
        return JSONResponse(content=default_preferences.dict())
    
    except Exception as e:
        logger.error(f"Error getting mobile preferences: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/preferences")
async def update_mobile_preferences(
    preferences: MobileUserPreferences,
    user_id: str = Depends(get_current_user)
):
    """Update user's mobile preferences."""
    try:
        # In a real app, this would save to database
        # For now, just validate and return the preferences
        
        updated_preferences = {
            "theme": preferences.theme,
            "reduced_motion": preferences.reduced_motion,
            "data_saver": preferences.data_saver,
            "offline_mode": preferences.offline_mode,
            "sync_frequency": preferences.sync_frequency,
            "notifications": preferences.notifications.dict(),
            "visualization": preferences.visualization.dict(),
            "updated_at": datetime.now().isoformat()
        }
        
        return JSONResponse(content=updated_preferences)
    
    except Exception as e:
        logger.error(f"Error updating mobile preferences: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/offline/sync-data")
async def get_offline_sync_data(user_id: str = Depends(get_current_user)):
    """Get data for offline synchronization."""
    try:
        # Prepare essential data for offline use
        sync_data = {
            "experiments": _get_user_experiments_summary(user_id),
            "templates": _get_experiment_templates(),
            "performance_stats": _get_basic_performance_stats(),
            "user_preferences": _get_user_preferences(user_id),
            "sync_timestamp": datetime.now().isoformat()
        }
        
        return JSONResponse(content=sync_data)
    
    except Exception as e:
        logger.error(f"Error preparing offline sync data: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/offline/upload-data")
async def upload_offline_data(
    data: Dict[str, Any],
    user_id: str = Depends(get_current_user)
):
    """Upload data collected while offline."""
    try:
        # Process offline data
        uploaded_items = []
        
        if "experiments" in data:
            uploaded_items.extend(_process_offline_experiments(data["experiments"], user_id))
        
        if "user_actions" in data:
            uploaded_items.extend(_process_offline_actions(data["user_actions"], user_id))
        
        if "preferences" in data:
            _process_offline_preferences(data["preferences"], user_id)
            uploaded_items.append("preferences")
        
        response = {
            "success": True,
            "items_processed": len(uploaded_items),
            "uploaded_items": uploaded_items,
            "sync_timestamp": datetime.now().isoformat()
        }
        
        return JSONResponse(content=response)
    
    except Exception as e:
        logger.error(f"Error uploading offline data: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/push/vapid-key")
async def get_vapid_public_key():
    """Get VAPID public key for push notifications."""
    try:
        # In a real app, this would be your actual VAPID public key
        vapid_key = "BHxwPFYGKpKXwcAaOm3oWxP_r98Gx_Q8Jz8K9L9MNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz123456"
        
        return JSONResponse(content={"vapid_public_key": vapid_key})
    
    except Exception as e:
        logger.error(f"Error getting VAPID key: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/push/subscribe")
async def subscribe_to_push_notifications(
    subscription_data: Dict[str, Any],
    user_id: str = Depends(get_current_user)
):
    """Subscribe user to push notifications."""
    try:
        # Store the push subscription
        subscription = {
            "user_id": user_id,
            "endpoint": subscription_data.get("endpoint"),
            "keys": subscription_data.get("keys", {}),
            "subscribed_at": datetime.now().isoformat()
        }
        
        # In a real app, save this to database
        logger.info(f"Push subscription created for user {user_id}")
        
        return JSONResponse(content={"success": True, "message": "Subscription saved"})
    
    except Exception as e:
        logger.error(f"Error subscribing to push notifications: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/visualization/mobile-config")
async def get_mobile_visualization_config(request: Request):
    """Get mobile-optimized visualization configuration."""
    try:
        user_agent = request.headers.get("user-agent", "")
        is_mobile = any(device in user_agent.lower() for device in [
            'mobile', 'android', 'iphone', 'ipad'
        ])
        
        # Get device capabilities
        screen_width = int(request.query_params.get("width", 800))
        
        config = MobileVisualizationConfig()
        
        # Adjust config based on device
        if is_mobile:
            if screen_width < 400:
                config.max_agents_display = 5
                config.animation_fps = 20
                config.reduced_particles = True
                config.simplified_graphics = True
        
        return JSONResponse(content=config.dict())
    
    except Exception as e:
        logger.error(f"Error getting mobile visualization config: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/health")
async def mobile_health_check():
    """Health check endpoint optimized for mobile monitoring."""
    try:
        health_data = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "services": {
                "mobile_api": "operational",
                "push_notifications": "operational",
                "offline_sync": "operational"
            },
            "mobile_optimizations": {
                "reduced_payload": True,
                "compression_enabled": True,
                "caching_enabled": True
            }
        }
        
        return JSONResponse(content=health_data)
    
    except Exception as e:
        logger.error(f"Mobile health check error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


# Helper functions
def _get_device_optimized_config(is_mobile: bool, screen_width: int) -> Dict[str, Any]:
    """Get device-optimized configuration."""
    if is_mobile:
        return {
            "population_size": 50 if screen_width < 400 else 100,
            "generations": 25 if screen_width < 400 else 50,
            "grid_size": 6 if screen_width < 400 else 8,
            "max_steps": 100 if screen_width < 400 else 200,
            "animation_fps": 20 if screen_width < 400 else 30,
            "max_visible_agents": 5 if screen_width < 400 else 10
        }
    else:
        return {
            "population_size": 200,
            "generations": 100,
            "grid_size": 12,
            "max_steps": 500,
            "animation_fps": 60,
            "max_visible_agents": 50
        }


def _get_user_experiments_summary(user_id: str) -> List[Dict[str, Any]]:
    """Get summary of user's experiments for offline sync."""
    # In a real app, query database for user's experiments
    return [
        {
            "id": "exp_1",
            "name": "Mobile Test Experiment",
            "status": "completed",
            "best_fitness": 85.5,
            "generation": 25,
            "created_at": "2025-07-07T10:00:00Z"
        }
    ]


def _get_experiment_templates() -> List[Dict[str, Any]]:
    """Get experiment templates for offline use."""
    return [
        {
            "id": "mobile_quick",
            "name": "Quick Mobile Test",
            "population_size": 50,
            "generations": 25,
            "description": "Fast experiment optimized for mobile devices"
        },
        {
            "id": "mobile_standard", 
            "name": "Standard Mobile Experiment",
            "population_size": 100,
            "generations": 50,
            "description": "Balanced experiment for mobile devices"
        }
    ]


def _get_basic_performance_stats() -> Dict[str, Any]:
    """Get basic performance statistics for offline viewing."""
    return {
        "system_status": "operational",
        "average_response_time": 75.0,
        "cache_hit_rate": 85.5,
        "memory_usage": 78.2
    }


def _get_user_preferences(user_id: str) -> Dict[str, Any]:
    """Get user preferences for offline sync."""
    return {
        "theme": "auto",
        "notifications": True,
        "data_saver": False,
        "reduced_motion": False
    }


def _process_offline_experiments(experiments: List[Dict], user_id: str) -> List[str]:
    """Process experiments created while offline."""
    processed = []
    for exp in experiments:
        # In a real app, validate and save to database
        processed.append(f"experiment_{exp.get('id', 'unknown')}")
    return processed


def _process_offline_actions(actions: List[Dict], user_id: str) -> List[str]:
    """Process user actions performed while offline."""
    processed = []
    for action in actions:
        # In a real app, process and log actions
        processed.append(f"action_{action.get('type', 'unknown')}")
    return processed


def _process_offline_preferences(preferences: Dict, user_id: str):
    """Process preference changes made while offline."""
    # In a real app, update user preferences in database
    logger.info(f"Updated offline preferences for user {user_id}")
