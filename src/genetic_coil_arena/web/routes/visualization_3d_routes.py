"""
Phase 4B: 3D Visualization REST API Routes
FastAPI routes for 3D visualization data and streaming.
"""

from fastapi import APIRouter, HTTPException, Depends, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from typing import List, Dict, Any, Optional
import json
import asyncio
import logging
from datetime import datetime

from ..core.auth import get_current_user
from .database import get_db_connection
from .visualization_3d import (
    Visualization3DEngine, 
    FitnessLandscape3D, 
    PopulationCloud3D,
    Point3D
)
from .models import User

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/visualization3d", tags=["3D Visualization"])

# Initialize 3D visualization engine
viz_engine = Visualization3DEngine()

# WebSocket connection manager for real-time streaming
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.experiment_subscribers: Dict[str, List[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, experiment_id: Optional[str] = None):
        await websocket.accept()
        self.active_connections.append(websocket)
        if experiment_id:
            if experiment_id not in self.experiment_subscribers:
                self.experiment_subscribers[experiment_id] = []
            self.experiment_subscribers[experiment_id].append(websocket)

    def disconnect(self, websocket: WebSocket, experiment_id: Optional[str] = None):
        self.active_connections.remove(websocket)
        if experiment_id and experiment_id in self.experiment_subscribers:
            if websocket in self.experiment_subscribers[experiment_id]:
                self.experiment_subscribers[experiment_id].remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast_to_experiment(self, message: str, experiment_id: str):
        if experiment_id in self.experiment_subscribers:
            for connection in self.experiment_subscribers[experiment_id]:
                try:
                    await connection.send_text(message)
                except:
                    # Remove dead connections
                    self.disconnect(connection, experiment_id)

manager = ConnectionManager()


@router.get("/experiments/{experiment_id}/fitness-landscape")
async def get_fitness_landscape_3d(
    experiment_id: str,
    generation: Optional[int] = None,
    resolution: int = 50,
    current_user: User = Depends(get_current_user)
):
    """Get 3D fitness landscape data for an experiment."""
    try:
        # Get experiment data from database
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Verify experiment exists and user has access
            cursor.execute("""
                SELECT id, name, status, best_fitness 
                FROM experiments 
                WHERE id = ? AND user_id = ?
            """, (experiment_id, current_user.id))
            
            experiment = cursor.fetchone()
            if not experiment:
                raise HTTPException(status_code=404, detail="Experiment not found")
            
            # Get generation data
            if generation is None:
                cursor.execute("""
                    SELECT MAX(generation) FROM experiment_results 
                    WHERE experiment_id = ?
                """, (experiment_id,))
                generation = cursor.fetchone()[0] or 0
            
            cursor.execute("""
                SELECT generation, individual_data, fitness 
                FROM experiment_results 
                WHERE experiment_id = ? AND generation = ?
                ORDER BY fitness DESC
            """, (experiment_id, generation))
            
            results = cursor.fetchall()
            
            if not results:
                raise HTTPException(status_code=404, detail="No data found for generation")
            
            # Generate 3D fitness landscape
            landscape = viz_engine.create_fitness_landscape_3d(
                experiment_id=experiment_id,
                generation=generation,
                data=results,
                resolution=resolution
            )
            
            # Convert to Three.js format
            threejs_data = viz_engine.export_to_threejs(landscape)
            
            return JSONResponse(content={
                "experiment_id": experiment_id,
                "generation": generation,
                "landscape": threejs_data,
                "metadata": {
                    "resolution": resolution,
                    "timestamp": datetime.now().isoformat(),
                    "bounds": landscape.bounds
                }
            })
            
    except Exception as e:
        logger.error(f"Error generating 3D fitness landscape: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/experiments/{experiment_id}/population-cloud")
async def get_population_cloud_3d(
    experiment_id: str,
    generation: Optional[int] = None,
    show_elite: bool = True,
    current_user: User = Depends(get_current_user)
):
    """Get 3D population cloud visualization."""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Verify experiment exists and user has access
            cursor.execute("""
                SELECT id, name, status 
                FROM experiments 
                WHERE id = ? AND user_id = ?
            """, (experiment_id, current_user.id))
            
            experiment = cursor.fetchone()
            if not experiment:
                raise HTTPException(status_code=404, detail="Experiment not found")
            
            # Get generation data
            if generation is None:
                cursor.execute("""
                    SELECT MAX(generation) FROM experiment_results 
                    WHERE experiment_id = ?
                """, (experiment_id,))
                generation = cursor.fetchone()[0] or 0
            
            cursor.execute("""
                SELECT individual_data, fitness 
                FROM experiment_results 
                WHERE experiment_id = ? AND generation = ?
                ORDER BY fitness DESC
            """, (experiment_id, generation))
            
            results = cursor.fetchall()
            
            if not results:
                raise HTTPException(status_code=404, detail="No population data found")
            
            # Generate 3D population cloud
            cloud = viz_engine.create_population_cloud_3d(
                experiment_id=experiment_id,
                generation=generation,
                population_data=results,
                show_elite=show_elite
            )
            
            # Convert to Three.js format
            threejs_data = viz_engine.export_population_to_threejs(cloud)
            
            return JSONResponse(content={
                "experiment_id": experiment_id,
                "generation": generation,
                "population": threejs_data,
                "metadata": {
                    "elite_count": len(cloud.elite_indices),
                    "diversity_score": cloud.diversity_score,
                    "best_fitness": cloud.best_fitness,
                    "timestamp": datetime.now().isoformat()
                }
            })
            
    except Exception as e:
        logger.error(f"Error generating 3D population cloud: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/experiments/{experiment_id}/evolution-trajectory")
async def get_evolution_trajectory_3d(
    experiment_id: str,
    max_generations: int = 100,
    current_user: User = Depends(get_current_user)
):
    """Get 3D evolution trajectory showing fitness progression over generations."""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Verify experiment exists
            cursor.execute("""
                SELECT id, name FROM experiments 
                WHERE id = ? AND user_id = ?
            """, (experiment_id, current_user.id))
            
            experiment = cursor.fetchone()
            if not experiment:
                raise HTTPException(status_code=404, detail="Experiment not found")
            
            # Get evolution data
            cursor.execute("""
                SELECT generation, AVG(fitness) as avg_fitness, 
                       MAX(fitness) as best_fitness, MIN(fitness) as worst_fitness
                FROM experiment_results 
                WHERE experiment_id = ?
                GROUP BY generation
                ORDER BY generation
                LIMIT ?
            """, (experiment_id, max_generations))
            
            trajectory_data = cursor.fetchall()
            
            if not trajectory_data:
                raise HTTPException(status_code=404, detail="No trajectory data found")
            
            # Generate 3D trajectory
            trajectory = viz_engine.create_evolution_trajectory_3d(
                experiment_id=experiment_id,
                trajectory_data=trajectory_data
            )
            
            return JSONResponse(content={
                "experiment_id": experiment_id,
                "trajectory": trajectory,
                "metadata": {
                    "generations": len(trajectory_data),
                    "timestamp": datetime.now().isoformat()
                }
            })
            
    except Exception as e:
        logger.error(f"Error generating evolution trajectory: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.websocket("/stream/{experiment_id}")
async def websocket_stream_3d(websocket: WebSocket, experiment_id: str):
    """WebSocket endpoint for real-time 3D visualization streaming."""
    await manager.connect(websocket, experiment_id)
    try:
        while True:
            # Listen for client messages (configuration changes, etc.)
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message.get("type") == "subscribe_fitness_landscape":
                # Stream fitness landscape updates
                await stream_fitness_landscape(websocket, experiment_id, message.get("config", {}))
            elif message.get("type") == "subscribe_population":
                # Stream population updates
                await stream_population_cloud(websocket, experiment_id, message.get("config", {}))
            elif message.get("type") == "ping":
                await websocket.send_text(json.dumps({"type": "pong"}))
                
    except WebSocketDisconnect:
        manager.disconnect(websocket, experiment_id)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket, experiment_id)


async def stream_fitness_landscape(websocket: WebSocket, experiment_id: str, config: Dict[str, Any]):
    """Stream real-time fitness landscape updates."""
    try:
        resolution = config.get("resolution", 30)
        update_interval = config.get("interval", 5.0)  # seconds
        
        while True:
            try:
                # Get latest generation data
                with get_db_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        SELECT MAX(generation) FROM experiment_results 
                        WHERE experiment_id = ?
                    """, (experiment_id,))
                    
                    latest_generation = cursor.fetchone()[0]
                    if latest_generation is not None:
                        cursor.execute("""
                            SELECT generation, individual_data, fitness 
                            FROM experiment_results 
                            WHERE experiment_id = ? AND generation = ?
                        """, (experiment_id, latest_generation))
                        
                        results = cursor.fetchall()
                        if results:
                            # Generate landscape
                            landscape = viz_engine.create_fitness_landscape_3d(
                                experiment_id=experiment_id,
                                generation=latest_generation,
                                data=results,
                                resolution=resolution
                            )
                            
                            # Send update
                            update = {
                                "type": "fitness_landscape_update",
                                "experiment_id": experiment_id,
                                "generation": latest_generation,
                                "data": viz_engine.export_to_threejs(landscape),
                                "timestamp": datetime.now().isoformat()
                            }
                            
                            await websocket.send_text(json.dumps(update))
                
                await asyncio.sleep(update_interval)
                
            except Exception as e:
                logger.error(f"Error in fitness landscape streaming: {e}")
                break
                
    except Exception as e:
        logger.error(f"Stream fitness landscape error: {e}")


async def stream_population_cloud(websocket: WebSocket, experiment_id: str, config: Dict[str, Any]):
    """Stream real-time population cloud updates."""
    try:
        show_elite = config.get("show_elite", True)
        update_interval = config.get("interval", 3.0)  # seconds
        
        while True:
            try:
                # Get latest population data
                with get_db_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        SELECT MAX(generation) FROM experiment_results 
                        WHERE experiment_id = ?
                    """, (experiment_id,))
                    
                    latest_generation = cursor.fetchone()[0]
                    if latest_generation is not None:
                        cursor.execute("""
                            SELECT individual_data, fitness 
                            FROM experiment_results 
                            WHERE experiment_id = ? AND generation = ?
                        """, (experiment_id, latest_generation))
                        
                        results = cursor.fetchall()
                        if results:
                            # Generate population cloud
                            cloud = viz_engine.create_population_cloud_3d(
                                experiment_id=experiment_id,
                                generation=latest_generation,
                                population_data=results,
                                show_elite=show_elite
                            )
                            
                            # Send update
                            update = {
                                "type": "population_cloud_update",
                                "experiment_id": experiment_id,
                                "generation": latest_generation,
                                "data": viz_engine.export_population_to_threejs(cloud),
                                "metadata": {
                                    "elite_count": len(cloud.elite_indices),
                                    "diversity_score": cloud.diversity_score,
                                    "best_fitness": cloud.best_fitness
                                },
                                "timestamp": datetime.now().isoformat()
                            }
                            
                            await websocket.send_text(json.dumps(update))
                
                await asyncio.sleep(update_interval)
                
            except Exception as e:
                logger.error(f"Error in population streaming: {e}")
                break
                
    except Exception as e:
        logger.error(f"Stream population cloud error: {e}")


@router.post("/experiments/{experiment_id}/generate-animation")
async def generate_evolution_animation(
    experiment_id: str,
    animation_config: Dict[str, Any],
    current_user: User = Depends(get_current_user)
):
    """Generate evolution animation data for 3D visualization."""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Verify experiment exists
            cursor.execute("""
                SELECT id, name FROM experiments 
                WHERE id = ? AND user_id = ?
            """, (experiment_id, current_user.id))
            
            experiment = cursor.fetchone()
            if not experiment:
                raise HTTPException(status_code=404, detail="Experiment not found")
            
            # Get animation configuration
            start_generation = animation_config.get("start_generation", 0)
            end_generation = animation_config.get("end_generation", 100)
            frame_interval = animation_config.get("frame_interval", 5)
            animation_type = animation_config.get("type", "fitness_landscape")
            
            # Generate animation frames
            frames = []
            for generation in range(start_generation, end_generation + 1, frame_interval):
                cursor.execute("""
                    SELECT generation, individual_data, fitness 
                    FROM experiment_results 
                    WHERE experiment_id = ? AND generation = ?
                """, (experiment_id, generation))
                
                results = cursor.fetchall()
                if results:
                    if animation_type == "fitness_landscape":
                        landscape = viz_engine.create_fitness_landscape_3d(
                            experiment_id=experiment_id,
                            generation=generation,
                            data=results,
                            resolution=animation_config.get("resolution", 30)
                        )
                        frame_data = viz_engine.export_to_threejs(landscape)
                    elif animation_type == "population_cloud":
                        cloud = viz_engine.create_population_cloud_3d(
                            experiment_id=experiment_id,
                            generation=generation,
                            population_data=results,
                            show_elite=animation_config.get("show_elite", True)
                        )
                        frame_data = viz_engine.export_population_to_threejs(cloud)
                    
                    frames.append({
                        "generation": generation,
                        "data": frame_data,
                        "timestamp": datetime.now().isoformat()
                    })
            
            animation_data = {
                "experiment_id": experiment_id,
                "animation_type": animation_type,
                "frames": frames,
                "config": animation_config,
                "metadata": {
                    "total_frames": len(frames),
                    "duration_estimate": len(frames) * 0.1,  # 100ms per frame
                    "created_at": datetime.now().isoformat()
                }
            }
            
            return JSONResponse(content=animation_data)
            
    except Exception as e:
        logger.error(f"Error generating evolution animation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Health check endpoint for 3D visualization service
@router.get("/health")
async def health_check():
    """Health check for 3D visualization service."""
    return {
        "status": "healthy",
        "service": "3D Visualization API",
        "timestamp": datetime.now().isoformat(),
        "active_connections": len(manager.active_connections),
        "experiment_streams": len(manager.experiment_subscribers)
    }
