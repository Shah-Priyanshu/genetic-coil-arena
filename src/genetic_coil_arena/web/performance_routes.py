"""
Phase 4D: Performance Optimization API Routes
REST API endpoints for performance monitoring and optimization
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import Dict, Any, Optional
from datetime import datetime
import logging
import asyncio

from .performance_optimizer import performance_optimizer

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/performance", tags=["Performance Optimization"])


@router.get("/status")
async def get_performance_status():
    """Get comprehensive performance status."""
    try:
        status = performance_optimizer.get_system_status()
        return JSONResponse(content=status)
    except Exception as e:
        logger.error(f"Error getting performance status: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/metrics")
async def get_performance_metrics():
    """Get current performance metrics."""
    try:
        metrics = performance_optimizer.monitor.get_performance_summary()
        return JSONResponse(content=metrics)
    except Exception as e:
        logger.error(f"Error getting performance metrics: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/cache/stats")
async def get_cache_stats():
    """Get cache performance statistics."""
    try:
        stats = performance_optimizer.parallel_engine.cache.get_stats()
        return JSONResponse(content=stats)
    except Exception as e:
        logger.error(f"Error getting cache stats: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/cache/clear")
async def clear_cache():
    """Clear all cache entries."""
    try:
        performance_optimizer.parallel_engine.cache.clear()
        return JSONResponse(content={"message": "Cache cleared successfully"})
    except Exception as e:
        logger.error(f"Error clearing cache: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/memory")
async def get_memory_status():
    """Get current memory usage statistics."""
    try:
        memory_stats = performance_optimizer.memory_optimizer.get_memory_stats()
        return JSONResponse(content=memory_stats)
    except Exception as e:
        logger.error(f"Error getting memory status: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/memory/optimize")
async def optimize_memory():
    """Trigger memory optimization."""
    try:
        performance_optimizer.memory_optimizer.optimize_memory()
        return JSONResponse(content={"message": "Memory optimization completed"})
    except Exception as e:
        logger.error(f"Error optimizing memory: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/start")
async def start_performance_optimization(background_tasks: BackgroundTasks):
    """Start performance optimization systems."""
    try:
        if not performance_optimizer.is_running:
            background_tasks.add_task(performance_optimizer.start_optimization)
            return JSONResponse(content={"message": "Performance optimization started"})
        else:
            return JSONResponse(content={"message": "Performance optimization already running"})
    except Exception as e:
        logger.error(f"Error starting performance optimization: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/stop")
async def stop_performance_optimization():
    """Stop performance optimization systems."""
    try:
        if performance_optimizer.is_running:
            await performance_optimizer.stop_optimization()
            return JSONResponse(content={"message": "Performance optimization stopped"})
        else:
            return JSONResponse(content={"message": "Performance optimization not running"})
    except Exception as e:
        logger.error(f"Error stopping performance optimization: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/parallel/stats")
async def get_parallel_stats():
    """Get parallel processing statistics."""
    try:
        stats = performance_optimizer.parallel_engine.get_performance_stats()
        return JSONResponse(content=stats)
    except Exception as e:
        logger.error(f"Error getting parallel stats: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/benchmark")
async def run_performance_benchmark():
    """Run a quick performance benchmark."""
    try:
        import time
        import random
        
        start_time = time.time()
        
        # Simulate some computational work
        test_population = [random.random() for _ in range(1000)]
        
        def dummy_fitness(individual):
            return sum(x * x for x in [individual] * 100)
        
        # Test parallel processing
        result = await performance_optimizer.parallel_engine.evolve_population_parallel(
            test_population, dummy_fitness, generations=10
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        return JSONResponse(content={
            "benchmark_duration_seconds": round(duration, 3),
            "population_size": len(test_population),
            "individuals_per_second": round(len(test_population) / duration, 2),
            "cache_stats": performance_optimizer.parallel_engine.cache.get_stats(),
            "status": "completed"
        })
    except Exception as e:
        logger.error(f"Error running benchmark: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/health")
async def performance_health_check():
    """Health check for performance optimization system."""
    try:
        import psutil
        
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        
        status = "healthy"
        if cpu_percent > 90:
            status = "warning_high_cpu"
        elif memory.percent > 90:
            status = "warning_high_memory"
        
        return JSONResponse(content={
            "status": status,
            "service": "Performance Optimization",
            "cpu_usage_percent": cpu_percent,
            "memory_usage_percent": memory.percent,
            "optimization_running": performance_optimizer.is_running,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Performance health check error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/memory/stats")
async def get_memory_statistics():
    """Get detailed memory usage statistics."""
    try:
        memory_stats = performance_optimizer.memory_optimizer.get_memory_stats()
        return JSONResponse(content=memory_stats)
    except Exception as e:
        logger.error(f"Error getting memory statistics: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/dashboard")
async def get_performance_dashboard():
    """Get comprehensive performance dashboard data."""
    try:
        # Get system status
        system_status = performance_optimizer.get_system_status()
        
        # Get performance metrics
        metrics = performance_optimizer.monitor.get_performance_summary()
        
        # Get memory stats
        memory_stats = performance_optimizer.memory_optimizer.get_memory_stats()
        
        # Get cache stats
        cache_stats = performance_optimizer.parallel_engine.cache.get_stats()
        
        # Get parallel stats
        parallel_stats = performance_optimizer.parallel_engine.get_performance_stats()
        
        dashboard_data = {
            "system_performance": {
                "status": system_status,
                "metrics": metrics
            },
            "memory_usage": memory_stats,
            "cache_performance": cache_stats,
            "parallel_processing": parallel_stats,
            "optimization_active": performance_optimizer.is_running,
            "timestamp": datetime.now().isoformat()
        }
        
        return JSONResponse(content=dashboard_data)
    except Exception as e:
        logger.error(f"Error generating performance dashboard: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
