"""
Phase 4D: Performance & Scalability Optimization
High-performance computing enhancements for genetic algorithms
"""

import asyncio
import time
import psutil
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
from datetime import datetime
import logging
import json
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics for monitoring system performance."""
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    active_experiments: int
    generation_rate: float  # generations per second
    population_throughput: float  # individuals processed per second
    cache_hit_rate: float
    response_time_ms: float


class PerformanceMonitor:
    """Real-time performance monitoring and optimization."""
    
    def __init__(self):
        self.metrics_history: List[PerformanceMetrics] = []
        self.monitoring_active = False
        self.optimization_enabled = True
        
    def start_monitoring(self, interval: float = 1.0):
        """Start continuous performance monitoring."""
        self.monitoring_active = True
        
        def monitor_loop():
            while self.monitoring_active:
                try:
                    metrics = self._collect_metrics()
                    self.metrics_history.append(metrics)
                    
                    # Keep only last 1000 metrics
                    if len(self.metrics_history) > 1000:
                        self.metrics_history = self.metrics_history[-1000:]
                    
                    # Trigger optimization if needed
                    if self.optimization_enabled:
                        self._check_optimization_triggers(metrics)
                    
                    time.sleep(interval)
                except Exception as e:
                    logger.error(f"Performance monitoring error: {e}")
                    time.sleep(interval)
        
        thread = threading.Thread(target=monitor_loop, daemon=True)
        thread.start()
        logger.info("Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop performance monitoring."""
        self.monitoring_active = False
        logger.info("Performance monitoring stopped")
    
    def _collect_metrics(self) -> PerformanceMetrics:
        """Collect current system metrics."""
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            return PerformanceMetrics(
                timestamp=datetime.now(),
                cpu_usage=cpu_percent,
                memory_usage=memory.percent,
                disk_usage=disk.percent,
                active_experiments=0,  # Will be updated by experiment manager
                generation_rate=0.0,
                population_throughput=0.0,
                cache_hit_rate=0.0,
                response_time_ms=0.0
            )
        except Exception as e:
            logger.error(f"Error collecting metrics: {e}")
            return PerformanceMetrics(
                timestamp=datetime.now(),
                cpu_usage=0.0, memory_usage=0.0, disk_usage=0.0,
                active_experiments=0, generation_rate=0.0,
                population_throughput=0.0, cache_hit_rate=0.0,
                response_time_ms=0.0
            )
    
    def _check_optimization_triggers(self, metrics: PerformanceMetrics):
        """Check if optimization is needed based on metrics."""
        if metrics.cpu_usage > 80:
            logger.warning(f"High CPU usage detected: {metrics.cpu_usage}%")
        
        if metrics.memory_usage > 85:
            logger.warning(f"High memory usage detected: {metrics.memory_usage}%")
        
        if metrics.response_time_ms > 1000:
            logger.warning(f"High response time detected: {metrics.response_time_ms}ms")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary statistics."""
        if not self.metrics_history:
            return {"status": "no_data"}
        
        recent_metrics = self.metrics_history[-10:]  # Last 10 measurements
        
        avg_cpu = sum(m.cpu_usage for m in recent_metrics) / len(recent_metrics)
        avg_memory = sum(m.memory_usage for m in recent_metrics) / len(recent_metrics)
        avg_response = sum(m.response_time_ms for m in recent_metrics) / len(recent_metrics)
        
        return {
            "status": "active",
            "measurements": len(self.metrics_history),
            "avg_cpu_usage": round(avg_cpu, 2),
            "avg_memory_usage": round(avg_memory, 2),
            "avg_response_time_ms": round(avg_response, 2),
            "last_update": self.metrics_history[-1].timestamp.isoformat()
        }


class CacheManager:
    """Advanced caching system for experiment data and results."""
    
    def __init__(self, max_size: int = 1000):
        self.cache: Dict[str, Any] = {}
        self.access_times: Dict[str, float] = {}
        self.hit_count = 0
        self.miss_count = 0
        self.max_size = max_size
        self._lock = threading.Lock()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self._lock:
            if key in self.cache:
                self.access_times[key] = time.time()
                self.hit_count += 1
                return self.cache[key]
            else:
                self.miss_count += 1
                return None
    
    def set(self, key: str, value: Any):
        """Set value in cache."""
        with self._lock:
            if len(self.cache) >= self.max_size:
                self._evict_lru()
            
            self.cache[key] = value
            self.access_times[key] = time.time()
    
    def _evict_lru(self):
        """Evict least recently used item."""
        if not self.access_times:
            return
        
        lru_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
        del self.cache[lru_key]
        del self.access_times[lru_key]
    
    def clear(self):
        """Clear all cache entries."""
        with self._lock:
            self.cache.clear()
            self.access_times.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.hit_count + self.miss_count
        hit_rate = (self.hit_count / total_requests * 100) if total_requests > 0 else 0
        
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "hit_rate_percent": round(hit_rate, 2),
            "memory_usage_estimate_mb": len(str(self.cache)) / 1024 / 1024
        }


class ParallelEvolutionEngine:
    """High-performance parallel evolution processing."""
    
    def __init__(self, max_workers: Optional[int] = None):
        self.max_workers = max_workers or min(32, (psutil.cpu_count() or 1) + 4)
        self.thread_pool = ThreadPoolExecutor(max_workers=self.max_workers)
        self.process_pool = ProcessPoolExecutor(max_workers=psutil.cpu_count())
        self.cache = CacheManager()
        self.total_tasks_processed = 0
        self._lock = threading.Lock()
        
    async def evolve_population_parallel(self, population: List[Any], 
                                       fitness_function: Callable,
                                       generations: int = 100) -> List[Any]:
        """Evolve population using parallel processing."""
        try:
            # Check cache first
            cache_key = f"pop_{hash(str(population))}_{generations}"
            cached_result = self.cache.get(cache_key)
            if cached_result:
                return cached_result
            
            # Parallel fitness evaluation
            fitness_futures = []
            for individual in population:
                future = self.thread_pool.submit(fitness_function, individual)
                fitness_futures.append(future)
            
            # Collect fitness values
            fitness_values = []
            for future in fitness_futures:
                try:
                    fitness = future.result(timeout=30.0)
                    fitness_values.append(fitness)
                    # Increment task counter
                    with self._lock:
                        self.total_tasks_processed += 1
                except Exception as e:
                    logger.error(f"Fitness evaluation error: {e}")
                    fitness_values.append(0.0)
            
            # Evolution logic would go here
            # For now, return the population as-is
            result = population.copy()
            
            # Cache the result
            self.cache.set(cache_key, result)
            
            return result
            
        except Exception as e:
            logger.error(f"Parallel evolution error: {e}")
            return population
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        return {
            "max_workers": self.max_workers,
            "cpu_count": psutil.cpu_count(),
            "cache_stats": self.cache.get_stats(),
            "total_tasks_processed": self.total_tasks_processed,
            "status": "active"
        }
    
    def shutdown(self):
        """Shutdown executor pools."""
        self.thread_pool.shutdown(wait=True)
        self.process_pool.shutdown(wait=True)


class MemoryOptimizer:
    """Memory usage optimization and garbage collection management."""
    
    def __init__(self):
        self.gc_enabled = True
        self.memory_threshold = 80.0  # Trigger cleanup at 80% memory
        
    def optimize_memory(self):
        """Optimize memory usage."""
        import gc
        
        try:
            # Force garbage collection
            if self.gc_enabled:
                collected = gc.collect()
                logger.info(f"Garbage collection freed {collected} objects")
            
            # Check memory usage
            memory = psutil.virtual_memory()
            if memory.percent > self.memory_threshold:
                logger.warning(f"High memory usage: {memory.percent}%")
                self._emergency_cleanup()
            
        except Exception as e:
            logger.error(f"Memory optimization error: {e}")
    
    def _emergency_cleanup(self):
        """Emergency memory cleanup procedures."""
        import gc
        
        # Force multiple GC cycles
        for _ in range(3):
            gc.collect()
        
        logger.info("Emergency memory cleanup completed")
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get current memory statistics."""
        memory = psutil.virtual_memory()
        return {
            "total_gb": round(memory.total / 1024**3, 2),
            "available_gb": round(memory.available / 1024**3, 2),
            "used_percent": memory.percent,
            "threshold_percent": self.memory_threshold,
            "status": "warning" if memory.percent > self.memory_threshold else "normal"
        }


class PerformanceOptimizer:
    """Main performance optimization coordinator."""
    
    def __init__(self):
        self.monitor = PerformanceMonitor()
        self.parallel_engine = ParallelEvolutionEngine()
        self.memory_optimizer = MemoryOptimizer()
        self.is_running = False
        
    async def start_optimization(self):
        """Start all performance optimization systems."""
        self.is_running = True
        
        # Start monitoring
        self.monitor.start_monitoring(interval=2.0)
        
        # Start periodic optimization
        asyncio.create_task(self._optimization_loop())
        
        logger.info("Performance optimization system started")
    
    async def stop_optimization(self):
        """Stop optimization systems."""
        self.is_running = False
        self.monitor.stop_monitoring()
        self.parallel_engine.shutdown()
        
        logger.info("Performance optimization system stopped")
    
    async def _optimization_loop(self):
        """Main optimization loop."""
        while self.is_running:
            try:
                # Memory optimization every 30 seconds
                self.memory_optimizer.optimize_memory()
                
                await asyncio.sleep(30)
                
            except Exception as e:
                logger.error(f"Optimization loop error: {e}")
                await asyncio.sleep(30)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        return {
            "performance_monitor": self.monitor.get_performance_summary(),
            "parallel_engine": self.parallel_engine.get_performance_stats(),
            "memory_optimizer": self.memory_optimizer.get_memory_stats(),
            "system_status": "running" if self.is_running else "stopped",
            "timestamp": datetime.now().isoformat()
        }


# Global performance optimizer instance
performance_optimizer = PerformanceOptimizer()
