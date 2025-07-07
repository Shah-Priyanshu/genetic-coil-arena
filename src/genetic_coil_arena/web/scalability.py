"""
Phase 4D: Scalability Enhancements
Advanced scalability features for high-load genetic algorithm processing
"""

import asyncio
import threading
import queue
import time
from typing import List, Dict, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import logging
import json
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ExperimentJob:
    """Job definition for experiment processing."""
    id: str
    experiment_config: Dict[str, Any]
    priority: int = 1  # Higher number = higher priority
    created_at: datetime = field(default_factory=datetime.now)
    status: str = "pending"  # pending, running, completed, failed
    result: Optional[Any] = None
    error: Optional[str] = None


class JobQueue:
    """Priority-based job queue for experiment processing."""
    
    def __init__(self, max_size: int = 1000):
        self._queue = queue.PriorityQueue(maxsize=max_size)
        self._jobs: Dict[str, ExperimentJob] = {}
        self._lock = threading.Lock()
        
    def add_job(self, job: ExperimentJob) -> bool:
        """Add job to queue."""
        try:
            with self._lock:
                # Priority queue expects (priority, item) tuples
                # Negative priority for max-heap behavior
                priority_item = (-job.priority, job.created_at, job)
                self._queue.put(priority_item, timeout=1.0)
                self._jobs[job.id] = job
                logger.info(f"Added job {job.id} to queue (priority {job.priority})")
                return True
        except queue.Full:
            logger.error(f"Job queue full, cannot add job {job.id}")
            return False
        except Exception as e:
            logger.error(f"Error adding job {job.id}: {e}")
            return False
    
    def get_job(self, timeout: float = 1.0) -> Optional[ExperimentJob]:
        """Get next job from queue."""
        try:
            priority_item = self._queue.get(timeout=timeout)
            _, _, job = priority_item
            with self._lock:
                job.status = "running"
            return job
        except queue.Empty:
            return None
        except Exception as e:
            logger.error(f"Error getting job from queue: {e}")
            return None
    
    def complete_job(self, job_id: str, result: Any = None, error: str = None):
        """Mark job as completed."""
        with self._lock:
            if job_id in self._jobs:
                job = self._jobs[job_id]
                job.status = "completed" if error is None else "failed"
                job.result = result
                job.error = error
                logger.info(f"Job {job_id} {job.status}")
    
    def get_queue_stats(self) -> Dict[str, Any]:
        """Get queue statistics."""
        with self._lock:
            total_jobs = len(self._jobs)
            status_counts = {}
            for job in self._jobs.values():
                status_counts[job.status] = status_counts.get(job.status, 0) + 1
            
            return {
                "queue_size": self._queue.qsize(),
                "total_jobs": total_jobs,
                "status_counts": status_counts,
                "queue_full": self._queue.full()
            }


class WorkerPool:
    """Pool of workers for processing experiment jobs."""
    
    def __init__(self, num_workers: int = 4):
        self.num_workers = num_workers
        self.workers: List[threading.Thread] = []
        self.job_queue = JobQueue()
        self.running = False
        self.processed_jobs = 0
        self.failed_jobs = 0
        
    def start(self):
        """Start worker pool."""
        self.running = True
        
        for i in range(self.num_workers):
            worker = threading.Thread(
                target=self._worker_loop,
                name=f"Worker-{i}",
                daemon=True
            )
            worker.start()
            self.workers.append(worker)
        
        logger.info(f"Started {self.num_workers} workers")
    
    def stop(self):
        """Stop worker pool."""
        self.running = False
        logger.info("Stopping worker pool...")
    
    def _worker_loop(self):
        """Main worker loop."""
        while self.running:
            try:
                job = self.job_queue.get_job(timeout=1.0)
                if job:
                    self._process_job(job)
                else:
                    time.sleep(0.1)  # Brief pause if no jobs
            except Exception as e:
                logger.error(f"Worker error: {e}")
                time.sleep(1.0)
    
    def _process_job(self, job: ExperimentJob):
        """Process a single job."""
        try:
            logger.info(f"Processing job {job.id}")
            
            # Simulate experiment processing
            start_time = time.time()
            
            # Extract experiment parameters
            config = job.experiment_config
            population_size = config.get("population_size", 100)
            generations = config.get("generations", 50)
            
            # Simulate genetic algorithm execution
            result = self._simulate_genetic_algorithm(population_size, generations)
            
            duration = time.time() - start_time
            
            # Complete job
            job_result = {
                "best_fitness": result,
                "duration_seconds": duration,
                "generations_completed": generations,
                "population_size": population_size
            }
            
            self.job_queue.complete_job(job.id, result=job_result)
            self.processed_jobs += 1
            
        except Exception as e:
            logger.error(f"Job processing error for {job.id}: {e}")
            self.job_queue.complete_job(job.id, error=str(e))
            self.failed_jobs += 1
    
    def _simulate_genetic_algorithm(self, population_size: int, generations: int) -> float:
        """Simulate genetic algorithm execution."""
        import random
        
        # Simulate evolution process
        best_fitness = 0.0
        for gen in range(generations):
            # Simulate fitness evaluation
            fitness_values = [random.random() for _ in range(population_size)]
            current_best = max(fitness_values)
            
            if current_best > best_fitness:
                best_fitness = current_best
            
            # Small delay to simulate computation
            time.sleep(0.001)
        
        return best_fitness
    
    def submit_job(self, job: ExperimentJob) -> bool:
        """Submit job for processing."""
        return self.job_queue.add_job(job)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get worker pool statistics."""
        queue_stats = self.job_queue.get_queue_stats()
        
        return {
            "num_workers": self.num_workers,
            "running": self.running,
            "processed_jobs": self.processed_jobs,
            "failed_jobs": self.failed_jobs,
            "success_rate": (self.processed_jobs / max(1, self.processed_jobs + self.failed_jobs)) * 100,
            "queue_stats": queue_stats
        }


class LoadBalancer:
    """Load balancing for distributed experiment processing."""
    
    def __init__(self):
        self.worker_pools: List[WorkerPool] = []
        self.current_pool_index = 0
        
    def add_worker_pool(self, pool: WorkerPool):
        """Add worker pool to load balancer."""
        self.worker_pools.append(pool)
        logger.info(f"Added worker pool, total pools: {len(self.worker_pools)}")
    
    def distribute_job(self, job: ExperimentJob) -> bool:
        """Distribute job to least loaded worker pool."""
        if not self.worker_pools:
            logger.error("No worker pools available")
            return False
        
        # Simple round-robin distribution
        # In production, could use more sophisticated load balancing
        selected_pool = self.worker_pools[self.current_pool_index]
        self.current_pool_index = (self.current_pool_index + 1) % len(self.worker_pools)
        
        return selected_pool.submit_job(job)
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics."""
        total_stats = {
            "total_pools": len(self.worker_pools),
            "total_workers": 0,
            "total_processed": 0,
            "total_failed": 0,
            "pool_stats": []
        }
        
        for i, pool in enumerate(self.worker_pools):
            pool_stats = pool.get_stats()
            total_stats["pool_stats"].append({
                "pool_id": i,
                **pool_stats
            })
            
            total_stats["total_workers"] += pool_stats["num_workers"]
            total_stats["total_processed"] += pool_stats["processed_jobs"]
            total_stats["total_failed"] += pool_stats["failed_jobs"]
        
        if total_stats["total_processed"] + total_stats["total_failed"] > 0:
            total_stats["overall_success_rate"] = (
                total_stats["total_processed"] / 
                (total_stats["total_processed"] + total_stats["total_failed"])
            ) * 100
        else:
            total_stats["overall_success_rate"] = 0.0
        
        return total_stats


class AdaptiveScaler:
    """Adaptive scaling based on system load."""
    
    def __init__(self, load_balancer: LoadBalancer):
        self.load_balancer = load_balancer
        self.monitoring = False
        self.scale_up_threshold = 80.0  # CPU percentage
        self.scale_down_threshold = 30.0
        self.max_pools = 8
        self.min_pools = 1
        
    def start_monitoring(self):
        """Start adaptive scaling monitoring."""
        self.monitoring = True
        
        def monitor_loop():
            while self.monitoring:
                try:
                    self._check_scaling_needs()
                    time.sleep(10.0)  # Check every 10 seconds
                except Exception as e:
                    logger.error(f"Scaling monitor error: {e}")
                    time.sleep(10.0)
        
        thread = threading.Thread(target=monitor_loop, daemon=True)
        thread.start()
        logger.info("Adaptive scaling monitoring started")
    
    def stop_monitoring(self):
        """Stop adaptive scaling monitoring."""
        self.monitoring = False
        logger.info("Adaptive scaling monitoring stopped")
    
    def _check_scaling_needs(self):
        """Check if scaling is needed."""
        try:
            import psutil
            
            cpu_percent = psutil.cpu_percent(interval=1.0)
            current_pools = len(self.load_balancer.worker_pools)
            
            # Scale up if high CPU and can add more pools
            if cpu_percent > self.scale_up_threshold and current_pools < self.max_pools:
                self._scale_up()
            
            # Scale down if low CPU and have extra pools
            elif cpu_percent < self.scale_down_threshold and current_pools > self.min_pools:
                self._scale_down()
                
        except Exception as e:
            logger.error(f"Error checking scaling needs: {e}")
    
    def _scale_up(self):
        """Add new worker pool."""
        try:
            new_pool = WorkerPool(num_workers=2)  # Start with 2 workers
            new_pool.start()
            self.load_balancer.add_worker_pool(new_pool)
            logger.info("Scaled up: Added new worker pool")
        except Exception as e:
            logger.error(f"Scale up error: {e}")
    
    def _scale_down(self):
        """Remove worker pool if possible."""
        try:
            if len(self.load_balancer.worker_pools) > self.min_pools:
                # Remove the last pool (simple strategy)
                pool_to_remove = self.load_balancer.worker_pools.pop()
                pool_to_remove.stop()
                logger.info("Scaled down: Removed worker pool")
        except Exception as e:
            logger.error(f"Scale down error: {e}")


class ScalabilityManager:
    """Main scalability management system."""
    
    def __init__(self):
        self.load_balancer = LoadBalancer()
        self.adaptive_scaler = AdaptiveScaler(self.load_balancer)
        self.is_running = False
        
        # Initialize with one worker pool
        initial_pool = WorkerPool(num_workers=4)
        self.load_balancer.add_worker_pool(initial_pool)
    
    async def start(self):
        """Start scalability system."""
        self.is_running = True
        
        # Start all worker pools
        for pool in self.load_balancer.worker_pools:
            pool.start()
        
        # Start adaptive scaling
        self.adaptive_scaler.start_monitoring()
        
        logger.info("Scalability management system started")
    
    async def stop(self):
        """Stop scalability system."""
        self.is_running = False
        
        # Stop adaptive scaling
        self.adaptive_scaler.stop_monitoring()
        
        # Stop all worker pools
        for pool in self.load_balancer.worker_pools:
            pool.stop()
        
        logger.info("Scalability management system stopped")
    
    def submit_experiment(self, experiment_config: Dict[str, Any], 
                         priority: int = 1) -> str:
        """Submit experiment for processing."""
        job_id = f"exp_{int(time.time() * 1000)}"
        job = ExperimentJob(
            id=job_id,
            experiment_config=experiment_config,
            priority=priority
        )
        
        success = self.load_balancer.distribute_job(job)
        if success:
            logger.info(f"Submitted experiment {job_id}")
            return job_id
        else:
            logger.error(f"Failed to submit experiment {job_id}")
            return ""
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        return {
            "system_running": self.is_running,
            "load_balancer_stats": self.load_balancer.get_system_stats(),
            "adaptive_scaling": {
                "monitoring": self.adaptive_scaler.monitoring,
                "scale_up_threshold": self.adaptive_scaler.scale_up_threshold,
                "scale_down_threshold": self.adaptive_scaler.scale_down_threshold
            },
            "timestamp": datetime.now().isoformat()
        }


# Global scalability manager instance
scalability_manager = ScalabilityManager()
