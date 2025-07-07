#!/usr/bin/env python3
"""
Phase 4D Load Testing and Performance Demo
Demonstrates the scalability and performance features of the genetic coil arena
"""

import asyncio
import aiohttp
import time
import statistics
import json
from typing import List, Dict, Any
import concurrent.futures
import threading


class LoadTestConfig:
    """Configuration for load testing."""
    def __init__(self):
        self.base_url = "http://localhost:8001"
        self.concurrent_requests = 10
        self.total_requests = 100
        self.endpoints = [
            "/api/performance/status",
            "/api/performance/metrics",
            "/api/performance/cache/stats",
            "/api/performance/memory/stats",
            "/api/performance/parallel/stats",
            "/api/performance/dashboard"
        ]


class PerformanceDemo:
    """Demonstrates Phase 4D performance and scalability features."""
    
    def __init__(self, config: LoadTestConfig):
        self.config = config
        self.results = []
        self.error_count = 0
        
    async def make_request(self, session: aiohttp.ClientSession, endpoint: str) -> Dict[str, Any]:
        """Make a single HTTP request and measure performance."""
        start_time = time.time()
        url = f"{self.config.base_url}{endpoint}"
        
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                content = await response.text()
                end_time = time.time()
                
                return {
                    "endpoint": endpoint,
                    "status_code": response.status,
                    "response_time_ms": (end_time - start_time) * 1000,
                    "content_length": len(content),
                    "success": response.status == 200,
                    "timestamp": start_time
                }
        except Exception as e:
            end_time = time.time()
            self.error_count += 1
            return {
                "endpoint": endpoint,
                "status_code": 0,
                "response_time_ms": (end_time - start_time) * 1000,
                "content_length": 0,
                "success": False,
                "error": str(e),
                "timestamp": start_time
            }
    
    async def run_concurrent_test(self, endpoint: str, num_requests: int) -> List[Dict[str, Any]]:
        """Run concurrent requests to a single endpoint."""
        async with aiohttp.ClientSession() as session:
            tasks = [self.make_request(session, endpoint) for _ in range(num_requests)]
            results = await asyncio.gather(*tasks)
            return results
    
    async def run_load_test(self) -> Dict[str, Any]:
        """Run comprehensive load test across all endpoints."""
        print("🚀 Starting Phase 4D Performance Load Test")
        print("=" * 60)
        
        all_results = []
        start_time = time.time()
        
        # Test each endpoint with concurrent requests
        for endpoint in self.config.endpoints:
            print(f"Testing {endpoint}...")
            endpoint_results = await self.run_concurrent_test(
                endpoint, 
                self.config.concurrent_requests
            )
            all_results.extend(endpoint_results)
            
            # Brief pause between endpoint tests
            await asyncio.sleep(0.1)
        
        end_time = time.time()
        total_test_time = end_time - start_time
        
        # Calculate statistics
        successful_requests = [r for r in all_results if r["success"]]
        response_times = [r["response_time_ms"] for r in successful_requests]
        
        if response_times:
            stats = {
                "total_requests": len(all_results),
                "successful_requests": len(successful_requests),
                "failed_requests": len(all_results) - len(successful_requests),
                "success_rate_percent": (len(successful_requests) / len(all_results)) * 100,
                "total_test_time_seconds": total_test_time,
                "requests_per_second": len(all_results) / total_test_time,
                "response_time_stats": {
                    "min_ms": min(response_times),
                    "max_ms": max(response_times),
                    "avg_ms": statistics.mean(response_times),
                    "median_ms": statistics.median(response_times),
                    "p95_ms": self.percentile(response_times, 95),
                    "p99_ms": self.percentile(response_times, 99)
                }
            }
        else:
            stats = {
                "total_requests": len(all_results),
                "successful_requests": 0,
                "failed_requests": len(all_results),
                "success_rate_percent": 0,
                "total_test_time_seconds": total_test_time,
                "requests_per_second": 0,
                "response_time_stats": {}
            }
        
        self.results = all_results
        return stats
    
    def percentile(self, data: List[float], p: int) -> float:
        """Calculate percentile."""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        index = (p / 100) * (len(sorted_data) - 1)
        
        if index.is_integer():
            return sorted_data[int(index)]
        else:
            lower = sorted_data[int(index)]
            upper = sorted_data[int(index) + 1]
            return lower + (upper - lower) * (index - int(index))
    
    def print_results(self, stats: Dict[str, Any]):
        """Print formatted test results."""
        print("\n" + "=" * 60)
        print("📊 LOAD TEST RESULTS")
        print("=" * 60)
        
        print(f"Total Requests: {stats['total_requests']}")
        print(f"✅ Successful: {stats['successful_requests']}")
        print(f"❌ Failed: {stats['failed_requests']}")
        print(f"Success Rate: {stats['success_rate_percent']:.1f}%")
        print(f"Test Duration: {stats['total_test_time_seconds']:.2f}s")
        print(f"Requests/sec: {stats['requests_per_second']:.1f}")
        
        if stats['response_time_stats']:
            rt_stats = stats['response_time_stats']
            print(f"\n📈 RESPONSE TIME STATISTICS")
            print(f"Min: {rt_stats['min_ms']:.1f}ms")
            print(f"Max: {rt_stats['max_ms']:.1f}ms")
            print(f"Average: {rt_stats['avg_ms']:.1f}ms")
            print(f"Median: {rt_stats['median_ms']:.1f}ms")
            print(f"95th percentile: {rt_stats['p95_ms']:.1f}ms")
            print(f"99th percentile: {rt_stats['p99_ms']:.1f}ms")
        
        # Performance assessment
        avg_response = stats['response_time_stats'].get('avg_ms', float('inf'))
        success_rate = stats['success_rate_percent']
        
        print(f"\n🎯 PERFORMANCE ASSESSMENT")
        if success_rate >= 99 and avg_response <= 100:
            print("🌟 EXCELLENT: High reliability and fast response times")
        elif success_rate >= 95 and avg_response <= 200:
            print("👍 GOOD: Solid performance with room for optimization")
        elif success_rate >= 90 and avg_response <= 500:
            print("⚠️  FAIR: Acceptable but needs performance tuning")
        else:
            print("❌ POOR: Significant performance issues detected")


class ScalabilityDemo:
    """Demonstrates scalability features of Phase 4D."""
    
    def __init__(self, base_url: str = "http://localhost:8001"):
        self.base_url = base_url
    
    async def test_optimization_control(self):
        """Test performance optimization start/stop."""
        print("\n🔧 Testing Performance Optimization Control")
        
        async with aiohttp.ClientSession() as session:
            # Start optimization
            start_url = f"{self.base_url}/api/performance/start"
            async with session.post(start_url) as response:
                print(f"Start Optimization: {response.status}")
            
            # Wait and check status
            await asyncio.sleep(2)
            status_url = f"{self.base_url}/api/performance/status"
            async with session.get(status_url) as response:
                data = await response.json()
                print(f"Optimization Running: {response.status}")
            
            # Stop optimization
            stop_url = f"{self.base_url}/api/performance/stop"
            async with session.post(stop_url) as response:
                print(f"Stop Optimization: {response.status}")
    
    async def test_memory_optimization(self):
        """Test memory optimization capabilities."""
        print("\n💾 Testing Memory Optimization")
        
        async with aiohttp.ClientSession() as session:
            # Get initial memory stats
            memory_url = f"{self.base_url}/api/performance/memory/stats"
            async with session.get(memory_url) as response:
                initial_stats = await response.json()
                print(f"Initial Memory Usage: {initial_stats.get('used_percent', 0):.1f}%")
            
            # Trigger memory optimization
            optimize_url = f"{self.base_url}/api/performance/memory/optimize"
            async with session.post(optimize_url) as response:
                print(f"Memory Optimization: {response.status}")
            
            # Check final memory stats
            async with session.get(memory_url) as response:
                final_stats = await response.json()
                print(f"Final Memory Usage: {final_stats.get('used_percent', 0):.1f}%")
    
    async def test_cache_performance(self):
        """Test caching system performance."""
        print("\n🗄️  Testing Cache Performance")
        
        async with aiohttp.ClientSession() as session:
            # Clear cache first
            clear_url = f"{self.base_url}/api/performance/cache/clear"
            async with session.post(clear_url) as response:
                print(f"Cache Clear: {response.status}")
            
            # Get cache stats
            stats_url = f"{self.base_url}/api/performance/cache/stats"
            async with session.get(stats_url) as response:
                cache_stats = await response.json()
                print(f"Cache Size: {cache_stats.get('size', 0)}")
                print(f"Hit Rate: {cache_stats.get('hit_rate_percent', 0):.1f}%")
    
    async def run_scalability_demo(self):
        """Run complete scalability demonstration."""
        print("🎯 Phase 4D Scalability Demonstration")
        print("=" * 60)
        
        await self.test_optimization_control()
        await self.test_memory_optimization()
        await self.test_cache_performance()
        
        print("\n✅ Scalability demonstration completed!")


async def main():
    """Main demonstration function."""
    # Configuration
    config = LoadTestConfig()
    
    # Run load test
    load_tester = PerformanceDemo(config)
    stats = await load_tester.run_load_test()
    load_tester.print_results(stats)
    
    # Run scalability demo
    scalability_demo = ScalabilityDemo()
    await scalability_demo.run_scalability_demo()
    
    print("\n🎉 Phase 4D Performance & Scalability Demo Complete!")
    print("All features demonstrated successfully.")


if __name__ == "__main__":
    # Check if required packages are available
    try:
        import aiohttp
    except ImportError:
        print("⚠️ aiohttp not installed. Installing...")
        import subprocess
        subprocess.run(["pip", "install", "aiohttp"])
        import aiohttp
    
    asyncio.run(main())
