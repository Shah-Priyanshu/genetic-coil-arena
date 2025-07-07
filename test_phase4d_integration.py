#!/usr/bin/env python3
"""
Phase 4D Integration Test Suite
Tests all performance and scalability features
"""

import requests
import json
import time
import asyncio
import threading
from typing import Dict, Any

# Test configuration
BASE_URL = "http://localhost:8001"
PERFORMANCE_BASE = f"{BASE_URL}/api/performance"


class Phase4DTestSuite:
    """Comprehensive test suite for Phase 4D Performance & Scalability features."""
    
    def __init__(self):
        self.results = []
        self.server_running = False
    
    def log_test(self, test_name: str, success: bool, details: str = ""):
        """Log test results."""
        status = "✅ PASS" if success else "❌ FAIL"
        self.results.append({
            "test": test_name,
            "success": success,
            "details": details
        })
        print(f"{status} {test_name}: {details}")
    
    def check_server_health(self) -> bool:
        """Check if the server is running and healthy."""
        try:
            response = requests.get(f"{BASE_URL}/", timeout=5)
            self.server_running = response.status_code == 200
            self.log_test("Server Health Check", self.server_running, 
                         f"Status: {response.status_code}")
            return self.server_running
        except Exception as e:
            self.log_test("Server Health Check", False, f"Error: {e}")
            return False
    
    def test_performance_status(self) -> bool:
        """Test performance status endpoint."""
        try:
            response = requests.get(f"{PERFORMANCE_BASE}/status", timeout=10)
            if response.status_code == 200:
                data = response.json()
                has_monitor = "performance_monitor" in data
                has_parallel = "parallel_engine" in data
                has_memory = "memory_optimizer" in data
                
                success = has_monitor and has_parallel and has_memory
                details = f"Components present: monitor={has_monitor}, parallel={has_parallel}, memory={has_memory}"
                self.log_test("Performance Status", success, details)
                return success
            else:
                self.log_test("Performance Status", False, f"HTTP {response.status_code}")
                return False
        except Exception as e:
            self.log_test("Performance Status", False, f"Error: {e}")
            return False
    
    def test_performance_metrics(self) -> bool:
        """Test performance metrics endpoint."""
        try:
            response = requests.get(f"{PERFORMANCE_BASE}/metrics", timeout=10)
            success = response.status_code == 200
            details = f"HTTP {response.status_code}, Data: {response.text[:100]}..."
            self.log_test("Performance Metrics", success, details)
            return success
        except Exception as e:
            self.log_test("Performance Metrics", False, f"Error: {e}")
            return False
    
    def test_cache_stats(self) -> bool:
        """Test cache statistics endpoint."""
        try:
            response = requests.get(f"{PERFORMANCE_BASE}/cache/stats", timeout=10)
            if response.status_code == 200:
                data = response.json()
                has_stats = all(key in data for key in ["size", "max_size", "hit_count", "miss_count"])
                details = f"Cache stats: {data}"
                self.log_test("Cache Statistics", has_stats, details)
                return has_stats
            else:
                self.log_test("Cache Statistics", False, f"HTTP {response.status_code}")
                return False
        except Exception as e:
            self.log_test("Cache Statistics", False, f"Error: {e}")
            return False
    
    def test_memory_stats(self) -> bool:
        """Test memory statistics endpoint."""
        try:
            response = requests.get(f"{PERFORMANCE_BASE}/memory/stats", timeout=10)
            success = response.status_code == 200
            if success:
                data = response.json()
                details = f"Memory stats available, keys: {list(data.keys())}"
            else:
                details = f"HTTP {response.status_code}"
            self.log_test("Memory Statistics", success, details)
            return success
        except Exception as e:
            self.log_test("Memory Statistics", False, f"Error: {e}")
            return False
    
    def test_optimization_control(self) -> bool:
        """Test performance optimization start/stop control."""
        try:
            # Test start optimization
            start_response = requests.post(f"{PERFORMANCE_BASE}/start", timeout=10)
            start_success = start_response.status_code == 200
            
            if start_success:
                time.sleep(2)  # Let it run for a moment
                
                # Test stop optimization
                stop_response = requests.post(f"{PERFORMANCE_BASE}/stop", timeout=10)
                stop_success = stop_response.status_code == 200
                
                success = start_success and stop_success
                details = f"Start: {start_response.status_code}, Stop: {stop_response.status_code}"
            else:
                success = False
                details = f"Start failed: {start_response.status_code}"
            
            self.log_test("Optimization Control", success, details)
            return success
        except Exception as e:
            self.log_test("Optimization Control", False, f"Error: {e}")
            return False
    
    def test_parallel_stats(self) -> bool:
        """Test parallel processing statistics."""
        try:
            response = requests.get(f"{PERFORMANCE_BASE}/parallel/stats", timeout=10)
            if response.status_code == 200:
                data = response.json()
                has_workers = "max_workers" in data
                has_tasks = "total_tasks_processed" in data
                success = has_workers and has_tasks
                details = f"Parallel stats: workers={data.get('max_workers')}, tasks={data.get('total_tasks_processed')}"
                self.log_test("Parallel Statistics", success, details)
                return success
            else:
                self.log_test("Parallel Statistics", False, f"HTTP {response.status_code}")
                return False
        except Exception as e:
            self.log_test("Parallel Statistics", False, f"Error: {e}")
            return False
    
    def test_memory_optimization(self) -> bool:
        """Test memory optimization endpoint."""
        try:
            response = requests.post(f"{PERFORMANCE_BASE}/memory/optimize", timeout=15)
            success = response.status_code == 200
            details = f"HTTP {response.status_code}, Response: {response.text}"
            self.log_test("Memory Optimization", success, details)
            return success
        except Exception as e:
            self.log_test("Memory Optimization", False, f"Error: {e}")
            return False
    
    def test_cache_operations(self) -> bool:
        """Test cache clear operation."""
        try:
            response = requests.post(f"{PERFORMANCE_BASE}/cache/clear", timeout=10)
            success = response.status_code == 200
            details = f"HTTP {response.status_code}, Response: {response.text}"
            self.log_test("Cache Operations", success, details)
            return success
        except Exception as e:
            self.log_test("Cache Operations", False, f"Error: {e}")
            return False
    
    def test_dashboard_endpoints(self) -> bool:
        """Test performance dashboard endpoint."""
        try:
            response = requests.get(f"{PERFORMANCE_BASE}/dashboard", timeout=10)
            success = response.status_code == 200
            if success:
                data = response.json()
                has_sections = all(key in data for key in [
                    "system_performance", "memory_usage", "cache_performance"
                ])
                details = f"Dashboard sections present: {has_sections}"
                success = has_sections
            else:
                details = f"HTTP {response.status_code}"
            self.log_test("Performance Dashboard", success, details)
            return success
        except Exception as e:
            self.log_test("Performance Dashboard", False, f"Error: {e}")
            return False
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run the complete Phase 4D test suite."""
        print("🚀 Starting Phase 4D Integration Test Suite")
        print("=" * 60)
        
        # Check server health first
        if not self.check_server_health():
            print("❌ Server not healthy, aborting tests")
            return self.get_summary()
        
        # Core performance tests
        tests = [
            self.test_performance_status,
            self.test_performance_metrics,
            self.test_cache_stats,
            self.test_memory_stats,
            self.test_optimization_control,
            self.test_parallel_stats,
            self.test_memory_optimization,
            self.test_cache_operations,
            self.test_dashboard_endpoints,
        ]
        
        # Run all tests
        for test in tests:
            try:
                test()
                time.sleep(0.5)  # Brief pause between tests
            except Exception as e:
                print(f"❌ Test failed with exception: {e}")
        
        return self.get_summary()
    
    def get_summary(self) -> Dict[str, Any]:
        """Get test results summary."""
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r["success"])
        failed_tests = total_tests - passed_tests
        
        print("\n" + "=" * 60)
        print("📊 PHASE 4D TEST SUMMARY")
        print("=" * 60)
        print(f"Total Tests: {total_tests}")
        print(f"✅ Passed: {passed_tests}")
        print(f"❌ Failed: {failed_tests}")
        print(f"Success Rate: {(passed_tests/total_tests*100):.1f}%" if total_tests > 0 else "No tests run")
        
        if failed_tests > 0:
            print("\n❌ Failed Tests:")
            for result in self.results:
                if not result["success"]:
                    print(f"  - {result['test']}: {result['details']}")
        
        summary = {
            "total_tests": total_tests,
            "passed": passed_tests,
            "failed": failed_tests,
            "success_rate": (passed_tests/total_tests*100) if total_tests > 0 else 0,
            "server_running": self.server_running,
            "results": self.results
        }
        
        return summary


def main():
    """Run the Phase 4D integration tests."""
    suite = Phase4DTestSuite()
    summary = suite.run_all_tests()
    
    if summary["success_rate"] >= 90:
        print("\n🎉 Phase 4D Integration: EXCELLENT")
    elif summary["success_rate"] >= 70:
        print("\n👍 Phase 4D Integration: GOOD")
    elif summary["success_rate"] >= 50:
        print("\n⚠️  Phase 4D Integration: NEEDS WORK")
    else:
        print("\n❌ Phase 4D Integration: CRITICAL ISSUES")
    
    return summary


if __name__ == "__main__":
    main()
