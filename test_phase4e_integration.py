#!/usr/bin/env python3
"""
Phase 4E Integration Test Suite
Tests all mobile and PWA features
"""

import requests
import json
import time
from typing import Dict, Any, List

# Test configuration
BASE_URL = "http://localhost:8001"
MOBILE_BASE = f"{BASE_URL}/api/mobile"


class Phase4ETestSuite:
    """Comprehensive test suite for Phase 4E Mobile & PWA features."""
    
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
    
    def test_mobile_health_endpoint(self) -> bool:
        """Test mobile health check endpoint."""
        try:
            response = requests.get(f"{MOBILE_BASE}/health", timeout=10)
            if response.status_code == 200:
                data = response.json()
                has_status = "status" in data and data["status"] == "healthy"
                has_services = "services" in data
                has_optimizations = "mobile_optimizations" in data
                
                success = has_status and has_services and has_optimizations
                details = f"Status: {data.get('status')}, Services: {has_services}, Optimizations: {has_optimizations}"
                self.log_test("Mobile Health Check", success, details)
                return success
            else:
                self.log_test("Mobile Health Check", False, f"HTTP {response.status_code}")
                return False
        except Exception as e:
            self.log_test("Mobile Health Check", False, f"Error: {e}")
            return False
    
    def test_device_info_endpoint(self) -> bool:
        """Test device information endpoint."""
        try:
            response = requests.get(f"{MOBILE_BASE}/device-info", timeout=10)
            if response.status_code == 200:
                data = response.json()
                required_fields = ["device_type", "screen_width", "screen_height", "user_agent", "is_mobile"]
                has_all_fields = all(field in data for field in required_fields)
                has_recommendations = "recommended_config" in data
                
                success = has_all_fields and has_recommendations
                details = f"Fields present: {has_all_fields}, Recommendations: {has_recommendations}, Device: {data.get('device_type')}"
                self.log_test("Device Info", success, details)
                return success
            else:
                self.log_test("Device Info", False, f"HTTP {response.status_code}")
                return False
        except Exception as e:
            self.log_test("Device Info", False, f"Error: {e}")
            return False
    
    def test_mobile_experiment_config(self) -> bool:
        """Test mobile experiment configuration."""
        try:
            mobile_config = {
                "name": "Mobile Test Experiment",
                "population_size": 50,
                "generations": 25,
                "mutation_rate": 0.1,
                "crossover_rate": 0.8,
                "elite_size": 3,
                "grid_size": 8,
                "max_steps": 200,
                "mobile_optimized": True
            }
            
            response = requests.post(
                f"{MOBILE_BASE}/experiment/mobile-config", 
                json=mobile_config,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                has_mobile_opts = data.get("mobile_optimized", False)
                has_batch_size = "batch_size" in data
                reasonable_size = data.get("population_size", 0) <= 200
                
                success = has_mobile_opts and has_batch_size and reasonable_size
                details = f"Mobile optimized: {has_mobile_opts}, Batch size: {has_batch_size}, Pop size: {data.get('population_size')}"
                self.log_test("Mobile Experiment Config", success, details)
                return success
            else:
                self.log_test("Mobile Experiment Config", False, f"HTTP {response.status_code}")
                return False
        except Exception as e:
            self.log_test("Mobile Experiment Config", False, f"Error: {e}")
            return False
    
    def test_mobile_visualization_config(self) -> bool:
        """Test mobile visualization configuration."""
        try:
            params = {"width": "360", "height": "640"}
            response = requests.get(
                f"{MOBILE_BASE}/visualization/mobile-config",
                params=params,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                has_mobile_opts = data.get("reduced_particles", False)
                has_touch_controls = data.get("touch_controls", False)
                reasonable_fps = 15 <= data.get("animation_fps", 0) <= 60
                
                success = has_mobile_opts and has_touch_controls and reasonable_fps
                details = f"Reduced particles: {has_mobile_opts}, Touch: {has_touch_controls}, FPS: {data.get('animation_fps')}"
                self.log_test("Mobile Visualization Config", success, details)
                return success
            else:
                self.log_test("Mobile Visualization Config", False, f"HTTP {response.status_code}")
                return False
        except Exception as e:
            self.log_test("Mobile Visualization Config", False, f"Error: {e}")
            return False
    
    def test_offline_sync_data(self) -> bool:
        """Test offline sync data endpoint."""
        try:
            response = requests.get(f"{MOBILE_BASE}/offline/sync-data", timeout=10)
            if response.status_code == 200:
                data = response.json()
                required_sections = ["experiments", "templates", "performance_stats", "user_preferences"]
                has_all_sections = all(section in data for section in required_sections)
                has_timestamp = "sync_timestamp" in data
                
                success = has_all_sections and has_timestamp
                details = f"Sections: {has_all_sections}, Timestamp: {has_timestamp}"
                self.log_test("Offline Sync Data", success, details)
                return success
            else:
                self.log_test("Offline Sync Data", False, f"HTTP {response.status_code}")
                return False
        except Exception as e:
            self.log_test("Offline Sync Data", False, f"Error: {e}")
            return False
    
    def test_vapid_key_endpoint(self) -> bool:
        """Test VAPID key endpoint for push notifications."""
        try:
            response = requests.get(f"{MOBILE_BASE}/push/vapid-key", timeout=10)
            if response.status_code == 200:
                data = response.json()
                has_vapid_key = "vapid_public_key" in data
                key_length = len(data.get("vapid_public_key", ""))
                valid_key = key_length > 50  # VAPID keys are typically long
                
                success = has_vapid_key and valid_key
                details = f"VAPID key present: {has_vapid_key}, Key length: {key_length}"
                self.log_test("VAPID Key", success, details)
                return success
            else:
                self.log_test("VAPID Key", False, f"HTTP {response.status_code}")
                return False
        except Exception as e:
            self.log_test("VAPID Key", False, f"Error: {e}")
            return False
    
    def test_pwa_manifest(self) -> bool:
        """Test PWA manifest file."""
        try:
            response = requests.get(f"{BASE_URL}/manifest.json", timeout=10)
            if response.status_code == 200:
                data = response.json()
                required_fields = ["name", "short_name", "start_url", "display", "theme_color"]
                has_required = all(field in data for field in required_fields)
                has_icons = "icons" in data and len(data["icons"]) > 0
                
                success = has_required and has_icons
                details = f"Required fields: {has_required}, Icons: {len(data.get('icons', []))}, Display: {data.get('display')}"
                self.log_test("PWA Manifest", success, details)
                return success
            else:
                self.log_test("PWA Manifest", False, f"HTTP {response.status_code}")
                return False
        except Exception as e:
            self.log_test("PWA Manifest", False, f"Error: {e}")
            return False
    
    def test_service_worker(self) -> bool:
        """Test service worker file."""
        try:
            response = requests.get(f"{BASE_URL}/sw.js", timeout=10)
            if response.status_code == 200:
                content = response.text
                has_cache_logic = "cache" in content.lower()
                has_fetch_handler = "fetch" in content.lower()
                has_install_handler = "install" in content.lower()
                
                success = has_cache_logic and has_fetch_handler and has_install_handler
                details = f"Cache logic: {has_cache_logic}, Fetch handler: {has_fetch_handler}, Install handler: {has_install_handler}"
                self.log_test("Service Worker", success, details)
                return success
            else:
                self.log_test("Service Worker", False, f"HTTP {response.status_code}")
                return False
        except Exception as e:
            self.log_test("Service Worker", False, f"Error: {e}")
            return False
    
    def test_offline_page(self) -> bool:
        """Test offline page."""
        try:
            response = requests.get(f"{BASE_URL}/offline.html", timeout=10)
            if response.status_code == 200:
                content = response.text.lower()
                has_offline_message = "offline" in content
                has_retry_functionality = "try again" in content or "retry" in content
                has_styling = "<style>" in content
                
                success = has_offline_message and has_retry_functionality and has_styling
                details = f"Offline message: {has_offline_message}, Retry: {has_retry_functionality}, Styled: {has_styling}"
                self.log_test("Offline Page", success, details)
                return success
            else:
                self.log_test("Offline Page", False, f"HTTP {response.status_code}")
                return False
        except Exception as e:
            self.log_test("Offline Page", False, f"Error: {e}")
            return False
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run the complete Phase 4E test suite."""
        print("🚀 Starting Phase 4E Mobile & PWA Integration Test Suite")
        print("=" * 70)
        
        # Check server health first
        if not self.check_server_health():
            print("❌ Server not healthy, aborting tests")
            return self.get_summary()
        
        # Mobile API tests
        tests = [
            self.test_mobile_health_endpoint,
            self.test_device_info_endpoint,
            self.test_mobile_experiment_config,
            self.test_mobile_visualization_config,
            self.test_offline_sync_data,
            self.test_vapid_key_endpoint,
            # PWA tests
            self.test_pwa_manifest,
            self.test_service_worker,
            self.test_offline_page,
        ]
        
        # Run all tests
        for test in tests:
            try:
                test()
                time.sleep(0.3)  # Brief pause between tests
            except Exception as e:
                print(f"❌ Test failed with exception: {e}")
        
        return self.get_summary()
    
    def get_summary(self) -> Dict[str, Any]:
        """Get test results summary."""
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r["success"])
        failed_tests = total_tests - passed_tests
        
        print("\n" + "=" * 70)
        print("📊 PHASE 4E TEST SUMMARY")
        print("=" * 70)
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
    """Run the Phase 4E integration tests."""
    suite = Phase4ETestSuite()
    summary = suite.run_all_tests()
    
    if summary["success_rate"] >= 90:
        print("\n🎉 Phase 4E Integration: EXCELLENT")
    elif summary["success_rate"] >= 70:
        print("\n👍 Phase 4E Integration: GOOD")
    elif summary["success_rate"] >= 50:
        print("\n⚠️  Phase 4E Integration: NEEDS WORK")
    else:
        print("\n❌ Phase 4E Integration: CRITICAL ISSUES")
    
    return summary


if __name__ == "__main__":
    main()
