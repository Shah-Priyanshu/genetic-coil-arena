"""
Test script for Phase 4C Collaboration Features
Demonstrates team management, notifications, and version control functionality.
"""

import requests
import json
import time
from datetime import datetime

BASE_URL = "http://localhost:8000"

def test_health_endpoint():
    """Test the system health endpoint."""
    print("🔍 Testing System Health...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            health_data = response.json()
            print("✅ System Health Check:")
            print(f"   Status: {health_data.get('status', 'unknown')}")
            print(f"   Version: {health_data.get('version', 'unknown')}")
            
            components = health_data.get('components', {})
            for component, status in components.items():
                status_icon = "✅" if status else "❌"
                print(f"   {status_icon} {component}: {status}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def test_team_management():
    """Test team management functionality."""
    print("\n🤝 Testing Team Management...")
    
    # Test team creation
    team_data = {
        "name": "Demo Research Team",
        "description": "A demonstration team for Phase 4C testing"
    }
    
    try:
        response = requests.post(f"{BASE_URL}/api/v1/teams/", json=team_data)
        if response.status_code in [200, 201]:
            print("✅ Team creation endpoint available")
            team_info = response.json()
            print(f"   Created team: {team_info.get('name', 'Unknown')}")
            return True
        else:
            print(f"❌ Team creation failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Team management test error: {e}")
        return False

def test_notifications():
    """Test notification system."""
    print("\n🔔 Testing Notifications...")
    
    try:
        response = requests.get(f"{BASE_URL}/api/notifications/")
        if response.status_code == 200:
            print("✅ Notifications endpoint available")
            notifications = response.json()
            print(f"   Found {len(notifications)} notifications")
            return True
        else:
            print(f"❌ Notifications test failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Notifications test error: {e}")
        return False

def test_version_control():
    """Test version control system."""
    print("\n🔄 Testing Version Control...")
    
    try:
        response = requests.get(f"{BASE_URL}/api/versions/")
        if response.status_code == 200:
            print("✅ Version control endpoint available")
            versions = response.json()
            print(f"   Found {len(versions)} experiment versions")
            return True
        else:
            print(f"❌ Version control test failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Version control test error: {e}")
        return False

def test_ui_endpoints():
    """Test UI endpoints."""
    print("\n🖥️ Testing UI Endpoints...")
    
    ui_endpoints = [
        "/teams/ui",
        "/templates-ui",
        "/batch-ui",
        "/visualization3d"
    ]
    
    available_uis = []
    for endpoint in ui_endpoints:
        try:
            response = requests.get(f"{BASE_URL}{endpoint}")
            if response.status_code == 200:
                available_uis.append(endpoint)
                print(f"✅ {endpoint} - Available")
            else:
                print(f"❌ {endpoint} - Not available ({response.status_code})")
        except Exception as e:
            print(f"❌ {endpoint} - Error: {e}")
    
    return len(available_uis) > 0

def main():
    """Run all tests for Phase 4C features."""
    print("🧬 Phase 4C Collaboration Features Test Suite")
    print("=" * 50)
    
    start_time = time.time()
    
    # Run tests
    tests = [
        ("System Health", test_health_endpoint),
        ("Team Management", test_team_management),
        ("Notifications", test_notifications),
        ("Version Control", test_version_control),
        ("UI Endpoints", test_ui_endpoints)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
    
    # Summary
    elapsed = time.time() - start_time
    print("\n" + "=" * 50)
    print(f"📊 Test Summary: {passed}/{total} tests passed")
    print(f"⏱️ Time elapsed: {elapsed:.2f} seconds")
    
    if passed == total:
        print("🎉 All Phase 4C collaboration features are working!")
    else:
        print("⚠️ Some features need attention")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
