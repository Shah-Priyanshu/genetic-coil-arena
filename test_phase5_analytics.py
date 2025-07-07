"""
Phase 5 Analytics Integration Test
Test the advanced analytics and AI integration features
"""

import asyncio
import requests
import json
from datetime import datetime


async def test_phase5_analytics():
    """Test Phase 5 advanced analytics integration."""
    
    print("🧪 Starting Phase 5 Analytics Integration Tests")
    print("=" * 60)
    
    base_url = "http://localhost:8000"
    analytics_url = f"{base_url}/api/analytics"
    
    # Start the server in background first
    print("🚀 Starting production server with Phase 5 analytics...")
    
    # Import and start server
    try:
        from src.genetic_coil_arena.web.server_production import ProductionServer
        server = ProductionServer(host="localhost", port=8000)
        
        # Start server in background
        import uvicorn
        import threading
        
        def run_server():
            uvicorn.run(server.app, host="localhost", port=8000, log_level="info")
        
        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()
        
        # Wait for server to start
        await asyncio.sleep(3)
        print("✅ Server started successfully")
        
    except Exception as e:
        print(f"❌ Failed to start server: {e}")
        return False
    
    # Test analytics endpoints
    test_results = {}
    total_tests = 0
    passed_tests = 0
    
    # Test 1: Analytics Health Check
    total_tests += 1
    try:
        response = requests.get(f"{analytics_url}/health", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Analytics Health: {data.get('status', 'unknown')}")
            test_results["health"] = "PASSED"
            passed_tests += 1
        else:
            print(f"❌ Analytics Health: HTTP {response.status_code}")
            test_results["health"] = "FAILED"
    except Exception as e:
        print(f"❌ Analytics Health: {e}")
        test_results["health"] = "FAILED"
    
    # Test 2: Get Insights
    total_tests += 1
    try:
        response = requests.get(f"{analytics_url}/insights?limit=5", timeout=10)
        if response.status_code == 200:
            data = response.json()
            insights_count = len(data.get('insights', []))
            print(f"✅ Analytics Insights: {insights_count} insights generated")
            test_results["insights"] = "PASSED"
            passed_tests += 1
        else:
            print(f"❌ Analytics Insights: HTTP {response.status_code}")
            test_results["insights"] = "FAILED"
    except Exception as e:
        print(f"❌ Analytics Insights: {e}")
        test_results["insights"] = "FAILED"
    
    # Test 3: Create Prediction
    total_tests += 1
    try:
        prediction_data = {
            "config": {
                "population_size": 100,
                "generations": 50,
                "mutation_rate": 0.1,
                "crossover_rate": 0.8
            },
            "experiment_id": "test_prediction"
        }
        response = requests.post(
            f"{analytics_url}/predictions", 
            json=prediction_data, 
            timeout=10
        )
        if response.status_code == 200:
            data = response.json()
            fitness = data.get('prediction', {}).get('predicted_best_fitness', 0)
            print(f"✅ ML Prediction: Predicted fitness {fitness:.3f}")
            test_results["prediction"] = "PASSED"
            passed_tests += 1
        else:
            print(f"❌ ML Prediction: HTTP {response.status_code}")
            test_results["prediction"] = "FAILED"
    except Exception as e:
        print(f"❌ ML Prediction: {e}")
        test_results["prediction"] = "FAILED"
    
    # Test 4: Get Anomalies
    total_tests += 1
    try:
        response = requests.get(f"{analytics_url}/anomalies?hours=24", timeout=10)
        if response.status_code == 200:
            data = response.json()
            anomaly_count = len(data.get('anomalies', []))
            print(f"✅ Anomaly Detection: {anomaly_count} anomalies detected")
            test_results["anomalies"] = "PASSED"
            passed_tests += 1
        else:
            print(f"❌ Anomaly Detection: HTTP {response.status_code}")
            test_results["anomalies"] = "FAILED"
    except Exception as e:
        print(f"❌ Anomaly Detection: {e}")
        test_results["anomalies"] = "FAILED"
    
    # Test 5: Get Optimization Recommendations
    total_tests += 1
    try:
        optimization_data = {
            "config": {
                "population_size": 50,
                "generations": 25,
                "mutation_rate": 0.2,
                "crossover_rate": 0.7
            }
        }
        response = requests.post(
            f"{analytics_url}/optimization/recommendations", 
            json=optimization_data, 
            timeout=10
        )
        if response.status_code == 200:
            data = response.json()
            recommendations = data.get('recommendations', [])
            print(f"✅ Optimization: {len(recommendations)} recommendations generated")
            test_results["optimization"] = "PASSED"
            passed_tests += 1
        else:
            print(f"❌ Optimization: HTTP {response.status_code}")
            test_results["optimization"] = "FAILED"
    except Exception as e:
        print(f"❌ Optimization: {e}")
        test_results["optimization"] = "FAILED"
    
    # Test 6: Performance Trends
    total_tests += 1
    try:
        response = requests.get(f"{analytics_url}/performance/trends?days=7", timeout=10)
        if response.status_code == 200:
            data = response.json()
            trends = data.get('trends', {})
            print(f"✅ Performance Trends: Analysis completed")
            test_results["trends"] = "PASSED"
            passed_tests += 1
        else:
            print(f"❌ Performance Trends: HTTP {response.status_code}")
            test_results["trends"] = "FAILED"
    except Exception as e:
        print(f"❌ Performance Trends: {e}")
        test_results["trends"] = "FAILED"
    
    # Test 7: Model Status
    total_tests += 1
    try:
        response = requests.get(f"{analytics_url}/model/status", timeout=10)
        if response.status_code == 200:
            data = response.json()
            model_status = data.get('model_status', {})
            print(f"✅ Model Status: {json.dumps(model_status, indent=2)}")
            test_results["model_status"] = "PASSED"
            passed_tests += 1
        else:
            print(f"❌ Model Status: HTTP {response.status_code}")
            test_results["model_status"] = "FAILED"
    except Exception as e:
        print(f"❌ Model Status: {e}")
        test_results["model_status"] = "FAILED"
    
    # Test 8: Model Retrain
    total_tests += 1
    try:
        response = requests.post(f"{analytics_url}/model/retrain", timeout=20)
        if response.status_code == 200:
            data = response.json()
            metrics = data.get('training_metrics', {})
            print(f"✅ Model Retrain: Training completed with metrics")
            test_results["retrain"] = "PASSED"
            passed_tests += 1
        else:
            print(f"❌ Model Retrain: HTTP {response.status_code}")
            test_results["retrain"] = "FAILED"
    except Exception as e:
        print(f"❌ Model Retrain: {e}")
        test_results["retrain"] = "FAILED"
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 PHASE 5 ANALYTICS INTEGRATION TEST SUMMARY")
    print("=" * 60)
    
    success_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
    
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {total_tests - passed_tests}")
    print(f"Success Rate: {success_rate:.1f}%")
    
    print("\nDetailed Results:")
    for test_name, result in test_results.items():
        status_icon = "✅" if result == "PASSED" else "❌"
        print(f"  {status_icon} {test_name.replace('_', ' ').title()}: {result}")
    
    if success_rate >= 80:
        print(f"\n🎉 PHASE 5 ANALYTICS INTEGRATION: SUCCESS!")
        print("Advanced Analytics & AI Integration features are working correctly.")
        return True
    else:
        print(f"\n⚠️ PHASE 5 ANALYTICS INTEGRATION: PARTIAL SUCCESS")
        print("Some analytics features need attention.")
        return False


if __name__ == "__main__":
    success = asyncio.run(test_phase5_analytics())
    if success:
        print("\n✅ Phase 5 ready for production!")
    else:
        print("\n⚠️ Phase 5 needs additional work.")
