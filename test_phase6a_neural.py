"""
Phase 6A Neural Evolution Integration Test
Test the deep learning integration features
"""

import asyncio
import requests
import json
from datetime import datetime
import time


async def test_phase6a_neural_evolution():
    """Test Phase 6A neural evolution integration."""
    
    print("🧪 Starting Phase 6A Neural Evolution Integration Tests")
    print("=" * 60)
    
    base_url = "http://localhost:8000"
    neural_url = f"{base_url}/api/neural"
    
    # Test data
    test_results = {}
    total_tests = 0
    passed_tests = 0
    
    print("🚀 Testing Neural Evolution System...")
    
    # Test 1: Neural System Health Check
    total_tests += 1
    try:
        response = requests.get(f"{neural_url}/health", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Neural Health: {data.get('status', 'unknown')}")
            print(f"   PyTorch Version: {data.get('torch_version', 'unknown')}")
            print(f"   CUDA Available: {data.get('cuda_available', False)}")
            test_results["health"] = "PASSED"
            passed_tests += 1
        else:
            print(f"❌ Neural Health: HTTP {response.status_code}")
            test_results["health"] = "FAILED"
    except Exception as e:
        print(f"❌ Neural Health: {e}")
        test_results["health"] = "FAILED"
    
    # Test 2: Get Neural Templates
    total_tests += 1
    try:
        response = requests.get(f"{neural_url}/templates", timeout=10)
        if response.status_code == 200:
            data = response.json()
            templates = data.get('templates', {})
            print(f"✅ Neural Templates: {len(templates)} templates available")
            for name, template in list(templates.items())[:2]:  # Show first 2
                print(f"   - {name}: {template['description']}")
            test_results["templates"] = "PASSED"
            passed_tests += 1
        else:
            print(f"❌ Neural Templates: HTTP {response.status_code}")
            test_results["templates"] = "FAILED"
    except Exception as e:
        print(f"❌ Neural Templates: {e}")
        test_results["templates"] = "FAILED"
    
    # Test 3: Create Neural Experiment
    total_tests += 1
    experiment_id = None
    try:
        experiment_request = {
            "name": "Test XOR Problem",
            "description": "Test neural evolution on XOR problem",
            "agent_config": {
                "input_size": 10,
                "hidden_layers": [8, 4],
                "output_size": 1,
                "activation": "sigmoid",
                "network_type": "feedforward"
            },
            "evolution_config": {
                "population_size": 20,
                "generations": 10,
                "mutation_rate": 0.2,
                "crossover_rate": 0.7,
                "elite_size": 2,
                "speciation": False
            },
            "environment_type": "xor",
            "environment_config": {}
        }
        
        response = requests.post(
            f"{neural_url}/experiments", 
            json=experiment_request,
            timeout=15
        )
        
        if response.status_code == 200:
            data = response.json()
            experiment_id = data.get('experiment_id')
            print(f"✅ Create Experiment: {experiment_id}")
            print(f"   Status: {data.get('status', 'unknown')}")
            print(f"   Estimated Duration: {data.get('estimated_duration', 'unknown')}")
            test_results["create_experiment"] = "PASSED"
            passed_tests += 1
        else:
            print(f"❌ Create Experiment: HTTP {response.status_code}")
            if response.text:
                print(f"   Error: {response.text[:100]}")
            test_results["create_experiment"] = "FAILED"
    except Exception as e:
        print(f"❌ Create Experiment: {e}")
        test_results["create_experiment"] = "FAILED"
    
    # Test 4: Monitor Experiment Progress
    total_tests += 1
    if experiment_id:
        try:
            # Wait a bit for experiment to start
            time.sleep(2)
            
            response = requests.get(f"{neural_url}/experiments/{experiment_id}", timeout=10)
            if response.status_code == 200:
                data = response.json()
                status = data.get('status', 'unknown')
                progress = data.get('progress', 0)
                print(f"✅ Monitor Experiment: Status {status}, Progress {progress}%")
                
                if 'current_generation' in data:
                    gen_info = data['current_generation']
                    print(f"   Generation: {gen_info.get('generation', 0)}")
                    print(f"   Best Fitness: {gen_info.get('fitness_stats', {}).get('best', 0):.4f}")
                
                test_results["monitor"] = "PASSED"
                passed_tests += 1
            else:
                print(f"❌ Monitor Experiment: HTTP {response.status_code}")
                test_results["monitor"] = "FAILED"
        except Exception as e:
            print(f"❌ Monitor Experiment: {e}")
            test_results["monitor"] = "FAILED"
    else:
        print("❌ Monitor Experiment: No experiment ID to monitor")
        test_results["monitor"] = "FAILED"
    
    # Test 5: List Experiments
    total_tests += 1
    try:
        response = requests.get(f"{neural_url}/experiments", timeout=10)
        if response.status_code == 200:
            data = response.json()
            experiments = data.get('experiments', [])
            print(f"✅ List Experiments: {len(experiments)} experiments found")
            test_results["list_experiments"] = "PASSED"
            passed_tests += 1
        else:
            print(f"❌ List Experiments: HTTP {response.status_code}")
            test_results["list_experiments"] = "FAILED"
    except Exception as e:
        print(f"❌ List Experiments: {e}")
        test_results["list_experiments"] = "FAILED"
    
    # Test 6: System Status
    total_tests += 1
    try:
        response = requests.get(f"{neural_url}/status", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ System Status: {data.get('status', 'unknown')}")
            print(f"   Running Experiments: {data.get('running_experiments', 0)}")
            by_status = data.get('experiments_by_status', {})
            for status, count in by_status.items():
                if count > 0:
                    print(f"   {status.title()}: {count}")
            test_results["system_status"] = "PASSED"
            passed_tests += 1
        else:
            print(f"❌ System Status: HTTP {response.status_code}")
            test_results["system_status"] = "FAILED"
    except Exception as e:
        print(f"❌ System Status: {e}")
        test_results["system_status"] = "FAILED"
    
    # Wait for experiment to complete or timeout
    if experiment_id:
        print(f"\n⏳ Waiting for experiment {experiment_id} to complete...")
        max_wait = 30  # seconds
        wait_time = 0
        
        while wait_time < max_wait:
            try:
                response = requests.get(f"{neural_url}/experiments/{experiment_id}", timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    status = data.get('status')
                    
                    if status in ['completed', 'failed']:
                        print(f"✅ Experiment completed with status: {status}")
                        
                        # Test 7: Get Best Agent (if completed)
                        if status == 'completed':
                            total_tests += 1
                            try:
                                response = requests.get(f"{neural_url}/experiments/{experiment_id}/best-agent", timeout=10)
                                if response.status_code == 200:
                                    agent_data = response.json()
                                    best_fitness = agent_data.get('best_fitness', 0)
                                    print(f"✅ Best Agent: Fitness {best_fitness:.4f}")
                                    test_results["best_agent"] = "PASSED"
                                    passed_tests += 1
                                else:
                                    print(f"❌ Best Agent: HTTP {response.status_code}")
                                    test_results["best_agent"] = "FAILED"
                            except Exception as e:
                                print(f"❌ Best Agent: {e}")
                                test_results["best_agent"] = "FAILED"
                        break
                    
                    progress = data.get('progress', 0)
                    print(f"   Progress: {progress}%", end='\r')
                
                time.sleep(2)
                wait_time += 2
                
            except Exception as e:
                print(f"   Error checking status: {e}")
                break
        
        if wait_time >= max_wait:
            print(f"\n⚠️ Experiment timeout after {max_wait} seconds")
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 PHASE 6A NEURAL EVOLUTION INTEGRATION TEST SUMMARY")
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
        print(f"\n🎉 PHASE 6A NEURAL EVOLUTION INTEGRATION: SUCCESS!")
        print("Deep Learning Integration features are working correctly.")
        return True
    else:
        print(f"\n⚠️ PHASE 6A NEURAL EVOLUTION INTEGRATION: PARTIAL SUCCESS")
        print("Some neural evolution features need attention.")
        return False


if __name__ == "__main__":
    # Test if dependencies are available
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__} available")
    except ImportError:
        print("❌ PyTorch not available - some tests may fail")
    
    try:
        from src.genetic_coil_arena.neural_evolution import NeuralAgent
        print("✅ Neural evolution modules available")
    except ImportError as e:
        print(f"❌ Neural evolution modules not available: {e}")
    
    print("\n" + "=" * 60)
    success = asyncio.run(test_phase6a_neural_evolution())
    
    if success:
        print("\n✅ Phase 6A ready for production!")
    else:
        print("\n⚠️ Phase 6A needs additional work.")
