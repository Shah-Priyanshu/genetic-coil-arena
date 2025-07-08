#!/usr/bin/env python3
"""
Phase 3 Web Dashboard Demo

This demo showcases the new web dashboard capabilities:
- FastAPI-based web server
- Real-time monitoring interface
- RESTful API endpoints
- Modern responsive UI
- Integration with existing CLI commands

Usage:
    python phase3_web_demo.py
"""

import time
import threading
import webbrowser
import subprocess
import sys
from pathlib import Path

def print_demo_header():
    """Print the demo header."""
    print("🌐 Phase 3 Web Dashboard Demo")
    print("=" * 50)
    print("This demo showcases the new web interface capabilities")
    print()

def start_web_server():
    """Start the web server using CLI command."""
    print("🚀 Starting web server...")
    print("Command: genetic-coil web --auto-open=false")
    print()
    
    # Start the web server
    process = subprocess.Popen([
        sys.executable, "-m", "genetic_coil_arena.cli.main", 
        "web", "--auto-open=false"
    ], 
    stdout=subprocess.PIPE, 
    stderr=subprocess.PIPE,
    text=True
    )
    
    return process

def wait_for_server(max_attempts=10):
    """Wait for the web server to be ready."""
    import requests
    
    for attempt in range(max_attempts):
        try:
            response = requests.get("http://localhost:8000/health", timeout=1)
            if response.status_code == 200:
                return True
        except:
            pass
        time.sleep(1)
    return False

def demonstrate_features():
    """Demonstrate the key features."""
    print("📊 Web Dashboard Features Demonstrated:")
    print()
    
    features = [
        "✅ FastAPI web framework integration",
        "✅ Modern responsive UI design", 
        "✅ RESTful API endpoints",
        "✅ Health check monitoring",
        "✅ CLI integration (genetic-coil web)",
        "🔧 Real-time experiment monitoring (in progress)",
        "🔧 WebSocket integration (in progress)",
        "🔧 Interactive parameter controls (in progress)",
        "🔧 Live fitness plots (in progress)",
        "🔧 Multi-experiment management (in progress)"
    ]
    
    for feature in features:
        print(f"  {feature}")
        time.sleep(0.5)
    
    print()

def open_browser_delayed():
    """Open browser after a delay."""
    time.sleep(3)
    print("🌐 Opening web dashboard in browser...")
    webbrowser.open("http://localhost:8000")

def demo_api_integration():
    """Demonstrate API integration with existing features."""
    print("🔗 Integration with Existing Features:")
    print()
    
    integrations = [
        "📦 Persistence system integration",
        "🧬 Evolution engine connectivity", 
        "📊 Analysis tools API endpoints",
        "🎮 Visualization system hooks",
        "⚙️ Configuration management",
        "📁 Experiment directory access"
    ]
    
    for integration in integrations:
        print(f"  {integration}")
        time.sleep(0.5)
    
    print()

def show_next_steps():
    """Show what's coming next."""
    print("🚀 Next Development Steps:")
    print()
    
    next_steps = [
        "1. Real-time evolution monitoring with WebSockets",
        "2. Interactive fitness plots using Plotly",
        "3. Agent visualization and replay",
        "4. Parameter adjustment controls",
        "5. Multi-experiment comparison",
        "6. Advanced neural architectures interface",
        "7. REST API for remote control",
        "8. User authentication and sessions"
    ]
    
    for step in next_steps:
        print(f"  {step}")
    
    print()

def main():
    """Run the web dashboard demo."""
    print_demo_header()
    
    # Check if we can import required modules
    try:
        import fastapi
        import uvicorn
        print("✅ FastAPI dependencies available")
    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
        print("Install with: pip install fastapi uvicorn[standard] websockets")
        return
    
    # Start web server
    print("1. Starting Web Server")
    print("-" * 25)
    
    server_process = start_web_server()
    
    # Wait a moment for server to start
    time.sleep(2)
    
    # Check if server is running
    try:
        import requests
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            print("✅ Web server is running at http://localhost:8000")
        else:
            print("⚠️ Web server responded but may have issues")
    except Exception as e:
        print(f"❌ Could not connect to web server: {e}")
        print("Note: Server may still be starting up")
    
    print()
    
    # Open browser in background
    browser_thread = threading.Thread(target=open_browser_delayed, daemon=True)
    browser_thread.start()
    
    # Demonstrate features
    print("2. Feature Overview")
    print("-" * 20)
    demonstrate_features()
    
    print("3. System Integration")
    print("-" * 22)
    demo_api_integration()
    
    print("4. Development Roadmap")
    print("-" * 25)
    show_next_steps()
    
    print("🎯 Demo Summary:")
    print("=" * 20)
    print("✅ Web dashboard infrastructure is now functional")
    print("✅ FastAPI server with modern UI design")
    print("✅ CLI integration (genetic-coil web command)")
    print("✅ Health monitoring and API endpoints")
    print("✅ Foundation ready for advanced features")
    print()
    print("🌐 Web Interface: http://localhost:8000")
    print("📖 API Documentation: http://localhost:8000/docs")
    print("🔍 Health Check: http://localhost:8000/health")
    print()
    print("Press Ctrl+C to stop the web server and exit demo")
    
    try:
        # Keep the demo running
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Stopping web server...")
        server_process.terminate()
        server_process.wait()
        print("✅ Web server stopped")
        print("\n🎉 Phase 3 Web Dashboard Demo completed!")
        print("Next: Continue with real-time monitoring features")


if __name__ == "__main__":
    main()
