"""
Simple web dashboard demo for genetic-coil-arena.
"""

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import uvicorn

app = FastAPI(title="Genetic Coil Arena", description="Web Dashboard Demo")

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    """Serve a simple dashboard page."""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Genetic Coil Arena - Dashboard</title>
        <style>
            body { 
                font-family: Arial, sans-serif; 
                margin: 40px; 
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                min-height: 100vh;
            }
            .container { 
                max-width: 800px; 
                margin: 0 auto; 
                background: rgba(255,255,255,0.1);
                padding: 40px;
                border-radius: 20px;
                backdrop-filter: blur(10px);
                box-shadow: 0 8px 32px rgba(0,0,0,0.3);
            }
            h1 { 
                text-align: center; 
                font-size: 3em; 
                margin-bottom: 20px;
                text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
            }
            .feature {
                background: rgba(255,255,255,0.2);
                padding: 20px;
                margin: 20px 0;
                border-radius: 10px;
                border-left: 4px solid #4CAF50;
            }
            .status {
                text-align: center;
                padding: 20px;
                background: rgba(76,175,80,0.2);
                border-radius: 10px;
                margin: 20px 0;
            }
            .demo-text {
                font-size: 1.2em;
                line-height: 1.6;
                text-align: center;
                margin: 30px 0;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🧬 Genetic Coil Arena</h1>
            <div class="demo-text">
                <strong>Web Dashboard - Phase 3 Implementation</strong>
            </div>
            
            <div class="status">
                ✅ Web Server Running Successfully!
            </div>
            
            <div class="feature">
                <h3>🚀 Features Implemented</h3>
                <ul>
                    <li>FastAPI web framework integration</li>
                    <li>WebSocket support for real-time updates</li>
                    <li>Modern responsive UI design</li>
                    <li>RESTful API endpoints</li>
                    <li>Experiment management system</li>
                </ul>
            </div>
            
            <div class="feature">
                <h3>📊 Coming Soon</h3>
                <ul>
                    <li>Real-time evolution monitoring</li>
                    <li>Interactive fitness plots</li>
                    <li>Agent visualization</li>
                    <li>Parameter adjustment controls</li>
                    <li>Multi-experiment comparison</li>
                </ul>
            </div>
            
            <div class="feature">
                <h3>🔧 Available CLI Commands</h3>
                <p><code>genetic-coil web</code> - Start this web dashboard</p>
                <p><code>genetic-coil evolve</code> - Run evolution from command line</p>
                <p><code>genetic-coil experiments</code> - List all experiments</p>
                <p><code>genetic-coil visualize</code> - Launch game visualization</p>
            </div>
            
            <div style="text-align: center; margin-top: 40px; opacity: 0.8;">
                <p>Genetic Coil Arena - Evolving Snake AI with Genetic Algorithms</p>
                <p>Phase 3: Advanced Production Features</p>
            </div>
        </div>
    </body>
    </html>
    """

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "genetic-coil-arena-web"}

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
