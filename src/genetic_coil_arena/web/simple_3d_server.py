"""
Phase 4B: Simple 3D Visualization Server
Basic FastAPI server for testing 3D visualization features.
"""

import uvicorn
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import only the 3D visualization router
try:
    from .visualization_3d_ui_router import router as viz_3d_ui_router
    UI_AVAILABLE = True
except ImportError as e:
    logger.warning(f"3D UI not available: {e}")
    UI_AVAILABLE = False

def create_simple_app() -> FastAPI:
    """Create a simple FastAPI app for 3D visualization testing."""
    
    app = FastAPI(
        title="Genetic Coil Arena - 3D Visualization",
        description="Phase 4B: 3D Visualization Testing Server",
        version="4B.0.0"
    )
    
    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE"],
        allow_headers=["*"],
    )
    
    @app.get("/", response_class=HTMLResponse)
    async def root():
        """Main landing page."""
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Genetic Coil Arena - Phase 4B</title>
            <style>
                body { 
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; 
                    margin: 0; 
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    min-height: 100vh; 
                    display: flex; 
                    align-items: center; 
                    justify-content: center; 
                    color: white;
                }
                .container { 
                    text-align: center; 
                    background: rgba(255,255,255,0.1); 
                    padding: 3rem; 
                    border-radius: 20px; 
                    backdrop-filter: blur(10px);
                }
                .logo { font-size: 3rem; margin-bottom: 1rem; }
                .subtitle { font-size: 1.2rem; margin-bottom: 2rem; }
                .btn { 
                    padding: 12px 24px; 
                    margin: 0.5rem; 
                    border: none; 
                    border-radius: 8px; 
                    background: white; 
                    color: #333; 
                    text-decoration: none; 
                    display: inline-block;
                    font-weight: 600; 
                    transition: transform 0.2s;
                }
                .btn:hover { transform: translateY(-2px); }
                .btn-primary { background: #4CAF50; color: white; }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="logo">🌌 Phase 4B: 3D Visualization Lab</div>
                <div class="subtitle">Advanced 3D Visualizations for Evolutionary Computing</div>
                
                <div>
                    <a href="/visualization3d/" class="btn btn-primary">🚀 Launch 3D Visualizations</a>
                    <a href="/health" class="btn">📊 Health Check</a>
                </div>
                
                <div style="margin-top: 2rem; text-align: left; max-width: 500px;">
                    <h3>✅ Phase 4B Features</h3>
                    <ul>
                        <li>✅ 3D Visualization Engine</li>
                        <li>✅ Three.js Integration</li>
                        <li>✅ Interactive UI Dashboard</li>
                        <li>🚧 Real-time Streaming (API Ready)</li>
                        <li>🚧 WebSocket Support (API Ready)</li>
                        <li>🚧 Animation Export (API Ready)</li>
                    </ul>
                </div>
            </div>
        </body>
        </html>
        """
    
    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {
            "status": "healthy",
            "version": "Phase 4B.0.0",
            "features": {
                "3d_visualization_ui": UI_AVAILABLE,
                "three_js_integration": True,
                "interactive_dashboard": True
            }
        }
    
    # Include 3D visualization UI router if available
    if UI_AVAILABLE:
        app.include_router(viz_3d_ui_router)
        logger.info("✅ 3D Visualization UI integrated")
    else:
        logger.warning("⚠️ 3D Visualization UI not available")
    
    return app

def run_simple_server(host: str = "localhost", port: int = 8000):
    """Run the simple 3D visualization server."""
    app = create_simple_app()
    
    logger.info(f"🚀 Starting Phase 4B 3D Visualization Server on http://{host}:{port}")
    logger.info("🌌 Navigate to /visualization3d/ for 3D visualizations")
    
    uvicorn.run(app, host=host, port=port, log_level="info")

if __name__ == "__main__":
    run_simple_server()
