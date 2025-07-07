"""
Production-ready FastAPI server for Genetic Coil Arena.
Phase 3 Final: Complete integration with authentication, database, analytics, and experiment management.
"""

from typing import Optional
import asyncio
import uvicorn
import logging
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from fastapi.security import HTTPBearer
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager

# Import all modules
try:
    from .api_real import router as experiment_router
    from .auth_routes import auth_router
    from .visualization import viz_router
    from .templates_routes import templates_router
    from .templates_ui import templates_ui_router
    from .batch_routes import batch_router
    from .batch_ui import batch_ui_router
    from .comparison_routes import router as comparison_router
    from .scheduling_routes import router as scheduling_router
    from .visualization_3d_routes import router as viz_3d_router
    from .visualization_3d_ui_router import router as viz_3d_ui_router
    # Phase 4C: Collaboration Features
    from .team_management_routes import router as team_router
    from .team_management_ui_router import router as team_ui_router
    from .notifications_routes import router as notifications_router
    from .version_control_routes import router as version_control_router
    # Phase 4D: Performance & Scalability
    from .performance_routes import router as performance_router
    from .scalability import ScalabilityManager
    from .performance_optimizer import PerformanceOptimizer
    # Phase 4E: Mobile & PWA
    from .mobile_routes import router as mobile_router
    # Phase 5: Advanced Analytics & AI Integration
    from .analytics_routes import analytics_router
    from .analytics import analytics_router as basic_analytics_router
    from .database import db
    from .auth import get_current_user
    from ..config import Config
    COMPONENTS_AVAILABLE = True
    AUTH_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Some components not available: {e}")
    COMPONENTS_AVAILABLE = False
    AUTH_AVAILABLE = False
    # Define fallback dependencies
    def get_current_user():
        return {"id": "test_user", "username": "test"}
    # Define a minimal Config class as fallback
    class Config:
        def __init__(self):
            pass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

security = HTTPBearer()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle application startup and shutdown."""
    # Startup
    logger.info("🚀 Starting Genetic Coil Arena Production Server")
    
    if COMPONENTS_AVAILABLE:
        # Initialize database
        try:
            await db.initialize()
            logger.info("✅ Database initialized successfully")
        except Exception as e:
            logger.error(f"❌ Database initialization failed: {e}")
        
        # Initialize Performance & Scalability components
        try:
            # Start background monitoring and workers
            logger.info("🔧 Initializing Performance & Scalability components...")
            # The components are initialized in ProductionServer.__init__()
            logger.info("✅ Performance & Scalability components ready")
        except Exception as e:
            logger.error(f"❌ Performance/Scalability initialization failed: {e}")
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down Genetic Coil Arena")
    if COMPONENTS_AVAILABLE:
        try:
            await db.close()
            logger.info("✅ Database connection closed")
        except Exception as e:
            logger.error(f"❌ Database shutdown error: {e}")


class ProductionServer:
    """Production-ready web server with full feature integration."""
    
    def __init__(self, host: str = "localhost", port: int = 8000, config = None):
        self.host = host
        self.port = port
        self.config = config or Config()
        
        # Initialize Phase 4D components if available
        if COMPONENTS_AVAILABLE:
            try:
                self.performance_optimizer = PerformanceOptimizer()
                self.scalability_manager = ScalabilityManager()
                logger.info("✅ Performance and Scalability components initialized")
            except Exception as e:
                logger.error(f"❌ Failed to initialize Performance/Scalability: {e}")
                self.performance_optimizer = None
                self.scalability_manager = None
        else:
            self.performance_optimizer = None
            self.scalability_manager = None
        
        # Create FastAPI app with lifespan management
        self.app = FastAPI(
            title="Genetic Coil Arena",
            description="Production-ready evolutionary computing platform with Performance & Scalability",
            version="1.0.0",
            lifespan=lifespan
        )
        
        # Configure CORS for production
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=[
                "http://localhost:8000",
                "http://127.0.0.1:8000",
                # Add your production domains here
            ],
            allow_credentials=True,
            allow_methods=["GET", "POST", "PUT", "DELETE"],
            allow_headers=["*"],
        )
        
        # Include all routers
        self._setup_routes()
    
    def _setup_routes(self):
        """Set up all application routes."""
        
        # Root route with authentication check
        @self.app.get("/", response_class=HTMLResponse)
        async def root():
            """Main landing page with authentication redirect."""
            html_content = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>Genetic Coil Arena</title>
                <meta charset="utf-8">
                <meta name="viewport" content="width=device-width, initial-scale=1">
                <style>
                    body { 
                        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
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
                        max-width: 600px;
                    }
                    .logo { font-size: 3rem; margin-bottom: 1rem; }
                    .subtitle { font-size: 1.2rem; margin-bottom: 2rem; opacity: 0.9; }
                    .buttons { display: flex; gap: 1rem; justify-content: center; flex-wrap: wrap; }
                    .btn { 
                        padding: 12px 24px; 
                        border: none; 
                        border-radius: 8px; 
                        background: white; 
                        color: #333; 
                        text-decoration: none; 
                        font-weight: 600; 
                        transition: transform 0.2s;
                    }
                    .btn:hover { transform: translateY(-2px); }
                    .btn-primary { background: #4CAF50; color: white; }
                    .features { 
                        margin-top: 2rem; 
                        text-align: left; 
                        max-width: 500px; 
                        margin-left: auto; 
                        margin-right: auto; 
                    }
                    .feature { margin: 1rem 0; padding: 0.5rem 0; }
                    .feature::before { content: "✅ "; margin-right: 0.5rem; }
                </style>
            </head>
            <body>
                <div class="container">
                    <div class="logo">🧬 Genetic Coil Arena</div>
                    <div class="subtitle">Professional Evolutionary Computing Platform</div>                    <div class="buttons">
                        <a href="/auth/login" class="btn btn-primary">Login</a>
                        <a href="/auth/register" class="btn">Register</a>
                        <a href="/templates-ui" class="btn">🧬 Templates</a>
                        <a href="/batch-ui" class="btn">⚡ Batch Processing</a>
                        <a href="/api/comparison/dashboard" class="btn">📊 Compare Experiments</a>
                        <a href="/api/scheduling/dashboard" class="btn">📅 Schedule Experiments</a>
                        <a href="/visualization3d" class="btn">🌌 3D Visualizations</a>
                        <a href="/teams/ui" class="btn">🤝 Team Management</a>
                        <a href="/dashboard" class="btn">Dashboard</a>
                        <a href="/analytics" class="btn">Analytics</a>
                    </div>
                    
                    <div class="features">                        <div class="feature">Real-time experiment monitoring</div>
                        <div class="feature">Advanced analytics and visualization</div>
                        <div class="feature">Team collaboration and sharing</div>
                        <div class="feature">Experiment version control</div>
                        <div class="feature">Email and SMS notifications</div>
                        <div class="feature">3D visualizations and streaming</div>
                        <div class="feature">Persistent experiment storage</div>
                        <div class="feature">User authentication and management</div>
                        <div class="feature">WebSocket live updates</div>
                        <div class="feature">Production-ready deployment</div>
                    </div>
                </div>
            </body>
            </html>
            """
            return HTMLResponse(content=html_content)
          # Health check endpoint
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint for monitoring."""
            db_status = False
            if COMPONENTS_AVAILABLE:
                try:
                    db_status = hasattr(db, 'is_connected') and db.is_connected()
                except:
                    db_status = False
            
            return {
                "status": "healthy",
                "version": "1.0.0",                "components": {
                    "database": db_status,
                    "authentication": AUTH_AVAILABLE,
                    "experiments": True,
                    "analytics": COMPONENTS_AVAILABLE,
                    "team_management": COMPONENTS_AVAILABLE,
                    "notifications": COMPONENTS_AVAILABLE,
                    "version_control": COMPONENTS_AVAILABLE,
                    "3d_visualization": COMPONENTS_AVAILABLE
                }
            }
        
        # Phase 4E: PWA Support Routes
        @self.app.get("/manifest.json")
        async def serve_manifest():
            """Serve PWA manifest file."""
            import os
            manifest_path = os.path.join(os.path.dirname(__file__), "static", "manifest.json")
            if os.path.exists(manifest_path):
                with open(manifest_path, 'r') as f:
                    manifest_content = f.read()
                return Response(content=manifest_content, media_type="application/json")
            else:
                # Return default manifest if file doesn't exist
                default_manifest = {
                    "name": "Genetic Coil Arena",
                    "short_name": "CoilArena",
                    "start_url": "/",
                    "display": "standalone",
                    "theme_color": "#667eea",
                    "background_color": "#ffffff"
                }
                return JSONResponse(content=default_manifest)
        
        @self.app.get("/sw.js")
        async def serve_service_worker():
            """Serve service worker file."""
            import os
            sw_path = os.path.join(os.path.dirname(__file__), "static", "js", "sw.js")
            if os.path.exists(sw_path):
                with open(sw_path, 'r') as f:
                    sw_content = f.read()
                return Response(content=sw_content, media_type="application/javascript")
            else:
                return Response(content="// Service worker not available", media_type="application/javascript")
        
        @self.app.get("/offline.html")
        async def serve_offline_page():
            """Serve offline page."""
            import os
            offline_path = os.path.join(os.path.dirname(__file__), "static", "offline.html")
            if os.path.exists(offline_path):
                with open(offline_path, 'r') as f:
                    offline_content = f.read()
                return HTMLResponse(content=offline_content)
            else:
                return HTMLResponse(content="<h1>Offline</h1><p>You are currently offline.</p>")

        # Protected dashboard route
        if AUTH_AVAILABLE:
            @self.app.get("/dashboard-protected")
            async def protected_dashboard(current_user: dict = Depends(get_current_user)):
                """Protected dashboard that requires authentication."""
                return RedirectResponse(url="/dashboard")
        else:
            @self.app.get("/dashboard-protected")
            async def protected_dashboard():
                """Protected dashboard (fallback without auth)."""
                return RedirectResponse(url="/dashboard")
        
        if COMPONENTS_AVAILABLE:
            # Include all feature routers
            self.app.include_router(auth_router)
            self.app.include_router(experiment_router)
            self.app.include_router(basic_analytics_router)
            self.app.include_router(analytics_router)
            self.app.include_router(viz_router)
            self.app.include_router(templates_router)
            self.app.include_router(templates_ui_router)
            self.app.include_router(batch_router)
            self.app.include_router(batch_ui_router)
            self.app.include_router(comparison_router)
            self.app.include_router(scheduling_router)
            self.app.include_router(viz_3d_router)
            self.app.include_router(viz_3d_ui_router)
            # Phase 4C: Collaboration Features
            self.app.include_router(team_router)
            self.app.include_router(team_ui_router)
            self.app.include_router(notifications_router)
            self.app.include_router(version_control_router)
            # Phase 4D: Performance & Scalability
            self.app.include_router(performance_router)
            # Phase 4E: Mobile & PWA
            self.app.include_router(mobile_router)
            
            logger.info("✅ All routes integrated successfully - Phase 4C Collaboration + Phase 4D Performance + Phase 4E Mobile + Phase 5 Analytics Features included")
        else:
            logger.warning("⚠️ Running with limited functionality due to missing components")
    
    def run(self, **kwargs):
        """Run the production server."""
        # Production server configuration
        server_config = {
            "app": self.app,
            "host": self.host,
            "port": self.port,
            "log_level": "info",
            "access_log": True,
            "workers": 1,  # For development; increase for production
            **kwargs
        }
        
        logger.info(f"🚀 Starting server on http://{self.host}:{self.port}")
        uvicorn.run(**server_config)
    
    async def run_async(self, **kwargs):
        """Run the server asynchronously."""
        server_config = {
            "app": self.app,
            "host": self.host,
            "port": self.port,
            "log_level": "info",
            **kwargs
        }
        
        config = uvicorn.Config(**server_config)
        server = uvicorn.Server(config)
        await server.serve()


def create_production_app(config = None) -> FastAPI:
    """Factory function to create production FastAPI app."""
    server = ProductionServer(config=config)
    return server.app


def run_production_server(host: str = "localhost", port: int = 8000, config = None):
    """Run the production server."""
    server = ProductionServer(host=host, port=port, config=config)
    server.run()


if __name__ == "__main__":
    # Run production server
    run_production_server()
