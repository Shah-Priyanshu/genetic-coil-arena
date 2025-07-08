"""
Genetic Coil Arena - Unified Production Server
Complete evolutionary computing platform with all integrated features.
"""

import asyncio
import uvicorn
import logging
import os
from pathlib import Path
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Core components
from .core.database import DatabaseManager
from .core.file_storage import FileStorageManager
from .core.security_middleware import SecurityMiddleware, InputValidationMiddleware

# Import all route modules - each with fallback handling
# Core API routes
try:
    from .routes.experiments_routes import router as experiment_router
except ImportError:
    experiment_router = None
    logger.warning("experiments_routes not available")

try:
    from .routes.auth_routes import auth_router
except ImportError:
    auth_router = None
    logger.warning("auth_routes not available")

# Analytics and Visualization
try:
    from .routes.analytics_routes import analytics_router
except ImportError:
    analytics_router = None
    logger.warning("analytics_routes not available")

try:
    from .visualization import viz_router
except ImportError:
    viz_router = None
    logger.warning("visualization not available")

# Neural Evolution
try:
    from .routes.neural_routes import neural_router
except ImportError:
    neural_router = None
    logger.warning("neural_routes not available")

# Templates and Batch Processing
try:
    from .routes.templates_routes import templates_router
except ImportError:
    templates_router = None
    logger.warning("templates_routes not available")

try:
    from .ui.templates_ui import templates_ui_router
except ImportError:
    templates_ui_router = None
    logger.warning("templates_ui not available")

try:
    from .routes.batch_routes import batch_router
except ImportError:
    batch_router = None
    logger.warning("batch_routes not available")

try:
    from .ui.batch_ui import batch_ui_router
except ImportError:
    batch_ui_router = None
    logger.warning("batch_ui not available")

# Advanced Features
try:
    from .routes.comparison_routes import router as comparison_router
except ImportError:
    comparison_router = None
    logger.warning("comparison_routes not available")

try:
    from .routes.scheduling_routes import router as scheduling_router
except ImportError:
    scheduling_router = None
    logger.warning("scheduling_routes not available")

try:
    from .routes.team_management_routes import router as team_router
except ImportError:
    team_router = None
    logger.warning("team_management_routes not available")

try:
    from .ui.team_management_ui_router import router as team_ui_router
except ImportError:
    team_ui_router = None
    logger.warning("team_management_ui_router not available")

try:
    from .routes.notifications_routes import router as notifications_router
except ImportError:
    notifications_router = None
    logger.warning("notifications_routes not available")

try:
    from .routes.version_control_routes import router as version_control_router
except ImportError:
    version_control_router = None
    logger.warning("version_control_routes not available")

# Performance and Mobile
try:
    from .routes.performance_routes import router as performance_router
except ImportError:
    performance_router = None
    logger.warning("performance_routes not available")

try:
    from .routes.mobile_routes import router as mobile_router
except ImportError:
    mobile_router = None
    logger.warning("mobile_routes not available")

# 3D Visualization
try:
    from .routes.visualization_3d_routes import router as viz_3d_router
except ImportError:
    viz_3d_router = None
    logger.warning("visualization_3d_routes not available")

try:
    from .ui.visualization_3d_ui_router import router as viz_3d_ui_router
except ImportError:
    viz_3d_ui_router = None
    logger.warning("visualization_3d_ui_router not available")

# Optional performance components
try:
    from .scalability import ScalabilityManager
except ImportError:
    ScalabilityManager = None
    logger.warning("scalability not available")

try:
    from .performance_optimizer import PerformanceOptimizer
except ImportError:
    PerformanceOptimizer = None
    logger.warning("performance_optimizer not available")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    logger.info("🚀 Starting Genetic Coil Arena Unified Server")
    
    # Initialize core components
    try:
        # Initialize file storage
        file_storage_manager = FileStorageManager()
        app.state.file_storage = file_storage_manager
        logger.info("✅ File storage initialized")
        
        # Initialize database
        db = DatabaseManager()
        app.state.db = db
        logger.info("✅ Database initialized")
        
        # Initialize performance components
        if ScalabilityManager:
            app.state.scalability_manager = ScalabilityManager()
        if PerformanceOptimizer:
            app.state.performance_optimizer = PerformanceOptimizer()
            logger.info("✅ Performance components initialized")
        
        logger.info("🎉 All systems ready - Genetic Coil Arena is operational!")
        
    except Exception as e:
        logger.error(f"❌ Startup error: {e}")
        raise
    
    yield
    
    # Cleanup
    logger.info("🔄 Shutting down Genetic Coil Arena...")
    try:
        if hasattr(app.state, 'scalability_manager') and hasattr(app.state.scalability_manager, 'shutdown'):
            try:
                await app.state.scalability_manager.shutdown()
            except Exception:
                # If shutdown is not async, try sync version
                if hasattr(app.state.scalability_manager, 'shutdown'):
                    app.state.scalability_manager.shutdown()
        logger.info("✅ Graceful shutdown complete")
    except Exception as e:
        logger.error(f"❌ Shutdown error: {e}")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    
    app = FastAPI(
        title="Genetic Coil Arena",
        description="Complete evolutionary computing platform with neural evolution, analytics, and visualization",
        version="6.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan
    )
    
    # Configure CORS with environment-based origins
    allowed_origins = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:8080").split(",")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["*"],
    )
    
    # Add security middleware
    app.add_middleware(SecurityMiddleware)
    app.add_middleware(InputValidationMiddleware)
    
    # Mount static files
    static_path = Path(__file__).parent / "static"
    if static_path.exists():
        app.mount("/static", StaticFiles(directory=str(static_path)), name="static")
        logger.info("✅ Static files mounted")
    
    # Include all available routers
    routers_included = 0
    
    # Core experiment management
    if experiment_router:
        app.include_router(experiment_router, prefix="/api", tags=["experiments"])
        routers_included += 1
    if auth_router:
        app.include_router(auth_router, prefix="/api", tags=["authentication"])
        routers_included += 1
    
    # Analytics and Neural Evolution
    if analytics_router:
        app.include_router(analytics_router, prefix="/api", tags=["analytics"])
        routers_included += 1
    if neural_router:
        app.include_router(neural_router, prefix="/api", tags=["neural-evolution"])
        routers_included += 1
    
    # Visualization
    if viz_router:
        app.include_router(viz_router, prefix="/api", tags=["visualization"])
        routers_included += 1
    if viz_3d_router:
        app.include_router(viz_3d_router, prefix="/api", tags=["3d-visualization"])
        routers_included += 1
    if viz_3d_ui_router:
        app.include_router(viz_3d_ui_router, prefix="/ui", tags=["3d-ui"])
        routers_included += 1
    
    # Templates and Batch Processing
    if templates_router:
        app.include_router(templates_router, prefix="/api", tags=["templates"])
        routers_included += 1
    if templates_ui_router:
        app.include_router(templates_ui_router, prefix="/ui", tags=["templates-ui"])
        routers_included += 1
    if batch_router:
        app.include_router(batch_router, prefix="/api", tags=["batch"])
        routers_included += 1
    if batch_ui_router:
        app.include_router(batch_ui_router, prefix="/ui", tags=["batch-ui"])
        routers_included += 1
    
    # Collaboration and Team Management
    if team_router:
        app.include_router(team_router, prefix="/api", tags=["teams"])
        routers_included += 1
    if team_ui_router:
        app.include_router(team_ui_router, prefix="/ui", tags=["teams-ui"])
        routers_included += 1
    if notifications_router:
        app.include_router(notifications_router, prefix="/api", tags=["notifications"])
        routers_included += 1
    if version_control_router:
        app.include_router(version_control_router, prefix="/api", tags=["version-control"])
        routers_included += 1
    
    # Advanced Features
    if comparison_router:
        app.include_router(comparison_router, prefix="/api", tags=["comparison"])
        routers_included += 1
    if scheduling_router:
        app.include_router(scheduling_router, prefix="/api", tags=["scheduling"])
        routers_included += 1
    if performance_router:
        app.include_router(performance_router, prefix="/api", tags=["performance"])
        routers_included += 1
    if mobile_router:
        app.include_router(mobile_router, prefix="/api", tags=["mobile"])
        routers_included += 1
    
    logger.info(f"✅ {routers_included} API routes integrated")
    
    # Health check endpoint
    @app.get("/health")
    async def health_check():
        """System health check."""
        return {
            "status": "healthy",
            "service": "Genetic Coil Arena",
            "version": "6.0.0",
            "routers_included": routers_included,
            "features": {
                "experiments": experiment_router is not None,
                "analytics": analytics_router is not None,
                "neural_evolution": neural_router is not None,
                "visualization": viz_router is not None,
                "collaboration": team_router is not None,
                "mobile": mobile_router is not None
            }
        }
    
    # Root endpoint - redirect to docs
    @app.get("/", response_class=HTMLResponse)
    async def root():
        """Root endpoint with platform overview."""
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Genetic Coil Arena</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
                .container { max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
                h1 { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }
                .feature { background: #ecf0f1; padding: 15px; margin: 10px 0; border-radius: 5px; }
                .api-link { display: inline-block; background: #3498db; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px; margin: 5px; }
                .api-link:hover { background: #2980b9; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🧬 Genetic Coil Arena</h1>
                <p><strong>Complete Evolutionary Computing Platform</strong></p>
                <p>Advanced genetic algorithms, neural evolution, and AI-powered analytics for research and production.</p>
                
                <h2>🚀 Platform Features</h2>
                <div class="feature">
                    <strong>🧠 Neural Evolution:</strong> PyTorch-based neuroevolution with NEAT-like speciation
                </div>
                <div class="feature">
                    <strong>📊 Advanced Analytics:</strong> ML-powered insights and predictive modeling
                </div>
                <div class="feature">
                    <strong>🔬 Experiment Management:</strong> Complete lifecycle management with persistence
                </div>
                <div class="feature">
                    <strong>👥 Collaboration:</strong> Team management and experiment sharing
                </div>
                <div class="feature">
                    <strong>📱 Mobile Support:</strong> Progressive Web App with offline capabilities
                </div>
                <div class="feature">
                    <strong>🎯 Performance:</strong> Scalable architecture with load balancing
                </div>
                
                <h2>🔗 API Access</h2>
                <a href="/docs" class="api-link">📖 API Documentation</a>
                <a href="/redoc" class="api-link">📚 ReDoc</a>
                <a href="/health" class="api-link">💓 Health Check</a>
                
                <h2>🎯 Quick Start</h2>
                <pre style="background: #2c3e50; color: #ecf0f1; padding: 15px; border-radius: 5px; overflow-x: auto;">
# Install
pip install -e .

# Run server
python -m src.genetic_coil_arena.web.server

# Access APIs
curl http://localhost:8000/health
curl http://localhost:8000/api/neural/health
curl http://localhost:8000/api/analytics/health
                </pre>
            </div>
        </body>
        </html>
        """
    
    return app


class GeneticCoilArenaServer:
    """Unified server manager for Genetic Coil Arena."""
    
    def __init__(self, host: str = "localhost", port: int = 8000):
        self.host = host
        self.port = port
        self.app = create_app()
    
    def run(self, **kwargs):
        """Run the production server."""
        server_config = {
            "app": self.app,
            "host": self.host,
            "port": self.port,
            "log_level": "info",
            "access_log": True,
            "workers": 1,  # Increase for production
            **kwargs
        }
        
        logger.info(f"🚀 Starting Genetic Coil Arena on http://{self.host}:{self.port}")
        logger.info("📖 API Documentation: http://localhost:8000/docs")
        logger.info("🧠 Neural Evolution: http://localhost:8000/api/neural/health")
        logger.info("📊 Analytics: http://localhost:8000/api/analytics/health")
        
        uvicorn.run(**server_config)


def main():
    """Main entry point."""
    server = GeneticCoilArenaServer()
    server.run()


if __name__ == "__main__":
    main()
