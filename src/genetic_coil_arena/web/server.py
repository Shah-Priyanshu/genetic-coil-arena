"""
FastAPI web server for the genetic-coil-arena dashboard.
Updated to use production server with full integration.
"""

import asyncio
import uvicorn
from fastapi import FastAPI
from typing import Optional, Any

# Import production server
try:
    from .server_production import ProductionServer, create_production_app
    production_available = True
except ImportError:
    # Fallback to basic server
    try:
        from .api_real import router
        fallback_available = True
    except ImportError:
        router = None
        fallback_available = False
    production_available = False

from ..config import Config


class WebServer:
    """Web server for the genetic algorithm dashboard."""
    
    def __init__(self, host: str = "localhost", port: int = 8000, config: Optional[Config] = None):
        self.host = host
        self.port = port
        self.config = config or Config()
        
        if production_available:
            # Use production server
            self.production_server = ProductionServer(host=host, port=port, config=self.config)
            self.app = self.production_server.app
        else:
            # Fallback to basic server
            self.app = self._create_basic_app()
    
    def _create_basic_app(self) -> FastAPI:
        """Create basic FastAPI app as fallback."""
        from fastapi.middleware.cors import CORSMiddleware
        
        app = FastAPI(
            title="Genetic Coil Arena",
            description="Web dashboard for genetic algorithm experiments",
            version="1.0.0"
        )
        
        # Add CORS middleware
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Include API routes if available
        if fallback_available and router:
            app.include_router(router)
        
        @app.get("/")
        async def root():
            return {"message": "Genetic Coil Arena Dashboard (Basic Mode)"}
        
        return app
    
    def run(self, **kwargs: Any):
        """Run the web server."""
        if production_available:
            self.production_server.run(**kwargs)
        else:
            # Basic server configuration
            server_config = {
                "app": self.app,
                "host": self.host,
                "port": self.port,
                "log_level": "info",
                **kwargs
            }
            uvicorn.run(**server_config)
    
    async def run_async(self, **kwargs: Any):
        """Run the web server asynchronously."""
        if production_available:
            await self.production_server.run_async(**kwargs)
        else:
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


def create_app(config: Optional[Config] = None) -> FastAPI:
    """Factory function to create FastAPI app."""
    if production_available:
        return create_production_app(config)
    else:
        server = WebServer(config=config)
        return server.app


def run_server(host: str = "localhost", port: int = 8000, config: Optional[Config] = None):
    """Convenience function to run the web server."""
    server = WebServer(host=host, port=port, config=config)
    server.run()


if __name__ == "__main__":
    # Run server directly
    run_server()
