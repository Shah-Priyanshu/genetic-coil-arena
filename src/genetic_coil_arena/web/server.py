"""
FastAPI web server for the genetic-coil-arena dashboard.
"""

import asyncio
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from .api import router
from ..config import Config


class WebServer:
    """Web server for the genetic algorithm dashboard."""
    
    def __init__(self, host: str = "localhost", port: int = 8000, config: Config = None):
        self.host = host
        self.port = port
        self.config = config or Config()
        
        # Create FastAPI app
        self.app = FastAPI(
            title="Genetic Coil Arena",
            description="Web dashboard for genetic algorithm experiments",
            version="1.0.0"
        )
        
        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # In production, specify allowed origins
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Include API routes
        self.app.include_router(router)
        
        # Add startup/shutdown handlers
        self.app.add_event_handler("startup", self.startup)
        self.app.add_event_handler("shutdown", self.shutdown)
    
    async def startup(self):
        """Called when the server starts."""
        print(f"🚀 Genetic Coil Arena dashboard starting on http://{self.host}:{self.port}")
        print("📊 Web interface available at: http://localhost:8000")
        print("🔌 WebSocket endpoint: ws://localhost:8000/ws")
    
    async def shutdown(self):
        """Called when the server shuts down."""
        print("🛑 Shutting down Genetic Coil Arena dashboard")
    
    def run(self, **kwargs):
        """Run the web server."""
        # Merge default config with provided kwargs
        server_config = {
            "app": self.app,
            "host": self.host,
            "port": self.port,
            "log_level": "info",
            **kwargs
        }
        
        uvicorn.run(**server_config)
    
    async def run_async(self, **kwargs):
        """Run the web server asynchronously."""
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


def create_app(config: Config = None) -> FastAPI:
    """Factory function to create FastAPI app."""
    server = WebServer(config=config)
    return server.app


def run_server(host: str = "localhost", port: int = 8000, config: Config = None):
    """Convenience function to run the web server."""
    server = WebServer(host=host, port=port, config=config)
    server.run()


if __name__ == "__main__":
    # Run server directly
    run_server()
