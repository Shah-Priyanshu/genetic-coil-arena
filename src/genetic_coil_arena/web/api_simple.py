"""
Simplified FastAPI routes for the web dashboard.
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse
from typing import List

# Simple router setup
router = APIRouter()

@router.get("/")
async def dashboard():
    """Serve the main dashboard HTML."""
    return HTMLResponse(content="""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Genetic Coil Arena - Dashboard</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .container { max-width: 1200px; margin: 0 auto; }
            .card { border: 1px solid #ddd; border-radius: 8px; padding: 20px; margin: 10px 0; }
            .btn { padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; }
            .btn-primary { background-color: #007bff; color: white; }
            .status { padding: 5px 10px; border-radius: 4px; font-size: 12px; }
            .running { background-color: #28a745; color: white; }
            .completed { background-color: #6c757d; color: white; }
            .failed { background-color: #dc3545; color: white; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🧬 Genetic Coil Arena</h1>
            <p>Evolutionary Snake AI Dashboard</p>
            
            <div class="card">
                <h2>Start New Experiment</h2>
                <form id="experiment-form">
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
                        <div>
                            <label>Experiment Name:</label><br>
                            <input type="text" id="name" required style="width: 100%; padding: 8px; margin: 5px 0;">
                        </div>
                        <div>
                            <label>Population Size:</label><br>
                            <input type="number" id="population_size" value="50" min="10" max="500" style="width: 100%; padding: 8px; margin: 5px 0;">
                        </div>
                        <div>
                            <label>Generations:</label><br>
                            <input type="number" id="generations" value="100" min="1" max="1000" style="width: 100%; padding: 8px; margin: 5px 0;">
                        </div>
                        <div>
                            <label>Mutation Rate:</label><br>
                            <input type="number" id="mutation_rate" value="0.1" min="0" max="1" step="0.01" style="width: 100%; padding: 8px; margin: 5px 0;">
                        </div>
                    </div>
                    <button type="submit" class="btn btn-primary" style="margin-top: 10px;">Start Experiment</button>
                </form>
            </div>
            
            <div class="card">
                <h2>Active Experiments</h2>
                <div id="experiments-list">
                    <p>No experiments running. Start one above!</p>
                </div>
            </div>
        </div>
        
        <script>
            // Simple demo functionality
            document.getElementById('experiment-form').addEventListener('submit', function(e) {
                e.preventDefault();
                alert('Experiment system is being set up. This is a basic demo.');
            });
        </script>
    </body>
    </html>
    """, media_type="text/html")

@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "genetic-coil-arena-web"}

@router.get("/experiments")
async def list_experiments():
    """List all experiments."""
    return {"experiments": []}

# Export router for use in server
__all__ = ["router"]
