"""
Phase 4A: Experiment Comparison API Routes
REST API endpoints for experiment comparison and analysis.
"""

import logging
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response, HTMLResponse
from pydantic import BaseModel

from .comparison import ExperimentComparison
from .database import DatabaseManager

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/comparison", tags=["experiment-comparison"])


class ComparisonRequest(BaseModel):
    """Request model for experiment comparison."""
    experiment_ids: List[str]
    include_detailed_analysis: bool = True


class ExportRequest(BaseModel):
    """Request model for exporting comparison data."""
    experiment_ids: List[str]
    format: str = "json"  # json, csv


# Dependency to get comparison service
async def get_comparison_service():
    """Get experiment comparison service."""
    from .api import experiment_manager  # Import here to avoid circular dependency
    db_manager = DatabaseManager()
    await db_manager.init_db()
    return ExperimentComparison(experiment_manager, db_manager)


@router.post("/compare")
async def compare_experiments(request: ComparisonRequest):
    """Compare multiple experiments and return detailed analysis."""
    try:
        if len(request.experiment_ids) < 2:
            raise HTTPException(status_code=400, detail="At least 2 experiments required for comparison")
        
        # Get the service
        comparison_service = await get_comparison_service()
        
        # Generate comparison report
        report = await comparison_service.generate_comparison_report(request.experiment_ids)
        
        logger.info(f"Generated comparison report for {len(request.experiment_ids)} experiments")
        return report
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to compare experiments: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate comparison")


@router.get("/experiments/{experiment_id}/similar")
async def find_similar_experiments(
    experiment_id: str,
    limit: int = Query(5, ge=1, le=20)
):
    """Find experiments similar to the given experiment."""
    try:
        comparison_service = await get_comparison_service()
        
        # Get all experiments
        all_experiments = comparison_service.experiment_manager.list_experiments()
        
        # Filter out the target experiment and return others
        similar = [exp for exp in all_experiments if exp.id != experiment_id][:limit]
        
        return {
            "target_experiment_id": experiment_id,
            "similar_experiments": [
                {
                    "id": exp.id,
                    "name": exp.name,
                    "status": exp.status,
                    "similarity_score": 0.85,  # Placeholder score
                    "best_fitness": exp.best_fitness
                }
                for exp in similar
            ]
        }
        
    except Exception as e:
        logger.error(f"Failed to find similar experiments: {e}")
        raise HTTPException(status_code=500, detail="Failed to find similar experiments")


@router.post("/export")
async def export_comparison(request: ExportRequest):
    """Export comparison data in various formats."""
    try:
        comparison_service = await get_comparison_service()
        
        # Generate comparison
        comparison_result = await comparison_service.compare_experiments(request.experiment_ids)
        
        # Export data
        exported_data = comparison_service.export_comparison_data(comparison_result, request.format)
        
        # Return appropriate response
        if request.format == "json":
            return Response(
                content=exported_data,
                media_type="application/json",
                headers={"Content-Disposition": "attachment; filename=experiment_comparison.json"}
            )
        elif request.format == "csv":
            return Response(
                content=exported_data,
                media_type="text/csv",
                headers={"Content-Disposition": "attachment; filename=experiment_comparison.csv"}
            )
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported format: {request.format}")
            
    except Exception as e:
        logger.error(f"Failed to export comparison: {e}")
        raise HTTPException(status_code=500, detail="Failed to export comparison data")


@router.get("/rankings/{metric}")
async def get_metric_rankings(
    metric: str,
    limit: int = Query(10, ge=1, le=50)
):
    """Get top experiments ranked by a specific metric."""
    try:
        comparison_service = await get_comparison_service()
        
        # Get all experiments
        all_experiments = comparison_service.experiment_manager.list_experiments()
        
        if not all_experiments:
            return {"metric": metric, "rankings": []}
        
        # For now, rank by best_fitness (in a full implementation, support all metrics)
        if metric == "best_fitness":
            ranked = sorted(all_experiments, key=lambda x: x.best_fitness, reverse=True)[:limit]
            
            return {
                "metric": metric,
                "rankings": [
                    {
                        "rank": i + 1,
                        "experiment_id": exp.id,
                        "name": exp.name,
                        "value": exp.best_fitness,
                        "status": exp.status
                    }
                    for i, exp in enumerate(ranked)
                ]
            }
        else:
            raise HTTPException(status_code=400, detail=f"Metric '{metric}' not supported yet")
            
    except Exception as e:
        logger.error(f"Failed to get rankings: {e}")
        raise HTTPException(status_code=500, detail="Failed to get metric rankings")


@router.get("/dashboard")
async def comparison_dashboard():
    """Serve the experiment comparison dashboard."""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Experiment Comparison - Genetic Coil Arena</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <style>
            * { box-sizing: border-box; margin: 0; padding: 0; }
            body { 
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                color: #333;
            }
            .container { max-width: 1400px; margin: 0 auto; padding: 20px; }
            
            .header {
                background: rgba(255, 255, 255, 0.95);
                backdrop-filter: blur(10px);
                border-radius: 16px;
                padding: 30px;
                margin-bottom: 30px;
                box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
                text-align: center;
            }
            .header h1 { color: #4f46e5; font-size: 2.5rem; margin-bottom: 10px; }
            .header p { color: #6b7280; font-size: 1.1rem; }
            
            .grid { display: grid; grid-template-columns: 1fr 2fr; gap: 30px; margin-bottom: 30px; }
            
            .card {
                background: rgba(255, 255, 255, 0.95);
                backdrop-filter: blur(10px);
                border-radius: 16px;
                padding: 25px;
                box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
            }
            
            .card h2 { margin-bottom: 20px; color: #374151; font-size: 1.5rem; }
            
            .experiment-selector {
                margin-bottom: 20px;
            }
            
            .experiment-item {
                display: flex;
                align-items: center;
                padding: 12px;
                border: 2px solid #e5e7eb;
                border-radius: 8px;
                margin-bottom: 8px;
                cursor: pointer;
                transition: all 0.2s;
            }
            .experiment-item:hover { border-color: #4f46e5; background: #f8fafc; }
            .experiment-item.selected { border-color: #4f46e5; background: #ede9fe; }
            
            .experiment-item input[type="checkbox"] {
                margin-right: 12px;
                width: 16px;
                height: 16px;
            }
            
            .btn {
                background: linear-gradient(135deg, #4f46e5, #7c3aed);
                color: white;
                border: none;
                padding: 12px 24px;
                border-radius: 8px;
                cursor: pointer;
                font-size: 16px;
                font-weight: 500;
                transition: transform 0.2s, box-shadow 0.2s;
                width: 100%;
                margin-top: 10px;
            }
            .btn:hover {
                transform: translateY(-2px);
                box-shadow: 0 4px 12px rgba(79, 70, 229, 0.4);
            }
            .btn:disabled {
                opacity: 0.5;
                cursor: not-allowed;
                transform: none;
                box-shadow: none;
            }
            
            .comparison-results {
                grid-column: 1 / -1;
                display: none;
            }
            .comparison-results.show { display: block; }
            
            .metrics-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 20px;
                margin-bottom: 30px;
            }
            
            .metric-card {
                background: white;
                border-radius: 12px;
                padding: 20px;
                box-shadow: 0 4px 16px rgba(0, 0, 0, 0.1);
            }
            
            .metric-card h3 {
                color: #374151;
                margin-bottom: 15px;
                font-size: 1.2rem;
            }
            
            .chart-container {
                position: relative;
                height: 250px;
                margin-bottom: 20px;
            }
            
            .recommendations {
                background: linear-gradient(135deg, #10b981, #059669);
                color: white;
                border-radius: 12px;
                padding: 20px;
                margin-top: 20px;
            }
            
            .recommendations h3 {
                margin-bottom: 15px;
                font-size: 1.3rem;
            }
            
            .recommendations ul {
                list-style: none;
                padding: 0;
            }
            
            .recommendations li {
                padding: 8px 0;
                border-bottom: 1px solid rgba(255, 255, 255, 0.2);
            }
            .recommendations li:last-child { border-bottom: none; }
            
            .loading {
                text-align: center;
                padding: 40px;
                color: #6b7280;
            }
            
            .error {
                background: #fee2e2;
                color: #dc2626;
                padding: 15px;
                border-radius: 8px;
                margin: 10px 0;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>📊 Experiment Comparison</h1>
                <p>Advanced side-by-side analysis and performance comparison</p>
            </div>
            
            <div class="grid">
                <div class="card">
                    <h2>🔬 Select Experiments</h2>
                    <div id="experimentSelector" class="experiment-selector">
                        <div class="loading">Loading experiments...</div>
                    </div>
                    <button id="compareBtn" class="btn" disabled>Compare Selected (0)</button>
                </div>
                
                <div class="card">
                    <h2>⚡ Quick Actions</h2>
                    <button class="btn" onclick="selectTopPerformers()">Select Top 3 Performers</button>
                    <button class="btn" onclick="selectRecent()">Select 3 Most Recent</button>
                    <button class="btn" onclick="clearSelection()">Clear Selection</button>
                    
                    <h3 style="margin-top: 30px; margin-bottom: 15px;">📈 Global Rankings</h3>
                    <div id="globalRankings">
                        <div class="loading">Loading rankings...</div>
                    </div>
                </div>
            </div>
            
            <div id="comparisonResults" class="card comparison-results">
                <h2>📊 Comparison Results</h2>
                <div id="comparisonContent"></div>
            </div>
        </div>
        
        <script>
            let experiments = [];
            let selectedExperiments = new Set();
            
            // Load experiments
            async function loadExperiments() {
                try {
                    const response = await fetch('/api/experiments');
                    experiments = await response.json();
                    renderExperimentSelector();
                    loadGlobalRankings();
                } catch (error) {
                    document.getElementById('experimentSelector').innerHTML = 
                        '<div class="error">Failed to load experiments</div>';
                }
            }
            
            function renderExperimentSelector() {
                const container = document.getElementById('experimentSelector');
                
                if (experiments.length === 0) {
                    container.innerHTML = '<p style="text-align: center; color: #6b7280;">No experiments found</p>';
                    return;
                }
                
                container.innerHTML = experiments.map(exp => `
                    <div class="experiment-item" onclick="toggleExperiment('${exp.id}')">
                        <input type="checkbox" id="exp-${exp.id}" onchange="toggleExperiment('${exp.id}')">
                        <div>
                            <strong>${exp.name}</strong><br>
                            <small>Status: ${exp.status} | Best Fitness: ${exp.best_fitness.toFixed(3)}</small>
                        </div>
                    </div>
                `).join('');
                
                updateCompareButton();
            }
            
            function toggleExperiment(expId) {
                const checkbox = document.getElementById(`exp-${expId}`);
                const item = checkbox.parentElement;
                
                if (selectedExperiments.has(expId)) {
                    selectedExperiments.delete(expId);
                    checkbox.checked = false;
                    item.classList.remove('selected');
                } else {
                    selectedExperiments.add(expId);
                    checkbox.checked = true;
                    item.classList.add('selected');
                }
                
                updateCompareButton();
            }
            
            function updateCompareButton() {
                const btn = document.getElementById('compareBtn');
                const count = selectedExperiments.size;
                
                btn.textContent = `Compare Selected (${count})`;
                btn.disabled = count < 2;
            }
            
            function selectTopPerformers() {
                clearSelection();
                const sorted = [...experiments].sort((a, b) => b.best_fitness - a.best_fitness);
                sorted.slice(0, 3).forEach(exp => toggleExperiment(exp.id));
            }
            
            function selectRecent() {
                clearSelection();
                const sorted = [...experiments].sort((a, b) => new Date(b.start_time) - new Date(a.start_time));
                sorted.slice(0, 3).forEach(exp => toggleExperiment(exp.id));
            }
            
            function clearSelection() {
                selectedExperiments.clear();
                document.querySelectorAll('.experiment-item').forEach(item => {
                    item.classList.remove('selected');
                    item.querySelector('input').checked = false;
                });
                updateCompareButton();
            }
            
            // Compare experiments
            document.getElementById('compareBtn').onclick = async function() {
                if (selectedExperiments.size < 2) return;
                
                const results = document.getElementById('comparisonResults');
                const content = document.getElementById('comparisonContent');
                
                content.innerHTML = '<div class="loading">Analyzing experiments...</div>';
                results.classList.add('show');
                
                try {
                    const response = await fetch('/api/comparison/compare', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            experiment_ids: Array.from(selectedExperiments),
                            include_detailed_analysis: true
                        })
                    });
                    
                    if (!response.ok) {
                        throw new Error('Comparison failed');
                    }
                    
                    const data = await response.json();
                    renderComparisonResults(data);
                    
                } catch (error) {
                    content.innerHTML = '<div class="error">Failed to compare experiments: ' + error.message + '</div>';
                }
            };
            
            function renderComparisonResults(data) {
                const content = document.getElementById('comparisonContent');
                
                // Create metrics visualization
                const metricsHtml = `
                    <div class="metrics-grid">
                        <div class="metric-card">
                            <h3>🏆 Performance Rankings</h3>
                            <div class="chart-container">
                                <canvas id="performanceChart"></canvas>
                            </div>
                        </div>
                        <div class="metric-card">
                            <h3>⚡ Efficiency Analysis</h3>
                            <div class="chart-container">
                                <canvas id="efficiencyChart"></canvas>
                            </div>
                        </div>
                        <div class="metric-card">
                            <h3>📈 Convergence Comparison</h3>
                            <div class="chart-container">
                                <canvas id="convergenceChart"></canvas>
                            </div>
                        </div>
                        <div class="metric-card">
                            <h3>📊 Summary Statistics</h3>
                            <div>
                                <p><strong>Experiments:</strong> ${data.experiment_count}</p>
                                <p><strong>Best Fitness:</strong> ${data.summary.metrics?.best_fitness?.max?.toFixed(3) || 'N/A'}</p>
                                <p><strong>Avg Fitness:</strong> ${data.summary.metrics?.best_fitness?.mean?.toFixed(3) || 'N/A'}</p>
                                <p><strong>Fitness Range:</strong> ${data.statistical_analysis?.fitness_range?.toFixed(3) || 'N/A'}</p>
                            </div>
                        </div>
                    </div>
                    
                    <div class="recommendations">
                        <h3>💡 Recommendations</h3>
                        <ul>
                            ${data.recommendations.map(rec => `<li>${rec}</li>`).join('')}
                        </ul>
                    </div>
                `;
                
                content.innerHTML = metricsHtml;
                
                // Create charts
                createPerformanceChart(data);
                createEfficiencyChart(data);
                createConvergenceChart(data);
            }
            
            function createPerformanceChart(data) {
                const ctx = document.getElementById('performanceChart').getContext('2d');
                const rankings = data.rankings.best_fitness || [];
                
                new Chart(ctx, {
                    type: 'bar',
                    data: {
                        labels: rankings.map(r => r.experiment_name || r.experiment_id.slice(0, 8)),
                        datasets: [{
                            label: 'Best Fitness',
                            data: rankings.map(r => r.score),
                            backgroundColor: 'rgba(79, 70, 229, 0.6)',
                            borderColor: 'rgba(79, 70, 229, 1)',
                            borderWidth: 1
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        scales: {
                            y: { beginAtZero: true }
                        }
                    }
                });
            }
            
            function createEfficiencyChart(data) {
                const ctx = document.getElementById('efficiencyChart').getContext('2d');
                const rankings = data.rankings.efficiency || [];
                
                new Chart(ctx, {
                    type: 'doughnut',
                    data: {
                        labels: rankings.map(r => r.experiment_name || r.experiment_id.slice(0, 8)),
                        datasets: [{
                            data: rankings.map(r => r.score),
                            backgroundColor: [
                                'rgba(16, 185, 129, 0.6)',
                                'rgba(59, 130, 246, 0.6)',
                                'rgba(245, 158, 11, 0.6)',
                                'rgba(239, 68, 68, 0.6)',
                                'rgba(139, 92, 246, 0.6)'
                            ]
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false
                    }
                });
            }
            
            function createConvergenceChart(data) {
                const ctx = document.getElementById('convergenceChart').getContext('2d');
                const rankings = data.rankings.convergence_rate || [];
                
                new Chart(ctx, {
                    type: 'line',
                    data: {
                        labels: rankings.map((_, i) => `Rank ${i + 1}`),
                        datasets: [{
                            label: 'Convergence Rate',
                            data: rankings.map(r => r.score),
                            borderColor: 'rgba(245, 158, 11, 1)',
                            backgroundColor: 'rgba(245, 158, 11, 0.1)',
                            tension: 0.4
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        scales: {
                            y: { beginAtZero: true }
                        }
                    }
                });
            }
            
            async function loadGlobalRankings() {
                try {
                    const response = await fetch('/api/comparison/rankings/best_fitness?limit=5');
                    const data = await response.json();
                    
                    const container = document.getElementById('globalRankings');
                    container.innerHTML = data.rankings.map((rank, i) => `
                        <div style="padding: 8px; border-bottom: 1px solid #e5e7eb;">
                            <strong>#${rank.rank}</strong> ${rank.name} 
                            <span style="color: #6b7280;">(${rank.value.toFixed(3)})</span>
                        </div>
                    `).join('');
                    
                } catch (error) {
                    document.getElementById('globalRankings').innerHTML = 
                        '<div class="error">Failed to load rankings</div>';
                }
            }
            
            // Initialize
            loadExperiments();
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)
