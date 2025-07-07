"""
Enhanced dashboard with advanced analytics and interactive visualizations.
Phase 3C: Advanced Analytics & Visualization Features
"""

from fastapi.responses import HTMLResponse
from fastapi import APIRouter

viz_router = APIRouter()

@viz_router.get("/analytics")
async def analytics_dashboard():
    """Serve the advanced analytics dashboard."""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Genetic Coil Arena - Advanced Analytics</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <style>
            body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 0; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; }
            .container { max-width: 1600px; margin: 0 auto; padding: 20px; }
            .header { text-align: center; margin-bottom: 40px; color: white; }
            .nav-tabs { display: flex; background: rgba(255,255,255,0.1); border-radius: 12px; padding: 8px; margin-bottom: 30px; }
            .tab { flex: 1; padding: 15px; text-align: center; border-radius: 8px; cursor: pointer; color: white; font-weight: 600; transition: all 0.3s; }
            .tab.active { background: white; color: #667eea; }
            .tab:hover { background: rgba(255,255,255,0.2); }
            .card { background: white; border-radius: 12px; padding: 25px; margin: 20px 0; box-shadow: 0 8px 32px rgba(0,0,0,0.1); }
            .chart-container { position: relative; height: 400px; margin: 20px 0; }
            .chart-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(500px, 1fr)); gap: 30px; }
            .metric-card { background: linear-gradient(45deg, #667eea, #764ba2); color: white; border-radius: 12px; padding: 20px; text-align: center; }
            .metric-value { font-size: 36px; font-weight: bold; margin-bottom: 10px; }
            .metric-label { font-size: 14px; opacity: 0.9; }
            .analytics-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 30px; }
            .experiment-selector { margin-bottom: 30px; }
            .experiment-selector select { padding: 12px; border-radius: 8px; border: 2px solid #ddd; font-size: 16px; }
            .comparison-section { background: #f8f9fa; border-radius: 12px; padding: 25px; margin: 20px 0; }
            .agent-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 15px; }
            .agent-card { background: white; border-radius: 8px; padding: 15px; border-left: 4px solid #667eea; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>📊 Genetic Coil Arena - Advanced Analytics</h1>
                <p>Deep Insights into Evolutionary Performance</p>
            </div>
            
            <div class="nav-tabs">
                <div class="tab active" onclick="showTab('overview')">📈 Overview</div>
                <div class="tab" onclick="showTab('fitness')">🏆 Fitness Analysis</div>
                <div class="tab" onclick="showTab('agents')">🤖 Agent Details</div>
                <div class="tab" onclick="showTab('comparison')">⚖️ Comparison</div>
            </div>
            
            <div class="experiment-selector">
                <label style="color: white; font-weight: 600; margin-right: 15px;">Select Experiment:</label>
                <select id="experimentSelect" onchange="loadExperimentData()">
                    <option value="">Loading experiments...</option>
                </select>
            </div>
            
            <!-- Overview Tab -->
            <div id="tab-overview" class="tab-content">
                <div class="analytics-grid">
                    <div class="metric-card">
                        <div class="metric-value" id="currentGeneration">0</div>
                        <div class="metric-label">Current Generation</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value" id="bestFitness">0.0</div>
                        <div class="metric-label">Best Fitness</div>
                    </div>
                </div>
                
                <div class="card">
                    <h2>🎯 Experiment Progress Overview</h2>
                    <div class="chart-container">
                        <canvas id="overviewChart"></canvas>
                    </div>
                </div>
            </div>
            
            <!-- Fitness Analysis Tab -->
            <div id="tab-fitness" class="tab-content" style="display: none;">
                <div class="chart-grid">
                    <div class="card">
                        <h3>📈 Fitness Evolution</h3>
                        <div class="chart-container">
                            <canvas id="fitnessChart"></canvas>
                        </div>
                    </div>
                    <div class="card">
                        <h3>🔀 Population Diversity</h3>
                        <div class="chart-container">
                            <canvas id="diversityChart"></canvas>
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- Agent Details Tab -->
            <div id="tab-agents" class="tab-content" style="display: none;">
                <div class="card">
                    <h2>🤖 Top Performing Agents</h2>
                    <div id="agentsList" class="agent-grid">
                        <div class="agent-card">Loading agent data...</div>
                    </div>
                </div>
            </div>
            
            <!-- Comparison Tab -->
            <div id="tab-comparison" class="tab-content" style="display: none;">
                <div class="comparison-section">
                    <h2>⚖️ Experiment Comparison</h2>
                    <div style="margin-bottom: 20px;">
                        <label style="font-weight: 600;">Compare with:</label>
                        <select id="compareSelect" onchange="loadComparison()">
                            <option value="">Select experiment to compare...</option>
                        </select>
                    </div>
                    <div id="comparisonResults">
                        <p>Select an experiment to compare with the current one.</p>
                    </div>
                </div>
            </div>
        </div>
        
        <script>
            let currentExperimentId = null;
            let charts = {};
            
            // Tab switching
            function showTab(tabName) {
                document.querySelectorAll('.tab-content').forEach(tab => tab.style.display = 'none');
                document.querySelectorAll('.tab').forEach(tab => tab.classList.remove('active'));
                document.getElementById(`tab-${tabName}`).style.display = 'block';
                document.querySelector(`[onclick="showTab('${tabName}')"]`).classList.add('active');
                
                // Load data for specific tabs
                if (tabName === 'fitness' && currentExperimentId) {
                    loadFitnessCharts();
                } else if (tabName === 'agents' && currentExperimentId) {
                    loadAgentDetails();
                } else if (tabName === 'comparison' && currentExperimentId) {
                    loadComparisonOptions();
                }
            }
            
            // Load experiments
            async function loadExperiments() {
                try {
                    const response = await fetch('/experiments');
                    const data = await response.json();
                    
                    const select = document.getElementById('experimentSelect');
                    const compareSelect = document.getElementById('compareSelect');
                    
                    select.innerHTML = '<option value="">Select an experiment...</option>';
                    compareSelect.innerHTML = '<option value="">Select experiment to compare...</option>';
                    
                    data.experiments.forEach(exp => {
                        const option = new Option(`${exp.name} (${exp.id.substring(0, 8)}...)`, exp.id);
                        const compareOption = new Option(`${exp.name} (${exp.id.substring(0, 8)}...)`, exp.id);
                        select.add(option);
                        compareSelect.add(compareOption);
                    });
                    
                    if (data.experiments.length > 0) {
                        currentExperimentId = data.experiments[0].id;
                        select.value = currentExperimentId;
                        loadExperimentData();
                    }
                } catch (error) {
                    console.error('Failed to load experiments:', error);
                }
            }
            
            // Load experiment data
            async function loadExperimentData() {
                const experimentId = document.getElementById('experimentSelect').value;
                if (!experimentId) return;
                
                currentExperimentId = experimentId;
                
                try {
                    const response = await fetch(`/experiments/${experimentId}`);
                    const data = await response.json();
                    
                    // Update overview metrics
                    document.getElementById('currentGeneration').textContent = data.generation || 0;
                    document.getElementById('bestFitness').textContent = (data.best_fitness || 0).toFixed(2);
                    
                    // Create overview chart
                    createOverviewChart();
                    
                } catch (error) {
                    console.error('Failed to load experiment data:', error);
                }
            }
            
            // Create overview chart
            function createOverviewChart() {
                const ctx = document.getElementById('overviewChart').getContext('2d');
                
                if (charts.overview) {
                    charts.overview.destroy();
                }
                
                // Demo data - in real implementation, this would come from API
                const generations = Array.from({length: 50}, (_, i) => i + 1);
                const bestFitness = generations.map(g => 10 + g * 1.5 + Math.sin(g/5) * 5);
                const avgFitness = generations.map(g => 8 + g * 1.2 + Math.sin(g/7) * 3);
                
                charts.overview = new Chart(ctx, {
                    type: 'line',
                    data: {
                        labels: generations,
                        datasets: [{
                            label: 'Best Fitness',
                            data: bestFitness,
                            borderColor: '#667eea',
                            backgroundColor: 'rgba(102, 126, 234, 0.1)',
                            fill: true,
                            tension: 0.4
                        }, {
                            label: 'Average Fitness',
                            data: avgFitness,
                            borderColor: '#764ba2',
                            backgroundColor: 'rgba(118, 75, 162, 0.1)',
                            fill: true,
                            tension: 0.4
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: {
                            title: {
                                display: true,
                                text: 'Fitness Evolution Over Generations'
                            }
                        },
                        scales: {
                            x: {
                                title: {
                                    display: true,
                                    text: 'Generation'
                                }
                            },
                            y: {
                                title: {
                                    display: true,
                                    text: 'Fitness Score'
                                }
                            }
                        }
                    }
                });
            }
            
            // Load fitness charts
            async function loadFitnessCharts() {
                if (!currentExperimentId) return;
                
                try {
                    const response = await fetch(`/analytics/experiment/${currentExperimentId}/fitness-history`);
                    const data = await response.json();
                    
                    // Fitness chart
                    const fitnessCtx = document.getElementById('fitnessChart').getContext('2d');
                    if (charts.fitness) charts.fitness.destroy();
                    
                    charts.fitness = new Chart(fitnessCtx, {
                        type: 'line',
                        data: {
                            labels: data.generations,
                            datasets: [{
                                label: 'Best Fitness',
                                data: data.best_fitness,
                                borderColor: '#28a745',
                                backgroundColor: 'rgba(40, 167, 69, 0.1)',
                                fill: false
                            }, {
                                label: 'Average Fitness',
                                data: data.avg_fitness,
                                borderColor: '#007bff',
                                backgroundColor: 'rgba(0, 123, 255, 0.1)',
                                fill: false
                            }, {
                                label: 'Worst Fitness',
                                data: data.worst_fitness,
                                borderColor: '#dc3545',
                                backgroundColor: 'rgba(220, 53, 69, 0.1)',
                                fill: false
                            }]
                        },
                        options: {
                            responsive: true,
                            maintainAspectRatio: false,
                            plugins: {
                                title: {
                                    display: true,
                                    text: 'Detailed Fitness Analysis'
                                }
                            }
                        }
                    });
                    
                    // Diversity chart
                    const diversityCtx = document.getElementById('diversityChart').getContext('2d');
                    if (charts.diversity) charts.diversity.destroy();
                    
                    charts.diversity = new Chart(diversityCtx, {
                        type: 'line',
                        data: {
                            labels: data.generations,
                            datasets: [{
                                label: 'Population Diversity',
                                data: data.diversity,
                                borderColor: '#ffc107',
                                backgroundColor: 'rgba(255, 193, 7, 0.2)',
                                fill: true,
                                tension: 0.4
                            }]
                        },
                        options: {
                            responsive: true,
                            maintainAspectRatio: false,
                            plugins: {
                                title: {
                                    display: true,
                                    text: 'Population Diversity Over Time'
                                }
                            },
                            scales: {
                                y: {
                                    beginAtZero: true,
                                    max: 1
                                }
                            }
                        }
                    });
                    
                } catch (error) {
                    console.error('Failed to load fitness charts:', error);
                }
            }
            
            // Load agent details
            async function loadAgentDetails() {
                if (!currentExperimentId) return;
                
                try {
                    const response = await fetch(`/analytics/experiment/${currentExperimentId}/agents`);
                    const data = await response.json();
                    
                    const agentsList = document.getElementById('agentsList');
                    agentsList.innerHTML = data.agents.map((agent, index) => `
                        <div class="agent-card">
                            <h4>Agent #${index + 1}</h4>
                            <p><strong>Fitness:</strong> ${agent.fitness}</p>
                            <p><strong>Avg Score:</strong> ${agent.avg_score}</p>
                            <p><strong>Survival:</strong> ${agent.survival_time}s</p>
                            <p><strong>Games:</strong> ${agent.games_played}</p>
                        </div>
                    `).join('');
                    
                } catch (error) {
                    console.error('Failed to load agent details:', error);
                }
            }
            
            // Load comparison options
            function loadComparisonOptions() {
                // Implementation for comparison loading
                document.getElementById('comparisonResults').innerHTML = `
                    <div class="card">
                        <h3>📊 Comparison Analysis</h3>
                        <p>Advanced comparison features will be implemented here, including:</p>
                        <ul>
                            <li>Side-by-side fitness progression</li>
                            <li>Convergence speed analysis</li>
                            <li>Population diversity comparison</li>
                            <li>Performance benchmarking</li>
                        </ul>
                    </div>
                `;
            }
            
            // Initialize
            document.addEventListener('DOMContentLoaded', function() {
                loadExperiments();
                
                // Auto-refresh every 30 seconds
                setInterval(loadExperiments, 30000);
            });
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content, media_type="text/html")
