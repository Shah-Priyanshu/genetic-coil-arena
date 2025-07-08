"""
Phase 4A: Batch Processing Dashboard
Advanced web interface for managing batch experiments.
"""

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

batch_ui_router = APIRouter(prefix="/batch-ui", tags=["batch-ui"])


@batch_ui_router.get("/", response_class=HTMLResponse)
async def batch_dashboard():
    """Serve the batch processing dashboard."""
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Batch Processing - Genetic Coil Arena</title>
        <style>
            * { 
                margin: 0; 
                padding: 0; 
                box-sizing: border-box; 
            }
            
            body { 
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                min-height: 100vh; 
                color: #333;
            }
            
            .header {
                background: rgba(255,255,255,0.1);
                backdrop-filter: blur(10px);
                padding: 1rem 2rem;
                color: white;
                box-shadow: 0 2px 20px rgba(0,0,0,0.1);
                display: flex;
                justify-content: space-between;
                align-items: center;
            }
            
            .header-left h1 {
                font-size: 2rem;
                margin-bottom: 0.5rem;
            }
            
            .header-left p {
                opacity: 0.9;
                font-size: 1.1rem;
            }
            
            .header-right {
                display: flex;
                gap: 1rem;
            }
            
            .container {
                max-width: 1400px;
                margin: 2rem auto;
                padding: 0 2rem;
            }
            
            .stats-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 1rem;
                margin-bottom: 2rem;
            }
            
            .stat-card {
                background: white;
                border-radius: 12px;
                padding: 1.5rem;
                box-shadow: 0 4px 20px rgba(0,0,0,0.1);
                text-align: center;
            }
            
            .stat-value {
                font-size: 2.5rem;
                font-weight: 700;
                margin-bottom: 0.5rem;
            }
            
            .stat-label {
                color: #666;
                font-size: 0.9rem;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }
            
            .stat-card.running .stat-value { color: #3b82f6; }
            .stat-card.completed .stat-value { color: #10b981; }
            .stat-card.failed .stat-value { color: #ef4444; }
            .stat-card.total .stat-value { color: #8b5cf6; }
            
            .main-content {
                display: grid;
                grid-template-columns: 1fr 400px;
                gap: 2rem;
                margin-bottom: 2rem;
            }
            
            .batches-section {
                background: white;
                border-radius: 12px;
                padding: 1.5rem;
                box-shadow: 0 4px 20px rgba(0,0,0,0.1);
            }
            
            .section-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 1.5rem;
                padding-bottom: 1rem;
                border-bottom: 1px solid #e5e5e5;
            }
            
            .section-title {
                font-size: 1.3rem;
                font-weight: 700;
            }
            
            .batch-list {
                display: flex;
                flex-direction: column;
                gap: 1rem;
            }
            
            .batch-item {
                border: 1px solid #e5e5e5;
                border-radius: 8px;
                padding: 1rem;
                transition: all 0.2s;
                cursor: pointer;
            }
            
            .batch-item:hover {
                border-color: #667eea;
                background: #f8f9ff;
            }
            
            .batch-header {
                display: flex;
                justify-content: space-between;
                align-items: flex-start;
                margin-bottom: 0.5rem;
            }
            
            .batch-name {
                font-weight: 600;
                font-size: 1.1rem;
            }
            
            .batch-status {
                padding: 0.25rem 0.75rem;
                border-radius: 20px;
                font-size: 0.8rem;
                font-weight: 600;
                text-transform: uppercase;
            }
            
            .status-created { background: #e0e7ff; color: #3730a3; }
            .status-running { background: #dbeafe; color: #1e40af; }
            .status-paused { background: #fef3c7; color: #92400e; }
            .status-completed { background: #d1fae5; color: #065f46; }
            .status-failed { background: #fee2e2; color: #991b1b; }
            .status-cancelled { background: #f3f4f6; color: #374151; }
            
            .batch-description {
                color: #666;
                font-size: 0.9rem;
                margin-bottom: 1rem;
                line-height: 1.4;
            }
            
            .batch-progress {
                margin-bottom: 1rem;
            }
            
            .progress-bar {
                width: 100%;
                height: 8px;
                background: #e5e5e5;
                border-radius: 4px;
                overflow: hidden;
                margin-bottom: 0.5rem;
            }
            
            .progress-fill {
                height: 100%;
                background: linear-gradient(90deg, #667eea, #764ba2);
                transition: width 0.3s ease;
            }
            
            .progress-text {
                font-size: 0.85rem;
                color: #666;
                display: flex;
                justify-content: space-between;
            }
            
            .batch-actions {
                display: flex;
                gap: 0.5rem;
                flex-wrap: wrap;
            }
            
            .btn {
                padding: 0.5rem 1rem;
                border: none;
                border-radius: 6px;
                cursor: pointer;
                font-weight: 600;
                transition: all 0.2s;
                text-decoration: none;
                display: inline-block;
                text-align: center;
                font-size: 0.85rem;
            }
            
            .btn-sm {
                padding: 0.25rem 0.75rem;
                font-size: 0.8rem;
            }
            
            .btn-primary {
                background: #667eea;
                color: white;
            }
            
            .btn-primary:hover {
                background: #5a67d8;
            }
            
            .btn-success {
                background: #10b981;
                color: white;
            }
            
            .btn-warning {
                background: #f59e0b;
                color: white;
            }
            
            .btn-danger {
                background: #ef4444;
                color: white;
            }
            
            .btn-secondary {
                background: #f7f7f7;
                color: #666;
                border: 1px solid #ddd;
            }
            
            .sidebar {
                display: flex;
                flex-direction: column;
                gap: 1.5rem;
            }
            
            .quick-actions {
                background: white;
                border-radius: 12px;
                padding: 1.5rem;
                box-shadow: 0 4px 20px rgba(0,0,0,0.1);
            }
            
            .system-monitor {
                background: white;
                border-radius: 12px;
                padding: 1.5rem;
                box-shadow: 0 4px 20px rgba(0,0,0,0.1);
            }
            
            .monitor-item {
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 0.5rem 0;
                border-bottom: 1px solid #f0f0f0;
            }
            
            .monitor-item:last-child {
                border-bottom: none;
            }
            
            .monitor-label {
                font-size: 0.9rem;
                color: #666;
            }
            
            .monitor-value {
                font-weight: 600;
                color: #333;
            }
            
            .loading {
                text-align: center;
                padding: 2rem;
                color: #666;
            }
            
            .empty-state {
                text-align: center;
                padding: 3rem 2rem;
                color: #666;
            }
            
            .empty-state h3 {
                margin-bottom: 1rem;
            }
            
            .modal {
                display: none;
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background: rgba(0,0,0,0.5);
                z-index: 1000;
                align-items: center;
                justify-content: center;
            }
            
            .modal.show {
                display: flex;
            }
            
            .modal-content {
                background: white;
                border-radius: 12px;
                padding: 2rem;
                max-width: 600px;
                width: 90%;
                max-height: 80vh;
                overflow-y: auto;
            }
            
            .modal-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 1.5rem;
                padding-bottom: 1rem;
                border-bottom: 1px solid #eee;
            }
            
            .close {
                background: none;
                border: none;
                font-size: 1.5rem;
                cursor: pointer;
                color: #999;
            }
            
            .form-group {
                margin-bottom: 1rem;
            }
            
            .form-group label {
                display: block;
                margin-bottom: 0.5rem;
                font-weight: 600;
            }
            
            .form-group input, .form-group textarea, .form-group select {
                width: 100%;
                padding: 0.75rem;
                border: 2px solid #e0e0e0;
                border-radius: 6px;
                font-size: 1rem;
            }
            
            .experiment-builder {
                border: 1px solid #e5e5e5;
                border-radius: 8px;
                padding: 1rem;
                margin-bottom: 1rem;
            }
            
            .experiment-header {
                display: flex;
                justify-content: between;
                align-items: center;
                margin-bottom: 1rem;
            }
            
            .remove-experiment {
                background: #ef4444;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 0.25rem 0.5rem;
                cursor: pointer;
                font-size: 0.8rem;
            }
            
            @media (max-width: 1024px) {
                .main-content {
                    grid-template-columns: 1fr;
                }
                
                .stats-grid {
                    grid-template-columns: repeat(2, 1fr);
                }
            }
            
            @media (max-width: 768px) {
                .stats-grid {
                    grid-template-columns: 1fr;
                }
                
                .header {
                    flex-direction: column;
                    gap: 1rem;
                    text-align: center;
                }
            }
        </style>
    </head>
    <body>
        <div class="header">
            <div class="header-left">
                <h1>⚡ Batch Processing</h1>
                <p>Manage multiple experiments simultaneously</p>
            </div>
            <div class="header-right">
                <button class="btn btn-primary" onclick="showCreateBatchModal()">
                    + Create Batch
                </button>
                <button class="btn btn-secondary" onclick="refreshData()">
                    🔄 Refresh
                </button>
            </div>
        </div>
        
        <div class="container">
            <!-- Statistics Overview -->
            <div class="stats-grid" id="statsGrid">
                <div class="stat-card total">
                    <div class="stat-value" id="totalBatches">-</div>
                    <div class="stat-label">Total Batches</div>
                </div>
                <div class="stat-card running">
                    <div class="stat-value" id="runningBatches">-</div>
                    <div class="stat-label">Running</div>
                </div>
                <div class="stat-card completed">
                    <div class="stat-value" id="completedBatches">-</div>
                    <div class="stat-label">Completed</div>
                </div>
                <div class="stat-card failed">
                    <div class="stat-value" id="failedBatches">-</div>
                    <div class="stat-label">Failed</div>
                </div>
            </div>
            
            <div class="main-content">
                <div class="batches-section">
                    <div class="section-header">
                        <h2 class="section-title">Batch Jobs</h2>
                        <div>
                            <select id="statusFilter" onchange="filterBatches()">
                                <option value="">All Status</option>
                                <option value="created">Created</option>
                                <option value="running">Running</option>
                                <option value="paused">Paused</option>
                                <option value="completed">Completed</option>
                                <option value="failed">Failed</option>
                                <option value="cancelled">Cancelled</option>
                            </select>
                        </div>
                    </div>
                    <div id="batchList" class="batch-list">
                        <div class="loading">Loading batches...</div>
                    </div>
                </div>
                
                <div class="sidebar">
                    <div class="quick-actions">
                        <h3 class="section-title">Quick Actions</h3>
                        <div style="display: flex; flex-direction: column; gap: 0.5rem; margin-top: 1rem;">
                            <button class="btn btn-primary" onclick="showCreateBatchModal()">
                                + Create New Batch
                            </button>
                            <button class="btn btn-secondary" onclick="showTemplatesBrowser()">
                                📋 Browse Templates
                            </button>
                            <button class="btn btn-secondary" onclick="exportBatchData()">
                                📤 Export Data
                            </button>
                        </div>
                    </div>
                    
                    <div class="system-monitor">
                        <h3 class="section-title">System Monitor</h3>
                        <div id="systemStats">
                            <div class="monitor-item">
                                <span class="monitor-label">Global Running</span>
                                <span class="monitor-value" id="globalRunning">-</span>
                            </div>
                            <div class="monitor-item">
                                <span class="monitor-label">Max Concurrent</span>
                                <span class="monitor-value" id="maxConcurrent">-</span>
                            </div>
                            <div class="monitor-item">
                                <span class="monitor-label">Total Experiments</span>
                                <span class="monitor-value" id="totalExperiments">-</span>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Create Batch Modal -->
        <div id="createBatchModal" class="modal">
            <div class="modal-content">
                <div class="modal-header">
                    <h2>Create New Batch</h2>
                    <button class="close" onclick="closeModal()">&times;</button>
                </div>
                <form id="createBatchForm">
                    <div class="form-group">
                        <label for="batchName">Batch Name</label>
                        <input type="text" id="batchName" required placeholder="My Experiment Batch">
                    </div>
                    <div class="form-group">
                        <label for="batchDescription">Description</label>
                        <textarea id="batchDescription" placeholder="Describe this batch of experiments..."></textarea>
                    </div>
                    <div class="form-group">
                        <label for="maxConcurrentInput">Max Concurrent Experiments</label>
                        <input type="number" id="maxConcurrentInput" value="3" min="1" max="10">
                    </div>
                    <div class="form-group">
                        <label>
                            <input type="checkbox" id="autoStart"> Auto-start batch immediately
                        </label>
                    </div>
                    
                    <h3>Experiments</h3>
                    <div id="experimentsContainer">
                        <!-- Experiments will be added here -->
                    </div>
                    <button type="button" class="btn btn-secondary" onclick="addExperiment()">
                        + Add Experiment
                    </button>
                    
                    <div style="margin-top: 2rem; display: flex; gap: 1rem;">
                        <button type="button" class="btn btn-secondary" onclick="closeModal()">Cancel</button>
                        <button type="submit" class="btn btn-primary">Create Batch</button>
                    </div>
                </form>
            </div>
        </div>
        
        <script>
            let batches = [];
            let templates = [];
            let systemStats = {};
            
            // Load data on page load
            document.addEventListener('DOMContentLoaded', () => {
                loadTemplates();
                loadBatches();
                loadSystemStats();
                
                // Refresh data every 5 seconds
                setInterval(() => {
                    loadBatches();
                    loadSystemStats();
                }, 5000);
            });
            
            async function loadTemplates() {
                try {
                    const response = await fetch('/templates/');
                    templates = await response.json();
                } catch (error) {
                    console.error('Error loading templates:', error);
                }
            }
            
            async function loadBatches() {
                try {
                    const response = await fetch('/batch/');
                    if (!response.ok) throw new Error('Failed to load batches');
                    
                    batches = await response.json();
                    displayBatches(batches);
                    updateStatistics();
                } catch (error) {
                    console.error('Error loading batches:', error);
                    document.getElementById('batchList').innerHTML = 
                        '<div class="loading">Failed to load batches</div>';
                }
            }
            
            async function loadSystemStats() {
                try {
                    const response = await fetch('/batch/status/overview');
                    if (!response.ok) throw new Error('Failed to load system stats');
                    
                    systemStats = await response.json();
                    updateSystemMonitor();
                } catch (error) {
                    console.error('Error loading system stats:', error);
                }
            }
            
            function updateStatistics() {
                const stats = {
                    total: batches.length,
                    running: batches.filter(b => b.status === 'running').length,
                    completed: batches.filter(b => b.status === 'completed').length,
                    failed: batches.filter(b => b.status === 'failed').length
                };
                
                document.getElementById('totalBatches').textContent = stats.total;
                document.getElementById('runningBatches').textContent = stats.running;
                document.getElementById('completedBatches').textContent = stats.completed;
                document.getElementById('failedBatches').textContent = stats.failed;
            }
            
            function updateSystemMonitor() {
                if (systemStats.system_stats) {
                    document.getElementById('globalRunning').textContent = 
                        systemStats.system_stats.global_running_experiments;
                    document.getElementById('maxConcurrent').textContent = 
                        systemStats.system_stats.max_global_concurrent;
                }
                
                if (systemStats.user_stats) {
                    document.getElementById('totalExperiments').textContent = 
                        systemStats.user_stats.total_experiments;
                }
            }
            
            function displayBatches(batchesToShow) {
                const container = document.getElementById('batchList');
                
                if (batchesToShow.length === 0) {
                    container.innerHTML = `
                        <div class="empty-state">
                            <h3>No batches found</h3>
                            <p>Create your first batch to get started!</p>
                        </div>
                    `;
                    return;
                }
                
                const batchesHTML = batchesToShow.map(batch => `
                    <div class="batch-item" onclick="viewBatchDetails('${batch.batch_id}')">
                        <div class="batch-header">
                            <div class="batch-name">${batch.name}</div>
                            <div class="batch-status status-${batch.status}">${batch.status}</div>
                        </div>
                        
                        <div class="batch-description">${batch.description}</div>
                        
                        <div class="batch-progress">
                            <div class="progress-bar">
                                <div class="progress-fill" style="width: ${batch.progress_percentage}%"></div>
                            </div>
                            <div class="progress-text">
                                <span>${batch.completed_experiments}/${batch.total_experiments} completed</span>
                                <span>${batch.progress_percentage.toFixed(1)}%</span>
                            </div>
                        </div>
                        
                        <div class="batch-actions" onclick="event.stopPropagation()">
                            ${getBatchActionButtons(batch)}
                        </div>
                    </div>
                `).join('');
                
                container.innerHTML = batchesHTML;
            }
            
            function getBatchActionButtons(batch) {
                let buttons = '';
                
                if (batch.status === 'created') {
                    buttons += `<button class="btn btn-sm btn-success" onclick="startBatch('${batch.batch_id}')">▶ Start</button>`;
                } else if (batch.status === 'running') {
                    buttons += `<button class="btn btn-sm btn-warning" onclick="pauseBatch('${batch.batch_id}')">⏸ Pause</button>`;
                } else if (batch.status === 'paused') {
                    buttons += `<button class="btn btn-sm btn-success" onclick="resumeBatch('${batch.batch_id}')">▶ Resume</button>`;
                }
                
                if (['created', 'running', 'paused'].includes(batch.status)) {
                    buttons += `<button class="btn btn-sm btn-danger" onclick="cancelBatch('${batch.batch_id}')">✖ Cancel</button>`;
                }
                
                buttons += `<button class="btn btn-sm btn-secondary" onclick="viewBatchDetails('${batch.batch_id}')">👁 Details</button>`;
                
                return buttons;
            }
            
            function filterBatches() {
                const status = document.getElementById('statusFilter').value;
                const filtered = status ? batches.filter(b => b.status === status) : batches;
                displayBatches(filtered);
            }
            
            function showCreateBatchModal() {
                document.getElementById('createBatchModal').classList.add('show');
                addExperiment(); // Add first experiment by default
            }
            
            function closeModal() {
                document.getElementById('createBatchModal').classList.remove('show');
                document.getElementById('createBatchForm').reset();
                document.getElementById('experimentsContainer').innerHTML = '';
            }
            
            function addExperiment() {
                const container = document.getElementById('experimentsContainer');
                const experimentIndex = container.children.length;
                
                const experimentHTML = `
                    <div class="experiment-builder" data-index="${experimentIndex}">
                        <div class="experiment-header">
                            <h4>Experiment ${experimentIndex + 1}</h4>
                            <button type="button" class="remove-experiment" onclick="removeExperiment(${experimentIndex})">Remove</button>
                        </div>
                        <div class="form-group">
                            <label>Experiment Name</label>
                            <input type="text" name="exp_name_${experimentIndex}" required placeholder="Experiment ${experimentIndex + 1}">
                        </div>
                        <div class="form-group">
                            <label>Template</label>
                            <select name="exp_template_${experimentIndex}" required>
                                <option value="">Select Template</option>
                                ${templates.map(t => `<option value="${t.id}">${t.name}</option>`).join('')}
                            </select>
                        </div>
                        <div class="form-group">
                            <label>Priority</label>
                            <select name="exp_priority_${experimentIndex}">
                                <option value="1">Low</option>
                                <option value="2" selected>Normal</option>
                                <option value="3">High</option>
                            </select>
                        </div>
                    </div>
                `;
                
                container.insertAdjacentHTML('beforeend', experimentHTML);
            }
            
            function removeExperiment(index) {
                const experiment = document.querySelector(`[data-index="${index}"]`);
                if (experiment) {
                    experiment.remove();
                }
            }
            
            async function startBatch(batchId) {
                try {
                    const response = await fetch(`/batch/${batchId}/start`, { method: 'POST' });
                    if (!response.ok) throw new Error('Failed to start batch');
                    
                    await loadBatches();
                } catch (error) {
                    alert('Failed to start batch: ' + error.message);
                }
            }
            
            async function pauseBatch(batchId) {
                try {
                    const response = await fetch(`/batch/${batchId}/pause`, { method: 'POST' });
                    if (!response.ok) throw new Error('Failed to pause batch');
                    
                    await loadBatches();
                } catch (error) {
                    alert('Failed to pause batch: ' + error.message);
                }
            }
            
            async function resumeBatch(batchId) {
                try {
                    const response = await fetch(`/batch/${batchId}/resume`, { method: 'POST' });
                    if (!response.ok) throw new Error('Failed to resume batch');
                    
                    await loadBatches();
                } catch (error) {
                    alert('Failed to resume batch: ' + error.message);
                }
            }
            
            async function cancelBatch(batchId) {
                if (!confirm('Are you sure you want to cancel this batch?')) return;
                
                try {
                    const response = await fetch(`/batch/${batchId}/cancel`, { method: 'POST' });
                    if (!response.ok) throw new Error('Failed to cancel batch');
                    
                    await loadBatches();
                } catch (error) {
                    alert('Failed to cancel batch: ' + error.message);
                }
            }
            
            function viewBatchDetails(batchId) {
                // Navigate to detailed view
                window.location.href = `/batch/${batchId}`;
            }
            
            function refreshData() {
                loadBatches();
                loadSystemStats();
            }
            
            function showTemplatesBrowser() {
                window.location.href = '/templates-ui/';
            }
            
            function exportBatchData() {
                // Export batch data as JSON
                const dataStr = JSON.stringify(batches, null, 2);
                const dataBlob = new Blob([dataStr], {type: 'application/json'});
                const url = URL.createObjectURL(dataBlob);
                
                const link = document.createElement('a');
                link.href = url;
                link.download = `batch-data-${new Date().toISOString().split('T')[0]}.json`;
                link.click();
                
                URL.revokeObjectURL(url);
            }
            
            // Handle form submission
            document.getElementById('createBatchForm').addEventListener('submit', async (e) => {
                e.preventDefault();
                
                const formData = new FormData(e.target);
                const experiments = [];
                
                // Collect experiment data
                const container = document.getElementById('experimentsContainer');
                for (let i = 0; i < container.children.length; i++) {
                    const name = formData.get(`exp_name_${i}`);
                    const template = formData.get(`exp_template_${i}`);
                    const priority = parseInt(formData.get(`exp_priority_${i}`)) || 2;
                    
                    if (name && template) {
                        experiments.push({
                            name: name,
                            template_id: template,
                            priority: priority,
                            dependencies: []
                        });
                    }
                }
                
                if (experiments.length === 0) {
                    alert('Please add at least one experiment');
                    return;
                }
                
                const batchData = {
                    name: document.getElementById('batchName').value,
                    description: document.getElementById('batchDescription').value,
                    max_concurrent: parseInt(document.getElementById('maxConcurrentInput').value),
                    auto_start: document.getElementById('autoStart').checked,
                    experiments: experiments
                };
                
                try {
                    const response = await fetch('/batch/', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(batchData)
                    });
                    
                    if (!response.ok) throw new Error('Failed to create batch');
                    
                    closeModal();
                    await loadBatches();
                    
                } catch (error) {
                    alert('Failed to create batch: ' + error.message);
                }
            });
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)
