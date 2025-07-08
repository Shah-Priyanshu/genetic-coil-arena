"""
Phase 4A: Experiment Templates Dashboard
Beautiful web interface for browsing and using experiment templates.
"""

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

templates_ui_router = APIRouter(prefix="/templates-ui", tags=["templates-ui"])


@templates_ui_router.get("/", response_class=HTMLResponse)
async def templates_dashboard():
    """Serve the experiment templates dashboard."""
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Experiment Templates - Genetic Coil Arena</title>
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
            }
            
            .header h1 {
                font-size: 2rem;
                margin-bottom: 0.5rem;
            }
            
            .header p {
                opacity: 0.9;
                font-size: 1.1rem;
            }
            
            .container {
                max-width: 1200px;
                margin: 2rem auto;
                padding: 0 2rem;
            }
            
            .filters {
                background: white;
                border-radius: 12px;
                padding: 1.5rem;
                margin-bottom: 2rem;
                box-shadow: 0 4px 20px rgba(0,0,0,0.1);
                display: flex;
                gap: 1rem;
                flex-wrap: wrap;
                align-items: center;
            }
            
            .filter-group {
                display: flex;
                flex-direction: column;
                gap: 0.5rem;
            }
            
            .filter-group label {
                font-weight: 600;
                color: #555;
                font-size: 0.9rem;
            }
            
            select, input {
                padding: 0.5rem;
                border: 2px solid #e0e0e0;
                border-radius: 6px;
                font-size: 0.9rem;
                transition: border-color 0.2s;
            }
            
            select:focus, input:focus {
                outline: none;
                border-color: #667eea;
            }
            
            .templates-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
                gap: 1.5rem;
                margin-bottom: 2rem;
            }
            
            .template-card {
                background: white;
                border-radius: 12px;
                padding: 1.5rem;
                box-shadow: 0 4px 20px rgba(0,0,0,0.1);
                transition: transform 0.3s ease, box-shadow 0.3s ease;
                cursor: pointer;
                position: relative;
                overflow: hidden;
            }
            
            .template-card:hover {
                transform: translateY(-5px);
                box-shadow: 0 8px 30px rgba(0,0,0,0.15);
            }
            
            .template-header {
                display: flex;
                align-items: flex-start;
                justify-content: space-between;
                margin-bottom: 1rem;
            }
            
            .template-title {
                font-size: 1.3rem;
                font-weight: 700;
                color: #333;
                margin-bottom: 0.5rem;
            }
            
            .difficulty-badge {
                padding: 0.25rem 0.75rem;
                border-radius: 20px;
                font-size: 0.8rem;
                font-weight: 600;
                text-transform: uppercase;
            }
            
            .difficulty-beginner {
                background: #e8f5e8;
                color: #2e7d2e;
            }
            
            .difficulty-intermediate {
                background: #fff3cd;
                color: #856404;
            }
            
            .difficulty-advanced {
                background: #f8d7da;
                color: #721c24;
            }
            
            .template-description {
                color: #666;
                margin-bottom: 1rem;
                line-height: 1.5;
            }
            
            .template-meta {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 1rem;
                font-size: 0.9rem;
                color: #888;
            }
            
            .tags {
                display: flex;
                flex-wrap: wrap;
                gap: 0.5rem;
                margin-bottom: 1rem;
            }
            
            .tag {
                background: #f0f0f0;
                color: #666;
                padding: 0.25rem 0.5rem;
                border-radius: 12px;
                font-size: 0.8rem;
            }
            
            .template-actions {
                display: flex;
                gap: 0.5rem;
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
                font-size: 0.9rem;
            }
            
            .btn-primary {
                background: #667eea;
                color: white;
            }
            
            .btn-primary:hover {
                background: #5a67d8;
                transform: translateY(-1px);
            }
            
            .btn-secondary {
                background: #f7f7f7;
                color: #666;
                border: 1px solid #ddd;
            }
            
            .btn-secondary:hover {
                background: #eaeaea;
            }
            
            .loading {
                text-align: center;
                padding: 2rem;
                color: white;
                font-size: 1.1rem;
            }
            
            .error {
                background: #f8d7da;
                color: #721c24;
                padding: 1rem;
                border-radius: 6px;
                margin: 1rem 0;
            }
            
            .empty-state {
                text-align: center;
                padding: 3rem 2rem;
                background: white;
                border-radius: 12px;
                box-shadow: 0 4px 20px rgba(0,0,0,0.1);
            }
            
            .empty-state h3 {
                color: #666;
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
                max-width: 500px;
                width: 90%;
                max-height: 80vh;
                overflow-y: auto;
            }
            
            .modal-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 1rem;
                padding-bottom: 1rem;
                border-bottom: 1px solid #eee;
            }
            
            .modal-title {
                font-size: 1.5rem;
                font-weight: 700;
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
                color: #333;
            }
            
            .form-group input, .form-group textarea {
                width: 100%;
                padding: 0.75rem;
                border: 2px solid #e0e0e0;
                border-radius: 6px;
                font-size: 1rem;
            }
            
            .form-group textarea {
                resize: vertical;
                min-height: 100px;
            }
            
            @media (max-width: 768px) {
                .templates-grid {
                    grid-template-columns: 1fr;
                }
                
                .filters {
                    flex-direction: column;
                    align-items: stretch;
                }
                
                .filter-group {
                    width: 100%;
                }
            }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🧬 Experiment Templates</h1>
            <p>Choose from pre-configured experiments or create your own</p>
        </div>
        
        <div class="container">
            <div class="filters">
                <div class="filter-group">
                    <label for="categoryFilter">Category</label>
                    <select id="categoryFilter">
                        <option value="">All Categories</option>
                        <option value="quick_start">Quick Start</option>
                        <option value="performance_test">Performance Test</option>
                        <option value="long_evolution">Long Evolution</option>
                        <option value="comparison_study">Comparison Study</option>
                        <option value="parameter_sweep">Parameter Sweep</option>
                        <option value="baseline_benchmark">Baseline Benchmark</option>
                    </select>
                </div>
                
                <div class="filter-group">
                    <label for="difficultyFilter">Difficulty</label>
                    <select id="difficultyFilter">
                        <option value="">All Levels</option>
                        <option value="beginner">Beginner</option>
                        <option value="intermediate">Intermediate</option>
                        <option value="advanced">Advanced</option>
                    </select>
                </div>
                
                <div class="filter-group">
                    <label for="searchFilter">Search</label>
                    <input type="text" id="searchFilter" placeholder="Search templates...">
                </div>
                
                <div class="filter-group">
                    <label>&nbsp;</label>
                    <button class="btn btn-primary" onclick="refreshTemplates()">🔄 Refresh</button>
                </div>
            </div>
            
            <div id="templatesContainer">
                <div class="loading">Loading templates...</div>
            </div>
        </div>
        
        <!-- Create Experiment Modal -->
        <div id="createExperimentModal" class="modal">
            <div class="modal-content">
                <div class="modal-header">
                    <h2 class="modal-title">Create Experiment</h2>
                    <button class="close" onclick="closeModal()">&times;</button>
                </div>
                <form id="createExperimentForm">
                    <div class="form-group">
                        <label for="experimentName">Experiment Name</label>
                        <input type="text" id="experimentName" required placeholder="My Evolution Experiment">
                    </div>
                    <div class="form-group">
                        <label for="experimentDescription">Description (optional)</label>
                        <textarea id="experimentDescription" placeholder="Describe your experiment..."></textarea>
                    </div>
                    <div class="template-actions">
                        <button type="button" class="btn btn-secondary" onclick="closeModal()">Cancel</button>
                        <button type="submit" class="btn btn-primary">Create Experiment</button>
                    </div>
                </form>
            </div>
        </div>
        
        <script>
            let templates = [];
            let selectedTemplate = null;
            
            // Load templates on page load
            document.addEventListener('DOMContentLoaded', loadTemplates);
            
            // Filter event listeners
            document.getElementById('categoryFilter').addEventListener('change', filterTemplates);
            document.getElementById('difficultyFilter').addEventListener('change', filterTemplates);
            document.getElementById('searchFilter').addEventListener('input', filterTemplates);
            
            async function loadTemplates() {
                try {
                    const response = await fetch('/templates/');
                    if (!response.ok) throw new Error('Failed to load templates');
                    
                    templates = await response.json();
                    displayTemplates(templates);
                } catch (error) {
                    console.error('Error loading templates:', error);
                    document.getElementById('templatesContainer').innerHTML = 
                        '<div class="error">Failed to load templates. Please try again.</div>';
                }
            }
            
            function displayTemplates(templatesToShow) {
                const container = document.getElementById('templatesContainer');
                
                if (templatesToShow.length === 0) {
                    container.innerHTML = `
                        <div class="empty-state">
                            <h3>No templates found</h3>
                            <p>Try adjusting your filters or create a custom template.</p>
                        </div>
                    `;
                    return;
                }
                
                const templatesHTML = templatesToShow.map(template => `
                    <div class="template-card" onclick="selectTemplate('${template.id}')">
                        <div class="template-header">
                            <div class="template-title">${template.name}</div>
                            <div class="difficulty-badge difficulty-${template.difficulty_level}">
                                ${template.difficulty_level}
                            </div>
                        </div>
                        
                        <div class="template-description">${template.description}</div>
                        
                        <div class="template-meta">
                            <span>⏱️ ${template.estimated_duration}</span>
                            <span>📁 ${template.category.replace('_', ' ')}</span>
                        </div>
                        
                        <div class="tags">
                            ${template.tags.map(tag => `<span class="tag">${tag}</span>`).join('')}
                        </div>
                        
                        <div class="template-actions" onclick="event.stopPropagation()">
                            <button class="btn btn-primary" onclick="createFromTemplate('${template.id}')">
                                🚀 Use Template
                            </button>
                            <button class="btn btn-secondary" onclick="viewTemplate('${template.id}')">
                                👁️ View Details
                            </button>
                        </div>
                    </div>
                `).join('');
                
                container.innerHTML = `<div class="templates-grid">${templatesHTML}</div>`;
            }
            
            function filterTemplates() {
                const category = document.getElementById('categoryFilter').value;
                const difficulty = document.getElementById('difficultyFilter').value;
                const search = document.getElementById('searchFilter').value.toLowerCase();
                
                let filtered = templates.filter(template => {
                    const matchesCategory = !category || template.category === category;
                    const matchesDifficulty = !difficulty || template.difficulty_level === difficulty;
                    const matchesSearch = !search || 
                        template.name.toLowerCase().includes(search) ||
                        template.description.toLowerCase().includes(search) ||
                        template.tags.some(tag => tag.toLowerCase().includes(search));
                    
                    return matchesCategory && matchesDifficulty && matchesSearch;
                });
                
                displayTemplates(filtered);
            }
            
            function selectTemplate(templateId) {
                selectedTemplate = templateId;
                // Could add visual selection feedback here
            }
            
            function createFromTemplate(templateId) {
                selectedTemplate = templateId;
                document.getElementById('createExperimentModal').classList.add('show');
                
                // Pre-fill form with template name
                const template = templates.find(t => t.id === templateId);
                if (template) {
                    document.getElementById('experimentName').value = 
                        `${template.name} - ${new Date().toLocaleDateString()}`;
                }
            }
            
            function viewTemplate(templateId) {
                // Navigate to template details
                window.location.href = `/templates/${templateId}`;
            }
            
            function closeModal() {
                document.getElementById('createExperimentModal').classList.remove('show');
                document.getElementById('createExperimentForm').reset();
            }
            
            function refreshTemplates() {
                document.getElementById('templatesContainer').innerHTML = 
                    '<div class="loading">Refreshing templates...</div>';
                loadTemplates();
            }
            
            // Handle form submission
            document.getElementById('createExperimentForm').addEventListener('submit', async (e) => {
                e.preventDefault();
                
                if (!selectedTemplate) {
                    alert('Please select a template first');
                    return;
                }
                
                const experimentName = document.getElementById('experimentName').value;
                const description = document.getElementById('experimentDescription').value;
                
                try {
                    const response = await fetch(`/templates/${selectedTemplate}/create-experiment`, {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({
                            experiment_name: experimentName,
                            custom_config: description ? { description } : null
                        })
                    });
                    
                    if (!response.ok) throw new Error('Failed to create experiment');
                    
                    const result = await response.json();
                    alert('Experiment created successfully!');
                    closeModal();
                    
                    // Redirect to experiments page
                    window.location.href = '/experiments';
                    
                } catch (error) {
                    console.error('Error creating experiment:', error);
                    alert('Failed to create experiment. Please try again.');
                }
            });
            
            // Close modal when clicking outside
            document.getElementById('createExperimentModal').addEventListener('click', (e) => {
                if (e.target.id === 'createExperimentModal') {
                    closeModal();
                }
            });
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)
