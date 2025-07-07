"""
Phase 4B: 3D Visualization UI
Interactive web interface for 3D experiment visualizations.
"""

from fastapi import Request
from fastapi.responses import HTMLResponse
from typing import Optional
import logging

logger = logging.getLogger(__name__)


def create_3d_visualization_dashboard(request: Request, user: Optional[dict] = None) -> str:
    """Create the main 3D visualization dashboard HTML."""
      user_info = ""
    if user:
        username = user.get("username", "User") if isinstance(user, dict) else "User"
        user_info = f"""
        <div class="user-info">
            <span>Welcome, {username}!</span>
            <a href="/auth/logout" class="btn btn-outline">Logout</a>
        </div>
        """
    
    return f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>3D Visualization Dashboard - Genetic Coil Arena</title>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
        <style>
            * {{
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }}
            
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #1e3c72, #2a5298);
                color: white;
                overflow-x: hidden;
            }}
            
            .container {{
                max-width: 1400px;
                margin: 0 auto;
                padding: 20px;
            }}
            
            .header {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 20px 0;
                border-bottom: 2px solid rgba(255, 255, 255, 0.1);
                margin-bottom: 30px;
            }}
            
            .header h1 {{
                font-size: 2.5rem;
                background: linear-gradient(45deg, #4fc3f7, #29b6f6, #03a9f4);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
            }}
            
            .user-info {{
                display: flex;
                gap: 15px;
                align-items: center;
            }}
            
            .btn {{
                padding: 10px 20px;
                border: none;
                border-radius: 8px;
                cursor: pointer;
                text-decoration: none;
                display: inline-block;
                transition: all 0.3s ease;
                font-weight: 500;
            }}
            
            .btn-primary {{
                background: linear-gradient(45deg, #4fc3f7, #29b6f6);
                color: white;
            }}
            
            .btn-outline {{
                border: 2px solid #4fc3f7;
                color: #4fc3f7;
                background: transparent;
            }}
            
            .btn:hover {{
                transform: translateY(-2px);
                box-shadow: 0 5px 15px rgba(0, 0, 0, 0.3);
            }}
            
            .dashboard-grid {{
                display: grid;
                grid-template-columns: 300px 1fr;
                gap: 20px;
                height: calc(100vh - 200px);
            }}
            
            .sidebar {{
                background: rgba(255, 255, 255, 0.1);
                border-radius: 12px;
                padding: 20px;
                backdrop-filter: blur(10px);
                border: 1px solid rgba(255, 255, 255, 0.2);
            }}
            
            .sidebar h3 {{
                margin-bottom: 20px;
                color: #4fc3f7;
            }}
            
            .control-group {{
                margin-bottom: 20px;
            }}
            
            .control-group label {{
                display: block;
                margin-bottom: 5px;
                font-weight: 500;
            }}
            
            .control-group select,
            .control-group input {{
                width: 100%;
                padding: 8px 12px;
                border: 1px solid rgba(255, 255, 255, 0.3);
                border-radius: 6px;
                background: rgba(255, 255, 255, 0.1);
                color: white;
                backdrop-filter: blur(5px);
            }}
            
            .control-group select option {{
                background: #1e3c72;
                color: white;
            }}
            
            .visualization-container {{
                background: rgba(0, 0, 0, 0.2);
                border-radius: 12px;
                position: relative;
                overflow: hidden;
                border: 1px solid rgba(255, 255, 255, 0.2);
            }}
            
            #threejs-canvas {{
                width: 100%;
                height: 100%;
                display: block;
            }}
            
            .status-bar {{
                position: absolute;
                top: 10px;
                left: 10px;
                background: rgba(0, 0, 0, 0.7);
                padding: 10px;
                border-radius: 6px;
                font-size: 0.9rem;
            }}
            
            .legend {{
                position: absolute;
                bottom: 10px;
                right: 10px;
                background: rgba(0, 0, 0, 0.7);
                padding: 15px;
                border-radius: 6px;
                max-width: 200px;
            }}
            
            .legend-item {{
                display: flex;
                align-items: center;
                margin-bottom: 5px;
            }}
            
            .legend-color {{
                width: 20px;
                height: 20px;
                border-radius: 3px;
                margin-right: 10px;
            }}
            
            .loading {{
                position: absolute;
                top: 50%;
                left: 50%;
                transform: translate(-50%, -50%);
                text-align: center;
            }}
            
            .spinner {{
                border: 4px solid rgba(255, 255, 255, 0.3);
                border-radius: 50%;
                border-top: 4px solid #4fc3f7;
                width: 50px;
                height: 50px;
                animation: spin 1s linear infinite;
                margin: 0 auto 10px;
            }}
            
            @keyframes spin {{
                0% {{ transform: rotate(0deg); }}
                100% {{ transform: rotate(360deg); }}
            }}
            
            .error {{
                background: rgba(255, 69, 58, 0.2);
                border: 1px solid rgba(255, 69, 58, 0.5);
                padding: 15px;
                border-radius: 8px;
                margin: 20px 0;
            }}
            
            .tab-container {{
                margin-bottom: 20px;
            }}
            
            .tabs {{
                display: flex;
                border-bottom: 2px solid rgba(255, 255, 255, 0.2);
            }}
            
            .tab {{
                padding: 10px 20px;
                cursor: pointer;
                border-bottom: 2px solid transparent;
                transition: all 0.3s ease;
            }}
            
            .tab.active {{
                border-bottom-color: #4fc3f7;
                color: #4fc3f7;
            }}
            
            .streaming-controls {{
                display: flex;
                gap: 10px;
                margin-top: 15px;
            }}
            
            .btn-small {{
                padding: 6px 12px;
                font-size: 0.9rem;
            }}
            
            .metrics {{
                background: rgba(255, 255, 255, 0.1);
                border-radius: 8px;
                padding: 15px;
                margin-top: 20px;
            }}
            
            .metric {{
                display: flex;
                justify-content: space-between;
                margin-bottom: 10px;
            }}
            
            .metric:last-child {{
                margin-bottom: 0;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🌌 3D Visualization Lab</h1>
                {user_info}
            </div>
            
            <div class="dashboard-grid">
                <div class="sidebar">
                    <h3>📊 Visualization Controls</h3>
                    
                    <div class="control-group">
                        <label for="experiment-select">Experiment:</label>
                        <select id="experiment-select">
                            <option value="">Select an experiment...</option>
                        </select>
                    </div>
                    
                    <div class="tab-container">
                        <div class="tabs">
                            <div class="tab active" data-tab="landscape">Landscape</div>
                            <div class="tab" data-tab="population">Population</div>
                            <div class="tab" data-tab="trajectory">Trajectory</div>
                        </div>
                    </div>
                    
                    <div id="landscape-controls" class="tab-content">
                        <div class="control-group">
                            <label for="generation-select">Generation:</label>
                            <select id="generation-select">
                                <option value="">Latest</option>
                            </select>
                        </div>
                        
                        <div class="control-group">
                            <label for="resolution-slider">Resolution:</label>
                            <input type="range" id="resolution-slider" min="20" max="100" value="50">
                            <span id="resolution-value">50</span>
                        </div>
                        
                        <div class="control-group">
                            <label for="color-mapping">Color Mapping:</label>
                            <select id="color-mapping">
                                <option value="fitness">Fitness</option>
                                <option value="generation">Generation</option>
                                <option value="diversity">Diversity</option>
                            </select>
                        </div>
                    </div>
                    
                    <div id="population-controls" class="tab-content" style="display: none;">
                        <div class="control-group">
                            <label for="pop-generation-select">Generation:</label>
                            <select id="pop-generation-select">
                                <option value="">Latest</option>
                            </select>
                        </div>
                        
                        <div class="control-group">
                            <label>
                                <input type="checkbox" id="show-elite" checked> Show Elite Individuals
                            </label>
                        </div>
                        
                        <div class="control-group">
                            <label>
                                <input type="checkbox" id="show-connections"> Show Population Connections
                            </label>
                        </div>
                    </div>
                    
                    <div id="trajectory-controls" class="tab-content" style="display: none;">
                        <div class="control-group">
                            <label for="max-generations">Max Generations:</label>
                            <input type="number" id="max-generations" value="100" min="10" max="1000">
                        </div>
                        
                        <div class="control-group">
                            <label for="trajectory-type">Trajectory Type:</label>
                            <select id="trajectory-type">
                                <option value="fitness">Fitness Evolution</option>
                                <option value="diversity">Diversity Evolution</option>
                                <option value="combined">Combined View</option>
                            </select>
                        </div>
                    </div>
                    
                    <div class="streaming-controls">
                        <button id="start-stream" class="btn btn-primary btn-small">🔴 Start Stream</button>
                        <button id="stop-stream" class="btn btn-outline btn-small" disabled>⏹️ Stop</button>
                    </div>
                    
                    <div class="metrics">
                        <h4>📈 Live Metrics</h4>
                        <div class="metric">
                            <span>Generation:</span>
                            <span id="current-generation">-</span>
                        </div>
                        <div class="metric">
                            <span>Best Fitness:</span>
                            <span id="best-fitness">-</span>
                        </div>
                        <div class="metric">
                            <span>Diversity:</span>
                            <span id="diversity-score">-</span>
                        </div>
                        <div class="metric">
                            <span>Population Size:</span>
                            <span id="population-size">-</span>
                        </div>
                    </div>
                    
                    <div class="control-group">
                        <button id="export-animation" class="btn btn-outline" style="width: 100%; margin-top: 15px;">
                            🎬 Export Animation
                        </button>
                    </div>
                </div>
                
                <div class="visualization-container">
                    <canvas id="threejs-canvas"></canvas>
                    
                    <div class="status-bar">
                        <div id="status-text">Ready</div>
                    </div>
                    
                    <div class="legend">
                        <h4>Legend</h4>
                        <div class="legend-item">
                            <div class="legend-color" style="background: rgb(255, 215, 0);"></div>
                            <span>Elite Individuals</span>
                        </div>
                        <div class="legend-item">
                            <div class="legend-color" style="background: linear-gradient(to right, red, green);"></div>
                            <span>Fitness Gradient</span>
                        </div>
                        <div class="legend-item">
                            <div class="legend-color" style="background: rgb(100, 150, 255);"></div>
                            <span>Regular Population</span>
                        </div>
                    </div>
                    
                    <div id="loading" class="loading" style="display: none;">
                        <div class="spinner"></div>
                        <div>Loading visualization...</div>
                    </div>
                </div>
            </div>
        </div>
        
        <script>
            // Global variables
            let scene, camera, renderer, controls;
            let currentVisualization = null;
            let websocket = null;
            let isStreaming = false;
            
            // Initialize Three.js
            function initThreeJS() {{
                const canvas = document.getElementById('threejs-canvas');
                const container = canvas.parentElement;
                
                // Scene setup
                scene = new THREE.Scene();
                scene.background = new THREE.Color(0x0a0a0a);
                
                // Camera setup
                camera = new THREE.PerspectiveCamera(75, container.clientWidth / container.clientHeight, 0.1, 1000);
                camera.position.set(5, 5, 5);
                
                // Renderer setup
                renderer = new THREE.WebGLRenderer({{ canvas: canvas, antialias: true }});
                renderer.setSize(container.clientWidth, container.clientHeight);
                renderer.shadowMap.enabled = true;
                renderer.shadowMap.type = THREE.PCFSoftShadowMap;
                
                // Controls
                controls = new THREE.OrbitControls(camera, renderer.domElement);
                controls.enableDamping = true;
                controls.dampingFactor = 0.05;
                
                // Lighting
                const ambientLight = new THREE.AmbientLight(0x404040, 0.6);
                scene.add(ambientLight);
                
                const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
                directionalLight.position.set(10, 10, 5);
                directionalLight.castShadow = true;
                scene.add(directionalLight);
                
                // Grid helper
                const gridHelper = new THREE.GridHelper(10, 10, 0x444444, 0x222222);
                scene.add(gridHelper);
                
                // Axes helper
                const axesHelper = new THREE.AxesHelper(5);
                scene.add(axesHelper);
                
                // Resize handler
                window.addEventListener('resize', onWindowResize);
                
                // Start render loop
                animate();
            }}
            
            function onWindowResize() {{
                const container = document.getElementById('threejs-canvas').parentElement;
                camera.aspect = container.clientWidth / container.clientHeight;
                camera.updateProjectionMatrix();
                renderer.setSize(container.clientWidth, container.clientHeight);
            }}
            
            function animate() {{
                requestAnimationFrame(animate);
                controls.update();
                renderer.render(scene, camera);
            }}
            
            // API functions
            async function loadExperiments() {{
                try {{
                    const response = await fetch('/api/v1/experiments/');
                    const data = await response.json();
                    
                    const select = document.getElementById('experiment-select');
                    select.innerHTML = '<option value="">Select an experiment...</option>';
                    
                    data.experiments.forEach(exp => {{
                        const option = document.createElement('option');
                        option.value = exp.id;
                        option.textContent = `${{exp.name}} (ID: ${{exp.id}})`;
                        select.appendChild(option);
                    }});
                }} catch (error) {{
                    console.error('Error loading experiments:', error);
                    updateStatus('Error loading experiments', 'error');
                }}
            }}
            
            async function loadVisualization() {{
                const experimentId = document.getElementById('experiment-select').value;
                if (!experimentId) return;
                
                const activeTab = document.querySelector('.tab.active').dataset.tab;
                
                showLoading(true);
                updateStatus('Loading visualization...', 'loading');
                
                try {{
                    let url, config = {{}};
                    
                    switch (activeTab) {{
                        case 'landscape':
                            url = `/api/v1/visualization3d/experiments/${{experimentId}}/fitness-landscape`;
                            const generation = document.getElementById('generation-select').value;
                            const resolution = document.getElementById('resolution-slider').value;
                            if (generation) config.generation = generation;
                            config.resolution = parseInt(resolution);
                            break;
                            
                        case 'population':
                            url = `/api/v1/visualization3d/experiments/${{experimentId}}/population-cloud`;
                            const popGeneration = document.getElementById('pop-generation-select').value;
                            const showElite = document.getElementById('show-elite').checked;
                            if (popGeneration) config.generation = popGeneration;
                            config.show_elite = showElite;
                            break;
                            
                        case 'trajectory':
                            url = `/api/v1/visualization3d/experiments/${{experimentId}}/evolution-trajectory`;
                            const maxGenerations = document.getElementById('max-generations').value;
                            config.max_generations = parseInt(maxGenerations);
                            break;
                    }}
                    
                    // Build query string
                    const params = new URLSearchParams(config).toString();
                    const fullUrl = params ? `${{url}}?${{params}}` : url;
                    
                    const response = await fetch(fullUrl);
                    const data = await response.json();
                    
                    if (data.error) {{
                        throw new Error(data.error);
                    }}
                    
                    renderVisualization(data, activeTab);
                    updateStatus('Visualization loaded successfully', 'success');
                    
                }} catch (error) {{
                    console.error('Error loading visualization:', error);
                    updateStatus(`Error: ${{error.message}}`, 'error');
                }} finally {{
                    showLoading(false);
                }}
            }}
            
            function renderVisualization(data, type) {{
                // Clear existing visualization
                clearScene();
                
                switch (type) {{
                    case 'landscape':
                        renderFitnessLandscape(data.landscape || data);
                        break;
                    case 'population':
                        renderPopulationCloud(data.population || data);
                        break;
                    case 'trajectory':
                        renderEvolutionTrajectory(data.trajectory || data);
                        break;
                }}
                
                // Update metrics if available
                if (data.metadata) {{
                    updateMetrics(data.metadata);
                }}
            }}
            
            function renderFitnessLandscape(landscapeData) {{
                if (!landscapeData.geometry || !landscapeData.geometry.vertices) {{
                    console.error('Invalid landscape data');
                    return;
                }}
                
                const vertices = landscapeData.geometry.vertices;
                const colors = landscapeData.geometry.colors || [];
                
                // Create geometry
                const geometry = new THREE.BufferGeometry();
                const positions = new Float32Array(vertices.length * 3);
                const colorsArray = new Float32Array(vertices.length * 3);
                
                vertices.forEach((vertex, i) => {{
                    positions[i * 3] = vertex.x * 10; // Scale for visibility
                    positions[i * 3 + 1] = vertex.z * 5; // Height from fitness
                    positions[i * 3 + 2] = vertex.y * 10;
                    
                    // Parse color
                    const color = colors[i] || 'rgb(100, 150, 255)';
                    const rgb = color.match(/\\d+/g) || [100, 150, 255];
                    colorsArray[i * 3] = parseInt(rgb[0]) / 255;
                    colorsArray[i * 3 + 1] = parseInt(rgb[1]) / 255;
                    colorsArray[i * 3 + 2] = parseInt(rgb[2]) / 255;
                }});
                
                geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
                geometry.setAttribute('color', new THREE.BufferAttribute(colorsArray, 3));
                
                // Create material
                const material = new THREE.PointsMaterial({{ 
                    size: 0.1, 
                    vertexColors: true,
                    transparent: true,
                    opacity: 0.8
                }});
                
                // Create points
                const points = new THREE.Points(geometry, material);
                scene.add(points);
                
                // Add wireframe if faces are available
                if (landscapeData.geometry.faces && landscapeData.geometry.faces.length > 0) {{
                    const faces = landscapeData.geometry.faces;
                    const indices = [];
                    
                    faces.forEach(face => {{
                        indices.push(face.a, face.b, face.c);
                    }});
                    
                    const indexedGeometry = new THREE.BufferGeometry();
                    indexedGeometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
                    indexedGeometry.setIndex(indices);
                    indexedGeometry.computeVertexNormals();
                    
                    const wireframeMaterial = new THREE.MeshBasicMaterial({{ 
                        color: 0x4fc3f7, 
                        wireframe: true, 
                        transparent: true, 
                        opacity: 0.3 
                    }});
                    
                    const wireframe = new THREE.Mesh(indexedGeometry, wireframeMaterial);
                    scene.add(wireframe);
                }}
                
                currentVisualization = {{ type: 'landscape', data: landscapeData }};
            }}
            
            function renderPopulationCloud(cloudData) {{
                if (!cloudData.points) {{
                    console.error('Invalid population cloud data');
                    return;
                }}
                
                const points = cloudData.points;
                
                points.forEach((point, i) => {{
                    const pos = point.position;
                    const metadata = point.metadata || {{}};
                    
                    // Create sphere for individual
                    const geometry = new THREE.SphereGeometry(0.1, 8, 6);
                    
                    // Color based on elite status or fitness
                    let color = 0x6699ff; // Default blue
                    if (metadata.is_elite) {{
                        color = 0xffd700; // Gold for elite
                    }} else if (point.color) {{
                        const rgb = point.color.match(/\\d+/g) || [102, 153, 255];
                        color = (parseInt(rgb[0]) << 16) + (parseInt(rgb[1]) << 8) + parseInt(rgb[2]);
                    }}
                    
                    const material = new THREE.MeshPhongMaterial({{ 
                        color: color,
                        transparent: true,
                        opacity: metadata.is_elite ? 1.0 : 0.8
                    }});
                    
                    const sphere = new THREE.Mesh(geometry, material);
                    sphere.position.set(pos.x * 10, pos.z * 5, pos.y * 10);
                    
                    // Add glow effect for elite individuals
                    if (metadata.is_elite) {{
                        const glowGeometry = new THREE.SphereGeometry(0.15, 8, 6);
                        const glowMaterial = new THREE.MeshBasicMaterial({{ 
                            color: 0xffd700, 
                            transparent: true, 
                            opacity: 0.3 
                        }});
                        const glow = new THREE.Mesh(glowGeometry, glowMaterial);
                        glow.position.copy(sphere.position);
                        scene.add(glow);
                    }}
                    
                    scene.add(sphere);
                }});
                
                // Update metrics
                if (cloudData.stats) {{
                    document.getElementById('population-size').textContent = cloudData.stats.population_size || '-';
                    document.getElementById('best-fitness').textContent = (cloudData.stats.best_fitness || 0).toFixed(4);
                    document.getElementById('diversity-score').textContent = (cloudData.stats.diversity_score || 0).toFixed(4);
                }}
                
                currentVisualization = {{ type: 'population', data: cloudData }};
            }}
            
            function renderEvolutionTrajectory(trajectoryData) {{
                if (!trajectoryData.points) {{
                    console.error('Invalid trajectory data');
                    return;
                }}
                
                const points = trajectoryData.points;
                
                // Create line geometry for trajectory
                const lineGeometry = new THREE.BufferGeometry();
                const linePositions = new Float32Array(points.length * 3);
                
                points.forEach((point, i) => {{
                    linePositions[i * 3] = point.x * 0.1; // Scale generation
                    linePositions[i * 3 + 1] = point.z * 5; // Best fitness height
                    linePositions[i * 3 + 2] = point.y * 5; // Average fitness depth
                }});
                
                lineGeometry.setAttribute('position', new THREE.BufferAttribute(linePositions, 3));
                
                const lineMaterial = new THREE.LineBasicMaterial({{ color: 0x4fc3f7, linewidth: 3 }});
                const line = new THREE.Line(lineGeometry, lineMaterial);
                scene.add(line);
                
                // Add points for each generation
                points.forEach((point, i) => {{
                    const geometry = new THREE.SphereGeometry(0.05, 6, 4);
                    const material = new THREE.MeshPhongMaterial({{ 
                        color: 0x29b6f6,
                        transparent: true,
                        opacity: 0.8
                    }});
                    
                    const sphere = new THREE.Mesh(geometry, material);
                    sphere.position.set(
                        point.x * 0.1,
                        point.z * 5,
                        point.y * 5
                    );
                    
                    scene.add(sphere);
                }});
                
                currentVisualization = {{ type: 'trajectory', data: trajectoryData }};
            }}
            
            function clearScene() {{
                // Remove all meshes and points (keep lights and helpers)
                const objectsToRemove = [];
                scene.traverse(child => {{
                    if (child instanceof THREE.Mesh || child instanceof THREE.Points || child instanceof THREE.Line) {{
                        if (child !== scene.getObjectByName('GridHelper') && 
                            child !== scene.getObjectByName('AxesHelper')) {{
                            objectsToRemove.push(child);
                        }}
                    }}
                }});
                
                objectsToRemove.forEach(obj => {{
                    scene.remove(obj);
                    if (obj.geometry) obj.geometry.dispose();
                    if (obj.material) obj.material.dispose();
                }});
            }}
            
            function updateStatus(message, type = 'info') {{
                const statusEl = document.getElementById('status-text');
                statusEl.textContent = message;
                statusEl.className = type;
            }}
            
            function showLoading(show) {{
                document.getElementById('loading').style.display = show ? 'block' : 'none';
            }}
            
            function updateMetrics(metadata) {{
                if (metadata.generation !== undefined) {{
                    document.getElementById('current-generation').textContent = metadata.generation;
                }}
                if (metadata.best_fitness !== undefined) {{
                    document.getElementById('best-fitness').textContent = metadata.best_fitness.toFixed(4);
                }}
                if (metadata.diversity_score !== undefined) {{
                    document.getElementById('diversity-score').textContent = metadata.diversity_score.toFixed(4);
                }}
                if (metadata.population_size !== undefined) {{
                    document.getElementById('population-size').textContent = metadata.population_size;
                }}
            }}
            
            // WebSocket streaming
            function startStreaming() {{
                const experimentId = document.getElementById('experiment-select').value;
                if (!experimentId) {{
                    alert('Please select an experiment first');
                    return;
                }}
                
                const wsUrl = `ws://localhost:8000/api/v1/visualization3d/stream/${{experimentId}}`;
                websocket = new WebSocket(wsUrl);
                
                websocket.onopen = () => {{
                    isStreaming = true;
                    document.getElementById('start-stream').disabled = true;
                    document.getElementById('stop-stream').disabled = false;
                    updateStatus('Streaming started', 'success');
                    
                    // Subscribe to updates based on active tab
                    const activeTab = document.querySelector('.tab.active').dataset.tab;
                    const config = {{ interval: 3.0 }};
                    
                    if (activeTab === 'landscape') {{
                        config.resolution = parseInt(document.getElementById('resolution-slider').value);
                        websocket.send(JSON.stringify({{ type: 'subscribe_fitness_landscape', config }}));
                    }} else if (activeTab === 'population') {{
                        config.show_elite = document.getElementById('show-elite').checked;
                        websocket.send(JSON.stringify({{ type: 'subscribe_population', config }}));
                    }}
                }};
                
                websocket.onmessage = (event) => {{
                    const data = JSON.parse(event.data);
                    
                    if (data.type === 'fitness_landscape_update') {{
                        renderVisualization(data, 'landscape');
                    }} else if (data.type === 'population_cloud_update') {{
                        renderVisualization(data, 'population');
                        if (data.metadata) updateMetrics(data.metadata);
                    }} else if (data.type === 'pong') {{
                        // Keep-alive response
                        console.log('WebSocket ping-pong OK');
                    }}
                }};
                
                websocket.onerror = (error) => {{
                    console.error('WebSocket error:', error);
                    updateStatus('Streaming error', 'error');
                    stopStreaming();
                }};
                
                websocket.onclose = () => {{
                    stopStreaming();
                }};
                
                // Send periodic pings
                const pingInterval = setInterval(() => {{
                    if (websocket && websocket.readyState === WebSocket.OPEN) {{
                        websocket.send(JSON.stringify({{ type: 'ping' }}));
                    }} else {{
                        clearInterval(pingInterval);
                    }}
                }}, 30000);
            }}
            
            function stopStreaming() {{
                if (websocket) {{
                    websocket.close();
                    websocket = null;
                }}
                isStreaming = false;
                document.getElementById('start-stream').disabled = false;
                document.getElementById('stop-stream').disabled = true;
                updateStatus('Streaming stopped', 'info');
            }}
            
            // Export animation
            async function exportAnimation() {{
                const experimentId = document.getElementById('experiment-select').value;
                if (!experimentId) {{
                    alert('Please select an experiment first');
                    return;
                }}
                
                const activeTab = document.querySelector('.tab.active').dataset.tab;
                const config = {{
                    type: activeTab === 'landscape' ? 'fitness_landscape' : 'population_cloud',
                    start_generation: 0,
                    end_generation: 100,
                    frame_interval: 5,
                    resolution: 30
                }};
                
                if (activeTab === 'population') {{
                    config.show_elite = document.getElementById('show-elite').checked;
                }}
                
                showLoading(true);
                updateStatus('Generating animation...', 'loading');
                
                try {{
                    const response = await fetch(`/api/v1/visualization3d/experiments/${{experimentId}}/generate-animation`, {{
                        method: 'POST',
                        headers: {{ 'Content-Type': 'application/json' }},
                        body: JSON.stringify(config)
                    }});
                    
                    const data = await response.json();
                    
                    if (data.error) {{
                        throw new Error(data.error);
                    }}
                    
                    // Download as JSON
                    const blob = new Blob([JSON.stringify(data, null, 2)], {{ type: 'application/json' }});
                    const url = URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = url;
                    a.download = `evolution_animation_${{experimentId}}.json`;
                    a.click();
                    URL.revokeObjectURL(url);
                    
                    updateStatus('Animation exported successfully', 'success');
                    
                }} catch (error) {{
                    console.error('Error exporting animation:', error);
                    updateStatus(`Export error: ${{error.message}}`, 'error');
                }} finally {{
                    showLoading(false);
                }}
            }}
            
            // Event listeners
            document.addEventListener('DOMContentLoaded', () => {{
                initThreeJS();
                loadExperiments();
                
                // Tab switching
                document.querySelectorAll('.tab').forEach(tab => {{
                    tab.addEventListener('click', () => {{
                        document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
                        document.querySelectorAll('.tab-content').forEach(c => c.style.display = 'none');
                        
                        tab.classList.add('active');
                        document.getElementById(`${{tab.dataset.tab}}-controls`).style.display = 'block';
                        
                        // Reload visualization for new tab
                        if (document.getElementById('experiment-select').value) {{
                            loadVisualization();
                        }}
                    }});
                }});
                
                // Control event listeners
                document.getElementById('experiment-select').addEventListener('change', loadVisualization);
                document.getElementById('generation-select').addEventListener('change', loadVisualization);
                document.getElementById('pop-generation-select').addEventListener('change', loadVisualization);
                document.getElementById('resolution-slider').addEventListener('input', (e) => {{
                    document.getElementById('resolution-value').textContent = e.target.value;
                }});
                document.getElementById('resolution-slider').addEventListener('change', loadVisualization);
                document.getElementById('color-mapping').addEventListener('change', loadVisualization);
                document.getElementById('show-elite').addEventListener('change', loadVisualization);
                document.getElementById('show-connections').addEventListener('change', loadVisualization);
                document.getElementById('max-generations').addEventListener('change', loadVisualization);
                document.getElementById('trajectory-type').addEventListener('change', loadVisualization);
                
                // Streaming controls
                document.getElementById('start-stream').addEventListener('click', startStreaming);
                document.getElementById('stop-stream').addEventListener('click', stopStreaming);
                
                // Export button
                document.getElementById('export-animation').addEventListener('click', exportAnimation);
            }});
        </script>
    </body>
    </html>
    """


def create_3d_visualization_routes(request: Request):
    """Create routes for 3D visualization dashboard."""
    
    from fastapi import APIRouter
    from fastapi.responses import HTMLResponse
    
    router = APIRouter()
    
    @router.get("/visualization3d", response_class=HTMLResponse)
    async def visualization_3d_dashboard():
        """3D Visualization Dashboard."""
        return create_3d_visualization_dashboard(request, None)
    
    return router
