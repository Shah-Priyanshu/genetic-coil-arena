"""
Advanced 3D Visualization Engine for Genetic Coil Arena
Phase 6B: Advanced Visualization & VR/AR Integration

This module provides advanced 3D visualization capabilities for neural networks,
population dynamics, and evolutionary processes.
"""

import numpy as np
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class VisualizationNode:
    """Represents a node in the 3D visualization space."""
    id: str
    position: Tuple[float, float, float]
    size: float = 1.0
    color: str = "#ff6b6b"
    node_type: str = "default"
    connections: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VisualizationEdge:
    """Represents an edge/connection in the 3D visualization."""
    source: str
    target: str
    weight: float = 1.0
    color: str = "#4ecdc4"
    edge_type: str = "default"
    animated: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VisualizationLayer:
    """Represents a layer in the 3D neural network visualization."""
    layer_id: str
    nodes: List[VisualizationNode]
    position_y: float
    layer_type: str = "hidden"
    activation_function: str = "relu"
    metadata: Dict[str, Any] = field(default_factory=dict)


class Neural3DVisualizer:
    """
    Advanced 3D visualizer for neural networks and evolutionary processes.
    """
    
    def __init__(self):
        self.nodes = {}
        self.edges = {}
        self.layers = {}
        self.animation_frames = []
        self.current_frame = 0
        
    def add_neural_network(self, network_id: str, network_config: Dict[str, Any]) -> Dict[str, Any]:
        """Add a neural network to the 3D visualization."""
        try:
            input_size = network_config.get('input_size', 10)
            hidden_layers = network_config.get('hidden_layers', [64, 32])
            output_size = network_config.get('output_size', 4)
            
            # Create layers
            layers = []
            layer_spacing = 5.0
            node_spacing = 2.0
            
            # Input layer
            input_layer = self._create_layer(
                layer_id=f"{network_id}_input",
                nodes_count=input_size,
                position_y=0,
                layer_type="input",
                node_spacing=node_spacing
            )
            layers.append(input_layer)
            
            # Hidden layers
            for i, hidden_size in enumerate(hidden_layers):
                hidden_layer = self._create_layer(
                    layer_id=f"{network_id}_hidden_{i}",
                    nodes_count=hidden_size,
                    position_y=(i + 1) * layer_spacing,
                    layer_type="hidden",
                    node_spacing=node_spacing
                )
                layers.append(hidden_layer)
            
            # Output layer
            output_layer = self._create_layer(
                layer_id=f"{network_id}_output",
                nodes_count=output_size,
                position_y=(len(hidden_layers) + 1) * layer_spacing,
                layer_type="output",
                node_spacing=node_spacing
            )
            layers.append(output_layer)
            
            # Create connections between layers
            edges = []
            for i in range(len(layers) - 1):
                current_layer = layers[i]
                next_layer = layers[i + 1]
                
                layer_edges = self._create_layer_connections(
                    current_layer, next_layer, network_id
                )
                edges.extend(layer_edges)
            
            # Store network visualization
            network_viz = {
                'network_id': network_id,
                'layers': layers,
                'edges': edges,
                'total_nodes': sum(len(layer.nodes) for layer in layers),
                'total_edges': len(edges),
                'created_at': datetime.now().isoformat(),
                'config': network_config
            }
            
            # Update internal storage
            for layer in layers:
                self.layers[layer.layer_id] = layer
                for node in layer.nodes:
                    self.nodes[node.id] = node
            
            for edge in edges:
                edge_id = f"{edge.source}_{edge.target}"
                self.edges[edge_id] = edge
            
            logger.info(f"Created 3D visualization for neural network {network_id}")
            return network_viz
            
        except Exception as e:
            logger.error(f"Error creating neural network visualization: {e}")
            raise
    
    def _create_layer(self, layer_id: str, nodes_count: int, position_y: float, 
                     layer_type: str, node_spacing: float) -> VisualizationLayer:
        """Create a visualization layer with nodes."""
        nodes = []
        
        # Calculate node positions in a grid or circular arrangement
        if nodes_count <= 10:
            # Linear arrangement for small layers
            start_x = -(nodes_count - 1) * node_spacing / 2
            for i in range(nodes_count):
                node = VisualizationNode(
                    id=f"{layer_id}_node_{i}",
                    position=(start_x + i * node_spacing, position_y, 0),
                    size=1.0,
                    color=self._get_layer_color(layer_type),
                    node_type=layer_type,
                    metadata={'layer_id': layer_id, 'node_index': i}
                )
                nodes.append(node)
        else:
            # Circular arrangement for larger layers
            radius = max(3.0, nodes_count * 0.3)
            for i in range(nodes_count):
                angle = 2 * np.pi * i / nodes_count
                x = radius * np.cos(angle)
                z = radius * np.sin(angle)
                
                node = VisualizationNode(
                    id=f"{layer_id}_node_{i}",
                    position=(x, position_y, z),
                    size=0.8,
                    color=self._get_layer_color(layer_type),
                    node_type=layer_type,
                    metadata={'layer_id': layer_id, 'node_index': i}
                )
                nodes.append(node)
        
        return VisualizationLayer(
            layer_id=layer_id,
            nodes=nodes,
            position_y=position_y,
            layer_type=layer_type,
            metadata={'nodes_count': nodes_count}
        )
    
    def _create_layer_connections(self, source_layer: VisualizationLayer, 
                                target_layer: VisualizationLayer, 
                                network_id: str) -> List[VisualizationEdge]:
        """Create connections between two layers."""
        edges = []
        
        for source_node in source_layer.nodes:
            for target_node in target_layer.nodes:
                # Simulate random weights for visualization
                weight = np.random.uniform(-1, 1)
                
                edge = VisualizationEdge(
                    source=source_node.id,
                    target=target_node.id,
                    weight=weight,
                    color=self._get_edge_color(weight),
                    edge_type="neural_connection",
                    metadata={
                        'network_id': network_id,
                        'source_layer': source_layer.layer_id,
                        'target_layer': target_layer.layer_id
                    }
                )
                edges.append(edge)
        
        return edges
    
    def _get_layer_color(self, layer_type: str) -> str:
        """Get color for layer type."""
        colors = {
            'input': '#4CAF50',    # Green
            'hidden': '#2196F3',   # Blue
            'output': '#FF9800'    # Orange
        }
        return colors.get(layer_type, '#9E9E9E')
    
    def _get_edge_color(self, weight: float) -> str:
        """Get color based on edge weight."""
        if weight > 0.5:
            return '#4CAF50'  # Strong positive - green
        elif weight > 0:
            return '#8BC34A'  # Weak positive - light green
        elif weight > -0.5:
            return '#FF9800'  # Weak negative - orange
        else:
            return '#F44336'  # Strong negative - red
    
    def update_network_activations(self, network_id: str, activations: Dict[str, List[float]]) -> Dict[str, Any]:
        """Update neural network with activation values."""
        try:
            updated_nodes = {}
            
            for layer_id, layer_activations in activations.items():
                layer = self.layers.get(f"{network_id}_{layer_id}")
                if layer:
                    for i, activation in enumerate(layer_activations):
                        if i < len(layer.nodes):
                            node = layer.nodes[i]
                            # Update node size and color based on activation
                            node.size = max(0.5, min(2.0, abs(activation) + 0.5))
                            node.color = self._get_activation_color(activation)
                            node.metadata['activation'] = activation
                            updated_nodes[node.id] = {
                                'size': node.size,
                                'color': node.color,
                                'activation': activation
                            }
            
            return {
                'network_id': network_id,
                'updated_nodes': updated_nodes,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error updating network activations: {e}")
            raise
    
    def _get_activation_color(self, activation: float) -> str:
        """Get color based on activation value."""
        # Normalize activation to 0-1 range for color mapping
        normalized = (activation + 1) / 2  # Assume activation range is -1 to 1
        normalized = max(0, min(1, normalized))
        
        # Color gradient from blue (low) to red (high)
        red = int(255 * normalized)
        blue = int(255 * (1 - normalized))
        return f"#{red:02x}00{blue:02x}"
    
    def create_population_visualization(self, population_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create 3D visualization of genetic algorithm population."""
        try:
            nodes = []
            edges = []
            
            # Arrange population in 3D grid
            grid_size = int(np.ceil(np.sqrt(len(population_data))))
            spacing = 3.0
            
            for i, individual in enumerate(population_data):
                x = (i % grid_size) * spacing - (grid_size * spacing) / 2
                z = (i // grid_size) * spacing - (grid_size * spacing) / 2
                y = 0
                
                # Get fitness for visualization
                fitness = individual.get('fitness', 0)
                
                node = VisualizationNode(
                    id=f"individual_{i}",
                    position=(x, y, z),
                    size=max(0.5, min(3.0, fitness / 100 + 0.5)),  # Scale by fitness
                    color=self._get_fitness_color(fitness),
                    node_type="individual",
                    metadata={
                        'fitness': fitness,
                        'generation': individual.get('generation', 0),
                        'individual_id': individual.get('id', i)
                    }
                )
                nodes.append(node)
                self.nodes[node.id] = node
            
            # Create connections between similar individuals
            for i, node_a in enumerate(nodes):
                for j, node_b in enumerate(nodes[i+1:], i+1):
                    if self._should_connect_individuals(node_a, node_b):
                        edge = VisualizationEdge(
                            source=node_a.id,
                            target=node_b.id,
                            weight=0.5,
                            color='#80808080',  # Semi-transparent gray
                            edge_type="similarity",
                            metadata={'connection_type': 'similarity'}
                        )
                        edges.append(edge)
                        edge_id = f"{edge.source}_{edge.target}"
                        self.edges[edge_id] = edge
            
            return {
                'population_id': f"pop_{datetime.now().isoformat()}",
                'nodes': [self._node_to_dict(node) for node in nodes],
                'edges': [self._edge_to_dict(edge) for edge in edges],
                'total_individuals': len(nodes),
                'total_connections': len(edges),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error creating population visualization: {e}")
            raise
    
    def _should_connect_individuals(self, node_a: VisualizationNode, node_b: VisualizationNode) -> bool:
        """Determine if two individuals should be connected in visualization."""
        # Connect if fitness values are similar (within 20% of each other)
        fitness_a = node_a.metadata.get('fitness', 0)
        fitness_b = node_b.metadata.get('fitness', 0)
        
        if fitness_a == 0 and fitness_b == 0:
            return False
        
        max_fitness = max(abs(fitness_a), abs(fitness_b))
        if max_fitness == 0:
            return False
        
        similarity = 1 - abs(fitness_a - fitness_b) / max_fitness
        return similarity > 0.8
    
    def _get_fitness_color(self, fitness: float) -> str:
        """Get color based on fitness value."""
        # Normalize fitness to color gradient
        # Assuming fitness range is 0-1000, adjust as needed
        normalized = min(1.0, max(0.0, fitness / 1000))
        
        # Gradient from red (low fitness) to green (high fitness)
        red = int(255 * (1 - normalized))
        green = int(255 * normalized)
        return f"#{red:02x}{green:02x}00"
    
    def _node_to_dict(self, node: VisualizationNode) -> Dict[str, Any]:
        """Convert node to dictionary for JSON serialization."""
        return {
            'id': node.id,
            'position': list(node.position),
            'size': node.size,
            'color': node.color,
            'type': node.node_type,
            'connections': node.connections,
            'metadata': node.metadata
        }
    
    def _edge_to_dict(self, edge: VisualizationEdge) -> Dict[str, Any]:
        """Convert edge to dictionary for JSON serialization."""
        return {
            'source': edge.source,
            'target': edge.target,
            'weight': edge.weight,
            'color': edge.color,
            'type': edge.edge_type,
            'animated': edge.animated,
            'metadata': edge.metadata
        }
    
    def get_visualization_data(self, visualization_id: str) -> Dict[str, Any]:
        """Get complete visualization data for rendering."""
        try:
            return {
                'visualization_id': visualization_id,
                'nodes': [self._node_to_dict(node) for node in self.nodes.values()],
                'edges': [self._edge_to_dict(edge) for edge in self.edges.values()],
                'layers': {layer_id: {
                    'layer_id': layer.layer_id,
                    'position_y': layer.position_y,
                    'layer_type': layer.layer_type,
                    'node_count': len(layer.nodes),
                    'metadata': layer.metadata
                } for layer_id, layer in self.layers.items()},
                'stats': {
                    'total_nodes': len(self.nodes),
                    'total_edges': len(self.edges),
                    'total_layers': len(self.layers)
                },
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting visualization data: {e}")
            raise


class VRExperienceManager:
    """
    Manager for VR/AR experiences and immersive visualization.
    """
    
    def __init__(self):
        self.active_sessions = {}
        self.vr_environments = {}
        
    def create_vr_session(self, user_id: str, session_config: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new VR session for evolutionary visualization."""
        try:
            session_id = f"vr_session_{user_id}_{datetime.now().isoformat()}"
            
            session = {
                'session_id': session_id,
                'user_id': user_id,
                'environment_type': session_config.get('environment_type', 'neural_space'),
                'immersion_level': session_config.get('immersion_level', 'standard'),
                'interaction_mode': session_config.get('interaction_mode', 'gaze_controller'),
                'created_at': datetime.now().isoformat(),
                'status': 'active',
                'position': [0, 1.6, 0],  # Default standing position
                'rotation': [0, 0, 0],    # Default rotation
                'config': session_config
            }
            
            self.active_sessions[session_id] = session
            
            logger.info(f"Created VR session {session_id} for user {user_id}")
            return session
            
        except Exception as e:
            logger.error(f"Error creating VR session: {e}")
            raise
    
    def get_vr_environment_data(self, session_id: str) -> Dict[str, Any]:
        """Get VR environment data for a session."""
        try:
            session = self.active_sessions.get(session_id)
            if not session:
                raise ValueError(f"VR session {session_id} not found")
            
            environment_type = session['environment_type']
            
            if environment_type == 'neural_space':
                return self._create_neural_vr_environment(session)
            elif environment_type == 'evolution_garden':
                return self._create_evolution_vr_environment(session)
            elif environment_type == 'data_cathedral':
                return self._create_data_vr_environment(session)
            else:
                return self._create_default_vr_environment(session)
                
        except Exception as e:
            logger.error(f"Error getting VR environment data: {e}")
            raise
    
    def _create_neural_vr_environment(self, session: Dict[str, Any]) -> Dict[str, Any]:
        """Create neural network VR environment."""
        return {
            'environment_id': f"neural_vr_{session['session_id']}",
            'type': 'neural_space',
            'objects': [
                {
                    'id': 'neural_grid',
                    'type': 'ground_grid',
                    'position': [0, 0, 0],
                    'scale': [50, 1, 50],
                    'color': '#1a1a2e',
                    'properties': {'grid_lines': True, 'transparency': 0.3}
                },
                {
                    'id': 'neural_core',
                    'type': 'sphere',
                    'position': [0, 5, 0],
                    'scale': [2, 2, 2],
                    'color': '#4fc3f7',
                    'properties': {'glow': True, 'pulsing': True}
                },
                {
                    'id': 'data_streams',
                    'type': 'particle_system',
                    'position': [0, 2, 0],
                    'properties': {
                        'particle_count': 1000,
                        'flow_pattern': 'spiral',
                        'colors': ['#64b5f6', '#81c784', '#ffb74d']
                    }
                }
            ],
            'lighting': {
                'ambient': '#0d47a1',
                'directional': '#1976d2',
                'point_lights': [
                    {'position': [10, 10, 10], 'color': '#64b5f6', 'intensity': 0.8},
                    {'position': [-10, 10, -10], 'color': '#81c784', 'intensity': 0.6}
                ]
            },
            'user_position': session['position'],
            'user_rotation': session['rotation'],
            'interaction_zones': [
                {
                    'id': 'neural_panel',
                    'type': 'ui_panel',
                    'position': [3, 2, 0],
                    'content': 'neural_controls'
                }
            ]
        }
    
    def _create_evolution_vr_environment(self, session: Dict[str, Any]) -> Dict[str, Any]:
        """Create evolutionary garden VR environment."""
        return {
            'environment_id': f"evolution_vr_{session['session_id']}",
            'type': 'evolution_garden',
            'objects': [
                {
                    'id': 'garden_terrain',
                    'type': 'terrain',
                    'position': [0, 0, 0],
                    'scale': [100, 5, 100],
                    'color': '#2e7d32',
                    'properties': {'texture': 'grass', 'height_variation': True}
                },
                {
                    'id': 'evolution_tree',
                    'type': 'tree_of_life',
                    'position': [0, 0, 0],
                    'scale': [15, 20, 15],
                    'properties': {'branching': True, 'growth_animation': True}
                },
                {
                    'id': 'population_flowers',
                    'type': 'population_display',
                    'position': [0, 0, 10],
                    'properties': {'individual_count': 100, 'fitness_colors': True}
                }
            ],
            'lighting': {
                'ambient': '#f1f8e9',
                'directional': '#ffecb3',
                'sky': {'type': 'procedural', 'time_of_day': 'golden_hour'}
            },
            'user_position': session['position'],
            'user_rotation': session['rotation']
        }
    
    def _create_data_vr_environment(self, session: Dict[str, Any]) -> Dict[str, Any]:
        """Create data cathedral VR environment."""
        return {
            'environment_id': f"data_vr_{session['session_id']}",
            'type': 'data_cathedral',
            'objects': [
                {
                    'id': 'cathedral_structure',
                    'type': 'architectural',
                    'position': [0, 0, 0],
                    'scale': [30, 40, 60],
                    'color': '#37474f',
                    'properties': {'style': 'gothic', 'materials': ['stone', 'glass']}
                },
                {
                    'id': 'data_pillars',
                    'type': 'data_columns',
                    'position': [0, 0, 0],
                    'properties': {'column_count': 12, 'data_visualization': True}
                },
                {
                    'id': 'stained_glass_analytics',
                    'type': 'stained_glass',
                    'position': [0, 15, -25],
                    'properties': {'pattern': 'analytics_mandala', 'dynamic_colors': True}
                }
            ],
            'lighting': {
                'ambient': '#1a1a2e',
                'colored_glass': True,
                'candles': {'count': 50, 'flicker': True}
            },
            'user_position': session['position'],
            'user_rotation': session['rotation']
        }
    
    def _create_default_vr_environment(self, session: Dict[str, Any]) -> Dict[str, Any]:
        """Create default VR environment."""
        return {
            'environment_id': f"default_vr_{session['session_id']}",
            'type': 'default',
            'objects': [
                {
                    'id': 'ground_plane',
                    'type': 'plane',
                    'position': [0, 0, 0],
                    'scale': [20, 1, 20],
                    'color': '#424242'
                }
            ],
            'lighting': {
                'ambient': '#ffffff',
                'directional': '#ffffff'
            },
            'user_position': session['position'],
            'user_rotation': session['rotation']
        }


# Create global instances
neural_3d_visualizer = Neural3DVisualizer()
vr_experience_manager = VRExperienceManager()
