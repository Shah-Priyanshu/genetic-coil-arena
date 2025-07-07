"""
Phase 6A: Deep Learning Integration for Genetic Coil Arena
Neural Evolution - Combining Neural Networks with Genetic Algorithms
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import logging
import json
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class NeuralAgentConfig:
    """Configuration for neural network-based agents."""
    input_size: int = 10
    hidden_layers: List[int] = field(default_factory=lambda: [64, 32])
    output_size: int = 4
    activation: str = "relu"
    dropout_rate: float = 0.1
    use_batch_norm: bool = True
    network_type: str = "feedforward"  # "feedforward", "lstm", "cnn"


@dataclass
class NeuroEvolutionConfig:
    """Configuration for neuro-evolution process."""
    population_size: int = 50
    generations: int = 100
    mutation_rate: float = 0.1
    mutation_strength: float = 0.02
    crossover_rate: float = 0.8
    elite_size: int = 5
    tournament_size: int = 3
    fitness_sharing: bool = True
    speciation: bool = True
    compatibility_threshold: float = 3.0


class NeuralNetwork(nn.Module):
    """Flexible neural network architecture for evolution."""
    
    def __init__(self, config: NeuralAgentConfig):
        super(NeuralNetwork, self).__init__()
        self.config = config
        self.layers = nn.ModuleList()
        
        # Build network architecture
        if config.network_type == "feedforward":
            self._build_feedforward()
        elif config.network_type == "lstm":
            self._build_lstm()
        elif config.network_type == "cnn":
            self._build_cnn()
        else:
            raise ValueError(f"Unknown network type: {config.network_type}")
    
    def _build_feedforward(self):
        """Build feedforward neural network."""
        sizes = [self.config.input_size] + self.config.hidden_layers + [self.config.output_size]
        
        for i in range(len(sizes) - 1):
            # Linear layer
            self.layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            
            # Add batch normalization (except for output layer)
            if i < len(sizes) - 2 and self.config.use_batch_norm:
                self.layers.append(nn.BatchNorm1d(sizes[i + 1]))
            
            # Add dropout (except for output layer)
            if i < len(sizes) - 2 and self.config.dropout_rate > 0:
                self.layers.append(nn.Dropout(self.config.dropout_rate))
    
    def _build_lstm(self):
        """Build LSTM-based network."""
        self.lstm = nn.LSTM(
            input_size=self.config.input_size,
            hidden_size=self.config.hidden_layers[0],
            num_layers=len(self.config.hidden_layers),
            batch_first=True,
            dropout=self.config.dropout_rate if len(self.config.hidden_layers) > 1 else 0
        )
        self.output_layer = nn.Linear(self.config.hidden_layers[-1], self.config.output_size)
    
    def _build_cnn(self):
        """Build CNN-based network."""
        # Assuming input is 2D (for grid-based environments)
        self.conv_layers = nn.ModuleList([
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.Conv2d(32, 64, kernel_size=3, padding=1)
        ])
        
        # Calculate flattened size (assuming 10x10 input after pooling)
        conv_output_size = 64 * 2 * 2  # After 3 max pooling operations
        
        self.fc_layers = nn.ModuleList([
            nn.Linear(conv_output_size, self.config.hidden_layers[0]),
            nn.Linear(self.config.hidden_layers[0], self.config.output_size)
        ])
    
    def forward(self, x):
        """Forward pass through the network."""
        if self.config.network_type == "feedforward":
            return self._forward_feedforward(x)
        elif self.config.network_type == "lstm":
            return self._forward_lstm(x)
        elif self.config.network_type == "cnn":
            return self._forward_cnn(x)
    
    def _forward_feedforward(self, x):
        """Forward pass for feedforward network."""
        current_x = x
        layer_idx = 0
        
        for i, layer in enumerate(self.layers):
            if isinstance(layer, nn.Linear):
                current_x = layer(current_x)
                layer_idx += 1
                
                # Apply activation (except for output layer)
                if layer_idx < len([l for l in self.layers if isinstance(l, nn.Linear)]):
                    if self.config.activation == "relu":
                        current_x = F.relu(current_x)
                    elif self.config.activation == "tanh":
                        current_x = torch.tanh(current_x)
                    elif self.config.activation == "sigmoid":
                        current_x = torch.sigmoid(current_x)
            else:
                # Batch norm or dropout
                current_x = layer(current_x)
        
        return current_x
    
    def _forward_lstm(self, x):
        """Forward pass for LSTM network."""
        # x should be (batch_size, sequence_length, input_size)
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # Add sequence dimension
        
        lstm_out, _ = self.lstm(x)
        # Take the last output
        output = self.output_layer(lstm_out[:, -1, :])
        return output
    
    def _forward_cnn(self, x):
        """Forward pass for CNN network."""
        # x should be (batch_size, channels, height, width)
        if len(x.shape) == 3:
            x = x.unsqueeze(1)  # Add channel dimension
        
        for conv_layer in self.conv_layers:
            x = F.relu(conv_layer(x))
            x = F.max_pool2d(x, 2)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Forward through FC layers
        for fc_layer in self.fc_layers[:-1]:
            x = F.relu(fc_layer(x))
        
        # Output layer
        x = self.fc_layers[-1](x)
        return x
    
    def get_weights(self) -> np.ndarray:
        """Get all network weights as a flat numpy array."""
        weights = []
        for param in self.parameters():
            weights.extend(param.data.cpu().numpy().flatten())
        return np.array(weights)
    
    def set_weights(self, weights: np.ndarray):
        """Set network weights from a flat numpy array."""
        idx = 0
        for param in self.parameters():
            param_size = param.numel()
            param_weights = weights[idx:idx + param_size]
            param.data = torch.tensor(param_weights.reshape(param.shape)).float()
            idx += param_size
    
    def copy(self) -> 'NeuralNetwork':
        """Create a deep copy of the network."""
        new_network = NeuralNetwork(self.config)
        new_network.load_state_dict(self.state_dict())
        return new_network


class NeuralAgent:
    """Neural network-based agent for evolution."""
    
    def __init__(self, config: NeuralAgentConfig, agent_id: Optional[str] = None):
        self.config = config
        self.agent_id = agent_id or f"neural_agent_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        self.network = NeuralNetwork(config)
        self.fitness = 0.0
        self.age = 0
        self.parent_ids = []
        self.species_id = None
        
        # Performance tracking
        self.evaluation_history = []
        self.complexity_score = self._calculate_complexity()
    
    def _calculate_complexity(self) -> float:
        """Calculate network complexity score."""
        total_params = sum(p.numel() for p in self.network.parameters())
        total_layers = len(list(self.network.parameters()))
        return total_params / 1000.0 + total_layers * 0.1
    
    def predict(self, inputs: np.ndarray) -> np.ndarray:
        """Make prediction using the neural network."""
        self.network.eval()
        with torch.no_grad():
            if isinstance(inputs, np.ndarray):
                inputs = torch.tensor(inputs, dtype=torch.float32)
            
            if len(inputs.shape) == 1:
                inputs = inputs.unsqueeze(0)  # Add batch dimension
            
            outputs = self.network(inputs)
            return outputs.cpu().numpy()
    
    def mutate(self, mutation_rate: float, mutation_strength: float):
        """Mutate the neural network weights."""
        for param in self.network.parameters():
            if torch.rand(1).item() < mutation_rate:
                # Add Gaussian noise to weights
                noise = torch.randn_like(param.data) * mutation_strength
                param.data += noise
                
                # Optionally clip weights to prevent explosion
                param.data = torch.clamp(param.data, -5.0, 5.0)
    
    def crossover(self, other: 'NeuralAgent', crossover_rate: float) -> Tuple['NeuralAgent', 'NeuralAgent']:
        """Perform crossover with another neural agent."""
        child1_config = self.config
        child2_config = other.config
        
        child1 = NeuralAgent(child1_config)
        child2 = NeuralAgent(child2_config)
        
        # Crossover weights
        self_weights = self.network.get_weights()
        other_weights = other.network.get_weights()
        
        if len(self_weights) == len(other_weights):
            # Uniform crossover
            mask = np.random.random(len(self_weights)) < crossover_rate
            
            child1_weights = np.where(mask, self_weights, other_weights)
            child2_weights = np.where(mask, other_weights, self_weights)
            
            child1.network.set_weights(child1_weights)
            child2.network.set_weights(child2_weights)
            
            # Set parent information
            child1.parent_ids = [self.agent_id, other.agent_id]
            child2.parent_ids = [self.agent_id, other.agent_id]
        else:
            # If architectures differ, create random weights
            logger.warning("Cannot crossover agents with different architectures")
            child1.parent_ids = [self.agent_id]
            child2.parent_ids = [other.agent_id]
        
        return child1, child2
    
    def calculate_compatibility(self, other: 'NeuralAgent') -> float:
        """Calculate compatibility distance for speciation."""
        if self.config.input_size != other.config.input_size or \
           self.config.output_size != other.config.output_size:
            return float('inf')  # Incompatible architectures
        
        # Weight difference
        self_weights = self.network.get_weights()
        other_weights = other.network.get_weights()
        
        if len(self_weights) != len(other_weights):
            return float('inf')
        
        weight_diff = np.mean(np.abs(self_weights - other_weights))
        
        # Architecture difference
        arch_diff = abs(len(self.config.hidden_layers) - len(other.config.hidden_layers))
        
        # Size difference
        self_size = sum(self.config.hidden_layers)
        other_size = sum(other.config.hidden_layers)
        size_diff = abs(self_size - other_size) / max(self_size, other_size, 1)
        
        # Combined compatibility score
        compatibility = weight_diff + arch_diff * 0.5 + size_diff * 0.3
        return compatibility
    
    def save(self, filepath: Path):
        """Save the neural agent to disk."""
        save_data = {
            'agent_id': self.agent_id,
            'config': {
                'input_size': self.config.input_size,
                'hidden_layers': self.config.hidden_layers,
                'output_size': self.config.output_size,
                'activation': self.config.activation,
                'dropout_rate': self.config.dropout_rate,
                'use_batch_norm': self.config.use_batch_norm,
                'network_type': self.config.network_type
            },
            'fitness': self.fitness,
            'age': self.age,
            'parent_ids': self.parent_ids,
            'species_id': self.species_id,
            'complexity_score': self.complexity_score,
            'evaluation_history': self.evaluation_history
        }
        
        # Save metadata
        with open(filepath.with_suffix('.json'), 'w') as f:
            json.dump(save_data, f, indent=2)
        
        # Save model weights
        torch.save(self.network.state_dict(), filepath.with_suffix('.pth'))
        
        logger.info(f"Neural agent {self.agent_id} saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: Path) -> 'NeuralAgent':
        """Load a neural agent from disk."""
        # Load metadata
        with open(filepath.with_suffix('.json'), 'r') as f:
            save_data = json.load(f)
        
        # Create config
        config_data = save_data['config']
        config = NeuralAgentConfig(**config_data)
        
        # Create agent
        agent = cls(config, save_data['agent_id'])
        agent.fitness = save_data['fitness']
        agent.age = save_data['age']
        agent.parent_ids = save_data['parent_ids']
        agent.species_id = save_data.get('species_id')
        agent.complexity_score = save_data.get('complexity_score', 0)
        agent.evaluation_history = save_data.get('evaluation_history', [])
        
        # Load model weights
        state_dict = torch.load(filepath.with_suffix('.pth'), map_location='cpu')
        agent.network.load_state_dict(state_dict)
        
        logger.info(f"Neural agent {agent.agent_id} loaded from {filepath}")
        return agent
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert agent to dictionary representation."""
        return {
            'agent_id': self.agent_id,
            'fitness': self.fitness,
            'age': self.age,
            'parent_ids': self.parent_ids,
            'species_id': self.species_id,
            'complexity_score': self.complexity_score,
            'config': {
                'input_size': self.config.input_size,
                'hidden_layers': self.config.hidden_layers,
                'output_size': self.config.output_size,
                'activation': self.config.activation,
                'dropout_rate': self.config.dropout_rate,
                'use_batch_norm': self.config.use_batch_norm,
                'network_type': self.config.network_type
            },
            'weights_shape': [param.shape for param in self.network.parameters()],
            'total_parameters': sum(p.numel() for p in self.network.parameters())
        }


class Species:
    """Species for neural agent speciation."""
    
    def __init__(self, species_id: str, representative: NeuralAgent):
        self.species_id = species_id
        self.representative = representative
        self.members = [representative]
        self.best_fitness = representative.fitness
        self.average_fitness = representative.fitness
        self.stagnation_count = 0
        self.last_improvement = 0
    
    def add_member(self, agent: NeuralAgent):
        """Add a member to the species."""
        self.members.append(agent)
        agent.species_id = self.species_id
        self._update_stats()
    
    def _update_stats(self):
        """Update species statistics."""
        if not self.members:
            return
        
        fitnesses = [member.fitness for member in self.members]
        self.average_fitness = np.mean(fitnesses)
        new_best = max(fitnesses)
        
        if new_best > self.best_fitness:
            self.best_fitness = new_best
            self.last_improvement = 0
            self.stagnation_count = 0
        else:
            self.stagnation_count += 1
    
    def should_be_removed(self, max_stagnation: int = 15) -> bool:
        """Check if species should be removed due to stagnation."""
        return self.stagnation_count > max_stagnation and len(self.members) > 1


# Global neural evolution utilities
def create_random_neural_agent(config: NeuralAgentConfig) -> NeuralAgent:
    """Create a neural agent with random weights."""
    agent = NeuralAgent(config)
    
    # Initialize weights with Xavier/Glorot initialization
    for param in agent.network.parameters():
        if len(param.shape) >= 2:
            nn.init.xavier_uniform_(param)
        else:
            nn.init.zeros_(param)
    
    return agent


def evaluate_neural_population(population: List[NeuralAgent], 
                             evaluation_function,
                             environment_config: Dict[str, Any]) -> List[NeuralAgent]:
    """Evaluate a population of neural agents."""
    logger.info(f"Evaluating population of {len(population)} neural agents")
    
    for i, agent in enumerate(population):
        try:
            # Evaluate agent in environment
            fitness = evaluation_function(agent, environment_config)
            agent.fitness = fitness
            agent.age += 1
            
            # Track evaluation history
            agent.evaluation_history.append({
                'generation': agent.age,
                'fitness': fitness,
                'timestamp': datetime.now().isoformat()
            })
            
            # Keep only last 100 evaluations
            if len(agent.evaluation_history) > 100:
                agent.evaluation_history = agent.evaluation_history[-100:]
                
        except Exception as e:
            logger.error(f"Error evaluating agent {agent.agent_id}: {e}")
            agent.fitness = 0.0
    
    return population
