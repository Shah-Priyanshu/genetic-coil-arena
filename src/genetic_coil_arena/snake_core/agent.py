"""Neural network agent for controlling the snake."""

import numpy as np
import numba  # type: ignore
from typing import Optional

from .game import Action


@numba.njit(cache=True)  # type: ignore
def _tanh(x: np.ndarray) -> np.ndarray:
    """Tanh activation function."""
    return np.tanh(x)


@numba.njit(cache=True)  # type: ignore
def _relu(x: np.ndarray) -> np.ndarray:
    """ReLU activation function."""
    return np.maximum(0, x)


@numba.njit(cache=True)  # type: ignore
def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Sigmoid activation function."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


@numba.njit(cache=True)  # type: ignore
def _forward_pass(observation: np.ndarray, weights1: np.ndarray, bias1: np.ndarray,
                 weights2: np.ndarray, bias2: np.ndarray, weights3: np.ndarray, 
                 bias3: np.ndarray, activation: int) -> np.ndarray:
    """Forward pass through 2-layer MLP with specified activation."""
    # Ensure all inputs are float32
    obs = observation.astype(np.float32)
    w1 = weights1.astype(np.float32)
    b1 = bias1.astype(np.float32)
    w2 = weights2.astype(np.float32)
    b2 = bias2.astype(np.float32)
    w3 = weights3.astype(np.float32)
    b3 = bias3.astype(np.float32)
    
    # First layer - ensure float32 throughout
    hidden1 = (np.dot(obs, w1) + b1).astype(np.float32)
    if activation == 0:  # tanh
        hidden1 = _tanh(hidden1).astype(np.float32)
    elif activation == 1:  # relu
        hidden1 = _relu(hidden1).astype(np.float32)
    else:  # sigmoid
        hidden1 = _sigmoid(hidden1).astype(np.float32)
    
    # Second layer - ensure float32 throughout
    hidden2 = (np.dot(hidden1, w2) + b2).astype(np.float32)
    if activation == 0:  # tanh
        hidden2 = _tanh(hidden2).astype(np.float32)
    elif activation == 1:  # relu
        hidden2 = _relu(hidden2).astype(np.float32)
    else:  # sigmoid
        hidden2 = _sigmoid(hidden2).astype(np.float32)
    
    # Output layer (no activation - raw logits)
    output = (np.dot(hidden2, w3) + b3).astype(np.float32)
    return output


class SnakeAgent:
    """Neural network agent for controlling the snake."""
    
    def __init__(self, input_size: int, hidden_size: int = 32, 
                 second_hidden_size: int = 16, activation: str = "tanh",
                 seed: Optional[int] = None):
        """Initialize the agent.
        
        Args:
            input_size: Size of observation vector
            hidden_size: Size of first hidden layer
            second_hidden_size: Size of second hidden layer  
            activation: Activation function ("tanh", "relu", "sigmoid")
            seed: Random seed for weight initialization
        """
        if seed is not None:
            np.random.seed(seed)
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.second_hidden_size = second_hidden_size
        self.output_size = 3  # Three actions: straight, left, right
        
        # Map activation string to int for numba
        self.activation_map = {"tanh": 0, "relu": 1, "sigmoid": 2}
        self.activation = self.activation_map.get(activation, 0)
        
        # Initialize weights with Xavier/Glorot initialization
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Initialize network weights."""
        # Xavier initialization
        scale1 = np.sqrt(2.0 / (self.input_size + self.hidden_size))
        scale2 = np.sqrt(2.0 / (self.hidden_size + self.second_hidden_size))
        scale3 = np.sqrt(2.0 / (self.second_hidden_size + self.output_size))
        
        self.weights1 = np.random.normal(0, scale1, (self.input_size, self.hidden_size)).astype(np.float32)
        self.bias1 = np.zeros(self.hidden_size, dtype=np.float32)
        
        self.weights2 = np.random.normal(0, scale2, (self.hidden_size, self.second_hidden_size)).astype(np.float32)
        self.bias2 = np.zeros(self.second_hidden_size, dtype=np.float32)
        
        self.weights3 = np.random.normal(0, scale3, (self.second_hidden_size, self.output_size)).astype(np.float32)
        self.bias3 = np.zeros(self.output_size, dtype=np.float32)
    
    def predict(self, observation: np.ndarray) -> Action:
        """Predict action given observation.
        
        Args:
            observation: Flattened observation vector
            
        Returns:
            Action to take
        """
        # Forward pass
        output = _forward_pass(
            observation.astype(np.float32),
            self.weights1, self.bias1,
            self.weights2, self.bias2, 
            self.weights3, self.bias3,
            self.activation
        )
        
        # Select action with highest probability
        action_idx = np.argmax(output)
        return Action(action_idx)
    
    def predict_probabilities(self, observation: np.ndarray) -> np.ndarray:
        """Get action probabilities (softmax output).
        
        Args:
            observation: Flattened observation vector
            
        Returns:
            Action probabilities
        """
        output = _forward_pass(
            observation.astype(np.float32),
            self.weights1, self.bias1,
            self.weights2, self.bias2,
            self.weights3, self.bias3, 
            self.activation
        )
        
        # Apply softmax
        exp_output = np.exp(output - np.max(output))  # Numerical stability
        return exp_output / np.sum(exp_output)
    
    def get_genome(self) -> np.ndarray:
        """Get flattened weight vector (genome) for genetic algorithm.
        
        Returns:
            Flattened array of all network weights and biases
        """
        return np.concatenate([
            self.weights1.flatten(),
            self.bias1,
            self.weights2.flatten(), 
            self.bias2,
            self.weights3.flatten(),
            self.bias3
        ])
    
    def set_genome(self, genome: np.ndarray) -> None:
        """Set network weights from flattened genome.
        
        Args:
            genome: Flattened array of weights and biases
        """
        idx = 0
        
        # First layer
        w1_size = self.input_size * self.hidden_size
        self.weights1 = genome[idx:idx+w1_size].reshape(self.input_size, self.hidden_size).astype(np.float32)
        idx += w1_size
        
        self.bias1 = genome[idx:idx+self.hidden_size].astype(np.float32)
        idx += self.hidden_size
        
        # Second layer
        w2_size = self.hidden_size * self.second_hidden_size
        self.weights2 = genome[idx:idx+w2_size].reshape(self.hidden_size, self.second_hidden_size).astype(np.float32)
        idx += w2_size
        
        self.bias2 = genome[idx:idx+self.second_hidden_size].astype(np.float32)
        idx += self.second_hidden_size
        
        # Output layer
        w3_size = self.second_hidden_size * self.output_size
        self.weights3 = genome[idx:idx+w3_size].reshape(self.second_hidden_size, self.output_size).astype(np.float32)
        idx += w3_size
        
        self.bias3 = genome[idx:idx+self.output_size].astype(np.float32)
    
    @property
    def genome_size(self) -> int:
        """Get size of genome (total number of parameters)."""
        return (self.input_size * self.hidden_size + self.hidden_size +
                self.hidden_size * self.second_hidden_size + self.second_hidden_size +
                self.second_hidden_size * self.output_size + self.output_size)
    
    def copy(self) -> "SnakeAgent":
        """Create a copy of this agent."""
        new_agent = SnakeAgent(
            self.input_size, 
            self.hidden_size,
            self.second_hidden_size, 
            list(self.activation_map.keys())[self.activation]
        )
        new_agent.set_genome(self.get_genome())
        return new_agent


def create_random_agent(input_size: int, hidden_size: int = 32, 
                       second_hidden_size: int = 16, activation: str = "tanh",
                       seed: Optional[int] = None) -> SnakeAgent:
    """Create a randomly initialized agent.
    
    Args:
        input_size: Size of observation vector
        hidden_size: Size of first hidden layer
        second_hidden_size: Size of second hidden layer
        activation: Activation function
        seed: Random seed
        
    Returns:
        Randomly initialized agent
    """
    return SnakeAgent(input_size, hidden_size, second_hidden_size, activation, seed)


def get_observation_size(board_size: int = 32, observation_radius: int = 5) -> int:
    """Calculate the observation vector size.
    
    Args:
        board_size: Size of game board
        observation_radius: Radius of observation window
        
    Returns:
        Size of observation vector
    """
    # Grid observation (2*radius+1)^2 + snake length (1) + food direction (2)
    grid_size = (2 * observation_radius + 1) ** 2
    return grid_size + 1 + 2
