"""Configuration management for Genetic Coil Arena."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional
import yaml


@dataclass
class GameConfig:
    """Game environment configuration."""
    
    board_size: int = 32
    observation_radius: int = 5  # 11x11 window
    # Episode cap, extended by 2 steps per body segment. At 500 a greedy food-seeker
    # times out in 58% of episodes (a ceiling); at 2000 it dies of its own mistakes first.
    max_steps: int = 2000
    # Die after this many steps without food (0 disables). Without it every policy
    # survives to max_steps on the torus and survival swamps food in the fitness.
    # The farthest food on a 32x32 torus is 32 steps away, so 100 leaves ~3x slack.
    starvation_steps: int = 100
    walls: bool = False  # True: leaving the board kills; False: board wraps (torus)
    food_reward: float = 1.0
    step_reward: float = 0.1
    death_penalty: float = -2.0
    early_death_threshold: int = 50
    

@dataclass 
class AgentConfig:
    """Neural network agent configuration."""
    
    hidden_size: int = 32
    second_hidden_size: int = 16
    activation: str = "tanh"
    

@dataclass
class EvolutionConfig:
    """Genetic algorithm configuration."""
    
    population_size: int = 256
    generations: int = 100
    tournament_size: int = 3
    elite_size: int = 13  # Top 5% of 256
    crossover_alpha: float = 0.3
    mutation_sigma: float = 0.2
    mutation_prob: float = 0.1
    early_stop_patience: int = 15
    

@dataclass
class EvaluationConfig:
    """Evaluation and performance configuration."""
    
    episodes_per_agent: int = 3
    parallel_processes: int = 8
    batch_size: int = 16
    deterministic_seed: Optional[int] = None
    # "food": food eaten, survival only breaks ties; "legacy": score + 0.1 * steps;
    # "return": sum of the game rewards (food_reward, step_reward, death_penalty)
    fitness_function: str = "food"
    

@dataclass
class LoggingConfig:
    """Logging and output configuration."""
    
    output_dir: Path = field(default_factory=lambda: Path("outputs"))
    log_level: str = "INFO"
    save_checkpoints: bool = True
    checkpoint_interval: int = 10
    save_best_agents: bool = True
    

@dataclass
class Config:
    """Main configuration container."""
    
    game: GameConfig = field(default_factory=GameConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    evolution: EvolutionConfig = field(default_factory=EvolutionConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    @classmethod
    def from_yaml(cls, path: Path) -> "Config":
        """Load configuration from YAML file."""
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        
        return cls(
            game=GameConfig(**data.get("game", {})),
            agent=AgentConfig(**data.get("agent", {})),
            evolution=EvolutionConfig(**data.get("evolution", {})),
            evaluation=EvaluationConfig(**data.get("evaluation", {})),
            logging=LoggingConfig(**data.get("logging", {})),
        )
    
    def to_yaml(self, path: Path) -> None:
        """Save configuration to YAML file."""
        data = {
            "game": self.game.__dict__,
            "agent": self.agent.__dict__, 
            "evolution": self.evolution.__dict__,
            "evaluation": self.evaluation.__dict__,
            "logging": {k: str(v) if isinstance(v, Path) else v 
                       for k, v in self.logging.__dict__.items()},
        }
        
        with open(path, "w") as f:
            yaml.dump(data, f, indent=2, default_flow_style=False)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "game": self.game.__dict__,
            "agent": self.agent.__dict__,
            "evolution": self.evolution.__dict__,
            "evaluation": self.evaluation.__dict__,
            "logging": self.logging.__dict__,
        }


def load_config(config_file: Optional[Path] = None) -> Config:
    """Load configuration from file or return default config.
    
    Args:
        config_file: Optional path to YAML config file
        
    Returns:
        Configuration object
    """
    if config_file and config_file.exists():
        return Config.from_yaml(config_file)
    else:
        return Config()
