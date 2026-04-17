from dataclasses import dataclass
import numpy as np

@dataclass
class MAPFInstance:
    grid: np.ndarray # (H, W) 0 is free, 1 is obstacle
    starts: np.ndarray
    goals: np.ndarray

@dataclass
class MAPFEnvConfig:
    width: int
    height: int
    n_agents: int
    obs_radius: int
    max_steps: int

@dataclass
class MAPFGeneratorConfig:
    width: int 
    height: int 
    n_agents: int 
    obstacle_ratio: float 
    seed: int 