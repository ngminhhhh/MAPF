import numpy as np
from dataclasses import dataclass
from collections import deque

from tqdm import tqdm   

@dataclass
class MAPFInstance:
    grid: np.ndarray # (H, W) 0 is free, 1 is obstacle
    starts: np.ndarray
    goals: np.ndarray

@dataclass
class MAPFGeneratorConfig:
    height: int 
    width: int 
    n_agents: int 
    obstacle_ratio: float 
    n_samples:int 

class MAPFInstanceGenerator:
    def __init__(self, cfg: MAPFGeneratorConfig, seed:int=42):
        self.cfg = cfg
        self.rng = np.random.RandomState(seed)

    def _is_grid_connected(self, grid: np.ndarray)-> bool:
        H, W = grid.shape
        empty_cells = np.argwhere(grid == 0)

        start_r, start_c = empty_cells[0]
        d_axis = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        visited = np.zeros_like(grid == 0, dtype=bool)
        q = deque()
        q.append((start_r, start_c))
        visited[start_r, start_c] = True

        visited_count = 0

        while q:
            r, c = q.popleft()
            visited_count += 1

            for dr, dc in d_axis:
                nr, nc = r + dr, c + dc

                if 0 <= nr < H and 0 <= nc < W: 
                    if not grid[nr, nc] and not visited[nr, nc]:
                        visited[nr, nc] = True
                        q.append((nr, nc))

        return visited_count == empty_cells.shape[0]

    def sample_instance(self) -> MAPFInstance:
        # * Get configs
        H, W = self.cfg.height, self.cfg.width
        n_agents = self.cfg.n_agents
        n_cells = W * H
        n_obs = int(self.cfg.obstacle_ratio * n_cells)

        # * Generate map
        grid = np.zeros((H, W), dtype=np.int8)
        obs_idx = self.rng.choice(n_cells, size=n_obs, replace=False)

        grid.reshape(-1)[obs_idx] = 1

        empty_cells = np.argwhere(grid == 0) 

        # * Get random start/end position of N agents
        agent_cell_idx = self.rng.choice(empty_cells.shape[0], size=2*n_agents, replace=False)
        agent_cells = empty_cells[agent_cell_idx]

        starts = agent_cells[:n_agents]
        goals = agent_cells[n_agents:]

        return MAPFInstance(
            grid=grid,
            starts=starts,
            goals=goals
        )
    
    def sample_instances(self) -> list[MAPFInstance]:
        instances = []
        n_samples = self.cfg.n_samples

        for idx in tqdm(range(n_samples), desc="Generating MAPF instances"):
            instance = self.sample_instance()
            success = self._is_grid_connected(instance.grid)

            while not success:
                instance = self.sample_instance()
                success = self._is_grid_connected(instance.grid)

            instances.append(instance)

        return instances

    def set_n_instances(self, n_instaces):
        self.cfg.n_samples = n_instaces