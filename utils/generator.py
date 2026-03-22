import numpy as np
from env.MAPF_type import *
from tqdm import tqdm
from dataclasses import dataclass
from collections import deque

class MAPFIntanceGenerator:
    def __init__(self, cfg: MAPFGeneratorConfig, base_seed:int=0):
        self.cfg = cfg
        self.base_seed = base_seed
        self.instance_idx = 0

    def _is_grid_connected(self, grid: np.ndarray) -> bool:
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
    
    def generate_instance(self, seed: int) -> MAPFInstance:
        rng = np.random.RandomState(seed)

        H, W = self.cfg.height, self.cfg.width
        n_agents = self.cfg.n_agents
        n_cells = H * W
        n_obs = int(self.cfg.obstacle_ratio * n_cells)

        while True:
            grid = np.zeros((H, W), dtype=np.int8)
            obs_idx = rng.choice(n_cells, size=n_obs, replace=False)
            grid.reshape(-1)[obs_idx] = 1

            if not self._is_grid_connected(grid):
                continue

            empty_cells = np.argwhere(grid == 0)
            idx = rng.choice(empty_cells.shape[0], size=2 * n_agents, replace=False)
            cells = empty_cells[idx]
            starts = cells[:n_agents]
            goals  = cells[n_agents:]

            return MAPFInstance(grid=grid, starts=starts, goals=goals)
    
    def next_instance(self) -> MAPFInstance:
        seed = self.base_seed + self.instance_idx
        instance = self.generate_instance(seed)
        self.instance_idx += 1

        return instance

    def sample_instances(self, n_samples) -> list[MAPFInstance]:
        instances = []
        for _ in tqdm(n_samples, desc="Generating MAPF instances"):
            instances.append(self.next_instance())
            
        return instances
    


