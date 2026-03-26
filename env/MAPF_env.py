import gymnasium as gym
from gymnasium import spaces
from env.MAPF_type import *
import numpy as np

from utils.generator import *

import pygame 
import heapq

def manhattan(r, c, gr, gc):
    return abs(r - gr) + abs(c - gc)

def A_star(grid: np.ndarray, start: np.ndarray, goal: np.ndarray):
    H, W = grid.shape
    sr, sc = start
    gr, gc = goal

    deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    open_heap = []
    heapq.heappush(open_heap, (manhattan(sr, sc, gr, gc), 0, (sr, sc))) # (f, g, node)

    par = {(sr, sc) : None}
    g_score = {(sr, sc) : 0}

    while open_heap:
        f, g, (r, c) = heapq.heappop(open_heap)

        if (r, c) == (gr, gc):
            path = []
            cur = (r, c)

            while cur is not None:
                path.append(cur)
                cur = par[cur]

            path.reverse()
            return path
        
        for (dr, dc) in deltas:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < H and 0 <= nc < W) or grid[nr, nc] == 1: continue

            tentative_g = g + 1

            if (nr, nc) not in g_score or tentative_g < g_score[(nr, nc)]:
                g_score[(nr, nc)] = tentative_g
                par[(nr, nc)] = (r, c)
                heapq.heappush(open_heap, (tentative_g + manhattan(nr, nc, gr, gc), tentative_g, (nr, nc)))

    return None

class MAPFRender:
    def __init__(self, cell_size=60, margin=20, fps=30, title="MAPF"):
        self.cell_size = cell_size
        self.margin = margin
        self.fps = fps
        self.title = title

        # Colors
        self.BG = (0, 0, 0)
        self.GRID_LINE = (45, 45, 45)
        self.FREE_CELL = (0, 0, 0)
        self.OBSTACLE = (90, 90, 90)

        self.AGENT = (220, 0, 0)
        self.AGENT_BORDER = (15, 15, 15)

        self.GOAL_TEXT = (255, 255, 255)
        self.AGENT_TEXT = (255, 255, 255)

        # State
        self._disabled = False
        self._inited = False
        self._screen = None
        self._clock = None
        self._font = None

        self._bg_surface = None
        self._bg_key = None

    def lazy_init(self, grid_shape):
        if self._disabled or self._inited:
            return

        pygame.init()
        pygame.display.set_caption(self.title)
        H, W = grid_shape
        w_px = W * self.cell_size + 2 * self.margin
        h_px = H * self.cell_size + 2 * self.margin

        self._screen = pygame.display.set_mode((w_px, h_px))
        self._clock = pygame.time.Clock()

        font_size = max(14, int(self.cell_size * 0.42))

        self._font = pygame.font.SysFont("JetBrains Mono", font_size)

        self._inited = True

    def _cell_rect(self, r, c):
        x = self.margin + c * self.cell_size
        y = self.margin + r * self.cell_size
        return pygame.Rect(x, y, self.cell_size, self.cell_size)

    def _build_background(self, grid, instance_idx):
        H, W = grid.shape
        key = (instance_idx, grid.shape[0], grid.shape[1], self.cell_size, self.margin)

        if self._bg_surface is not None and self._bg_key == key:
            return

        w_px = W * self.cell_size + 2 * self.margin
        h_px = H * self.cell_size + 2 * self.margin
        surf = pygame.Surface((w_px, h_px)).convert()
        surf.fill(self.BG)

        # Draw cells + grid lines
        for r in range(H):
            for c in range(W):
                rect = self._cell_rect(r, c)
                if grid[r, c] != 0:
                    pygame.draw.rect(surf, self.OBSTACLE, rect)
                else:
                    pygame.draw.rect(surf, self.FREE_CELL, rect)

                pygame.draw.rect(surf, self.GRID_LINE, rect, 1)

        self._bg_surface = surf
        self._bg_key = key

    def render(self, grid, agent_pos, agent_goal, instance_idx, mode="human"):
        if self._disabled:
            return None

        H, W = grid.shape
        self.lazy_init((H, W))
        if not self._inited:
            return None

        self._build_background(grid, instance_idx)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._disabled = True
                self.close()
                return None

        self._screen.blit(self._bg_surface, (0, 0))

        for i in range(agent_goal.shape[0]):
            gr, gc = int(agent_goal[i, 0]), int(agent_goal[i, 1])
            if 0 <= gr < H and 0 <= gc < W:
                rect = self._cell_rect(gr, gc)
                label = self._font.render(str(i + 1), True, self.GOAL_TEXT)
                lab_rect = label.get_rect(center=rect.center)
                self._screen.blit(label, lab_rect)

        pad = max(4, self.cell_size // 12)
        br = max(1, self.cell_size // 20)  
        for i in range(agent_pos.shape[0]):
            ar, ac = int(agent_pos[i, 0]), int(agent_pos[i, 1])
            if not (0 <= ar < H and 0 <= ac < W):
                continue

            cell = self._cell_rect(ar, ac)
            rect = pygame.Rect(cell.x + pad, cell.y + pad, cell.w - 2 * pad, cell.h - 2 * pad)

            pygame.draw.rect(self._screen, self.AGENT, rect, border_radius=br)
            pygame.draw.rect(self._screen, self.AGENT_BORDER, rect, 2, border_radius=br)

            label = self._font.render(str(i + 1), True, self.AGENT_TEXT)
            lab_rect = label.get_rect(center=rect.center)
            self._screen.blit(label, lab_rect)

        # Output
        pygame.display.flip()

        if mode == "human":
            self._clock.tick(self.fps)
            return None

        if mode == "rgb_array":
            frame = pygame.surfarray.array3d(self._screen).transpose(1, 0, 2)
            return frame

    def close(self):
        if self._inited:
            pygame.quit()

        self._inited = False
        self._screen = None
        self._clock = None
        self._font = None
        self._bg_surface = None
        self._bg_key = None

class MAPFEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps":5}

    def __init__(self, env_cfg: MAPFEnvConfig, instance_generator: MAPFGenerator, reward_fn, render_mode="human"):
        super().__init__()
        self.env_cfg = env_cfg
        self.instance_generator = instance_generator
        self.reward_fn = reward_fn

        self.render_mode = render_mode
        self._renderer = MAPFRender(cell_size=60, margin=20, fps=self.metadata.get("render_fps", 5))
        
        self.action_space = spaces.MultiDiscrete([5] * self.env_cfg.n_agents) # (n_agents, ) with value in ith is action of agents i in [0,4]
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.env_cfg.n_agents, self.env_cfg.obs_radius * 2 + 1, self.env_cfg.obs_radius * 2 + 1, 4), # n_channels = 4
            dtype=np.float32,
        )

        # * Instances controller
        self.grid: np.ndarray  # (H, W)
        self.agent_pos: np.ndarray  # (N, 2)
        self.agent_goal: np.ndarray # (N, 2)

        # * Instance info 
        self.step_count = 0

    def reset(self, *, seed: int|None = None, options: dict|None = None):
        '''
            @return:
                obs: first observation of each agents in instance
                info: infomation of instance
        '''
        super().reset(seed=seed)

        # * Get new instance 
        instance = self.instance_generator.generate_instance()

        # * Config new instance
        self.grid = instance.grid
        self.agent_pos = instance.starts.copy()
        self.agent_goal = instance.goals
        self.step_count = 0

        # * Build observation
        obs = self._build_observation()
        info = {
            "comm_edges": self._build_communication_graph(),
        }

        return obs, info
    
    def resolve_conflict(self, actions):
        deltas = np.array([
            [0, 0], # stay
            [-1, 0], # up
            [1, 0], # down
            [0, -1], # left
            [0, 1] # right
        ], dtype=np.int64)

        H, W = self.env_cfg.height, self.env_cfg.width

        old_pos = self.agent_pos.copy()
        new_pos = self.agent_pos + deltas[actions]

        N = self.env_cfg.n_agents
        
        # * Bound + Obstacle
        for i in range(N):
            r, c = new_pos[i]
            if not (0 <= r < H and 0 <= c < W) or self.grid[r, c] == 1:
                new_pos[i] = old_pos[i]

        changed = True #  * Is existed any agents reverted to old pos?
        iters = 0
        max_iters = N + 2

        while changed and iters < max_iters:
            changed = False 
            iters += 1

            _, inverse, counts = np.unique(new_pos, axis=0, return_inverse=True, return_counts=True)  # inverse: map agent to unique_cells indices
            collision_mask = counts[inverse] > 1 

            # * Vertex collision => stay in old pos
            if np.any(collision_mask):
                new_pos[collision_mask] = old_pos[collision_mask]
                changed = True

            moved = ~np.all(new_pos == old_pos, axis=1)
            moved_idx, = np.where(moved)

            # * Edge collision check
            if moved_idx.size >= 2:
                edge_owner = {}
                swap_agents = set()

                for i in moved_idx:
                    a = (int(old_pos[i, 0]), int(old_pos[i, 1]))
                    b = (int(new_pos[i, 0]), int(new_pos[i, 1]))

                    j = edge_owner.get((b, a))
                    if j is not None:
                        swap_agents.add(i)
                        swap_agents.add(j)
                    else:
                        edge_owner[(a, b)] = i

                if swap_agents:
                    swap_agents = np.fromiter(swap_agents, dtype=np.int64)
                    new_pos[swap_agents] = old_pos[swap_agents]
                    changed = True

        self.agent_pos = new_pos

    def step(self, actions: np.ndarray): # action (N, ) 
        self.step_count += 1

        old_pos = self.agent_pos.copy()

        self.resolve_conflict(actions)
        
        # * Return infomation
        at_goal = np.all(self.agent_pos == self.agent_goal, axis=1)
        terminated = bool(np.all(at_goal))
        truncated = self.step_count >= self.env_cfg.max_steps and not terminated

        reward = self.reward_fn(
                self,
                old_pos,
                self.agent_pos,
            )
        
        obs = self._build_observation()
        info = {
            "terminated": terminated,
            "truncated": truncated,
            "comm_edges": self._build_communication_graph(),
        }

        return obs, reward, terminated, truncated, info

    def _build_observation(self):
        H, W = self.env_cfg.height, self.env_cfg.width
        R = self.env_cfg.obs_radius

        obs = np.zeros((self.env_cfg.n_agents, 2*R + 1, 2*R + 1, 4), dtype=np.float32) # * (N, 2R + 1, 2R + 1, 4)

        # * Plane 0: obstacle
        obstacle_plane = (self.grid == 1).astype(np.float32) # (H, W)

        # * Plane 1: all agents
        agent_plane = np.zeros((H, W), dtype=np.float32) # (H, W)
        
        for i in range(self.env_cfg.n_agents):
            r, c = self.agent_pos[i]
            agent_plane[r, c] = 1.0

        for i in range(self.env_cfg.n_agents):
            ar, ac = self.agent_pos[i]
            gr, gc = self.agent_goal[i]

            r_min, r_max = ar - R, ar + R
            c_min, c_max = ac - R, ac + R

            for wr, r in enumerate(range(r_min, r_max + 1)):
                for wc, c in enumerate(range(c_min, c_max + 1)):
                    if 0 <= r < H and 0 <= c < W:
                        obs[i, wr, wc, 0] = obstacle_plane[r, c] # Plane 0: obstacle

                        if not (r == ar and c == ac):
                            obs[i, wr, wc, 1] = agent_plane[r, c]

                    else:
                        obs[i, wr, wc, 0] = 1.0

            # * Plane 2: goal projection
            dr, dc = gr - ar, gc - ac

            pr, pc = int(np.clip(dr, -R, R)), int(np.clip(dc, -R, R))
            wr_goal, wc_goal = pr + R, pc + R # [-R, R] => [0, 2R]
            
            obs[i, wr_goal, wc_goal, 2] = 1.0

            # * Plane 3: A* path
            Astar_path = A_star(self.grid, self.agent_pos[i], self.agent_goal[i])
            
            for (pr, pc) in Astar_path[1:]:
                wr, wc = pr - ar + R, pc - ac + R
                if 0 <= wr < 2*R + 1 and 0 <= wc < 2*R + 1:
                    obs[i, wr, wc, 3] = 1.0

        return obs # * (N, 2R + 1, 2R + 1, 4)

    def _build_communication_graph(self):
        pos = self.agent_pos                
        R = self.env_cfg.obs_radius

        # Pairwise relative positions: rel[i, j] = pos[j] - pos[i]
        rel = pos[None, :, :] - pos[:, None, :]   # (N, N, 2)

        # Square FOV mask
        in_fov = (
            (np.abs(rel[:, :, 0]) <= R) &
            (np.abs(rel[:, :, 1]) <= R)
        ) # (N, N)

        # Remove self-loops
        np.fill_diagonal(in_fov, False)

        # Convert boolean mask -> edge list [src, dst]
        src, dst = np.nonzero(in_fov)
        edges = np.stack([dst, src], axis=1).astype(np.int64)   # (E, 2) src send message to dst

        return edges

    def render(self):
        if self._renderer is None:
            return None
        return self._renderer.render(
            self.grid, self.agent_pos, self.agent_goal,
            instance_idx=self.current_instance_idx,
            mode=self.render_mode
        )

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


