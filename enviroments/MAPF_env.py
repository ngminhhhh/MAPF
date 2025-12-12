import gymnasium as gym
from gymnasium import spaces
from generator.Generator import *
from typing import Callable, Tuple
from model.A_star import A_star

# * Params of reward function
RewardFn = Callable[
    [
        "MAPFEnv",      
        np.ndarray,     # old_pos: (n_agents, 2)
        np.ndarray,     # new_pos: (n_agents, 2)
        bool,           # done (terminated or truncated)
    ],
    Tuple[float, np.ndarray]  # (global_reward, reward_per_agents)
]

class MAPFEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps":5}

    def __init__(self, instance_cfg: MAPFGeneratorConfig,
                 instances: list[MAPFInstance],
                 reward_fn: RewardFn | None = None, 
                 obs_radius:int=1, max_steps:int | None = None):
        super().__init__()
        self.cfg = instance_cfg
        self.reward_fn = reward_fn
        self.instances = instances

        self.H = self.cfg.height
        self.W = self.cfg.width
        self.n_agents = self.cfg.n_agents

        self.obs_radius = obs_radius
        self.obs_size = 2 * obs_radius + 1

        if max_steps is None:
            self.max_steps = 4 * (self.H + self.W)
        else:
            self.max_steps = max_steps

        self.action_space = spaces.MultiDiscrete([5] * self.n_agents) # (n_agents, ) with value in ith is action of agents i in [0,4]

        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.n_agents, self.obs_size, self.obs_size, 3), # n_channels = 3
            dtype=np.float32,
        )

        self.grid: np.ndarray | None = None
        self.agent_pos: np.ndarray | None = None
        self.agent_goal: np.ndarray | None = None
        self.step_count: int = 0
        self._dataset_idx: int = 0 # * dataset index pointer

    def reset(self, *, seed: int|None = None, options: dict | None=None):
        super().reset(seed=seed)

        # * Get instance
        idx = self._dataset_idx % len(self.instances)
        instance = self.instances[idx]
        self._dataset_idx += 1

        self.grid = instance.grid.copy()
        self.agent_pos = instance.starts.copy()
        self.agent_goal = instance.goals.copy()
        self.step_count = 0

        obs = self._build_observation()
        Astar_path = self._get_Astar_path()
        info = {
            "instance": instance,
            "Astar_paths": Astar_path
        }

        return obs, info
    
    def step(self, action):
        self.step_count += 1

        action = np.asarray(action, dtype=np.int64)

        deltas = np.array([
            [0, 0],    # stay
            [-1, 0],   # up
            [1, 0],    # down
            [0, -1],   # left
            [0, 1],    # right
        ], dtype=np.int64)

        H, W = self.H, self.W

        old_pos = self.agent_pos.copy()

        # * Proposed move
        proposed = self.agent_pos + deltas[action]
        new_pos = self.agent_pos.copy()

        # * Bound + Obstacle
        for i in range(self.n_agents):
            r, c = proposed[i]
            if 0 <= r < H and 0 <= c < W and self.grid[r, c] == 0:
                new_pos[i] = [r, c]
            else:
                new_pos[i] = self.agent_pos[i]

        # * Collision => all stay
        unique, counts = np.unique(new_pos, axis=0, return_counts=True)
        collision_cells = unique[counts > 1]
        if collision_cells.size > 0:
            for cell in collision_cells:
                mask = np.all(new_pos == cell, axis=1)
                new_pos[mask] = self.agent_pos[mask]

        # * Update state
        self.agent_pos = new_pos

        # * Terminated / truncated
        at_goal = np.all(self.agent_pos == self.agent_goal, axis=1)
        terminated = bool(np.all(at_goal))
        truncated = self.step_count >= self.max_steps and not terminated
        done = terminated or truncated

        # * Reward
        if self.reward_fn is not None:
            all_rewards, reward = self.reward_fn(
                self,
                old_pos,
                self.agent_pos,
                done,
            )
        else:
            all_rewards, reward = 0, 0

        # * Observation
        obs = self._build_observation()
        Astar_path = self._get_Astar_path()

        info = {
            "Astar_paths": Astar_path,
            "reward_per_agent": reward,
            "terminated": terminated,
            "truncated": truncated,
        }

        return obs, all_rewards, terminated, truncated, info

    def _get_Astar_path(self):
        '''
        @return:
            paths - List[List[Tuple[int, int]]]: path of all agents 
        '''
        paths = []
        for i in range(self.n_agents):
            sr, sc = self.agent_pos[i]
            gr, gc = self.agent_goal[i]
            path = A_star(self.grid, (sr, sc), (gr, gc))
            paths.append(path)
        
        return paths

    def _build_observation(self) -> np.ndarray:
        H, W = self.H, self.W
        R = self.obs_radius

        obs = np.zeros(
            (self.n_agents, self.obs_size, self.obs_size, 3),
            dtype=np.float32,
        )

        # * Plane 0: obstacle
        obstacle_plane = (self.grid == 1).astype(np.float32)

        # * Plane 1: all agents
        agent_plane = np.zeros((H, W), dtype=np.float32)
        for i in range(self.n_agents):
            r, c = self.agent_pos[i]
            agent_plane[r, c] = 1.0

        for i in range(self.n_agents):
            ar, ac = self.agent_pos[i]
            gr, gc = self.agent_goal[i]

            rmin = ar - R
            rmax = ar + R
            cmin = ac - R
            cmax = ac + R

            for wr, r in enumerate(range(rmin, rmax + 1)):
                for wc, c in enumerate(range(cmin, cmax + 1)):
                    if 0 <= r < H and 0 <= c < W:
                        # Plane 0: obstacle
                        obs[i, wr, wc, 0] = obstacle_plane[r, c]
                        
                        # Plane 1: other agents (không tính bản thân)
                        if not (r == ar and c == ac) and agent_plane[r, c] == 1.0:
                            obs[i, wr, wc, 1] = 1.0
                    else:
                        # out of bound -> obstacle
                        obs[i, wr, wc, 0] = 1.0

            # * Plane 2: goal projection 
            dr = gr - ar
            dc = gc - ac

            pr = int(np.clip(dr, -R, R))
            pc = int(np.clip(dc, -R, R))

            wr_goal = pr + R
            wc_goal = pc + R

            obs[i, wr_goal, wc_goal, 2] = 1.0

        return obs
