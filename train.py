import torch
import numpy as np

from env.MAPF_type import MAPFEnvConfig, MAPFGeneratorConfig
from env.MAPF_env import MAPFEnv
from utils.generator import MAPFGenerator

from model.MAPF_solver import Solver

if __name__ == "__main__":
    grid_height = 10
    grid_width = 10
    obstacle_ratio = 0.0
    obs_channels = 4
    n_agents = 4
    obs_radius = 3
    
    n_samples = 1
    max_steps = grid_height * grid_width

    generator_cfg = MAPFGeneratorConfig(width=grid_width,
                                        height=grid_height,
                                        n_agents=n_agents,
                                        obstacle_ratio=obstacle_ratio,
                                        n_samples=n_samples,
                                        seed=4096)
    
    env_cfg = MAPFEnvConfig(width=grid_width,
                            height=grid_height,
                            n_agents=n_agents,
                            obs_radius=obs_radius,
                            max_steps=max_steps)
    
    generator = MAPFGenerator(generator_cfg)
    env = MAPFEnv(env_cfg, generator, None)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Solver(obs_size=2*obs_radius+1, obs_channels=obs_channels, device=device).to(device)

    for i in range(n_samples):
        model.init_hidden_state(n_agents=n_agents)
        obs, info = env.reset()
        goal_vecs = (env.agent_goal - env.agent_pos).astype(np.float32) / np.array([grid_height, grid_width], dtype=np.float32)
        edges = info["comm_edges"]

        done = False
        while not done:
            actions, _ = model.act(obs, goal_vecs, edges)
            new_obs, rewards, terminated, truncated, info = env.step(actions)

            # New state
            obs = new_obs
            goal_vecs = (env.agent_goal - env.agent_pos).astype(np.float32) / np.array([grid_height, grid_width], dtype=np.float32)
            edges = info["comm_edges"]

            env.render()

