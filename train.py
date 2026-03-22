from env.MAPF_env import *

if __name__ == "__main__":
    # * Configuration params
    grid_width = 10
    grid_height = 10
    n_agents = 4
    obs_radius = 2

    n_samples = 50_000
    max_steps = grid_width * grid_height

    env_cfg = MAPFEnvConfig(width=grid_width, height=grid_height, n_agents=n_agents, obs_radius=obs_radius, max_steps=max_steps)
    