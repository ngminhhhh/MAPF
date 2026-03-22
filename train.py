from env.MAPF_env import *
from utils.generator import *

def reward_fn():
    return 0

if __name__ == "__main__":
    # * Configuration params
    grid_width = 10
    grid_height = 10
    obs_ratio = 0.2
    n_agents = 4
    obs_radius = 2
    base_seed = 10000

    n_samples = 50_000 # number of training samples
    max_steps = grid_width * grid_height # max step before interrupt

    env_cfg = MAPFEnvConfig(width=grid_width, height=grid_height, 
                            n_agents=n_agents, obs_radius=obs_radius, max_steps=max_steps)
    
    generator_cfg = MAPFGeneratorConfig(width=grid_width, height=grid_height, 
                                        n_agents=n_agents, obstacle_ratio=obs_ratio, n_samples=n_samples)
    
    generator = MAPFIntanceGenerator(cfg=generator_cfg, base_seed=base_seed)
    env = MAPFEnv(env_cfg=env_cfg, instance_generator=generator, reward_fn=reward_fn)

    # * Main:
    for i in range(n_samples):
        obs, _ = env.reset()

        print(obs)
        break

