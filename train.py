from env.MAPF_env import *
from utils.generator import *
from utils.replayBuffer import *

from model.MAPF_solver import Solver

import torch

if __name__ == "__main__":
    # * Instance's params
    grid_width = 10
    grid_height = 10
    obs_ratio = 0.2
    observation_channels = 4
    n_agents = 4
    observation_radius = 3

    n_samples = 1 # number of training samples
    max_steps = grid_width * grid_height # max step before interrupt

    # * Env + generator config
    env_cfg = MAPFEnvConfig(width=grid_width, height=grid_height, 
                            n_agents=n_agents, obs_radius=observation_radius, max_steps=max_steps)
    
    generator_cfg = MAPFGeneratorConfig(width=grid_width, height=grid_height, 
                                        n_agents=n_agents, obstacle_ratio=obs_ratio, n_samples=n_samples,
                                        seed=4096)
    
    # * Init generator + env
    generator = MAPFGenerator(cfg=generator_cfg)
    env = MAPFEnv(env_cfg=env_cfg, instance_generator=generator, reward_fn=None)

    buffer = ReplayBuffer(capacity=256, 
                          n_agents=n_agents, 
                          obs_size=(2*observation_radius+1), 
                          obs_channels=observation_channels)
    
    # * Model config
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # * cnn
    cnn_hidden_dim = 128
    n_cnn_blocks = 2
    kernel_size = 2*observation_radius +1
    # * gat
    gat_hidden_dim = 128
    n_gat_heads = 2
    # * mlp
    mlp_hidden_dim = 128
    n_mlp_layers = 2

    model = Solver(obs_width=2*observation_radius+1, obs_height=2*observation_radius+1, obs_channels=observation_channels,
                   cnn_hidden_dim=cnn_hidden_dim, n_cnn_blocks=n_cnn_blocks, kernel_size=kernel_size,
                   gat_hidden_dim=gat_hidden_dim, n_gat_heads=n_gat_heads,
                   mlp_hidden_dim=mlp_hidden_dim,n_mlp_layers=n_mlp_layers).to(device)

    # * Main:
    for i in range(n_samples):
        obs, info = env.reset()
        comm_edges = info["comm_edges"]
        done = False

        while not done:
            actions, log_probs, values = model.act(obs, comm_edges)
            print(actions)
            new_obs, rewards, terminated, truncated, _ = env.step(actions)

            done = terminated or truncated

            buffer.push(state=obs, action=actions, reward=rewards, done=np.full(n_agents, done, dtype=bool), log_prob=log_probs.cpu().numpy(), value=values.cpu().numpy())



            
