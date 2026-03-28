import torch

from env.MAPF_env import *
from utils.generator import *

from model.MAPF_solver import Solver
from model.reward import reward_fn

if __name__ == "__main__":
    # Instance params
    grid_width = 10
    grid_height = 10
    obs_ratio = 0.2
    observation_channels = 4
    n_agents = 4
    observation_radius = 3

    n_samples = 100
    max_steps = grid_width * grid_height

    # Save dirs
    ckpt_dir = "checkpoints"

    env_cfg = MAPFEnvConfig(
        width=grid_width,
        height=grid_height,
        n_agents=n_agents,
        obs_radius=observation_radius,
        max_steps=max_steps
    )

    generator_cfg = MAPFGeneratorConfig(
        width=grid_width,
        height=grid_height,
        n_agents=n_agents,
        obstacle_ratio=obs_ratio,
        n_samples=n_samples,
        seed=0
    )

    generator = MAPFGenerator(cfg=generator_cfg)
    env = MAPFEnv(env_cfg=env_cfg, instance_generator=generator, reward_fn=reward_fn)

    # Model config
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # CNN
    cnn_hidden_dim = 128
    n_cnn_blocks = 2
    kernel_size = 2 * observation_radius + 1
    goal_out_dim = 4

    # GAT
    gat_hidden_dim = 128
    n_gat_heads = 2

    # MLP
    mlp_hidden_dim = 128
    n_mlp_layers = 2

    model = Solver(
        obs_width=2 * observation_radius + 1,
        obs_height=2 * observation_radius + 1,
        obs_channels=observation_channels,
        cnn_hidden_dim=cnn_hidden_dim,
        n_cnn_blocks=n_cnn_blocks,
        kernel_size=kernel_size,
        goal_out_dim=goal_out_dim,
        gat_hidden_dim=gat_hidden_dim,
        n_gat_heads=n_gat_heads,
        mlp_hidden_dim=mlp_hidden_dim,
        n_mlp_layers=n_mlp_layers
    ).to(device)

    ckpt = torch.load("checkpoints/solver_step_1000.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    success_instances = 0

    for i in range(n_samples):
        obs, info = env.reset()
        comm_edges = info["comm_edges"]
        goal_vec = env.agent_goal - env.agent_pos
        done = False
        step = 0

        while not done:
            actions, log_probs, values = model.act(obs, goal_vec, comm_edges)

            new_obs, rewards, terminated, truncated, info = env.step(actions)

            done = terminated or truncated

            obs = new_obs
            comm_edges = info["comm_edges"]
            goal_vec = env.agent_goal - env.agent_pos

            step += 1

            env.render()

        agents_goal = env.agent_goal
        agents_pos = env.agent_pos

        reached = np.all(agents_pos == agents_goal, axis=1)
        n_reached = int(reached.sum())

        if n_reached == env_cfg.n_agents:
            success_instances += 1

        print(f"Episode {i}: reached {n_reached}/{len(agents_goal)} agents")

    print(f"success percent: {success_instances}")