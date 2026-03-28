import os
import numpy as np
import torch
from tqdm import tqdm

from env.MAPF_env import *
from utils.generator import *
from utils.replayBuffer import *
from loss.PPO_loss import train_step

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

    n_samples = 10_000
    max_steps = grid_width * grid_height

    # Save dirs
    ckpt_dir = "checkpoints"
    log_dir = "logs"
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Env + generator config
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
        seed=4096
    )

    # Init generator + env
    generator = MAPFGenerator(cfg=generator_cfg)
    env = MAPFEnv(env_cfg=env_cfg, instance_generator=generator, reward_fn=reward_fn)

    buffer = ReplayBuffer(
        capacity=256,
        n_agents=n_agents,
        obs_size=(2 * observation_radius + 1),
        obs_channels=observation_channels
    )

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

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    last_loss_value = np.nan
    saved_loss_steps = []
    saved_losses = []

    pbar = tqdm(range(1, n_samples + 1), desc="Training", unit="episode")

    for i in pbar:
        obs, info = env.reset()
        comm_edges = info["comm_edges"]
        goal_vec = env.agent_goal - env.agent_pos
        done = False

        episode_reward_sum = 0.0

        while not done:
            actions, log_probs, values = model.act(obs, goal_vec, comm_edges)

            new_obs, rewards, terminated, truncated, info = env.step(actions)
            episode_reward_sum += float(np.mean(rewards))

            done = terminated or truncated

            buffer.push(
                state=obs,
                goal_vec=goal_vec,
                action=actions,
                reward=rewards,
                done=np.full(n_agents, done, dtype=bool),
                log_prob=log_probs.cpu().numpy(),
                value=values.cpu().numpy(),
                edges=comm_edges
            )

            if buffer.is_full():
                next_goal_vec = env.agent_goal - env.agent_pos

                last_loss_value = train_step(
                    model=model,
                    optimizer=optimizer,
                    buffer=buffer,
                    next_state=new_obs,
                    next_goal_vec=next_goal_vec,
                    next_edges=info["comm_edges"],
                    gamma=0.99,
                    lam=0.95,
                    clip_eps=0.2,
                    value_coef=0.5,
                    entropy_coef=0.01,
                    n_epochs=4
                )

                buffer.reset()

            obs = new_obs
            comm_edges = info["comm_edges"]
            goal_vec = env.agent_goal - env.agent_pos

        if buffer.size > 0:
            last_loss_value = train_step(
                model=model,
                optimizer=optimizer,
                buffer=buffer,
                next_state=None,
                next_goal_vec=None,
                next_edges=None,
                gamma=0.99,
                lam=0.95,
                clip_eps=0.2,
                value_coef=0.5,
                entropy_coef=0.01,
                n_epochs=4
            )
            buffer.reset()

        # Update progress bar info
        pbar.set_postfix({
            "last_loss": f"{last_loss_value:.4f}" if not np.isnan(last_loss_value) else "nan",
            "ep_rew_mean": f"{episode_reward_sum:.4f}"
        })

        if i % 500 == 0:
            ckpt_path = os.path.join(ckpt_dir, f"solver_step_{i}.pt")
            torch.save({
                "sample_idx": i,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "last_loss": last_loss_value,
                "env_cfg": env_cfg,
                "generator_cfg": generator_cfg,
                "model_hparams": {
                    "obs_width": 2 * observation_radius + 1,
                    "obs_height": 2 * observation_radius + 1,
                    "obs_channels": observation_channels,
                    "cnn_hidden_dim": cnn_hidden_dim,
                    "n_cnn_blocks": n_cnn_blocks,
                    "kernel_size": kernel_size,
                    "goal_out_dim": goal_out_dim,
                    "gat_hidden_dim": gat_hidden_dim,
                    "n_gat_heads": n_gat_heads,
                    "mlp_hidden_dim": mlp_hidden_dim,
                    "n_mlp_layers": n_mlp_layers,
                }
            }, ckpt_path)


        if i % 100 == 0:
            saved_loss_steps.append(i)
            saved_losses.append(last_loss_value)

            np.save(os.path.join(log_dir, "loss_steps.npy"), np.array(saved_loss_steps, dtype=np.int32))
            np.save(os.path.join(log_dir, "last_losses.npy"), np.array(saved_losses, dtype=np.float32))