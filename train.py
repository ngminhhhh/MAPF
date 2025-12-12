from data_generator.Generator import *
from enviroments.MAPF_env import *
from model.architecture_1 import *

from torch.distributions import Categorical
from tqdm import tqdm
import numpy as np
import torch.nn as nn
import torch

if __name__ == "__main__":
    # * Instance configuration
    width = 10
    height = 10
    n_agents = 10
    obstacle_ratio = 0.2

    n_instances_pool = 5000
    n_episodes = 5000

    seed = 42

    cfg = MAPFGeneratorConfig(
        width=width,
        height=height,
        n_agents=n_agents,
        obstacle_ratio=obstacle_ratio,
        n_samples=n_instances_pool,  
    )

    # * Generate fixed pool of instances
    generator = MAPFInstanceGenerator(cfg, seed=seed)
    instances = generator.sample_instances(save_flag=False)

    env = MAPFEnv(instance_cfg=cfg, instances=instances, reward_fn=reward_fn, obs_radius=1)

    # * Solver configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    obs_channels = 3
    obs_size = 3
    cnn_out_dim = 64
    n_cnn_blocks = 3
    gru_out_dim = 128
    gat_hidden_dim = 256
    n_gat_layers = 2
    forecast_k = 4

    solver = MAPFSolver(
        cfg,
        device,
        obs_channels,
        obs_size,
        cnn_out_dim,
        n_cnn_blocks,
        gru_out_dim,
        gat_hidden_dim,
        n_gat_layers,
        forecast_k
    ).to(device)
    solver.train()

    # * Optimizer
    optimizer = torch.optim.Adam(solver.parameters(), lr=3e-4)
    gamma = 0.99
    value_coef = 0.5
    entropy_coef = 0.01

    episode_rewards = []
    episode_success_rates = []

    for i in tqdm(range(n_episodes), desc="Training episodes"):
        obs, info = env.reset()
        A_star_paths = info["Astar_paths"]

        done = False
        total_reward = 0.0
        step = 0

        solver.h_prev = None

        log_probs_list = []
        values_list = []
        rewards_list = []
        entropies_list = []

        while not done:
            # * Take action from policy
            logits, values = solver(obs, A_star_paths)   # logits: (1,N,5), values:(1,N)
            logits = logits.squeeze(0)   # (N,5)
            values = values.squeeze(0)   # (N,)

            # * Change to softmax - probability
            dist = Categorical(logits=logits)
            actions_tensor = dist.sample()              # (N,)
            log_probs = dist.log_prob(actions_tensor)   # (N,)
            entropy = dist.entropy()                    # (N,)

            actions = actions_tensor.cpu().numpy().astype(np.int64)

            # * Do action
            obs, reward, terminated, truncated, info = env.step(actions)
            A_star_paths = info["Astar_paths"]

            done = terminated or truncated
            total_reward += reward
            step += 1

            # * Save reward + log_probs
            reward_per_agent = info["reward_per_agent"]
            rewards = torch.from_numpy(reward_per_agent).to(device=device, dtype=torch.float32)

            log_probs_list.append(log_probs)
            values_list.append(values)
            rewards_list.append(rewards)
            entropies_list.append(entropy)

        # * success_rate per episode
        done_per_agent = info.get("done_per_agent", None)
        if done_per_agent is not None:
            success_rate = float(done_per_agent.mean())
        else:
            success_rate = 1.0 if terminated and not truncated else 0.0

        episode_rewards.append(total_reward)
        episode_success_rates.append(success_rate)

        # * train step
        log_probs_tensor = torch.stack(log_probs_list)     # (T, N)
        values_tensor = torch.stack(values_list)           # (T, N)
        rewards_tensor = torch.stack(rewards_list)         # (T, N)
        entropies_tensor = torch.stack(entropies_list)     # (T, N)

        T = rewards_tensor.shape[0]

        returns = torch.zeros_like(rewards_tensor, device=device)
        G = torch.zeros(n_agents, device=device)

        for t in reversed(range(T)):
            G = rewards_tensor[t] + gamma * G
            returns[t] = G

        advantages = returns - values_tensor.detach()

        # * Loss
        actor_loss = -(log_probs_tensor * advantages).mean()
        value_loss = nn.functional.mse_loss(values_tensor, returns)
        entropy_loss = -entropies_tensor.mean()

        loss = actor_loss + value_coef * value_loss + entropy_coef * entropy_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(solver.parameters(), 1.0)
        optimizer.step()

        if (i + 1) % 1000 == 0:
            checkpoint_path = f"mapf_solver_ep_{i+1}.pt"
            torch.save(
                {
                    "model_state_dict": solver.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": {
                        "width": width,
                        "height": height,
                        "n_agents": n_agents,
                        "obstacle_ratio": obstacle_ratio,
                        "obs_radius": 1,
                        "gamma": gamma,
                    },
                },
                checkpoint_path,
            )
            print(f"[Checkpoint] Saved checkpoint at episode {i+1} to {checkpoint_path}")

    np.savez(
        "training_metrics.npz",
        episode_rewards=np.array(episode_rewards, dtype=np.float32),
        episode_success_rates=np.array(episode_success_rates, dtype=np.float32),
    )
    print("Saved training metrics to training_metrics.npz")

    # * Final checkpoint
    checkpoint_path = "mapf_solver.pt"
    torch.save(
        {
            "model_state_dict": solver.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": {
                "width": width,
                "height": height,
                "n_agents": n_agents,
                "obstacle_ratio": obstacle_ratio,
                "obs_radius": 1,
                "gamma": gamma,
            },
        },
        checkpoint_path,
    )
    print(f"Saved checkpoint to {checkpoint_path}")
