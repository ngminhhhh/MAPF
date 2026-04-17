import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import trange

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from env.MAPF_env import MAPFEnv
from env.MAPF_type import MAPFEnvConfig, MAPFGeneratorConfig
from model.MAPF_solver import Solver
from utils.generator import MAPFGenerator


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained MAPF solver.")
    parser.add_argument("--grid_size", type=int, default=10)
    parser.add_argument("--obs_radius", type=int, default=2)
    parser.add_argument("--n_agents", type=int, default=4)
    parser.add_argument("--obstacle_ratio", type=float, default=0.2)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    n_samples = 100
    obstacle_ratio = 0.2
    max_steps = 20
    seed = 2026
    weights_path = ROOT_DIR / "weights" / "weights.pt"

    if not weights_path.is_file():
        raise FileNotFoundError(f"Model weights not found: {weights_path}")

    env_cfg = MAPFEnvConfig(
        width=args.grid_size,
        height=args.grid_size,
        n_agents=args.n_agents,
        obs_radius=args.obs_radius,
        max_steps=max_steps,
    )
    generator_cfg = MAPFGeneratorConfig(
        width=args.grid_size,
        height=args.grid_size,
        n_agents=args.n_agents,
        obstacle_ratio=obstacle_ratio,
        seed=seed,
    )

    generator = MAPFGenerator(cfg=generator_cfg)
    env = MAPFEnv(
        env_cfg=env_cfg,
        instance_generator=generator,
        render_mode="human",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Solver(obs_size=2 * args.obs_radius + 1, device=device).to(device)
    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()

    grid_scale = np.array([args.grid_size, args.grid_size], dtype=np.float32)
    success_count = 0
    total_reached_agents = 0
    total_steps = 0

    progress_bar = trange(n_samples, desc="Evaluating", unit="episode", file=sys.stdout)

    for episode_idx in progress_bar:
        obs, info = env.reset()
        comm_edges = info["comm_edges"]
        goal_vec = (env.agent_goal - env.agent_pos).astype(np.float32) / grid_scale
        hidden_state = model.init_hidden_state(args.n_agents)
        terminated = False
        truncated = False
        episode_steps = 0

        while not terminated and not truncated:
            actions, _, hidden_state = model.act(
                obs,
                goal_vec,
                comm_edges,
                hidden_state,
                deterministic=True,
            )
            obs, _, terminated, truncated, info = env.step(actions)
            comm_edges = info["comm_edges"]
            goal_vec = (env.agent_goal - env.agent_pos).astype(np.float32) / grid_scale
            episode_steps += 1
            env.render()

        reached = np.all(env.agent_pos == env.agent_goal, axis=1)
        reached_agents = int(reached.sum())

        total_reached_agents += reached_agents
        total_steps += episode_steps
        if reached_agents == args.n_agents:
            success_count += 1

        progress_bar.set_postfix(
            success=f"{success_count}/{episode_idx + 1}",
            avg_reached=f"{total_reached_agents / (episode_idx + 1):.2f}",
            avg_steps=f"{total_steps / (episode_idx + 1):.2f}",
        )

    progress_bar.close()
    env.close()

    success_rate = success_count / n_samples
    avg_reached_agents = total_reached_agents / n_samples
    avg_steps = total_steps / n_samples

    print(f"\nWeights      : {weights_path}")
    print(f"Success rate : {success_rate:.2%} ({success_count}/{n_samples})")
    print(f"Avg reached  : {avg_reached_agents:.2f}/{args.n_agents}")
    print(f"Avg steps    : {avg_steps:.2f}")
