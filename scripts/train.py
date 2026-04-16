import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import torch
import torch.nn as nn

from env.MAPF_env import MAPFEnv
from env.MAPF_type import MAPFEnvConfig, MAPFGeneratorConfig
from model.MAPF_solver import Solver
from utils.EECBS import eecbs
from utils.generator import MAPFGenerator


def build_env(
    *,
    grid_height: int,
    grid_width: int,
    n_agents: int,
    obstacle_ratio: float,
    obs_radius: int,
    n_samples: int,
    seed: int,
) -> MAPFEnv:
    generator_cfg = MAPFGeneratorConfig(
        width=grid_width,
        height=grid_height,
        n_agents=n_agents,
        obstacle_ratio=obstacle_ratio,
        n_samples=n_samples,
        seed=seed,
    )
    env_cfg = MAPFEnvConfig(
        width=grid_width,
        height=grid_height,
        n_agents=n_agents,
        obs_radius=obs_radius,
        max_steps=grid_height * grid_width,
    )
    return MAPFEnv(env_cfg, MAPFGenerator(generator_cfg), None)


if __name__ == "__main__":
    seed = 4096
    grid_height = 10
    grid_width = 10
    obstacle_ratio = 0.0
    n_agents = 4
    obs_radius = 3

    n_epochs = 20
    n_samples = 32
    learning_rate = 1e-3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    solver_dir = ROOT_DIR / "EECBS"
    checkpoint_dir = ROOT_DIR / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    torch.manual_seed(seed)
    np.random.seed(seed)

    env = build_env(
        grid_height=grid_height,
        grid_width=grid_width,
        n_agents=n_agents,
        obstacle_ratio=obstacle_ratio,
        obs_radius=obs_radius,
        n_samples=n_samples,
        seed=seed,
    )

    model = Solver(
        obs_size=2 * obs_radius + 1,
        obs_channels=env.observation_space.shape[-1],
        device=device,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    grid_scale = np.array([grid_height, grid_width], dtype=np.float32)

    for epoch in range(1, n_epochs + 1):
        model.train()
        epoch_loss = 0.0
        updated_instances = 0

        for instance_id in range(n_samples):
            obs, info = env.reset()
            expert_demo = eecbs(
                env.grid,
                env.agent_pos,
                env.agent_goal,
                instance_id=f"epoch{epoch}_instance{instance_id}",
                solver_dir=solver_dir,
            )

            if expert_demo.expert_actions.shape[0] == 0:
                continue

            model.init_hidden_state(n_agents)
            optimizer.zero_grad()
            total_loss = torch.zeros((), device=device)

            for expert_actions in expert_demo.expert_actions:
                goal_vecs = (env.agent_goal - env.agent_pos).astype(np.float32) / grid_scale
                logits, _ = model(obs, goal_vecs, info["comm_edges"])
                targets = torch.as_tensor(expert_actions, dtype=torch.long, device=device)
                total_loss = total_loss + criterion(logits, targets)
                obs, _, _, _, info = env.step(expert_actions)

            loss = total_loss / expert_demo.expert_actions.shape[0]
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            updated_instances += 1

        mean_loss = epoch_loss / updated_instances if updated_instances > 0 else 0.0
        print(f"epoch={epoch:03d} loss={mean_loss:.4f} updates={updated_instances}")

    torch.save(model.state_dict(), checkpoint_dir / "solver_il.pt")
    env.close()
