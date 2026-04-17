import torch
import sys
import argparse
from pathlib import Path
import yaml
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import copy

from concurrent.futures import ProcessPoolExecutor

# Path
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from env.MAPF_env import MAPFEnv
from env.MAPF_type import MAPFEnvConfig, MAPFGeneratorConfig
from utils.generator import MAPFGenerator
from model.MAPF_solver import Solver
from utils.EECBS import eecbs

TRAIN_CFG = ROOT_DIR / "configs" / "train.yaml"
MODEL_CFG = ROOT_DIR / "configs" / "model.yaml"
EECBS_CFG = ROOT_DIR / "configs" / "eecbs.yaml"
weight_dir = ROOT_DIR / "weights"

with open(EECBS_CFG, "r", encoding="utf-8") as f:
    eecbs_cfg = yaml.safe_load(f)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid_size", type=int, required=True)
    parser.add_argument("--obs_radius", type=int, required=True)
    parser.add_argument("--n_agents", type=int, required=True)
    parser.add_argument("--obstacle_ratio", type=float, required=True)
    parser.add_argument("--resume", type=str, default=None)

    args = parser.parse_args()

    return args.grid_size, args.obs_radius, args.n_agents, args.obstacle_ratio, args.resume

def get_expert_result(grid, agent_pos, agent_goal, instance_id):
    return eecbs(grid=grid, agent_pos=agent_pos, agent_goal=agent_goal,
                 instance_id=instance_id, cutoff_time=eecbs_cfg["cutoff_time"], suboptimality=eecbs_cfg["suboptimality"],
                 screen=eecbs_cfg["screen"], keep_files=False)

def generate_ep(generator_cfg, id):
    local_cfg = copy.deepcopy(generator_cfg)
    local_cfg.seed = generator_cfg.seed + id

    generator = MAPFGenerator(local_cfg)
    instance = generator.generate_instance()

    expert_res = get_expert_result(instance.grid, instance.starts, instance.goals, id)
    expert_actions = np.asarray(expert_res.expert_actions, dtype=np.int64)  # (T, n_agents)

    episode = {
        "grid": instance.grid,
        "agent_start": instance.starts,
        "agent_goal": instance.goals,
        "actions": expert_actions,
        "instances_idx": id,
    }

    return episode

def replay_ep(instance_id, env_cfg, generator_cfg, grid_scale):
    env = None

    try:
        ep = generate_ep(generator_cfg, instance_id)
        generator = MAPFGenerator(generator_cfg)
        env = MAPFEnv(env_cfg, generator)

        obs, info = env.reset(ep=ep)
        edges = info["comm_edges"]
        goal_vecs = (env.agent_goal - env.agent_pos) / grid_scale

        traj = {
            "obs": [],
            "goal_vecs": [],
            "edges": [],
            "actions": [],
        }

        actions = np.asarray(ep["actions"], dtype=np.int64)

        for t in range(actions.shape[0]):
            traj["obs"].append(np.array(obs, copy=True))
            traj["goal_vecs"].append(np.array(goal_vecs, copy=True))
            traj["edges"].append(np.array(edges, copy=True))
            traj["actions"].append(np.array(actions[t], copy=True))

            obs, _, _, _, info = env.step(actions[t])
            edges = info["comm_edges"]
            goal_vecs = (env.agent_goal - env.agent_pos) / grid_scale

        return {
            "success": True,
            "instance_id": instance_id,
            "traj": traj,
        }
    except Exception:
        return {
            "success": False,
            "instance_id": instance_id,
            "traj": None,
        }
    finally:
        if env is not None:
            env.close()

def compute_ep_loss(model, criterion, n_agents, traj):
    h_prev = model.init_hidden_state(n_agents=n_agents)
    ep_loss = 0.0

    T = len(traj["actions"])

    for t in range(T):
        obs = traj["obs"][t]
        goal_vecs = traj["goal_vecs"][t]
        edges = traj["edges"][t]
        target = torch.as_tensor(
            traj["actions"][t],
            dtype=torch.long,
            device=model.device
        )  # (N,)

        logits, value, h_next = model.forward(obs, goal_vecs, edges, h_prev)
        loss = criterion(logits, target)   # logits: (N, A), target: (N,)
        ep_loss = ep_loss + loss

        h_prev = h_next

    ep_loss = ep_loss / max(1, T)

    return ep_loss

def train_on_trajs(model, criterion, optimizer, n_agents, traj_batch):
    batch_loss = None
    for traj in traj_batch:
        ep_loss = compute_ep_loss(model, criterion, n_agents, traj)
        batch_loss = ep_loss if batch_loss is None else batch_loss + ep_loss

    batch_loss = batch_loss / len(traj_batch)

    optimizer.zero_grad()
    batch_loss.backward()
    optimizer.step()

    return batch_loss

def save_checkpoint(
    ckpt_path,
    model,
    optimizer,
    batch_idx,
    next_instance_idx,
):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "batch_idx": batch_idx,
        "next_instance_idx": next_instance_idx,
    }
    torch.save(checkpoint, ckpt_path)

def load_checkpoint(ckpt_path, model, optimizer, device):
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint

if __name__ == "__main__":
    # * Config
    grid_size, obs_radius, n_agents, obstacle_ratio, resume_path = parse_args()

    with open(TRAIN_CFG, "r", encoding="utf-8") as f:
        train_cfg = yaml.safe_load(f)
    
    train_seed = train_cfg["base_train_seed"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    weight_dir.mkdir(parents=True, exist_ok=True)

    # * Init environment & model
    env_cfg = MAPFEnvConfig(width=grid_size, height=grid_size, n_agents=n_agents, 
                               obs_radius=obs_radius, max_steps=grid_size*grid_size)

    generator_cfg = MAPFGeneratorConfig(width=grid_size, height=grid_size, n_agents=n_agents,
                                        obstacle_ratio=obstacle_ratio, seed=train_seed)
    

    model = Solver(obs_size=2*obs_radius + 1, device=device, config_path=MODEL_CFG).to(device)
    grid_scale = np.array([grid_size, grid_size])

    # * Loss & Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg["learning_rate"])

    batch_size = 8
    max_workers = 4
    save_point = 100

    n_train_samples = train_cfg["train_samples"]
    start_instance_idx = 0
    start_batch_idx = 0

    if resume_path is not None:
        checkpoint = load_checkpoint(resume_path, model, optimizer, device)
        start_instance_idx = checkpoint["next_instance_idx"]
        start_batch_idx = checkpoint["batch_idx"]
        print(
            f"Continue training from {resume_path} "
            f"(batch_idx={start_batch_idx}, next_instance_idx={start_instance_idx})"
        )

    model.train()
    batch_idx = start_batch_idx
    trained_samples = 0

    # * Main 
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        pbar = tqdm(
            total=n_train_samples,
            desc="Training",
            unit="episode",
        )

        while trained_samples < n_train_samples:
            success_trajs = []
            current_batch_size = min(batch_size, n_train_samples - trained_samples)

            while len(success_trajs) < current_batch_size:
                batch_attempts = min(max_workers, current_batch_size - len(success_trajs))
                batch_ids = list(range(start_instance_idx, start_instance_idx + batch_attempts))
                batch_results = list(
                    executor.map(
                        replay_ep,
                        batch_ids,
                        [env_cfg] * len(batch_ids),
                        [generator_cfg] * len(batch_ids),
                        [grid_scale] * len(batch_ids),
                    )
                )
                start_instance_idx += batch_attempts

                # Collect success trajectories
                for result in batch_results:
                    if result["success"]:
                        success_trajs.append(result["traj"])

            batch_loss = train_on_trajs(model, criterion, optimizer, n_agents, success_trajs)
            batch_idx += 1
            trained_samples += current_batch_size

            pbar.update(current_batch_size)
            pbar.set_postfix(loss=f"{batch_loss.item():.4f}")

            if batch_idx > 0 and batch_idx % save_point == 0:
                ckpt_path = weight_dir / f"checkpoint_batch_{batch_idx}.pt"
                save_checkpoint(
                    ckpt_path,
                    model,
                    optimizer,
                    batch_idx=batch_idx,
                    next_instance_idx=start_instance_idx,
                )
                print(f"Saved training checkpoint at batch {batch_idx} -> {ckpt_path}")

        final_ckpt_path = weight_dir / "weights.pt"
        save_checkpoint(
            final_ckpt_path,
            model,
            optimizer,
            batch_idx=batch_idx,
            next_instance_idx=start_instance_idx,
        )
        
        print(f"Saved final weights -> {final_ckpt_path}")
