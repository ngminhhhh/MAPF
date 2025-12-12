from generator.Generator import *
from enviroments.MAPF_env import *
from model.architecture_1 import *
from utils.visualizer import *
from utils.metrics import *

from pathlib import Path
import torch
from torch.distributions import Categorical

def load_instances(instance_dir: str) -> list[MAPFInstance]:
    instance_dir = Path(instance_dir)
    instances: list[MAPFInstance] = []

    for json_path in sorted(instance_dir.glob("instance_*.json")):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        grid = np.array(data["grid"], dtype=np.int8)
        starts = np.array(data["starts"], dtype=np.int32)
        goals = np.array(data["goals"], dtype=np.int32)

        instances.append(MAPFInstance(grid=grid, starts=starts, goals=goals))

    return instances

def instance_to_dict(inst: MAPFInstance) -> dict:
    H, W = inst.grid.shape
    return {
        "height": H,
        "width": W,
        "grid": inst.grid,              
        "starts": inst.starts.tolist(),
        "goals": inst.goals.tolist(),
    }

if __name__ == "__main__":
    # * Load instances
    instance_path = "./data/10x10-10/test"
    instances = load_instances(instance_dir=instance_path)

    width = 10
    height = 10
    n_agents = 10
    obstacle_ratio = 0.2
    n_instances = 100

    max_step = 20

    visualize_id = 10

    cfg = MAPFGeneratorConfig(
        width=width,
        height=height,
        n_agents=n_agents,
        obstacle_ratio=obstacle_ratio,
        n_samples=n_instances,  
    )

    # * Model init
    params_path = "mapf_solver.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

    checkpoint = torch.load(params_path, map_location=device)
    solver.load_state_dict(checkpoint["model_state_dict"])
    solver.to(device)
    solver.eval()

    # * Environment
    env = MAPFEnv(instance_cfg=cfg, instances=instances, reward_fn=reward_fn, obs_radius=1, max_steps=max_step)

    total_success_rate = 0.0

    for i in range(n_instances):
        obs, info = env.reset()
        A_star_paths = info["Astar_paths"]

        done = False
        solver.h_prev = None

        total_reward = 0.0
        step = 0

        # * Init path
        paths = [[] for _ in range(n_agents)]
        for agent_id in range(n_agents):
            r, c = env.agent_pos[agent_id]
            paths[agent_id].append((int(r), int(c)))

        while not done:
            logits, values = solver(obs, A_star_paths) 
            logits = logits.squeeze(0)
            values = values.squeeze(0)

            dist = Categorical(logits=logits)
            actions_tensor = dist.sample()

            actions = actions_tensor.cpu().numpy().astype(np.int64)

            obs, reward, terminated, truncated, info = env.step(actions)
            A_star_paths = info["Astar_paths"]

            for a in range(n_agents):
                r, c = env.agent_pos[a]
                paths[a].append((int(r), int(c)))

            done = terminated or truncated
            total_reward += reward
            step += 1

        instance_dict = instance_to_dict(instances[i])
        solution = {"paths": paths} 

        if i == visualize_id:
            visualize(instance_dict, solution)

        success = 0
        for a in range(n_agents):
            last_r, last_c = paths[a][-1]
            goal_r, goal_c = env.agent_goal[a]
            if last_r == goal_r and last_c == goal_c:
                success += 1

        success_rate = success / n_agents
        total_success_rate += success_rate

        print(f"Instance {i}'s reward={total_reward:.2f}, n_steps={step}, success_rate={success_rate}")
        
    print(f"AVG success rate: {total_success_rate / n_instances :.2f}")