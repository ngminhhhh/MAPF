import numpy as np
from env.MAPF_env import *

def reward_fn(env, old_pos, new_pos, *,
             goal_reward=5.0, progress_w=0.3, step_penalty=0.05, share_alpha=0.3):
    goals = env.agent_goal   # (N, 2)

    # Goal masks
    at_goal_old = np.all(old_pos == goals, axis=1)   # (N,)
    at_goal_new = np.all(new_pos == goals, axis=1)   # (N,)
    newly_reached_goal = (~at_goal_old) & at_goal_new

    # Manhattan progress
    old_dist = np.abs(old_pos - goals).sum(axis=1)   # (N,)
    new_dist = np.abs(new_pos - goals).sum(axis=1)   # (N,)
    delta = old_dist - new_dist                      # (N,)

    # Chỉ phạt step với agent chưa tới goal
    active = ~at_goal_new
    effective_step_penalty = step_penalty * active.astype(np.float32)

    # Self reward
    ind_rewards = (
        progress_w * delta.astype(np.float32)
        - effective_step_penalty
        + goal_reward * newly_reached_goal.astype(np.float32)
    ).astype(np.float32)

    # Aggregate rewards
    comm_graph = env.comm_graph
    n_agents = old_pos.shape[0]

    neigh_sum = np.zeros(n_agents, dtype=np.float32)
    neigh_count = np.ones(n_agents, dtype=np.float32)   # self-loop

    if comm_graph is not None and len(comm_graph) > 0:
        src = comm_graph[:, 0]
        dst = comm_graph[:, 1]

        np.add.at(neigh_sum, dst, ind_rewards[src])
        np.add.at(neigh_count, dst, 1.0)

    shared_rewards = (ind_rewards + neigh_sum) / neigh_count
    rewards = (1.0 - share_alpha) * ind_rewards + share_alpha * shared_rewards

    return rewards.astype(np.float32)