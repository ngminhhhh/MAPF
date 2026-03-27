import numpy as np
from env.MAPF_env import *

import numpy as np

def reward_fn(env, old_pos, new_pos,*, step_penalty=0.05, goal_reward=5.0, progress_w=0.3,
    stag_threshold=2, stag_lambda=0.02, share_alpha=0.3):
    goals = env.agent_goal   # (N, 2)

    # Manhattan progress
    old_dist = np.abs(old_pos - goals).sum(axis=1)   # (N,)
    new_dist = np.abs(new_pos - goals).sum(axis=1)   # (N,)
    delta = old_dist - new_dist                      # (N,)

    # Stagnation penalty
    no_progress = (delta <= 0)
    env.stagnation_count[no_progress] += 1
    env.stagnation_count[~no_progress] = 0

    overflow = np.maximum(env.stagnation_count - stag_threshold, 0)
    stag_penalty = stag_lambda * (overflow ** 2)

    # Goal reward
    reached_goal = np.all(new_pos == goals, axis=1).astype(np.float32)   # (N,)

    # self reward
    ind_rewards = (
        progress_w * delta.astype(np.float32)
        - step_penalty
        - stag_penalty.astype(np.float32)
        + goal_reward * reached_goal
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