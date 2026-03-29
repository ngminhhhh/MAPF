from utils.replayBuffer import ReplayBuffer

import torch
import torch.nn as nn
from torch.distributions import Categorical

device = "cuda" if torch.cuda.is_available() else "cpu"

def compute_gae(
    model: nn.Module,
    buffer: ReplayBuffer,
    values: torch.Tensor,
    next_state=None,
    next_goal_vec=None,
    next_edges=None,
    gamma: float = 0.99,
    lam: float = 0.95
):
    T, N = buffer.size, buffer.N

    rewards = torch.as_tensor(buffer.rewards[:T], dtype=torch.float32, device=device)   # (T, N)
    dones   = torch.as_tensor(buffer.dones[:T],   dtype=torch.float32, device=device)   # (T, N)

    # Normalize rewards
    rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

    advantages = torch.zeros((T, N), dtype=torch.float32, device=device)

    # Bootstrap value V(s_T)
    if next_state is not None and next_goal_vec is not None and next_edges is not None:
        next_state_t    = torch.as_tensor(next_state,    dtype=torch.float32, device=device)
        next_goal_vec_t = torch.as_tensor(next_goal_vec, dtype=torch.float32, device=device)
        next_edges_t    = torch.as_tensor(next_edges,    dtype=torch.float32, device=device)

        with torch.no_grad():
            _, next_values = model(next_state_t, next_goal_vec_t, next_edges_t)   # (N, 1)
            next_values = next_values.squeeze(-1)                                  # (N,)
    else:
        next_values = torch.zeros((N,), dtype=torch.float32, device=device)

    gae = torch.zeros((N,), dtype=torch.float32, device=device)

    for t in reversed(range(T)):
        if t == T - 1:
            next_v = next_values
        else:
            next_v = values[t + 1]

        delta = rewards[t] + gamma * next_v * (1.0 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1.0 - dones[t]) * gae
        advantages[t] = gae

    returns = advantages + values
    return advantages, returns

def A2C_loss(
    model: nn.Module,
    buffer: ReplayBuffer,
    next_state=None,
    next_goal_vec=None,
    next_edges=None,
    critic_weight: float = 0.5,
    entropy_weight: float = 0.01,
    gamma: float = 0.99
):
    T = buffer.size
    N = buffer.N

    values    = torch.zeros((T, N), dtype=torch.float32, device=device)
    log_probs = torch.zeros((T, N), dtype=torch.float32, device=device)
    entropies = torch.zeros((T, N), dtype=torch.float32, device=device)

    for t in range(T):
        actions   = torch.as_tensor(buffer.actions[t],   dtype=torch.int64,   device=device)

        logits, critic_values = model(buffer.states[t], buffer.goal_vecs[t], buffer.edges[t])
        dist = Categorical(logits=logits)

        values[t]    = critic_values.squeeze(-1)
        log_probs[t] = dist.log_prob(actions)
        entropies[t] = dist.entropy()

    A, R = compute_gae(
        model=model,
        buffer=buffer,
        values=values,
        next_state=next_state,
        next_goal_vec=next_goal_vec,
        next_edges=next_edges,
        gamma=gamma
    )

    # Normalize advantage
    A = (A - A.mean()) / (A.std() + 1e-8)

    # Actor loss
    L_actor = -(log_probs * A.detach()).mean()

    # Critic loss
    L_critic = critic_weight * ((values - R.detach()) ** 2).mean()

    # Entropy bonus
    L_entropy = -entropy_weight * entropies.mean()

    return L_actor + L_critic + L_entropy