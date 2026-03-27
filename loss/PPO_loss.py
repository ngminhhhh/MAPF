import numpy as np
from utils.replayBuffer import ReplayBuffer

import numpy as np
import torch
from torch.distributions import Categorical

device = "cuda" if torch.cuda.is_available() else "cpu"

def compute_gae(model, buffer, next_state=None, next_edges=None, gamma=0.99, lam=0.95):
    T = buffer.size
    N = buffer.N

    rewards = buffer.rewards[:T]   # (T, N)
    values  = buffer.values[:T]    # (T, N)
    dones   = buffer.dones[:T]     # (T, N)

    advantages = np.zeros((T, N), dtype=np.float32)
    gae = np.zeros((N,), dtype=np.float32)

    if next_state is None or next_edges is None:
        next_value = np.zeros((N,), dtype=np.float32)
    else:
        _, _, next_value = model.act(next_state, next_edges, deterministic=True)
        next_value = next_value.detach().cpu().numpy().astype(np.float32)

        next_value = next_value * (1.0 - dones[-1])

    for t in reversed(range(T)):
        if t == T - 1:
            value_next = next_value
        else:
            value_next = values[t + 1]

        delta = rewards[t] + gamma * value_next * (1.0 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1.0 - dones[t]) * gae
        advantages[t] = gae

    returns = advantages + values
    return advantages, returns

def evaluate_actions(model, buffer):
    T = buffer.size

    all_log_probs = []
    all_values = []
    all_entropies = []

    for t in range(T):
        act_t = torch.as_tensor(buffer.actions[t], dtype=torch.long, device=device)      # (N,)

        logits_t, values_t = model.forward(buffer.states[t], buffer.edges[t])   # logits: (N, A), values: (N, 1)
        values_t = values_t.squeeze(-1)                     # (N,)

        dist_t = Categorical(logits=logits_t)

        log_probs_t = dist_t.log_prob(act_t)   # (N,)
        entropy_t = dist_t.entropy()           # (N,)

        all_log_probs.append(log_probs_t)
        all_values.append(values_t)
        all_entropies.append(entropy_t)

    new_log_probs = torch.stack(all_log_probs, dim=0)   # (T, N)
    new_values = torch.stack(all_values, dim=0)         # (T, N)
    entropies = torch.stack(all_entropies, dim=0)       # (T, N)

    return new_log_probs, new_values, entropies

def ppo_loss(model, buffer, advantages, returns, clip_eps=0.2, value_coef=0.5, entropy_coef=0.01):

    old_log_probs = torch.as_tensor(buffer.log_probs[:buffer.size], dtype=torch.float32, device=device)
    advantages = torch.as_tensor(advantages, dtype=torch.float32, device=device)
    returns = torch.as_tensor(returns, dtype=torch.float32, device=device)

    new_log_probs, new_values, entropies = evaluate_actions(model, buffer)

    # PPO ratio
    ratios = torch.exp(new_log_probs - old_log_probs)   # (T, N)

    # Clipped surrogate objective
    surr1 = ratios * advantages
    surr2 = torch.clamp(ratios, 1.0 - clip_eps, 1.0 + clip_eps) * advantages

    policy_loss = -torch.min(surr1, surr2).mean()

    # Critic loss
    value_loss = ((new_values - returns) ** 2).mean()

    # Entropy bonus
    entropy_bonus = entropies.mean()

    loss = policy_loss + value_coef * value_loss - entropy_coef * entropy_bonus

    return loss

def train_step(model, optimizer, buffer,
               next_state, next_edges,
               *, gamma=0.99, lam=0.95,
               clip_eps=0.2, value_coef=0.5, entropy_coef=0.01,
               n_epochs=4):
    
    advantages, returns = compute_gae(model=model, buffer=buffer, next_state=next_state, next_edges=next_edges, 
                                      gamma=gamma, lam=lam)
    
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    last_loss = None
    for _ in range(n_epochs):
        loss = ppo_loss(model=model, buffer=buffer, advantages=advantages, returns=returns,
                        clip_eps=clip_eps, value_coef=value_coef, entropy_coef=entropy_coef)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        last_loss = loss.item()

    return last_loss

