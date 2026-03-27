import numpy as np

class ReplayBuffer:
    def __init__(self, capacity, n_agents: int, obs_size: int, obs_channels: int):
        self.capacity = capacity
        self.N = n_agents

        state_shape = (n_agents, obs_size, obs_size, obs_channels)

        # Rollout data
        self.states    = np.zeros((capacity, *state_shape), dtype=np.float32)  # (T, N, H, W, C)
        self.actions   = np.zeros((capacity, n_agents), dtype=np.int64)        # (T, N)
        self.rewards   = np.zeros((capacity, n_agents), dtype=np.float32)      # (T, N)
        self.dones     = np.zeros((capacity, n_agents), dtype=np.float32)      # (T, N)

        # PPO specific
        self.log_probs = np.zeros((capacity, n_agents), dtype=np.float32)      # (T, N)
        self.values    = np.zeros((capacity, n_agents), dtype=np.float32)      # (T, N)

        # Controller
        self.ptr = 0
        self.size = 0

    def push(self, state, action, reward, done, log_prob, value):
        """
            state    : (N, H, W, C)
            action   : (N,)
            reward   : (N,)
            done     : (N,)
            log_prob : (N,)
            value    : (N,)
        """
        self.states[self.ptr]    = state
        self.actions[self.ptr]   = action
        self.rewards[self.ptr]   = reward
        self.dones[self.ptr]     = done
        self.log_probs[self.ptr] = log_prob
        self.values[self.ptr]    = value

        self.ptr += 1
        self.size = min(self.size + 1, self.capacity)

    def reset(self):
        self.states.fill(0)
        self.actions.fill(0)
        self.rewards.fill(0)
        self.dones.fill(0)
        self.log_probs.fill(0)
        self.values.fill(0)

        self.ptr = 0
        self.size = 0

    def is_full(self):
        return self.ptr == self.capacity
        