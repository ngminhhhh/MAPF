import numpy as np

class ReplayBuffer:
    def __init__(self, capacity, n_agents: int, obs_size: int, obs_channels: int, goal_dim: int = 2):
        self.capacity = capacity
        self.N = n_agents
        self.goal_dim = goal_dim

        state_shape = (n_agents, obs_size, obs_size, obs_channels)

        # Rollout data
        self.states    = np.zeros((capacity, *state_shape), dtype=np.float32)   # (T, N, H, W, C)
        self.goal_vecs = np.zeros((capacity, n_agents, goal_dim), dtype=np.float32)  # (T, N, 2)
        self.actions   = np.zeros((capacity, n_agents), dtype=np.int64)         # (T, N)
        self.rewards   = np.zeros((capacity, n_agents), dtype=np.float32)       # (T, N)
        self.dones     = np.zeros((capacity, n_agents), dtype=np.float32)       # (T, N)

        # PPO specific
        self.log_probs = np.zeros((capacity, n_agents), dtype=np.float32)       # (T, N)
        self.values    = np.zeros((capacity, n_agents), dtype=np.float32)       # (T, N)

        # Communication graph per timestep
        # edges[t] has shape (E_t, 2), E_t can vary
        self.edges = [None] * capacity

        # Controller
        self.ptr = 0
        self.size = 0

    def push(self, state, goal_vec, action, reward, done, log_prob, value, edges):
        """
        state    : (N, H, W, C)
        goal_vec : (N, 2)
        action   : (N,)
        reward   : (N,)
        done     : (N,)
        log_prob : (N,)
        value    : (N,)
        edges    : (E, 2) edge list of current timestep
        """

        self.states[self.ptr]    = state
        self.goal_vecs[self.ptr] = goal_vec
        self.actions[self.ptr]   = action
        self.rewards[self.ptr]   = reward
        self.dones[self.ptr]     = done
        self.log_probs[self.ptr] = log_prob
        self.values[self.ptr]    = value
        self.edges[self.ptr]     = edges.copy() if edges is not None else None

        self.ptr += 1
        self.size = min(self.size + 1, self.capacity)

    def reset(self):
        self.states.fill(0)
        self.goal_vecs.fill(0)
        self.actions.fill(0)
        self.rewards.fill(0)
        self.dones.fill(0)
        self.log_probs.fill(0)
        self.values.fill(0)

        self.edges = [None] * self.capacity

        self.ptr = 0
        self.size = 0

    def is_full(self):
        return self.ptr == self.capacity