import numpy as np
import torch

class ReplayBuffer:
    def __init__(self, capacity, n_agents:int, obs_size:int, obs_channels:int):
        self.capacity = capacity
        self.N = n_agents
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        state_shape = (n_agents, obs_size, obs_size, obs_channels)

        # Transitions
        self.states      = np.zeros((capacity, *state_shape), dtype=np.float32)    # s_{t}   : (T, N, H, W, C)
        self.actions     = np.zeros((capacity, n_agents),     dtype=np.int64)      # a_{t}   : (T, N)
        self.rewards     = np.zeros((capacity, n_agents),     dtype=np.float32)    # r_{t}   : (T, N)
        self.next_states = np.zeros((capacity, *state_shape), dtype=np.float32)    # s_{t+1} : (T, N, H, W, C)
        self.dones       = np.zeros((capacity, n_agents),     dtype=bool)          # (T, N)
    
        # Controller
        self.ptr  = 0
        self.size = 0   

    def push(self, state, action, reward, next_state, done):
        self.states[self.ptr]      = state
        self.actions[self.ptr]     = action
        self.rewards[self.ptr]     = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr]       = done

        self.ptr                   = (self.ptr + 1) % self.capacity
        self.size                 += 1

    def reset(self):
        self.states.fill(0)
        self.actions.fill(0)
        self.rewards.fill(0)
        self.next_states.fill(0)
        self.dones.fill(0)

        self.size = 0
        self.ptr = 0

    def is_full(self):
        return self.size == self.capacity
        