import torch 
import torch.nn as nn

import torch.nn.functional as F
from torch.distributions import Categorical

import numpy as np
from env.MAPF_env import MAPFEnv

class Block(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size, padding=kernel_size//2),
            nn.ReLU(),
            nn.Conv2d(output_dim, output_dim, kernel_size, padding=kernel_size//2),
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.block(x)
    
class FeatureExtraction(nn.Module):
    def __init__(self, obs_channels: int, hidden_dim: int, n_blocks: int, kernel_size: int):
        super().__init__()

        backbone = []
        in_dim = obs_channels

        for _ in range(n_blocks):
            backbone.append(Block(input_dim=in_dim, output_dim=hidden_dim, kernel_size=kernel_size))
            in_dim = hidden_dim

        self.extractor = nn.Sequential(*backbone, nn.Flatten(start_dim=1))

    def forward(self, x):
        return self.extractor(x)
    
class Solver(nn.Module):
    def __init__(self, obs_channels: int, 
                 cnn_hidden_dim: int, cnn_n_blocks:int, kernel_size: int):
        super().__init__()

        self.feature_extractor = FeatureExtraction(obs_channels, cnn_hidden_dim, cnn_n_blocks, kernel_size)

    def forward(self, x):
        feats = self.feature_extractor(x)

        return feats
