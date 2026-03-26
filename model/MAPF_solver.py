import torch 
import torch.nn as nn

import torch.nn.functional as F
from torch.distributions import Categorical

import numpy as np
from env.MAPF_env import MAPFEnv

class CNNBlock(nn.Module):
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
            backbone.append(CNNBlock(input_dim=in_dim, output_dim=hidden_dim, kernel_size=kernel_size))
            in_dim = hidden_dim

        self.extractor = nn.Sequential(*backbone, nn.Flatten(start_dim=1))

    def forward(self, x):
        return self.extractor(x)

class GATHead(nn.Module):
    def __init__(self, in_dim:int, hidden_dim:int, out_dim:int):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        # linear projection
        self.msg_proj = nn.Linear(in_features=in_dim, out_features=hidden_dim, bias=False)
        self.self_proj = nn.Linear(in_features=in_dim, out_features=hidden_dim, bias=False)
        
        # attention head
        self.attn = nn.Linear(in_features=2*hidden_dim, out_features=1, bias=False)

        # aggregate with attention score
        self.aggr = nn.Sequential(
            nn.Linear(2 * hidden_dim, out_dim),
            nn.ReLU()
        )
    
    @staticmethod
    def _edge_softmax(scores, dst, num_nodes: int):
        """
            scores: (E,)
            dst:    (E,)
            return: alpha (E,), softmax edges with same dst
        """


    def forward(self, x, edges):
        '''
            x: (N, in_dim)
            edges: (E, 2)
            return: (N, hidden_dim)
        '''
        device = x.device
        N = x.size(0)
        edges = torch.from_numpy(edges).to(device=device, dtype=torch.long) # convert to torch

        src = edges[:, 0] # (E, )
        dst = edges[:, 1] # (E, )

        # * Project features
        msg_feats = self.msg_proj(x)
        self_feats = self.self_proj(x)

        # * Gather edges features
        msg_src = msg_feats[src] # (E, H)
        msg_dst = msg_feats[dst] # (E, H)

        # * Attention logits for each edge
        attn_input = torch.cat([msg_src, msg_dst], dim=-1) # (E, 2H)
        e = self.attn(attn_input).squeeze(-1) # (E,)
        e = F.leaky_relu(e, negative_slope=0.2) # negative_slope = 0.2 base on GAT paper

        # * Softmax
        alpha = self._edge_softmax(e, dst, N) # (E,)

        # * Aggregate information of neighbor
        weighted_msg = msg_src * alpha.unsqueeze(-1) # (E, H)
        neigh_feats = torch.zeros((N, self.hidden_dim), device=device)
        neigh_feats.index_add_(0, dst, weighted_msg)

        # * Aggregate self + neighbor
        out = self.aggr(torch.cat([self_feats, neigh_feats], dim=-1))

        return out


class FeatureAggregator(nn.Module):
    def __init__(self, in_dim:int, hidden_dim:int, n_heads:int):
        super().__init__()
        self.heads = nn.ModuleList([
            GATHead(in_dim, hidden_dim)
            for _ in range(n_heads)
        ])

    def forward(self, x, edges):
        outs = [head(x, edges) for head in self.heads]
        return torch.cat(outs, dim=-1) # (N, hidden_dim * n_heads)

class Solver(nn.Module):
    def __init__(self, obs_channels: int, 
                 cnn_hidden_dim: int, cnn_n_blocks:int, kernel_size: int):
        super().__init__()

        self.feature_extractor = FeatureExtraction(obs_channels, cnn_hidden_dim, cnn_n_blocks, kernel_size)

    def forward(self, x):
        feats = self.feature_extractor(x)

        return feats
