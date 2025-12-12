# 09/12/2025
import torch 
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from generator.Generator import *
from enviroments.MAPF_env import *
import numpy as np

def reward_fn(env:MAPFEnv, old_agent_pos: np.ndarray, new_agent_pos: np.ndarray, done:bool):
    goals = env.agent_goal  

    d_old = np.abs(old_agent_pos - goals).sum(axis=1)  
    d_new = np.abs(new_agent_pos - goals).sum(axis=1)

    per_agent = (d_old - d_new).astype(np.float32)

    per_agent -= 0.01

    at_goal = np.all(new_agent_pos == goals, axis=1)
    per_agent[at_goal] += 1.0

    global_reward = float(per_agent.mean())

    return global_reward, per_agent

class CNNBlock(nn.Module):
    def __init__(self, in_dim:int=3, out_dim: int=64, kernel_size:int=3, activation:type[nn.Module]=nn.ReLU):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=kernel_size, padding=kernel_size//2),
            nn.BatchNorm2d(out_dim),
            activation(),

            nn.Conv2d(out_dim, out_dim, kernel_size=kernel_size, padding=kernel_size//2),
            nn.BatchNorm2d(out_dim),
            activation(),
        )
    def forward(self, x):
        return self.net(x)

class AgentFeatureExtractor(nn.Module):
    def __init__(self, obs_channels:int=3, obs_size:int=3, cnn_out_dim:int=64, n_blocks:int=3, gru_out_dim:int=64):
        super().__init__()
        # * Build CNN extractor
        backbone = []

        in_dim = obs_channels
        for _ in range(n_blocks):
            backbone.append(CNNBlock(in_dim=in_dim, out_dim=cnn_out_dim, kernel_size=obs_size))
            in_dim = cnn_out_dim

        self.cnn_extractor = nn.Sequential(*backbone)

        # * Build GRU extractor
        flatten_dim = cnn_out_dim * obs_size * obs_size

        self.gru_extractor = nn.GRU(
            input_size=flatten_dim, 
            hidden_size=gru_out_dim,
            batch_first=True
        )

    def forward(self, obs, h_prev=None):
        if obs.ndim == 4:
            obs = obs.unsqueeze(0)

        B, N, H, W, C = obs.shape

        # * CNN
        x = obs.permute(0, 1, 4, 2, 3).reshape(B * N, C, H, W)
        x = self.cnn_extractor(x)
        x = x.reshape(B * N, -1).unsqueeze(1)

        # * GRU
        if h_prev is not None:
            h_out, h_new = self.gru_extractor(x, h_prev)
        else:
            h_out, h_new = self.gru_extractor(x)

        feats = h_out[:, -1, :].reshape(B, N, -1)  # (B, N, gru_out_dim)

        return feats, h_new

class GATLayer(nn.Module):
    def __init__(self, in_dim:int, out_dim:int, epsilon:float=0.2):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.a = nn.Linear(2 * out_dim, 1, bias=False)
        self.leaky_relu = nn.LeakyReLU(epsilon)

    def forward(self, x:torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape

        h = self.W(x)
        h_i = h.unsqueeze(2).expand(B, N, N, -1)
        h_j = h.unsqueeze(1).expand(B, N, N, -1)

        a_input = torch.cat([h_i, h_j], dim=-1)
        e = self.leaky_relu(self.a(a_input).squeeze(-1))
        e = e.masked_fill(adj == 0, float("-inf"))
        alpha = torch.softmax(e, dim=-1)

        out = torch.bmm(alpha, h)

        return out

class GAT(nn.Module):
    def __init__(self, feat_dim: int, hidden_dim: int = 128, n_layers: int = 2):
        super().__init__()

        layers = []
        in_dim = feat_dim
        for _ in range(n_layers):
            layers.append(GATLayer(in_dim=in_dim, out_dim=hidden_dim))
            in_dim = hidden_dim

        self.layers = nn.ModuleList(layers)

    def forward(self, feats: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
            feats: (B, N, F_in)
            adj:   (B, N, N)
            return: (B, N, hidden_dim)
        """
        x = feats
        for layer in self.layers:
            x = layer(x, adj)
            x = F.relu(x)
        return x


class MAPFSolver(nn.Module):
    def __init__(self, cfg:MAPFGeneratorConfig, device, obs_channels:int, obs_size:int, cnn_out_dim:int, n_cnn_blocks:int, gru_out_dim:int, gat_hidden_dim:int, n_gat_layers:int, forecast_k:int):
        super().__init__()
        
        self.cfg = cfg
        
        self.n_agents = self.cfg.n_agents
        self.graph = [[] for _ in range(self.n_agents)]
        self.adj = np.zeros((self.n_agents, self.n_agents), dtype=np.float32)
        self.forecast_k = forecast_k
    
        self.device = device

        self.extractor = AgentFeatureExtractor(obs_channels=obs_channels, obs_size=obs_size, cnn_out_dim=cnn_out_dim, n_blocks=n_cnn_blocks, gru_out_dim=gru_out_dim)
        self.aggregator = GAT(feat_dim=gru_out_dim, hidden_dim=gat_hidden_dim, n_layers=n_gat_layers)

        self.actor_head = nn.Linear(gat_hidden_dim, 5)
        self.critic_head = nn.Linear(gat_hidden_dim, 1)

        self.h_prev = None

        self.to(device)

    def _get_pos(self, path: list[Tuple[int, int]], t):
        if t >= len(path):
            return path[-1]
        return path[t]

    def build_conflict_graph(self, paths:list[list[Tuple[int, int]]], forecast_k:int):
        self.graph = [[] for _ in range(self.n_agents)] # * Reset 
        self.adj.fill(0.0)

        for i in range(self.n_agents):
            for j in range(self.n_agents):
                if i == j: continue
                
                for t in range(forecast_k):
                    if self._get_pos(paths[i], t) == self._get_pos(paths[j], t):
                        self.graph[i].append(j)
                        self.adj[i, j] = 1.0
                        break

        np.fill_diagonal(self.adj, 1.0)

    def forward(self, obs, paths):
        self.build_conflict_graph(paths, self.forecast_k)

        adj_tensor = torch.from_numpy(self.adj).float().unsqueeze(0).to(self.device)
        obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(self.device) 

        feats, h_new = self.extractor(obs_tensor, self.h_prev)
        feats = self.aggregator(feats, adj_tensor)

        self.h_prev = h_new

        logits = self.actor_head(feats)  
        values = self.critic_head(feats).squeeze(-1)

        return logits, values