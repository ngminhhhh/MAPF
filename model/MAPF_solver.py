from pathlib import Path

import torch 
import torch.nn as nn

import torch.nn.functional as F
import yaml
from torch.distributions import Categorical


MODEL_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "model.yaml"


def load_model_config(config_path=MODEL_CONFIG_PATH):
    with Path(config_path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class FeatureExtractor(nn.Module):
    def __init__(
        self,
        obs_channels: int,
        obs_size: int,
        hidden_dims: list[int],
        kernel_sizes: list[int],
        cnn_out_dim: int,
        goal_dim: int,
        gru_dim: int,
    ):

        super().__init__()

        backbone = []
        in_dim = obs_channels

        for out_dim, kernel_size in zip(hidden_dims, kernel_sizes):
            backbone.append(nn.Conv2d(in_channels  = in_dim, 
                                      out_channels = out_dim, 
                                      kernel_size  = kernel_size, 
                                      padding      = kernel_size//2))
            backbone.append(nn.ReLU())
            in_dim = out_dim

        flatten_dim = obs_size * obs_size * hidden_dims[-1]

        self.cnn_extractor = nn.Sequential(*backbone, 
                                           nn.Flatten(start_dim=1), 
                                           nn.Linear(flatten_dim, cnn_out_dim, False), 
                                           nn.ReLU()
                                        )

        self.goal_encoder = nn.Sequential(nn.Linear(in_features=2, out_features=8),
                                          nn.ReLU(),
                                          nn.Linear(in_features=8, out_features=goal_dim),
                                          nn.ReLU()
                                        )

        self.gru_extractor = nn.GRU(input_size   = cnn_out_dim+goal_dim, 
                                    hidden_size  = gru_dim, 
                                    num_layers   = 1,
                                    batch_first  = True
                                    )

    def forward(self, obs, goal_vecs, h_prev):
        obs_feats = self.cnn_extractor(obs)
        goal_feats = self.goal_encoder(goal_vecs)

        feats = torch.cat([obs_feats, goal_feats], dim=-1)
        feats = feats.unsqueeze(1)
        feats, h_next = self.gru_extractor(feats, h_prev)

        feats = feats.squeeze(1)

        return feats, h_next # (N, gru_dim)

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
        if scores.numel() == 0: return scores 

        max_per_dst = torch.full((num_nodes, ), -torch.inf, device=scores.device, dtype=scores.dtype)
        max_per_dst.scatter_reduce_(0, dst, scores, reduce="amax", include_self=True)
        exp_scores = torch.exp(scores - max_per_dst[dst])

        exp_denom = torch.zeros((num_nodes,), device=scores.device, dtype=scores.dtype)
        exp_denom.index_add_(0, dst, exp_scores)

        alpha = exp_scores / (exp_denom[dst] + 1e-12)

        return alpha # (E, )

    def forward(self, x, edges):
        '''
            x: (N, in_dim)
            edges: (E, 2)
            return: (N, hidden_dim)
        '''
        device = x.device
        N = x.size(0)
        edges = torch.as_tensor(edges, device=device, dtype=torch.long)

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
            GATHead(in_dim, hidden_dim, hidden_dim)
            for _ in range(n_heads)
        ])

    def forward(self, x, edges):
        outs = [head(x, edges) for head in self.heads]
        return torch.cat(outs, dim=-1) # (N, out_dim * n_heads)

class PolicyHead(nn.Module):
    def __init__(self, in_dim, hidden_dims):
        super().__init__()

        backbone = []
        for hidden_dim in hidden_dims:
            backbone.append(nn.Linear(in_features=in_dim, out_features=hidden_dim, bias=False))
            backbone.append(nn.ReLU())
            in_dim = hidden_dim
        
        self.mlp = nn.Sequential(*backbone)
        self.actor_head = nn.Linear(in_features=hidden_dims[-1], out_features=5)
        self.critic_head = nn.Linear(in_features=hidden_dims[-1], out_features=1)

    def forward(self, x):
        x = self.mlp(x)
        logits = self.actor_head(x)
        value = self.critic_head(x)

        return logits, value

class Solver(nn.Module):
    def __init__(self, obs_size: int, device, config_path=MODEL_CONFIG_PATH):
        super().__init__()
        self.obs_size = obs_size
        self.device = device
        cfg = load_model_config(config_path)

        self.obs_channels = cfg["obs_channels"]
        
        extractor_cfg = cfg["feature_extractor"]
        aggregator_cfg = cfg["graph_attention"]
        policy_cfg = cfg["policy_head"]

        self.cnn_hidden_dims = extractor_cfg["cnn_hidden_dims"]
        self.kernel_sizes = extractor_cfg["kernel_sizes"]
        self.cnn_out_dim = extractor_cfg["cnn_out_dim"]
        self.goal_out_dim = extractor_cfg["goal_out_dim"]
        self.gru_dim = extractor_cfg["gru_dim"]

        self.gat_hidden_dim = aggregator_cfg["hidden_dim"]
        self.n_gat_heads = aggregator_cfg["n_heads"]

        self.mlp_hidden_dim = policy_cfg["hidden_dims"]

        # * Model architecture
        self.extractor = FeatureExtractor(obs_channels=self.obs_channels,
                                          obs_size=obs_size,
                                          hidden_dims=self.cnn_hidden_dims,
                                          kernel_sizes=self.kernel_sizes,
                                          cnn_out_dim=self.cnn_out_dim,
                                          goal_dim=self.goal_out_dim,
                                          gru_dim=self.gru_dim)

        self.aggregator = FeatureAggregator(in_dim=self.gru_dim, 
                                            hidden_dim=self.gat_hidden_dim, 
                                            n_heads=self.n_gat_heads)

        self.head = PolicyHead(hidden_dims=self.mlp_hidden_dim,
                               in_dim=self.gat_hidden_dim*self.n_gat_heads)

    def init_hidden_state(self, n_agents: int):
        return torch.zeros(1, n_agents, self.gru_dim, device=self.device)

    def forward(self, obs, goal_vecs, edges, h_prev):
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        goal_vecs = torch.as_tensor(goal_vecs, dtype=torch.float32, device=self.device)
        obs = obs.permute(0, 3, 1, 2).contiguous()

        feats, h_next = self.extractor(obs, goal_vecs, h_prev)
        feats = self.aggregator(feats, edges)
        logits, value = self.head(feats)

        return logits, value, h_next

    @torch.no_grad()
    def act(self, obs, goal_vecs, edges, h_prev, deterministic=False):
        logits, values, h_next = self.forward(obs, goal_vecs, edges, h_prev)

        dist = Categorical(logits=logits)

        if deterministic:
            actions = torch.argmax(logits, dim=-1)
        else:
            actions = dist.sample()

        values = values.squeeze(-1) # (N, )

        actions = actions.cpu().numpy()

        return actions, values, h_next
