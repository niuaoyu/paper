import torch
import torch.nn as nn
import math
from .pointnet_util import index_points, square_distance

# Do pairwise cross-attention between Q, K, V
class CrossAttentionModule(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()

        self.fc1 = nn.Linear(d_model, d_model)
        self.fc2 = nn.Linear(d_model, d_model)
        self.fc3 = nn.Linear(d_model, d_model)

        self.fc_delta = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.fc_gamma = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(), 
            nn.Linear(d_model, d_model)
        )

        self.w_qs = nn.Linear(d_model, d_model, bias=False)
        self.w_ks = nn.Linear(d_model, d_model, bias=False)
        self.w_vs = nn.Linear(d_model, d_model, bias=False)
        self.out = nn.Linear(d_model, d_model)
        self.scale = 1.0 / math.sqrt(d_model)

    # def forward(
    #     self,
    #     query: torch.Tensor,                 # (B, N_q, d_model)
    #     key: torch.Tensor,                   # (B, N_k, d_model)
    #     value: torch.Tensor,                 # (B, N_k, d_model)
    # ) -> torch.Tensor:
    #     pre = query  # (B, N_q, d_model)
    #     # Compute distance between query and key points
    #     dists_qk = square_distance(query, key)  # (B, N_q, N_k)
    #     qk_knn_idx = dists_qk.argsort()[:, :, :self.neighbor]  # (B, N_q, k)
    #     qk_knn_feature = index_points(key, qk_knn_idx)  # (B, N_q, k, d_model)

    #     q, k, v = self.w_qs(self.fc1(query)), index_points(self.w_ks(self.fc2(key)), qk_knn_idx), index_points(self.w_vs(self.fc3(value)), qk_knn_idx)  # (B, N_q, d_model), (B, N_q, k, d_model), (B, N_q, k, d_model)

    #     pos_enc = self.fc_delta(query[:, :, None] - qk_knn_feature)  # (B, N_q, k, d_model)

    #     attn = self.fc_gamma(q[:, :, None] - k + pos_enc)  # (B, N_q, k, d_model)
    #     attn = torch.softmax(attn * self.scale, dim=-2)  # (B, N_q, k, d_model)

    #     res = torch.einsum('bmnf,bmnf->bmf', attn, v + pos_enc)  # (B, N_q, d_model)
    #     out = self.out(res) + pre  # (B, N_q, d_model)

    #     return out # (B, N_q, d_model)
    
    def forward(
        self,
        query: torch.Tensor,                 # (B, N_q, d_model)
        key: torch.Tensor,                   # (B, N_k, d_model)
        value: torch.Tensor,                 # (B, N_k, d_model)
    ) -> torch.Tensor:
        pre = query  # (B, N_q, d_model)

        q, k, v = self.w_qs(self.fc1(query)), self.w_ks(self.fc2(key)), self.w_vs(self.fc3(value))  # (B, N_q, d_model), (B, N_k, d_model), (B, N_k, d_model)

        pos_enc = self.fc_delta(query[:, :, None] -  key[:, None, :])  # (B, N_q, N_k, d_model)

        attn = self.fc_gamma(q[:, :, None] - k[:, None, :] + pos_enc)  # (B, N_q, N_k, d_model)
        attn = torch.softmax(attn * self.scale, dim=-2)  # (B, N_q, N_k, d_model)

        res = torch.einsum('bmnf,bmnf->bmf', attn, v[:, None, :, :] + pos_enc)  # (B, N_q, d_model)
        out = self.out(res) + pre  # (B, N_q, d_model)

        return out # (B, N_q, d_model)
