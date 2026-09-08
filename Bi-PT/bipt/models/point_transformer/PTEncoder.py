import torch
import torch.nn as nn
from .pointnet_util import PointNetFeaturePropagation, PointNetSetAbstraction2, square_distance, index_points
from .transformer import TransformerBlock

class TransitionDown(nn.Module):
    def __init__(self, nneighbor, channels, norm_type: str = "bn"):
        super().__init__()
        self.sa = PointNetSetAbstraction2(0, nneighbor, channels[0], channels[1:], group_all=False, knn=True, norm_type=norm_type)

    def forward(self, xyz, points):
        return self.sa(xyz, points)
    
class Backbone(nn.Module):
    def __init__(self, nblocks, nneighbor, d_points, transformer_dim, norm_type: str = "bn"):
        super().__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(d_points, 32),
            nn.ReLU(),
            nn.Linear(32, 32)
        )
        self.transformer1 = TransformerBlock(32, transformer_dim, nneighbor)
        self.transition_downs = nn.ModuleList()
        self.transformers = nn.ModuleList()
        for i in range(nblocks):
            channel = 32 * 2 ** (i + 1)
            self.transition_downs.append(
                TransitionDown(nneighbor, [channel // 2 + 3, channel, channel], norm_type=norm_type))
            self.transformers.append(TransformerBlock(channel, transformer_dim, nneighbor))
        self.nblocks = nblocks

    def forward(self, x):
        xyz = x[..., :3]
        points = self.transformer1(xyz, self.fc1(x[..., 3:]))[0]

        xyz_and_feats = [(xyz, points)]
        for i in range(self.nblocks):
            xyz, points = self.transition_downs[i](xyz, points)
            points = self.transformers[i](xyz, points)[0]
            xyz_and_feats.append((xyz, points))
        return points, xyz_and_feats
    
class PTEncoder(nn.Module):
    def __init__(self, nblocks=4, nneighbor=16,  d_points=3, transformer_dim=512, norm_type: str = "bn"):
        super().__init__()
        self.backbone = Backbone(nblocks, nneighbor, d_points, transformer_dim, norm_type=norm_type)

        self.fc2 = nn.Sequential(
            nn.Linear(32 * 2 ** nblocks, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 32 * 2 ** nblocks)
        )
        self.transformer2 = TransformerBlock(32 * 2 ** nblocks, transformer_dim, nneighbor)
        self.nblocks = nblocks

    def forward(self, x):
        points, xyz_and_feats = self.backbone(x)
        xyz0 = xyz_and_feats[-1][0]
        points0 = self.transformer2(xyz0, self.fc2(points))[0]

        return xyz0, points0, xyz_and_feats