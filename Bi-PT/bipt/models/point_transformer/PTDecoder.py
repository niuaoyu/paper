import torch
import torch.nn as nn
from .pointnet_util import PointNetFeaturePropagation
from .transformer import TransformerBlock

def _norm1d(norm_type: str, num_channels: int) -> nn.Module:
    norm_type = (norm_type or "bn").lower()
    if norm_type == "bn":
        return nn.BatchNorm1d(num_channels)
    if norm_type == "in":
        return nn.InstanceNorm1d(num_channels, affine=True, track_running_stats=False)
    raise ValueError(f"Unsupported norm_type='{norm_type}'. Use 'bn' or 'in'.")

class TransitionUp(nn.Module):
    def __init__(
        self,
        dim1, dim2, dim_out,
        norm_type: str = "bn",        # keep (used historically)
        fc_norm_type: str = None,     # keep (optional override for fc path)
        fp_norm_type: str = None,     # NEW: override for PointNetFeaturePropagation
    ):
        class SwapAxes(nn.Module):
            def forward(self, x):
                return x.transpose(1, 2)

        super().__init__()

        # Backward-compatible defaults:
        # - if older code only passed norm_type, fc_norm_type used to effectively be norm_type
        # - fp_norm_type used to effectively be norm_type
        if fc_norm_type is None:
            fc_norm_type = norm_type
        if fp_norm_type is None:
            fp_norm_type = norm_type

        self.fc1 = nn.Sequential(
            nn.Linear(dim1, dim_out),
            SwapAxes(),
            _norm1d(fc_norm_type, dim_out),   # FC norm is explicit now
            SwapAxes(),
            nn.ReLU(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(dim2, dim_out),
            SwapAxes(),
            _norm1d(fc_norm_type, dim_out),
            SwapAxes(),
            nn.ReLU(),
        )

        # FP norm is explicit now (NEW)
        self.fp = PointNetFeaturePropagation(-1, [], norm_type=fp_norm_type)

    def forward(self, xyz1, points1, xyz2, points2):
        feats1 = self.fc1(points1)
        feats2 = self.fc2(points2)

        feats1 = self.fp(
            xyz2.transpose(1, 2),
            xyz1.transpose(1, 2),
            None,
            feats1.transpose(1, 2)
        ).transpose(1, 2)

        return feats1 + feats2

class PTDecoder(nn.Module):
    def __init__(
        self,
        nblocks=4,
        nneighbor=16,
        transformer_dim=512,
        norm_type: str = "bn",             # unchanged
        fc_norm_type: str = None,          # NEW (but optional)
        fp_norm_type: str = None,          # NEW (but optional)
    ):
        super().__init__()
        self.nblocks = nblocks

        self.transition_ups1 = nn.ModuleList()
        self.transformers1 = nn.ModuleList()

        for i in reversed(range(nblocks)):
            channel = 32 * 2 ** i
            self.transition_ups1.append(
                TransitionUp(
                    channel * 2, channel, channel,
                    norm_type=norm_type,          # keep old behavior
                    fc_norm_type=fc_norm_type,    # optional override
                    fp_norm_type=fp_norm_type,    # optional override
                )
            )
            self.transformers1.append(TransformerBlock(channel, transformer_dim, nneighbor))

    def forward(self, xyz0, points0, xyz_and_feats):
        xyz = xyz0
        points = points0
        for i in range(self.nblocks):
            points = self.transition_ups1[i](xyz, points, xyz_and_feats[- i - 2][0], xyz_and_feats[- i - 2][1])
            xyz = xyz_and_feats[- i - 2][0]
            points = self.transformers1[i](xyz, points)[0]
        points_avg = points.mean(1)
        return points, points_avg
