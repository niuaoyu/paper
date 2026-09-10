from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def index_points(points: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Gather batched points/features with `[B, ...]` indices."""
    batch_indices = torch.arange(points.shape[0], device=points.device)
    batch_indices = batch_indices.view(-1, *([1] * (indices.ndim - 1)))
    return points[batch_indices, indices]


@torch.no_grad()
def farthest_point_sample(points: torch.Tensor, sample_count: int) -> torch.Tensor:
    """Deterministic batched farthest-point sampling implemented in PyTorch."""
    batch, point_count, _ = points.shape
    sample_count = min(sample_count, point_count)
    centroids = torch.empty(batch, sample_count, dtype=torch.long, device=points.device)
    distances = torch.full((batch, point_count), float("inf"), device=points.device)
    center = points.mean(dim=1, keepdim=True)
    farthest = (points - center).square().sum(dim=-1).argmax(dim=1)
    batch_indices = torch.arange(batch, device=points.device)
    for sample_index in range(sample_count):
        centroids[:, sample_index] = farthest
        centroid = points[batch_indices, farthest].unsqueeze(1)
        squared_distance = (points - centroid).square().sum(dim=-1)
        distances = torch.minimum(distances, squared_distance)
        farthest = distances.argmax(dim=1)
    return centroids


class SharedMLP(nn.Module):
    def __init__(self, channels: list[int]) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        for input_dim, output_dim in zip(channels[:-1], channels[1:]):
            layers.extend([nn.Linear(input_dim, output_dim), nn.ReLU(inplace=True)])
        self.layers = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.layers(features)


class SetAbstraction(nn.Module):
    """PointNet++-style FPS, kNN grouping, local MLP, and symmetric pooling."""

    def __init__(
        self,
        sample_count: int,
        neighbor_count: int,
        input_dim: int,
        mlp_channels: list[int],
    ) -> None:
        super().__init__()
        self.sample_count = sample_count
        self.neighbor_count = neighbor_count
        self.local_mlp = SharedMLP([input_dim + 3, *mlp_channels])

    def forward(
        self,
        xyz: torch.Tensor,
        point_features: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        centroid_indices = farthest_point_sample(xyz, self.sample_count)
        centroids = index_points(xyz, centroid_indices)
        neighbor_count = min(self.neighbor_count, xyz.shape[1])
        neighbor_indices = torch.cdist(centroids, xyz).topk(
            neighbor_count, dim=-1, largest=False, sorted=False
        ).indices
        grouped_xyz = index_points(xyz, neighbor_indices)
        local_xyz = grouped_xyz - centroids.unsqueeze(2)
        if point_features is None:
            local_features = local_xyz
        else:
            grouped_features = index_points(point_features, neighbor_indices)
            local_features = torch.cat([local_xyz, grouped_features], dim=-1)
        encoded = self.local_mlp(local_features)
        return centroids, encoded.max(dim=2).values


class PointNetPlusPlusEncoder(nn.Module):
    """Hierarchical point encoder compatible with the original global feature interface."""

    def __init__(self, feature_dim: int = 256) -> None:
        super().__init__()
        self.sa1 = SetAbstraction(256, 24, 0, [32, 64, 96])
        self.sa2 = SetAbstraction(64, 24, 96, [96, 128, 192])
        self.global_mlp = SharedMLP([192 + 3, 256, feature_dim])

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        return self.forward_hierarchy(points)[-1]

    def forward_hierarchy(
        self, points: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        xyz1, features1 = self.sa1(points, None)
        xyz2, features2 = self.sa2(xyz1, features1)
        global_points = self.global_mlp(torch.cat([xyz2, features2], dim=-1))
        return xyz1, features1, xyz2, features2, global_points.max(dim=1).values


def rasterize_occupancy(points: torch.Tensor, resolution: int = 64) -> torch.Tensor:
    """Rasterize canonical `[-1, 1]` points into a binary occupancy volume."""
    batch, point_count, _ = points.shape
    coordinates = ((points.clamp(-1.0, 1.0) + 1.0) * (resolution - 1) * 0.5).long()
    x, y, z = coordinates.unbind(dim=-1)
    linear = z * resolution * resolution + y * resolution + x
    occupancy = points.new_zeros(batch, resolution**3)
    occupancy.scatter_(1, linear, 1.0)
    return occupancy.view(batch, 1, resolution, resolution, resolution)


class VolumeEncoder(nn.Module):
    """64-cubed occupancy encoder with four MR-Net-style feature scales."""

    def __init__(self, channels: tuple[int, int, int, int] = (32, 64, 128, 192)) -> None:
        super().__init__()
        blocks = []
        input_channels = 1
        for scale, output_channels in enumerate(channels):
            stride = 1 if scale == 0 else 2
            convolution_count = 2 if scale == 0 else 3
            layers: list[nn.Module] = [
                nn.Conv3d(
                    input_channels,
                    output_channels,
                    kernel_size=3,
                    stride=stride,
                    padding=1,
                ),
                nn.ReLU(inplace=True),
            ]
            for _ in range(convolution_count - 1):
                layers.extend(
                    [
                        nn.Conv3d(
                            output_channels, output_channels, kernel_size=3, padding=1
                        ),
                        nn.ReLU(inplace=True),
                    ]
                )
            blocks.append(nn.Sequential(*layers))
            input_channels = output_channels
        self.blocks = nn.ModuleList(blocks)
        self.output_channels = channels

    def forward(self, points: torch.Tensor) -> list[torch.Tensor]:
        features = rasterize_occupancy(points)
        pyramid = []
        for block in self.blocks:
            features = block(features)
            pyramid.append(features)
        return pyramid


class PointFeatureProjector(nn.Module):
    """Project learned features from three point scales onto graph vertices."""

    def __init__(self, neighbor_count: int = 3, output_dim: int = 4) -> None:
        super().__init__()
        self.neighbor_count = neighbor_count
        self.adapters = nn.ModuleList(
            [nn.Linear(3, output_dim), nn.Linear(96, output_dim), nn.Linear(192, output_dim)]
        )

    def forward(
        self,
        vertices: torch.Tensor,
        point_scales: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> list[torch.Tensor]:
        projected = []
        for (xyz, features), adapter in zip(point_scales, self.adapters):
            encoded = adapter(features)
            neighbor_count = min(self.neighbor_count, xyz.shape[1])
            distances, indices = torch.cdist(vertices, xyz).topk(
                neighbor_count, dim=-1, largest=False, sorted=False
            )
            neighbors = index_points(encoded, indices)
            weights = distances.clamp_min(1e-6).reciprocal()
            weights = weights / weights.sum(dim=-1, keepdim=True)
            projected.append((neighbors * weights.unsqueeze(-1)).sum(dim=2))
        return projected


class GraphProjection3D(nn.Module):
    """Sample the volume pyramid and point hierarchy at each graph vertex."""

    def __init__(self, volume_channels: tuple[int, int, int, int]) -> None:
        super().__init__()
        self.point_projector = PointFeatureProjector()
        self.output_dim = 3 + sum(volume_channels) + 3 * 4

    def forward(
        self,
        vertices: torch.Tensor,
        volume_features: list[torch.Tensor],
        point_scales: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        grid = vertices.clamp(-1.0, 1.0).view(vertices.shape[0], -1, 1, 1, 3)
        volume_projection = []
        for features in volume_features:
            sampled = F.grid_sample(
                features,
                grid,
                mode="bilinear",
                padding_mode="border",
                align_corners=True,
            )
            volume_projection.append(sampled[:, :, :, 0, 0].transpose(1, 2))
        point_projection = self.point_projector(vertices, point_scales)
        return torch.cat([vertices, *volume_projection, *point_projection], dim=-1)


class GraphConv(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.self_linear = nn.Linear(input_dim, output_dim)
        self.neighbor_linear = nn.Linear(input_dim, output_dim, bias=False)

    def forward(self, features: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
        batch, vertex_count, channels = features.shape
        source = torch.cat([edges[:, 0], edges[:, 1]], dim=0)
        target = torch.cat([edges[:, 1], edges[:, 0]], dim=0)
        neighbors = features[:, source]
        aggregate = features.new_zeros(batch, vertex_count, channels)
        aggregate.index_add_(1, target, neighbors)
        degree = torch.bincount(target, minlength=vertex_count).clamp_min(1)
        aggregate = aggregate / degree.to(features.dtype).view(1, -1, 1)
        return self.self_linear(features) + self.neighbor_linear(aggregate)


class GraphDeformationStage(nn.Module):
    """Residual graph-convolution stage matching the depth of an MR-Net block."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        hidden_layers: int,
        bottleneck_dim: int | None = None,
        offset_scale: float = 0.25,
    ) -> None:
        super().__init__()
        self.input = GraphConv(input_dim, hidden_dim)
        self.layers = nn.ModuleList(
            [GraphConv(hidden_dim, hidden_dim) for _ in range(hidden_layers)]
        )
        self.bottleneck = (
            GraphConv(hidden_dim, bottleneck_dim) if bottleneck_dim is not None else None
        )
        self.output = GraphConv(bottleneck_dim or hidden_dim, 3)
        self.offset_scale = offset_scale

    def forward(
        self,
        features: torch.Tensor,
        vertices: torch.Tensor,
        edges: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = torch.relu(self.input(features, edges))
        residual = hidden
        for layer_index, layer in enumerate(self.layers):
            updated = torch.relu(layer(hidden, edges))
            if layer_index % 2 == 1:
                hidden = 0.5 * (updated + residual)
                residual = hidden
            else:
                hidden = updated
        output_features = hidden
        if self.bottleneck is not None:
            output_features = torch.relu(self.bottleneck(output_features, edges))
        offsets = self.offset_scale * torch.tanh(self.output(output_features, edges))
        return vertices + offsets, hidden


class MultiStageGraphDecoder(nn.Module):
    """Three projection/deformation blocks with 14, 15, and 16 graph layers."""

    def __init__(self, projection_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.stages = nn.ModuleList(
            [
                GraphDeformationStage(projection_dim, hidden_dim, hidden_layers=12),
                GraphDeformationStage(
                    projection_dim + hidden_dim, hidden_dim, hidden_layers=13
                ),
                GraphDeformationStage(
                    projection_dim + hidden_dim,
                    hidden_dim,
                    hidden_layers=13,
                    bottleneck_dim=hidden_dim // 2,
                ),
            ]
        )

    def forward(
        self,
        template_vertices: torch.Tensor,
        edges: torch.Tensor,
        projector: GraphProjection3D,
        volume_features: list[torch.Tensor],
        point_scales: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = volume_features[0].shape[0]
        vertices = template_vertices.unsqueeze(0).expand(batch, -1, -1)
        outputs = []
        previous_hidden = None
        for stage in self.stages:
            projected = projector(vertices, volume_features, point_scales)
            if previous_hidden is not None:
                projected = torch.cat([projected, previous_hidden], dim=-1)
            vertices, previous_hidden = stage(projected, vertices, edges)
            outputs.append(vertices)
        return outputs[0], outputs[1], outputs[2]


class MinimalMRNet(nn.Module):
    """Complete PyTorch MR-Net path with a backward-compatible public interface."""

    def __init__(self, feature_dim: int = 256, hidden_dim: int = 128) -> None:
        super().__init__()
        self.encoder = PointNetPlusPlusEncoder(feature_dim)
        self.volume_encoder = VolumeEncoder()
        self.projector = GraphProjection3D(self.volume_encoder.output_channels)
        self.decoder = MultiStageGraphDecoder(self.projector.output_dim, hidden_dim)

    def forward(
        self,
        points: torch.Tensor,
        template_vertices: torch.Tensor,
        edges: torch.Tensor,
        return_stages: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        xyz1, features1, xyz2, features2, _ = self.encoder.forward_hierarchy(points)
        point_scales = [(points, points), (xyz1, features1), (xyz2, features2)]
        volume_features = self.volume_encoder(points)
        stages = self.decoder(
            template_vertices,
            edges,
            self.projector,
            volume_features,
            point_scales,
        )
        return stages if return_stages else stages[-1]
