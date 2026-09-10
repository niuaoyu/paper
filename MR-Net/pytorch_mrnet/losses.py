from __future__ import annotations

import torch


def chamfer_distance(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    distances = torch.cdist(prediction, target, p=2).square()
    return distances.min(dim=2).values.mean() + distances.min(dim=1).values.mean()


def edge_length_loss(
    prediction: torch.Tensor,
    template_vertices: torch.Tensor,
    edges: torch.Tensor,
) -> torch.Tensor:
    predicted_lengths = torch.linalg.vector_norm(
        prediction[:, edges[:, 0]] - prediction[:, edges[:, 1]], dim=-1
    )
    template_lengths = torch.linalg.vector_norm(
        template_vertices[edges[:, 0]] - template_vertices[edges[:, 1]], dim=-1
    )
    return (predicted_lengths - template_lengths.unsqueeze(0)).square().mean()


def graph_laplacian(vertices: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
    single = vertices.ndim == 2
    if single:
        vertices = vertices.unsqueeze(0)
    batch, vertex_count, _ = vertices.shape
    source = torch.cat([edges[:, 0], edges[:, 1]], dim=0)
    target = torch.cat([edges[:, 1], edges[:, 0]], dim=0)
    aggregate = vertices.new_zeros(batch, vertex_count, 3)
    aggregate.index_add_(1, target, vertices[:, source])
    degree = torch.bincount(target, minlength=vertex_count).clamp_min(1)
    neighbor_mean = aggregate / degree.to(vertices.dtype).view(1, -1, 1)
    result = vertices - neighbor_mean
    return result.squeeze(0) if single else result


def laplacian_loss(
    prediction: torch.Tensor,
    template_vertices: torch.Tensor,
    edges: torch.Tensor,
) -> torch.Tensor:
    predicted_laplacian = graph_laplacian(prediction, edges)
    template_laplacian = graph_laplacian(template_vertices, edges).unsqueeze(0)
    return (predicted_laplacian - template_laplacian).square().mean()


def multi_stage_mesh_loss(
    predictions: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    target: torch.Tensor,
    template_vertices: torch.Tensor,
    edges: torch.Tensor,
    stage_weights: tuple[float, float, float] = (0.1, 0.3, 0.6),
    edge_weight: float = 0.2,
    laplacian_weight: float = 0.2,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Apply the existing surface and mesh regularizers at all three stages."""
    total = target.new_zeros(())
    metrics: dict[str, torch.Tensor] = {}
    for stage_index, (prediction, stage_weight) in enumerate(
        zip(predictions, stage_weights), start=1
    ):
        chamfer = chamfer_distance(prediction, target)
        edge = edge_length_loss(prediction, template_vertices, edges)
        laplacian = laplacian_loss(prediction, template_vertices, edges)
        stage_loss = chamfer + edge_weight * edge + laplacian_weight * laplacian
        total = total + stage_weight * stage_loss
        metrics[f"stage{stage_index}_chamfer"] = chamfer
        metrics[f"stage{stage_index}_edge"] = edge
        metrics[f"stage{stage_index}_laplacian"] = laplacian
    metrics["loss"] = total
    return total, metrics
