import torch

def _nearest_squared_distance(source, target, chunk_size):
    """分块计算 source 中每个点到 target 的最近平方距离。"""
    nearest = []
    for start in range(0, source.shape[1], chunk_size):
        source_chunk = source[:, start:start + chunk_size]
        distances = torch.cdist(source_chunk, target, p=2)
        nearest.append(distances.square().amin(dim=2))
    return torch.cat(nearest, dim=1)


def chamfer_distance(x, y, chunk_size=512):
    """
    简单的 Chamfer Distance 实现
    Args:
        x: (B, N, 3) 点云
        y: (B, M, 3) 点云
    Returns:
        loss: 标量损失
        _ : 占位符（保持与 pytorch3d 接口一致）
    """
    if x.ndim != 3 or y.ndim != 3 or x.shape[0] != y.shape[0]:
        raise ValueError('点云必须是批次匹配的 (B, N, 3) 和 (B, M, 3) 张量')

    # 分块避免一次创建 (B, N, M, 3) 张量，在 ODE 阶段显著降低显存峰值。
    min_dist_x_to_y = _nearest_squared_distance(x, y, chunk_size)
    min_dist_y_to_x = _nearest_squared_distance(y, x, chunk_size)

    loss = torch.mean(min_dist_x_to_y) + torch.mean(min_dist_y_to_x)

    return loss, None
