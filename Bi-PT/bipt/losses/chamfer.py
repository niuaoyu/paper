from typing import Optional, Tuple, Union
import torch

try:
    from pytorch3d.ops import knn_points
    from pytorch3d.loss import chamfer_distance as chamfer_distance_p3d
except Exception as e:
    raise ImportError(
        "pytorch3d is required for this implementation. "
        "Install PyTorch3D matching your CUDA/PyTorch versions."
    ) from e

# ============================================================
# Types
# ============================================================
TwoLabelInput = Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]
ThreeLabelInput = Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]


# ============================================================
# Reduction helpers (unchanged)
# ============================================================
def _validate_chamfer_reduction_inputs(
    batch_reduction: Union[str, None],
    point_reduction: Union[str, None],
) -> None:
    if batch_reduction is not None and batch_reduction not in ["mean", "sum"]:
        raise ValueError('batch_reduction must be one of ["mean", "sum"] or None')
    if point_reduction is not None and point_reduction not in ["mean", "sum", "max"]:
        raise ValueError('point_reduction must be one of ["mean", "sum", "max"] or None')
    if point_reduction is None and batch_reduction is not None:
        raise ValueError("batch_reduction must be None if point_reduction is None")


def _reduce_points(dists: torch.Tensor, point_reduction: Union[str, None]) -> torch.Tensor:
    # dists: (B, P)
    if point_reduction is None:
        return dists
    if point_reduction == "max":
        return dists.max(dim=1).values
    if point_reduction == "sum":
        return dists.sum(dim=1)
    if point_reduction == "mean":
        return dists.mean(dim=1)
    raise ValueError('point_reduction must be one of ["mean", "sum", "max"] or None')


def _apply_batch_reduction(cham: torch.Tensor, batch_reduction: Union[str, None]) -> torch.Tensor:
    # cham: (B,) or (B,P)
    if batch_reduction is None:
        return cham
    if batch_reduction == "sum":
        return cham.sum()
    if batch_reduction == "mean":
        return cham.mean() if cham.numel() > 0 else cham.new_tensor(0.0)
    raise ValueError('batch_reduction must be one of ["mean", "sum"] or None')


# ============================================================
# Label splitters
# ============================================================
def _split_two_labels(
    labels: TwoLabelInput,
    B: int,
    P: int,
    name: str,
    unordered_pairs: bool = True,
    allow_unbatched: bool = True,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Labels are VTK-style pairs per point:
      - Tensor (B,P,2) or (P,2) when B==1 (allow_unbatched)
      - or tuple (l1,l2) each (B,P)

    If unordered_pairs=True: canonicalize (a,b) and (b,a) -> (min,max).
    """
    if labels is None:
        return None

    if isinstance(labels, (tuple, list)):
        if len(labels) != 2:
            raise ValueError(f"{name}: expected (l1,l2) length 2.")
        l1, l2 = labels
        if l1.shape != (B, P) or l2.shape != (B, P):
            raise ValueError(
                f"{name}: expected l1,l2 shapes (B,P)={(B,P)}; got {tuple(l1.shape)} and {tuple(l2.shape)}."
            )
    else:
        if not torch.is_tensor(labels):
            raise ValueError(f"{name}: expected Tensor or (l1,l2).")

        if allow_unbatched and labels.ndim == 2 and labels.shape == (P, 2) and B == 1:
            labels = labels.unsqueeze(0)  # (1,P,2)

        if labels.shape != (B, P, 2):
            raise ValueError(f"{name}: expected Tensor shape (B,P,2)={(B,P,2)}; got {tuple(labels.shape)}.")

        l1 = labels[..., 0]
        l2 = labels[..., 1]

    if unordered_pairs:
        lo = torch.minimum(l1, l2)
        hi = torch.maximum(l1, l2)
        l1, l2 = lo, hi

    return l1, l2


def _split_three_labels(
    labels: ThreeLabelInput,
    B: int,
    P: int,
    name: str,
    unordered_triplets: bool = True,
    allow_unbatched: bool = True,
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Labels are per point triplets:
      - Tensor (B,P,3) or (P,3) when B==1 (allow_unbatched)
      - or tuple (l1,l2,l3) each (B,P)

    If unordered_triplets=True: canonicalize per point by sorting (order-invariant).
    """
    if labels is None:
        return None

    if isinstance(labels, (tuple, list)):
        if len(labels) != 3:
            raise ValueError(f"{name}: expected (l1,l2,l3) length 3.")
        l1, l2, l3 = labels
        if l1.shape != (B, P) or l2.shape != (B, P) or l3.shape != (B, P):
            raise ValueError(
                f"{name}: expected l1,l2,l3 shapes (B,P)={(B,P)}; got "
                f"{tuple(l1.shape)}, {tuple(l2.shape)}, {tuple(l3.shape)}."
            )
        if unordered_triplets:
            stacked = torch.stack([l1, l2, l3], dim=-1)  # (B,P,3)
            stacked = torch.sort(stacked, dim=-1).values
            l1, l2, l3 = stacked[..., 0], stacked[..., 1], stacked[..., 2]
        return l1, l2, l3

    if not torch.is_tensor(labels):
        raise ValueError(f"{name}: expected Tensor or (l1,l2,l3).")

    if allow_unbatched and labels.ndim == 2 and labels.shape == (P, 3) and B == 1:
        labels = labels.unsqueeze(0)  # (1,P,3)

    if labels.shape != (B, P, 3):
        raise ValueError(f"{name}: expected Tensor shape (B,P,3)={(B,P,3)}; got {tuple(labels.shape)}.")

    if unordered_triplets:
        labels = torch.sort(labels, dim=-1).values

    return labels[..., 0], labels[..., 1], labels[..., 2]


# ============================================================
# KNN helper (unchanged)
# ============================================================
def _knn1_sq_l2(x_sub: torch.Tensor, y_sub: torch.Tensor) -> torch.Tensor:
    """
    Compute 1-NN squared L2 distances using PyTorch3D.
    x_sub: (n,3), y_sub: (m,3) -> (n,)
    """
    x_sub = x_sub.contiguous()
    y_sub = y_sub.contiguous()
    out = knn_points(x_sub.unsqueeze(0), y_sub.unsqueeze(0), K=1, return_nn=False)
    return out.dists.squeeze(0).squeeze(-1)  # (n,)


# ============================================================
# Pair-label hierarchical NN (your existing semantics)
# ============================================================
def _hier_nn_two_pair_labels_knn(
    x: torch.Tensor,   # (B,N,3)
    y: torch.Tensor,   # (B,M,3)
    x_l1: torch.Tensor, x_l2: torch.Tensor,  # (B,N)
    y_l1: torch.Tensor, y_l2: torch.Tensor,  # (B,M)
) -> torch.Tensor:
    """
    Hierarchical NN distances for x->y:
      Tier1: exact pair match (l1,l2)
      Tier2: share either endpoint of the pair (one-label fallback)
      Tier3: global NN fallback

    Returns: (B,N) squared L2 distances.
    """
    B, N, _ = x.shape
    INF = 1e10
    out_all = x.new_empty((B, N))

    for b in range(B):
        xb, yb = x[b], y[b]              # (N,3), (M,3)
        xl1, xl2 = x_l1[b], x_l2[b]      # (N,), (N,)
        yl1, yl2 = y_l1[b], y_l2[b]      # (M,), (M,)

        global_best = _knn1_sq_l2(xb, yb)  # (N,)

        both_best = xb.new_full((N,), INF)
        has_both = torch.zeros((N,), device=xb.device, dtype=torch.bool)

        max_label = torch.max(torch.cat([yl1, yl2, xl1, xl2], dim=0)).item()
        base = int(max_label) + 1 if max_label >= 0 else 1

        code_y = yl1.to(torch.long) * base + yl2.to(torch.long)  # (M,)
        code_x = xl1.to(torch.long) * base + xl2.to(torch.long)  # (N,)

        for code in torch.unique(code_y):
            y_mask = (code_y == code)
            if not y_mask.any():
                continue
            x_mask = (code_x == code)
            if not x_mask.any():
                continue

            y_sub = yb[y_mask]     # (m,3)
            x_idx = x_mask.nonzero(as_tuple=False).squeeze(1)
            x_sub = xb[x_idx]      # (n,3)

            d = _knn1_sq_l2(x_sub, y_sub)  # (n,)
            both_best[x_idx] = d
            has_both[x_idx] = True

        one_best = xb.new_full((N,), INF)
        has_one = torch.zeros((N,), device=xb.device, dtype=torch.bool)

        remain = ~has_both
        if remain.any():
            region_ids = torch.unique(torch.cat([yl1, yl2], dim=0))
            for r in region_ids:
                y_mask = (yl1 == r) | (yl2 == r)
                if not y_mask.any():
                    continue
                y_sub = yb[y_mask]

                x_mask = remain & ((xl1 == r) | (xl2 == r))
                if not x_mask.any():
                    continue

                x_idx = x_mask.nonzero(as_tuple=False).squeeze(1)
                x_sub = xb[x_idx]

                d = _knn1_sq_l2(x_sub, y_sub)
                one_best[x_idx] = torch.minimum(one_best[x_idx], d)
                has_one[x_idx] = True

        out = torch.where(has_both, both_best, torch.where(has_one, one_best, global_best))
        out_all[b] = out

    return out_all  # (B,N)


# ============================================================
# 3-label hierarchical NN (exact triplet -> share 2 -> share 1 -> global)
# ============================================================
def _hier_nn_three_labels_knn(
    x: torch.Tensor,   # (B,N,3)
    y: torch.Tensor,   # (B,M,3)
    x_l1: torch.Tensor, x_l2: torch.Tensor, x_l3: torch.Tensor,  # (B,N)
    y_l1: torch.Tensor, y_l2: torch.Tensor, y_l3: torch.Tensor,  # (B,M)
) -> torch.Tensor:
    """
    Hierarchical NN distances for x->y based on unordered 3-label sets:
      Tier1: exact triplet match (l1,l2,l3)
      Tier2: share any 2 labels
      Tier3: share any 1 label
      Tier4: global NN fallback

    Returns: (B,N) squared L2 distances.
    """
    B, N, _ = x.shape
    INF = 1e10
    out_all = x.new_empty((B, N))

    for b in range(B):
        xb, yb = x[b], y[b]
        xl1, xl2, xl3 = x_l1[b], x_l2[b], x_l3[b]
        yl1, yl2, yl3 = y_l1[b], y_l2[b], y_l3[b]

        global_best = _knn1_sq_l2(xb, yb)  # (N,)

        # Tier1: exact triplet
        best3 = xb.new_full((N,), INF)
        has3 = torch.zeros((N,), device=xb.device, dtype=torch.bool)

        max_label = torch.max(torch.cat([xl1, xl2, xl3, yl1, yl2, yl3], dim=0)).item()
        base = int(max_label) + 1 if max_label >= 0 else 1
        base2 = base * base

        code_y3 = yl1.to(torch.long) * base2 + yl2.to(torch.long) * base + yl3.to(torch.long)  # (M,)
        code_x3 = xl1.to(torch.long) * base2 + xl2.to(torch.long) * base + xl3.to(torch.long)  # (N,)

        for code in torch.unique(code_y3):
            y_mask = (code_y3 == code)
            if not y_mask.any():
                continue
            x_mask = (code_x3 == code)
            if not x_mask.any():
                continue
            y_sub = yb[y_mask]
            x_idx = x_mask.nonzero(as_tuple=False).squeeze(1)
            x_sub = xb[x_idx]
            d = _knn1_sq_l2(x_sub, y_sub)
            best3[x_idx] = d
            has3[x_idx] = True

        # Tier2: share any 2 labels
        best2 = xb.new_full((N,), INF)
        has2 = torch.zeros((N,), device=xb.device, dtype=torch.bool)

        remain2 = ~has3
        if remain2.any():
            # y pair codes from its 3 combinations
            y_p12 = yl1.to(torch.long) * base + yl2.to(torch.long)
            y_p13 = yl1.to(torch.long) * base + yl3.to(torch.long)
            y_p23 = yl2.to(torch.long) * base + yl3.to(torch.long)

            # x pair codes
            x_p12 = xl1.to(torch.long) * base + xl2.to(torch.long)
            x_p13 = xl1.to(torch.long) * base + xl3.to(torch.long)
            x_p23 = xl2.to(torch.long) * base + xl3.to(torch.long)

            all_y_pair_codes = torch.unique(torch.cat([y_p12, y_p13, y_p23], dim=0))

            for code in all_y_pair_codes:
                y_mask = (y_p12 == code) | (y_p13 == code) | (y_p23 == code)
                if not y_mask.any():
                    continue
                y_sub = yb[y_mask]

                x_mask = remain2 & ((x_p12 == code) | (x_p13 == code) | (x_p23 == code))
                if not x_mask.any():
                    continue

                x_idx = x_mask.nonzero(as_tuple=False).squeeze(1)
                x_sub = xb[x_idx]

                d = _knn1_sq_l2(x_sub, y_sub)
                best2[x_idx] = torch.minimum(best2[x_idx], d)
                has2[x_idx] = True

        # Tier3: share any 1 label
        best1 = xb.new_full((N,), INF)
        has1 = torch.zeros((N,), device=xb.device, dtype=torch.bool)

        remain1 = ~has3 & ~has2
        if remain1.any():
            region_ids = torch.unique(torch.cat([yl1, yl2, yl3], dim=0))
            for r in region_ids:
                y_mask = (yl1 == r) | (yl2 == r) | (yl3 == r)
                if not y_mask.any():
                    continue
                y_sub = yb[y_mask]

                x_mask = remain1 & ((xl1 == r) | (xl2 == r) | (xl3 == r))
                if not x_mask.any():
                    continue

                x_idx = x_mask.nonzero(as_tuple=False).squeeze(1)
                x_sub = xb[x_idx]

                d = _knn1_sq_l2(x_sub, y_sub)
                best1[x_idx] = torch.minimum(best1[x_idx], d)
                has1[x_idx] = True

        out = torch.where(
            has3, best3,
            torch.where(has2, best2,
                        torch.where(has1, best1, global_best))
        )
        out_all[b] = out

    return out_all  # (B,N)


# ============================================================
# Chamfer entrypoints
#   - chamfer_distance_pair: pair semantic-aware
#   - chamfer_distance_triplet: 3-label semantic-aware
#   - chamfer_distance: auto dispatch
# ============================================================
def chamfer_distance_pair(
    x: torch.Tensor,
    y: torch.Tensor,
    batch_reduction: Union[str, None] = "mean",
    point_reduction: Union[str, None] = "mean",
    norm: int = 2,
    single_directional: bool = False,
    x_labels: TwoLabelInput = None,
    y_labels: TwoLabelInput = None,
    weight_x: float = 1.0,
    weight_y: float = 1.0,
):
    _validate_chamfer_reduction_inputs(batch_reduction, point_reduction)

    if norm != 2:
        raise ValueError("This implementation supports only norm=2 (squared L2).")
    if x.ndim != 3 or y.ndim != 3:
        raise ValueError("x and y must be (B,N,3) and (B,M,3).")
    if x.shape[0] != y.shape[0]:
        raise ValueError("x and y must have the same batch size.")
    if x.shape[2] != y.shape[2]:
        raise ValueError("x and y must have the same feature dimension (D).")

    if (x_labels is None) ^ (y_labels is None):
        raise ValueError("Either provide both x_labels and y_labels, or neither.")

    use_labels = (x_labels is not None) and (y_labels is not None)
    if not use_labels:
        loss, _ = chamfer_distance_p3d(
            x, y,
            batch_reduction=batch_reduction,
            point_reduction=point_reduction,
            norm=norm,
            single_directional=single_directional,
        )
        return loss, None

    B, N, _ = x.shape
    _, M, _ = y.shape

    x_l1, x_l2 = _split_two_labels(x_labels, B, N, "x_labels", unordered_pairs=True, allow_unbatched=True)
    y_l1, y_l2 = _split_two_labels(y_labels, B, M, "y_labels", unordered_pairs=True, allow_unbatched=True)

    d_x = _hier_nn_two_pair_labels_knn(x, y, x_l1, x_l2, y_l1, y_l2)  # (B,N)

    if single_directional:
        loss_x = _reduce_points(d_x, point_reduction)
        loss_x = _apply_batch_reduction(loss_x, batch_reduction)
        return loss_x, None

    d_y = _hier_nn_two_pair_labels_knn(y, x, y_l1, y_l2, x_l1, x_l2)  # (B,M)

    if point_reduction is None:
        return (d_x, d_y), None

    loss_x = _reduce_points(d_x, point_reduction)
    loss_y = _reduce_points(d_y, point_reduction)

    wx = float(weight_x)
    wy = float(weight_y)

    if point_reduction == "max":
        loss = torch.maximum(wx * loss_x, wy * loss_y)
    else:
        loss = wx * loss_x + wy * loss_y

    loss = _apply_batch_reduction(loss, batch_reduction)
    return loss, None


def chamfer_distance_triplet(
    x: torch.Tensor,
    y: torch.Tensor,
    batch_reduction: Union[str, None] = "mean",
    point_reduction: Union[str, None] = "mean",
    norm: int = 2,
    single_directional: bool = False,
    x_labels: ThreeLabelInput = None,
    y_labels: ThreeLabelInput = None,
    weight_x: float = 1.0,
    weight_y: float = 1.0,
):
    _validate_chamfer_reduction_inputs(batch_reduction, point_reduction)

    if norm != 2:
        raise ValueError("This implementation supports only norm=2 (squared L2).")
    if x.ndim != 3 or y.ndim != 3:
        raise ValueError("x and y must be (B,N,3) and (B,M,3).")
    if x.shape[0] != y.shape[0]:
        raise ValueError("x and y must have the same batch size.")
    if x.shape[2] != y.shape[2]:
        raise ValueError("x and y must have the same feature dimension (D).")

    if (x_labels is None) ^ (y_labels is None):
        raise ValueError("Either provide both x_labels and y_labels, or neither.")

    use_labels = (x_labels is not None) and (y_labels is not None)
    if not use_labels:
        loss, _ = chamfer_distance_p3d(
            x, y,
            batch_reduction=batch_reduction,
            point_reduction=point_reduction,
            norm=norm,
            single_directional=single_directional,
        )
        return loss, None

    B, N, _ = x.shape
    _, M, _ = y.shape

    x_l1, x_l2, x_l3 = _split_three_labels(x_labels, B, N, "x_labels", unordered_triplets=True, allow_unbatched=True)
    y_l1, y_l2, y_l3 = _split_three_labels(y_labels, B, M, "y_labels", unordered_triplets=True, allow_unbatched=True)

    d_x = _hier_nn_three_labels_knn(x, y, x_l1, x_l2, x_l3, y_l1, y_l2, y_l3)  # (B,N)

    if single_directional:
        loss_x = _reduce_points(d_x, point_reduction)
        loss_x = _apply_batch_reduction(loss_x, batch_reduction)
        return loss_x, None

    d_y = _hier_nn_three_labels_knn(y, x, y_l1, y_l2, y_l3, x_l1, x_l2, x_l3)  # (B,M)

    if point_reduction is None:
        return (d_x, d_y), None

    loss_x = _reduce_points(d_x, point_reduction)
    loss_y = _reduce_points(d_y, point_reduction)

    wx = float(weight_x)
    wy = float(weight_y)

    if point_reduction == "max":
        loss = torch.maximum(wx * loss_x, wy * loss_y)
    else:
        loss = wx * loss_x + wy * loss_y

    loss = _apply_batch_reduction(loss, batch_reduction)
    return loss, None


def chamfer_distance(
    x: torch.Tensor,
    y: torch.Tensor,
    batch_reduction: Union[str, None] = "mean",
    point_reduction: Union[str, None] = "mean",
    norm: int = 2,
    single_directional: bool = False,
    x_labels: Optional[Union[TwoLabelInput, ThreeLabelInput]] = None,
    y_labels: Optional[Union[TwoLabelInput, ThreeLabelInput]] = None,
    weight_x: float = 1.0,
    weight_y: float = 1.0,
):
    """
    Auto-dispatch:
      - if x_labels/y_labels are Tensor with last dim 2 -> pair mode
      - if last dim 3 -> triplet mode
      - if None -> plain PyTorch3D chamfer
    """
    _validate_chamfer_reduction_inputs(batch_reduction, point_reduction)

    if (x_labels is None) ^ (y_labels is None):
        raise ValueError("Either provide both x_labels and y_labels, or neither.")

    if x_labels is None and y_labels is None:
        loss, _ = chamfer_distance_p3d(
            x, y,
            batch_reduction=batch_reduction,
            point_reduction=point_reduction,
            norm=norm,
            single_directional=single_directional,
        )
        return loss, None

    # tuple/list inputs are ambiguous without shapes: infer by length
    if isinstance(x_labels, (tuple, list)) or isinstance(y_labels, (tuple, list)):
        if not (isinstance(x_labels, (tuple, list)) and isinstance(y_labels, (tuple, list))):
            raise ValueError("If using tuple/list label form, provide both x_labels and y_labels in tuple/list form.")
        if len(x_labels) == 2 and len(y_labels) == 2:
            return chamfer_distance_pair(
                x, y,
                batch_reduction=batch_reduction,
                point_reduction=point_reduction,
                norm=norm,
                single_directional=single_directional,
                x_labels=x_labels, y_labels=y_labels,
                weight_x=weight_x, weight_y=weight_y,
            )
        if len(x_labels) == 3 and len(y_labels) == 3:
            return chamfer_distance_triplet(
                x, y,
                batch_reduction=batch_reduction,
                point_reduction=point_reduction,
                norm=norm,
                single_directional=single_directional,
                x_labels=x_labels, y_labels=y_labels,
                weight_x=weight_x, weight_y=weight_y,
            )
        raise ValueError("Tuple/list labels must be length 2 (pair) or 3 (triplet).")

    # Tensor labels: infer by last dimension
    if not torch.is_tensor(x_labels) or not torch.is_tensor(y_labels):
        raise ValueError("x_labels/y_labels must be both Tensors or both tuples/lists.")

    # Allow unbatched (P,2)/(P,3) for B==1 happens inside splitters;
    # here we just infer expected K from the provided tensor.
    if x_labels.ndim == 3:
        k = x_labels.shape[-1]
    elif x_labels.ndim == 2:
        k = x_labels.shape[-1]
    else:
        raise ValueError(f"x_labels must have ndim 2 or 3, got {x_labels.ndim}.")

    if k == 2:
        return chamfer_distance_pair(
            x, y,
            batch_reduction=batch_reduction,
            point_reduction=point_reduction,
            norm=norm,
            single_directional=single_directional,
            x_labels=x_labels, y_labels=y_labels,
            weight_x=weight_x, weight_y=weight_y,
        )
    if k == 3:
        return chamfer_distance_triplet(
            x, y,
            batch_reduction=batch_reduction,
            point_reduction=point_reduction,
            norm=norm,
            single_directional=single_directional,
            x_labels=x_labels, y_labels=y_labels,
            weight_x=weight_x, weight_y=weight_y,
        )

    raise ValueError(f"Unsupported label dim k={k}. Only 2 (pair) or 3 (triplet) supported.")


# Edge Loss
def edge_loss(pred_points: torch.Tensor, faces: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    """
    Edge length loss to penalize high edge length
    Args:
        pred_points: FloatTensor of shape (B, N, 3)
        connectivity: LongTensor of shape (B, N, 3)
    Returns:
        loss: scalar tensor representing the edge length loss
    """
    device = pred_points.device

    # Make faces 2D: (T, 3)
    if faces.dim() == 3:
        faces = faces[0]
    faces = faces.to(device)           # (T, 3)

    # triangles [i, j, k] -> edges (i, j), (j, k), (k, i)
    i = faces[:, 0]
    j = faces[:, 1]
    k = faces[:, 2]

    edges = torch.stack([
        torch.stack([i, j], dim=-1),   # (T, 2)
        torch.stack([j, k], dim=-1),   # (T, 2)
        torch.stack([k, i], dim=-1)    # (T, 2)
    ], dim=0).reshape(-1, 2)           # (3T, 2)

    # undirected: sort & deduplicate
    edges, _ = torch.sort(edges, dim=1)    # (3T, 2)
    edges = torch.unique(edges, dim=0)     # (E, 2)

    # ----- simpler indexing instead of gather -----
    idx0 = edges[:, 0].to(device)   # (E,)
    idx1 = edges[:, 1].to(device)   # (E,)

    # pick endpoints for each batch
    p0 = pred_points[:, idx0, :]    # (B, E, 3)
    p1 = pred_points[:, idx1, :]    # (B, E, 3)

    # L2 distance per edge
    edge_lengths = torch.norm(p0 - p1, p=2, dim=-1)  # (B, E)

    if reduction == "mean":
        loss = edge_lengths.mean()
    elif reduction == "sum":
        loss = edge_lengths.sum()
    else:
        raise ValueError("reduction must be 'mean' or 'sum'")

    return loss
