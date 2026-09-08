"""Loss functions for Bi-PT.

The semantic-aware Chamfer distance restricts nearest-neighbor search by the
per-point chamber labels (paper Sec. 2.4), with graceful fallback to plain
PyTorch3D Chamfer when no labels are supplied.
"""

from .chamfer import (
    chamfer_distance,
    chamfer_distance_pair,
    chamfer_distance_triplet,
    edge_loss,
)

__all__ = [
    "chamfer_distance",
    "chamfer_distance_pair",
    "chamfer_distance_triplet",
    "edge_loss",
]
