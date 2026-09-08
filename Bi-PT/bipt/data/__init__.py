"""Data loading for Bi-PT (sparse point clouds + labeled ground-truth meshes)."""

from .dataset import (
    LabeledDataset,
    collate_batch,
    pc_normalize_xyz,
    build_spc_feat_bn6,
    build_spc_feat_bn8,
    build_atlas_feat_bn6,
    build_atlas_feat,
    load_vtp_points_and_labels,
    load_sparse_npz_points_and_pairs,
    load_vtp_faces,
)

__all__ = [
    "LabeledDataset",
    "collate_batch",
    "pc_normalize_xyz",
    "build_spc_feat_bn6",
    "build_spc_feat_bn8",
    "build_atlas_feat_bn6",
    "build_atlas_feat",
    "load_vtp_points_and_labels",
    "load_sparse_npz_points_and_pairs",
    "load_vtp_faces",
]
