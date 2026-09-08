"""Dataset and VTK / npz IO for Bi-PT.

Loads sparse point clouds (``.npz`` with keys ``pcs`` / ``pairs``) and labeled
ground-truth meshes (``.vtp`` with ``VertexBoundaryLabels`` / ``VertexLabels3``),
builds per-point input features (normalized xyz + centered xyz + scaled labels),
and exposes them through :class:`LabeledDataset` / :func:`collate_batch`.

See ``data/README.md`` for the expected directory layout and file formats.
"""
import os
import glob
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

import vtk
from vtk.util.numpy_support import vtk_to_numpy


def pc_normalize_xyz(sparse_pc: np.ndarray, target_pc: np.ndarray = None) -> np.ndarray:
    sparse_pc = sparse_pc.astype(np.float32)
    centroid = sparse_pc.mean(axis=0, keepdims=True)
    pc = sparse_pc - centroid

    x_max, x_min = np.max(pc[:, 0]), np.min(pc[:, 0])
    y_max, y_min = np.max(pc[:, 1]), np.min(pc[:, 1])
    z_max, z_min = np.max(pc[:, 2]), np.min(pc[:, 2])

    span_x = max(float(x_max - x_min), 1e-8)
    span_y = max(float(y_max - y_min), 1e-8)
    span_z = max(float(z_max - z_min), 1e-8)

    if target_pc is not None:
        t_pc = target_pc.astype(np.float32) - centroid
        t_pc[:, 0] = 1.7 * (t_pc[:, 0] - float(x_min)) / span_x - 0.85
        t_pc[:, 1] = 1.7 * (t_pc[:, 1] - float(y_min)) / span_y - 0.85
        t_pc[:, 2] = 1.7 * (t_pc[:, 2] - float(z_min)) / span_z - 0.85
        return t_pc

    pc[:, 0] = 1.7 * (pc[:, 0] - float(x_min)) / span_x - 0.85
    pc[:, 1] = 1.7 * (pc[:, 1] - float(y_min)) / span_y - 0.85
    pc[:, 2] = 1.7 * (pc[:, 2] - float(z_min)) / span_z - 0.85
    return pc


def _canonicalize_rows(labs: np.ndarray) -> np.ndarray:
    labs = np.asarray(labs)
    if labs.ndim == 1:
        labs = labs.reshape(-1, 1)
    if labs.ndim != 2:
        raise ValueError(f"labels must be 2D, got {labs.shape}")
    labs = labs.astype(np.int64)
    labs = np.sort(labs, axis=1)
    return labs


def build_spc_feat_bn6(xyz_np: np.ndarray) -> np.ndarray:
    xyz_norm = pc_normalize_xyz(xyz_np)
    featC = xyz_norm - xyz_norm.mean(0, keepdims=True)
    feat = np.concatenate([xyz_norm, featC], axis=1).astype(np.float32)  # (N,6)
    return feat


def build_atlas_feat_bn6(atlas_xyz_np: np.ndarray) -> np.ndarray:
    xyz_norm = pc_normalize_xyz(atlas_xyz_np)
    featC = xyz_norm - xyz_norm.mean(0, keepdims=True)
    feat = np.concatenate([xyz_norm, featC], axis=1).astype(np.float32)  # (N,6)
    return feat


def build_spc_feat_bn8(xyz_np: np.ndarray, pairs2_np: np.ndarray, label_scale: float = 10.0) -> np.ndarray:
    """
    spc ALWAYS BN8: [xyz_norm(3), featC(3), pair2(2)] => 8
    """
    xyz_norm = pc_normalize_xyz(xyz_np)
    featC = xyz_norm - xyz_norm.mean(0, keepdims=True)

    labs = _canonicalize_rows(pairs2_np)
    if labs.shape[1] < 2:
        raise ValueError(f"spc pairs must be (N,2). got {labs.shape}")
    labs = labs[:, :2].astype(np.float32)

    if label_scale is not None and label_scale > 0:
        labs = labs / float(label_scale)

    feat = np.concatenate([xyz_norm, featC, labs], axis=1).astype(np.float32)  # (N,8)
    if feat.shape[1] != 8:
        raise RuntimeError(f"spc feature dim must be 8, got {feat.shape[1]}")
    return feat


def build_atlas_feat(
    atlas_xyz_np: np.ndarray,
    atlas_feat_labels_np: np.ndarray,
    feat_label_dim: int,     # 2 => BN8, 3 => BN9
    label_scale: float = 10.0,
) -> np.ndarray:
    """
    atlas feature:
      - feat_label_dim=2 => [xyz_norm(3), featC(3), pair2(2)] => 8
      - feat_label_dim=3 => [xyz_norm(3), featC(3), lab3(3)]  => 9
    """
    xyz_norm = pc_normalize_xyz(atlas_xyz_np)
    featC = xyz_norm - xyz_norm.mean(0, keepdims=True)

    labs = _canonicalize_rows(atlas_feat_labels_np)
    if labs.shape[1] < feat_label_dim:
        raise ValueError(f"atlas feat labels need >= {feat_label_dim} comps, got {labs.shape}")
    labs = labs[:, :feat_label_dim].astype(np.float32)

    if label_scale is not None and label_scale > 0:
        labs = labs / float(label_scale)

    feat = np.concatenate([xyz_norm, featC, labs], axis=1).astype(np.float32)  # (N,6+feat_label_dim)
    expected = 6 + feat_label_dim
    if feat.shape[1] != expected:
        raise RuntimeError(f"atlas feature dim mismatch: got {feat.shape[1]} vs expected {expected}")
    return feat


def load_vtp_points_and_labels(vtp_path: str, array_name: str, min_dim: int) -> Tuple[np.ndarray, np.ndarray]:
    if not os.path.exists(vtp_path):
        raise FileNotFoundError(vtp_path)

    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(vtp_path)
    reader.Update()
    poly = reader.GetOutput()
    if poly is None or poly.GetPoints() is None:
        raise RuntimeError(f"No PolyData/Points in VTP: {vtp_path}")

    pts = vtk_to_numpy(poly.GetPoints().GetData()).astype(np.float32)

    pd = poly.GetPointData()
    if pd is None:
        raise RuntimeError(f"No point data in VTP: {vtp_path}")

    arr = pd.GetArray(array_name)
    if arr is None:
        names = [pd.GetArrayName(i) for i in range(pd.GetNumberOfArrays())]
        raise RuntimeError(f"Array '{array_name}' not found in {vtp_path}. Available: {names}")

    labs = vtk_to_numpy(arr)
    labs = _canonicalize_rows(labs)

    if labs.shape[1] < min_dim:
        raise RuntimeError(f"'{array_name}' must have >= {min_dim} comps. Got {labs.shape}")

    if labs.shape[0] != pts.shape[0]:
        raise RuntimeError(f"labels and points count mismatch in {vtp_path}")

    return pts, labs[:, :min_dim].astype(np.int64)


def load_sparse_npz_points_and_pairs(npz_path: str, pcs_key: str, pairs_key: str) -> Tuple[np.ndarray, np.ndarray]:
    npz = np.load(npz_path, allow_pickle=False)
    if pcs_key not in npz:
        raise KeyError(f"{npz_path}: missing '{pcs_key}'")
    if pairs_key not in npz:
        raise KeyError(f"{npz_path}: missing '{pairs_key}' (sparse must provide pair labels)")

    pcs = np.asarray(npz[pcs_key])
    pairs = np.asarray(npz[pairs_key])

    if pcs.ndim == 3 and pcs.shape[-1] == 3:
        pcs = pcs.reshape(-1, 3)
    if pairs.ndim == 3 and pairs.shape[-1] == 2:
        pairs = pairs.reshape(-1, 2)

    if pcs.ndim != 2 or pcs.shape[1] != 3:
        raise ValueError(f"{npz_path}: pcs expected (N,3), got {pcs.shape}")
    if pairs.ndim != 2 or pairs.shape[1] < 2:
        raise ValueError(f"{npz_path}: pairs expected (N,2), got {pairs.shape}")

    pairs = _canonicalize_rows(pairs)[:, :2]

    if pcs.shape[0] != pairs.shape[0]:
        raise ValueError(f"{npz_path}: pcs N={pcs.shape[0]} != pairs N={pairs.shape[0]}")

    return pcs.astype(np.float32), pairs.astype(np.int64)


def load_vtp_faces(vtp_path: str) -> np.ndarray:
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(vtp_path)
    reader.Update()
    polydata = reader.GetOutput()

    tri_filter = vtk.vtkTriangleFilter()
    tri_filter.SetInputData(polydata)
    tri_filter.Update()
    tri_mesh = tri_filter.GetOutput()

    polys = tri_mesh.GetPolys()
    np_arr = vtk_to_numpy(polys.GetData())
    n_cells = polys.GetNumberOfCells()
    np_arr = np_arr.reshape((n_cells, 4))
    faces = np_arr[:, 1:4].astype(np.int64)
    return faces


class LabeledDataset(Dataset):
    """
    spc: always BN8 from .npz (pcs + pairs2)
    gt:
      - always provides pair labels (for pair-CD)
      - if label_dim=3: also provides label3 (for label3-CD)
    """
    def __init__(
        self,
        sparse_dir: str,
        gt_dir: str,
        pcs_key: str,
        spc_pairs_key: str,
        gt_pair_array: str,
        gt_label3_array: str,
        label_dim: int,
        label_scale: float,
        no_label_feat: bool = False,
    ):
        sparse_files = sorted(glob.glob(os.path.join(sparse_dir, "*.npz")))
        gt_files = sorted(glob.glob(os.path.join(gt_dir, "*.vtp")))
        if len(sparse_files) == 0 or len(gt_files) == 0:
            raise RuntimeError(f"Empty data: sparse={len(sparse_files)} gt={len(gt_files)}")
        if len(sparse_files) != len(gt_files):
            raise RuntimeError(f"Count mismatch: sparse={len(sparse_files)} gt={len(gt_files)}")

        self.samples = []
        self.label_dim = int(label_dim)
        self.label_scale = float(label_scale)
        self.no_label_feat = bool(no_label_feat)

        for sp_file, gt_file in zip(sparse_files, gt_files):
            sp_xyz, sp_pairs = load_sparse_npz_points_and_pairs(sp_file, pcs_key=pcs_key, pairs_key=spc_pairs_key)
            sp_feat = build_spc_feat_bn6(sp_xyz) if self.no_label_feat else build_spc_feat_bn8(sp_xyz, sp_pairs, label_scale=self.label_scale)

            gt_xyz, gt_pairs = load_vtp_points_and_labels(gt_file, gt_pair_array, min_dim=2)
            gt_xyz_norm = pc_normalize_xyz(sp_xyz, gt_xyz).astype(np.float32)

            if self.label_dim == 3:
                _, gt_lab3 = load_vtp_points_and_labels(gt_file, gt_label3_array, min_dim=3)
            else:
                gt_lab3 = None

            self.samples.append((
                sp_feat.astype(np.float32),       # (Ns,8) or (Ns,6)
                gt_xyz_norm.astype(np.float32),   # (Na,3)
                gt_pairs.astype(np.int64),        # (Na,2)
                (gt_lab3.astype(np.int64) if gt_lab3 is not None else None),  # (Na,3) or None
            ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_batch(batch):
    sp_feat = torch.from_numpy(np.stack([b[0] for b in batch], axis=0))  # (B,Ns,8 or 6)
    gt_xyz = torch.from_numpy(np.stack([b[1] for b in batch], axis=0))   # (B,Na,3)
    gt_pairs = torch.from_numpy(np.stack([b[2] for b in batch], axis=0)) # (B,Na,2)
    has_lab3 = (batch[0][3] is not None)
    gt_lab3 = torch.from_numpy(np.stack([b[3] for b in batch], axis=0)) if has_lab3 else None
    return sp_feat, gt_xyz, gt_pairs, gt_lab3
