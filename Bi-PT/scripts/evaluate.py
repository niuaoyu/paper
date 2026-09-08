#!/usr/bin/env python3
"""
Bi-PT evaluation: geometric + mesh-quality metrics (parallel).

Parallel evaluation for Bi-PT predicted meshes (.vtp) vs GT (.vtp). Consumes the
``<case>_pred_l*.vtp`` files written by ``scripts/infer.py`` and reports the paper
Table-1 metrics: CD, EMD, P2F, NC, non-manifold vertex/edge/face ratios (NM-F is
the paper's ENF), and SI (self-intersection ratio).

Key behavior (label-driven, chamber-only):
  - --chambers "4,5,6" evaluates ONLY those chambers (no implicit "all").
  - Supports scalar (N,), pair (N,2), or triple (N,3) label arrays.
  - Submesh extraction is label-driven (point- or cell-labels).

IMPORTANT CHANGE (per your request):
  - NO surface sampling at all.
  - CD / NC / P2F are computed on the extracted geometry’s VERTICES (optionally deterministic subsampled).

Adds mesh quality metrics (per your code logic), for EVERY row (case/level/chamber or full mesh):
  - NM-V: non-manifold vertices ratio (Open3D)
  - NM-E: non-manifold edges ratio    (Open3D)
  - NM-F: non-manifold faces ratio    (adjacent face normal dot < 0 count / #faces)
  - SI  : self-intersection ratio     (torch-mesh-isect BVH collisions / #faces)

These are written into BOTH CSV and result.txt, and grouped by level/chamber.
"""

import os
import re
import csv
import math
import argparse
from typing import Dict, List, Tuple, Set, Union, Optional

import numpy as np
from scipy.spatial import cKDTree as KDTree
from scipy.optimize import linear_sum_assignment

import trimesh
import open3d as o3d

import vtk
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed


# -----------------------------
# VTP IO (VTK -> triangulated polydata -> trimesh)
# -----------------------------

def read_vtp_polydata(path: str) -> vtk.vtkPolyData:
    """Read a .vtp file into vtkPolyData."""
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    r = vtk.vtkXMLPolyDataReader()
    r.SetFileName(path)
    r.Update()
    poly = r.GetOutput()
    if poly is None or poly.GetNumberOfPoints() == 0:
        raise RuntimeError(f"Empty polydata: {path}")
    return poly


def triangulate_polydata(poly: vtk.vtkPolyData) -> vtk.vtkPolyData:
    """Ensure the surface consists of triangles."""
    tri = vtk.vtkTriangleFilter()
    tri.SetInputData(poly)
    tri.Update()
    out = tri.GetOutput()
    if out is None or out.GetNumberOfCells() == 0:
        raise RuntimeError("Triangulation produced empty mesh.")
    return out


def vtk_poly_to_trimesh(poly_tri: vtk.vtkPolyData) -> Tuple[trimesh.Trimesh, np.ndarray]:
    """
    Convert triangulated vtkPolyData -> trimesh.Trimesh (vertices + faces).
    Returns (mesh, faces_np) where faces_np is (F,3) vertex indices.
    """
    pts = vtk_to_numpy(poly_tri.GetPoints().GetData()).astype(np.float32)

    polys = poly_tri.GetPolys()
    polys.InitTraversal()
    idlist = vtk.vtkIdList()
    faces: List[List[int]] = []
    while polys.GetNextCell(idlist):
        if idlist.GetNumberOfIds() == 3:
            faces.append([idlist.GetId(0), idlist.GetId(1), idlist.GetId(2)])

    faces_np = np.asarray(faces, dtype=np.int64)
    if faces_np.size == 0:
        raise RuntimeError("No triangle faces found after triangulation.")
    mesh = trimesh.Trimesh(vertices=pts, faces=faces_np, process=False)
    return mesh, faces_np


def trimesh_to_vtk_poly_tri(mesh: trimesh.Trimesh) -> vtk.vtkPolyData:
    """Convert trimesh (triangles) -> vtkPolyData (triangles)."""
    poly = vtk.vtkPolyData()

    v = np.asarray(mesh.vertices, dtype=np.float32)
    pts = vtk.vtkPoints()
    pts.SetData(numpy_to_vtk(v, deep=True))
    poly.SetPoints(pts)

    f = np.asarray(mesh.faces, dtype=np.int64)
    cells = vtk.vtkCellArray()
    cells.Allocate(len(f))
    for (i0, i1, i2) in f:
        tri = vtk.vtkTriangle()
        tri.GetPointIds().SetId(0, int(i0))
        tri.GetPointIds().SetId(1, int(i1))
        tri.GetPointIds().SetId(2, int(i2))
        cells.InsertNextCell(tri)
    poly.SetPolys(cells)
    return poly


# -----------------------------
# Label extraction + submesh selection
# -----------------------------

def get_label_array(poly_tri: vtk.vtkPolyData, array_name: str, label_on: str) -> np.ndarray:
    """
    Read label array from vtkPolyData.
    label_on: "point" or "cell"
    Returns numpy array:
      - point labels: (V, k) or (V,)
      - cell labels : (F, k) or (F,)
    """
    if label_on == "point":
        arr = poly_tri.GetPointData().GetArray(array_name)
        if arr is None:
            raise KeyError(f"PointData missing array '{array_name}'")
        return vtk_to_numpy(arr)
    if label_on == "cell":
        arr = poly_tri.GetCellData().GetArray(array_name)
        if arr is None:
            raise KeyError(f"CellData missing array '{array_name}'")
        return vtk_to_numpy(arr)
    raise ValueError(f"Invalid label_on: {label_on}")


def membership_from_labels(labels: np.ndarray, chamber_id: int) -> np.ndarray:
    """
    labels can be:
      - (N,) scalar
      - (N,2) pair
      - (N,3) triple
    Returns membership mask: True if chamber_id appears in the label tuple.
    """
    lab = np.asarray(labels)
    if lab.ndim == 1:
        return (lab.astype(np.int64) == int(chamber_id))
    return np.any(lab.astype(np.int64) == int(chamber_id), axis=1)


def extract_submesh_by_chamber(
    mesh: trimesh.Trimesh,
    faces_np: np.ndarray,
    labels: np.ndarray,
    chamber_id: int,
    label_on: str
) -> trimesh.Trimesh:
    """
    Chamber-associated submesh.

    - label_on == "point": keep faces whose 3 vertices are ALL members of chamber_id
    - label_on == "cell" : keep faces whose cell-label tuple contains chamber_id
    """
    if label_on == "cell":
        face_keep = membership_from_labels(labels, chamber_id)  # (F,)
        if face_keep.shape[0] != faces_np.shape[0]:
            raise RuntimeError("Cell label count != face count after triangulation.")
    else:
        v_keep = membership_from_labels(labels, chamber_id)  # (V,)
        if v_keep.shape[0] != mesh.vertices.shape[0]:
            raise RuntimeError("Point label count != vertex count after triangulation.")
        face_keep = v_keep[faces_np].all(axis=1)

    faces_sub = faces_np[face_keep]
    if faces_sub.shape[0] == 0:
        return trimesh.Trimesh(vertices=np.zeros((0, 3), dtype=np.float32),
                               faces=np.zeros((0, 3), dtype=np.int64),
                               process=False)

    sub = trimesh.Trimesh(vertices=mesh.vertices, faces=faces_sub, process=False)
    sub.remove_unreferenced_vertices()
    return sub


# -----------------------------
# Deterministic seeds + subsampling (NO surface sampling)
# -----------------------------

def stable_task_seed(base_seed: int, case: str, level: str, chamber: Union[str, int]) -> int:
    b = (str(case) + "|" + str(level) + "|" + str(chamber)).encode("utf-8")
    s = int(np.frombuffer(b, dtype=np.uint8).sum())
    return int(base_seed + 1000 * s)


def subsample_rows(X: np.ndarray, max_n: Optional[int], rng: np.random.Generator) -> np.ndarray:
    """
    Deterministically subsample rows of X to at most max_n.
    If max_n is None or X is already small, returns X.
    """
    if max_n is None:
        return X
    n = int(X.shape[0])
    if n <= max_n:
        return X
    idx = rng.choice(n, size=int(max_n), replace=False)
    return X[idx]


# -----------------------------
# Metrics: CD / EMD / NC (vertex-based)
# -----------------------------

def chamfer_l2_symmetric_vertices(A: np.ndarray, B: np.ndarray, cd_sqrt: bool = False) -> float:
    """
    Symmetric Chamfer on vertex point sets.

    If cd_sqrt == False (default): mean squared NN
        mean_{a in A} ||a-NN_B(a)||^2 + mean_{b in B} ||b-NN_A(b)||^2

    If cd_sqrt == True: RMSE per direction then sum
        sqrt(mean_{a} ||a-NN_B(a)||^2) + sqrt(mean_{b} ||b-NN_A(b)||^2)
    """
    if A.shape[0] == 0 or B.shape[0] == 0:
        return float("nan")

    treeB = KDTree(B)
    dA, _ = treeB.query(A, k=1)  # Euclidean distances
    treeA = KDTree(A)
    dB, _ = treeA.query(B, k=1)

    if cd_sqrt:
        d1 = float(np.sqrt(np.mean(dA ** 2)))
        d2 = float(np.sqrt(np.mean(dB ** 2)))
        return d1 + d2
    else:
        # NOTE: this follows your current implementation (kept unchanged)
        return float(0.5 * np.mean(dA) + 0.5 * np.mean(dB))


def emd_hungarian(A: np.ndarray, B: np.ndarray) -> float:
    """
    EMD via Hungarian assignment (mean L2 distance).
    O(n^3). Use small n.
    """
    if A.shape[0] == 0 or B.shape[0] == 0:
        return float("nan")
    D = np.linalg.norm(A[:, None, :] - B[None, :, :], axis=-1)  # (n,n)
    r, c = linear_sum_assignment(D)
    return float(D[r, c].mean())


def normal_consistency_vertices(
    A_pts: np.ndarray, A_nrm: np.ndarray,
    B_pts: np.ndarray, B_nrm: np.ndarray
) -> float:
    """
    Symmetric NN-based cosine similarity of vertex normals.
    """
    if A_pts.shape[0] == 0 or B_pts.shape[0] == 0:
        return float("nan")

    treeB = KDTree(B_pts)
    _, idxB = treeB.query(A_pts, k=1)
    cos1 = np.sum(A_nrm * B_nrm[idxB], axis=1) / (
        (np.linalg.norm(A_nrm, axis=1) + 1e-12) * (np.linalg.norm(B_nrm[idxB], axis=1) + 1e-12)
    )

    treeA = KDTree(A_pts)
    _, idxA = treeA.query(B_pts, k=1)
    cos2 = np.sum(B_nrm * A_nrm[idxA], axis=1) / (
        (np.linalg.norm(B_nrm, axis=1) + 1e-12) * (np.linalg.norm(A_nrm[idxA], axis=1) + 1e-12)
    )

    return float(0.5 * (cos1.mean() + cos2.mean()))


# -----------------------------
# P2F (vertex-to-surface, VTK implicit distance) — NO sampling
# -----------------------------

def p2f_vertices_to_surface_sq(pts: np.ndarray, surface_poly_tri: vtk.vtkPolyData) -> float:
    """
    Mean vertex-to-surface distance (signed distance abs), averaged.
    (Kept as your current implementation; name still has "_sq" to avoid touching other code.)
    """
    n = int(pts.shape[0])
    if n == 0:
        return float("nan")

    ipd = vtk.vtkImplicitPolyDataDistance()
    ipd.SetInput(surface_poly_tri)

    acc = 0.0
    for p in pts:
        d = float(ipd.EvaluateFunction([float(p[0]), float(p[1]), float(p[2])]))
        acc += abs(d)
    return float(acc / max(n, 1))


def p2f_symmetric_vertices_sq(
    pred_mesh: trimesh.Trimesh,
    gt_mesh: trimesh.Trimesh,
    pred_pts: np.ndarray,
    gt_pts: np.ndarray,
) -> float:
    """
    Symmetric mean vertex-to-surface:
      mean_{v in pred_pts} dist(v, GT surface) + mean_{u in gt_pts} dist(u, Pred surface)
    """
    if pred_mesh.faces.shape[0] == 0 or gt_mesh.faces.shape[0] == 0:
        return float("nan")

    pred_poly = trimesh_to_vtk_poly_tri(pred_mesh)
    gt_poly   = trimesh_to_vtk_poly_tri(gt_mesh)

    dP = p2f_vertices_to_surface_sq(pred_pts, gt_poly)
    dG = p2f_vertices_to_surface_sq(gt_pts, pred_poly)
    return float(0.5 * dP + 0.5 * dG)


# -----------------------------
# Mesh quality: NM-V / NM-E / NM-F / SI  (your exact logic)
# -----------------------------

def calculate_non_manifold_edge_vertex_trimesh(gen_mesh: trimesh.Trimesh) -> Tuple[int, int]:
    """
    nm_edges: Open3D get_non_manifold_edges(allow_boundary_edges=False)
    nm_vertices: Open3D get_non_manifold_vertices()
    """
    gen_v = np.asarray(gen_mesh.vertices)
    gen_f = np.asarray(gen_mesh.faces)

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(gen_v)
    mesh.triangles = o3d.utility.Vector3iVector(gen_f)

    nm_edges = np.asarray(mesh.get_non_manifold_edges(allow_boundary_edges=False))
    nm_vertices = np.asarray(mesh.get_non_manifold_vertices())
    return int(nm_vertices.shape[0]), int(nm_edges.shape[0])


def calculate_non_manifold_face_trimesh(gen_mesh: trimesh.Trimesh) -> int:
    """
    NM-F count via adjacent face normal dot < 0 (count of face-adjacency pairs whose
    normals point in opposing directions).
    """
    f_adj = gen_mesh.face_adjacency
    fn = gen_mesh.face_normals

    nm_faces_count = 0
    if f_adj is not None and f_adj.shape[0] > 0:
        count = 0
        for i in range(f_adj.shape[0]):
            if fn[f_adj[i, 0]] @ fn[f_adj[i, 1]] < 0:
                count += 1
        nm_faces_count = int(count)
    return nm_faces_count


def si_bvh_trimesh(gen_mesh: trimesh.Trimesh, max_collisions: int = 8) -> Tuple[int, int, int, float]:
    """
    Self-intersection via torch-mesh-isect BVH (folded in from si.py).

    Returns (V, F, collisions, SI=collisions/F). Requires CUDA and the
    ``mesh_intersection`` extension; raises otherwise so the caller can record NaN.
    Robust to the single-collision case where ``squeeze()`` yields a 1-D array.
    """
    if gen_mesh.faces is None or gen_mesh.faces.shape[0] == 0:
        raise RuntimeError("Mesh has no faces.")

    import torch
    from mesh_intersection.bvh_search_tree import BVH

    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() is False (CUDA required for BVH SI).")

    vertices = torch.tensor(np.asarray(gen_mesh.vertices), dtype=torch.float32).cuda()
    faces = torch.tensor(gen_mesh.faces.astype(np.int64), dtype=torch.long).cuda()
    triangles = vertices[faces].unsqueeze(dim=0).contiguous()

    m = BVH(max_collisions=int(max_collisions))
    outputs = m(triangles).detach().cpu().numpy().squeeze()

    collisions_count = 0
    if outputs.ndim == 2 and outputs.shape[1] >= 2:
        collisions = outputs[outputs[:, 0] >= 0, :]
        collisions_count = int(collisions.shape[0])

    V = int(gen_mesh.vertices.shape[0])
    F = int(gen_mesh.faces.shape[0])
    SI = float(collisions_count / max(F, 1))
    return V, F, collisions_count, SI


def mesh_quality_ratios(
    gen_mesh: trimesh.Trimesh,
    compute_si: bool,
    max_collisions: int = 8,
    compute_nmve: bool = True,
) -> Dict[str, float]:
    """
    Returns ratios (not counts):
      NM-V = nm_vertices / nv   (Open3D; NaN if compute_nmve is False)
      NM-E = nm_edges    / ne   (Open3D; NaN if compute_nmve is False)
      NM-F = nm_faces    / nf
      SI   = collisions  / nf   (if compute_si and BVH/CUDA available, else NaN)

    ``compute_nmve=False`` skips the Open3D non-manifold vertex/edge call. That call
    is a native routine that can hard-crash (SIGSEGV) on some Open3D builds — a crash
    that Python ``try/except`` cannot catch, so it takes the whole worker down. Disable
    it (``--no-nmve``) to run the rest of the metrics on such environments.
    """
    if gen_mesh.faces is None or gen_mesh.faces.shape[0] == 0:
        return {"NM-V": float("nan"), "NM-E": float("nan"), "NM-F": float("nan"), "SI": float("nan")}

    nv = int(gen_mesh.vertices.shape[0])
    ne = int(gen_mesh.edges.shape[0]) if gen_mesh.edges is not None else 0
    nf = int(gen_mesh.faces.shape[0])

    # Open3D edge/vertex non-manifold (optional — can segfault on some builds)
    nm_v_cnt = nm_e_cnt = None
    if compute_nmve:
        try:
            nm_v_cnt, nm_e_cnt = calculate_non_manifold_edge_vertex_trimesh(gen_mesh)
        except Exception:
            nm_v_cnt, nm_e_cnt = 0, 0

    # Adjacent-face flips (NM-F)
    nm_f_cnt = calculate_non_manifold_face_trimesh(gen_mesh)

    # Self-intersection (SI) via BVH — only when requested; NaN on any failure.
    si = float("nan")
    if compute_si:
        try:
            _, _, _isect_cnt, si = si_bvh_trimesh(gen_mesh, max_collisions=max_collisions)
        except Exception:
            si = float("nan")

    # safe denominators
    nv_d = max(nv, 1)
    ne_d = max(ne, 1)
    nf_d = max(nf, 1)

    nmv = float(nm_v_cnt / nv_d) if nm_v_cnt is not None else float("nan")
    nme = float(nm_e_cnt / ne_d) if nm_e_cnt is not None else float("nan")
    nmf = float(nm_f_cnt / nf_d)

    return {"NM-V": nmv, "NM-E": nme, "NM-F": nmf, "SI": si}


# -----------------------------
# Evaluate one (case, level, chamber)
# -----------------------------

_METRIC_KEYS = ["CD", "EMD", "P2F", "NC", "NM-V", "NM-E", "NM-F", "SI"]
ChamberSpec = Union[str, int]  # "all" or an int chamber id


def evaluate_case_level_chamber(
    case: str,
    level: str,
    chamber: ChamberSpec,
    pred_root: str,
    gt_root: str,
    gt_suffix: str,
    seed: int,
    emd_samples: int,
    compute_emd_flag: bool,
    compute_si: bool,
    label_array: str,
    label_on: str,
    cd_sqrt: bool,
    max_points: Optional[int],
    max_p2f_points: Optional[int],
    max_collisions: int = 8,
    compute_nmve: bool = True,
) -> Dict[str, float]:
    """
    Evaluate metrics for one case/level/chamber.
    - chamber == "all": full mesh
    - else: label-driven submesh
    All geometry metrics are computed on (optionally subsampled) VERTICES (no surface sampling).
    Mesh quality is computed on the same extracted pred submesh.
    """
    pred_path = os.path.join(pred_root, f"{case}_pred_{level}.vtp")
    if not os.path.isfile(pred_path):
        raise FileNotFoundError(pred_path)

    gt_candidates = [
        os.path.join(gt_root, f"{case}{gt_suffix}"),
        os.path.join(gt_root, f"{case}.vtp"),
        os.path.join(gt_root, f"{case}_gt.vtp"),
    ]
    gt_path = None
    for p in gt_candidates:
        if os.path.isfile(p):
            gt_path = p
            break
    if gt_path is None:
        raise FileNotFoundError(f"GT not found for {case}. Tried: {gt_candidates}")

    # Load + triangulate
    pred_poly_tri = triangulate_polydata(read_vtp_polydata(pred_path))
    gt_poly_tri   = triangulate_polydata(read_vtp_polydata(gt_path))

    pred_tm, pred_faces = vtk_poly_to_trimesh(pred_poly_tri)
    gt_tm, gt_faces     = vtk_poly_to_trimesh(gt_poly_tri)

    # Optional submesh extraction
    if chamber != "all":
        ch_id = int(chamber)
        pred_labels = get_label_array(pred_poly_tri, label_array, label_on)
        gt_labels   = get_label_array(gt_poly_tri,   label_array, label_on)
        pred_tm = extract_submesh_by_chamber(pred_tm, pred_faces, pred_labels, ch_id, label_on)
        gt_tm   = extract_submesh_by_chamber(gt_tm,   gt_faces,   gt_labels,   ch_id, label_on)

    out: Dict[str, float] = {"case": case, "level": level, "chamber": str(chamber)}

    # Empty check
    if pred_tm.vertices.shape[0] == 0 or gt_tm.vertices.shape[0] == 0:
        out.update({k: float("nan") for k in _METRIC_KEYS})
        return out

    # Deterministic per-task RNG for subsampling
    task_seed = stable_task_seed(seed, case, level, chamber)
    rng = np.random.default_rng(task_seed)

    # Vertex points + vertex normals (NO sampling)
    P_all = np.asarray(pred_tm.vertices, dtype=np.float32)
    G_all = np.asarray(gt_tm.vertices, dtype=np.float32)
    Pn_all = np.asarray(pred_tm.vertex_normals, dtype=np.float32)
    Gn_all = np.asarray(gt_tm.vertex_normals, dtype=np.float32)

    # For CD/NC, optionally subsample to max_points
    P = subsample_rows(P_all, max_points, rng)
    G = subsample_rows(G_all, max_points, rng)

    # If we subsampled points, we should subsample normals with same indices.
    def subsample_with_idx(X: np.ndarray, N: np.ndarray, max_n: Optional[int], rng2: np.random.Generator):
        if max_n is None or X.shape[0] <= max_n:
            return X, N
        idx = rng2.choice(int(X.shape[0]), size=int(max_n), replace=False)
        return X[idx], N[idx]

    rng_nc = np.random.default_rng(task_seed + 17)
    P_nc, Pn = subsample_with_idx(P_all, Pn_all, max_points, rng_nc)
    G_nc, Gn = subsample_with_idx(G_all, Gn_all, max_points, rng_nc)

    # CD (vertex-based)
    out["CD"] = chamfer_l2_symmetric_vertices(P, G, cd_sqrt=cd_sqrt)

    # EMD (vertex-based, subsample to emd_samples and equal sizes)
    if compute_emd_flag:
        n = min(int(emd_samples), int(P_all.shape[0]), int(G_all.shape[0]))
        if n < 2:
            out["EMD"] = float("nan")
        else:
            rng_e = np.random.default_rng(task_seed + 33)
            P_e = subsample_rows(P_all, n, rng_e)
            G_e = subsample_rows(G_all, n, rng_e)
            out["EMD"] = emd_hungarian(P_e, G_e)
    else:
        out["EMD"] = float("nan")

    # NC
    out["NC"] = normal_consistency_vertices(P_nc, Pn, G_nc, Gn)

    # P2F (vertex-to-surface, optionally subsample vertices for speed)
    rng_p2f = np.random.default_rng(task_seed + 55)
    P_p2f = subsample_rows(P_all, max_p2f_points, rng_p2f)
    G_p2f = subsample_rows(G_all, max_p2f_points, rng_p2f)
    out["P2F"] = p2f_symmetric_vertices_sq(pred_tm, gt_tm, P_p2f, G_p2f)

    # Mesh quality on pred submesh
    q = mesh_quality_ratios(pred_tm, compute_si=compute_si, max_collisions=max_collisions,
                            compute_nmve=compute_nmve)
    out["NM-V"] = q["NM-V"]
    out["NM-E"] = q["NM-E"]
    out["NM-F"] = q["NM-F"]
    out["SI"]   = q["SI"]

    return out


# -----------------------------
# Prediction discovery + parsing
# -----------------------------

_PRED_PAT = re.compile(r"^(.+)_pred_(l\d+)\.vtp$")


def _level_key(level: str) -> int:
    m = re.match(r"^l(\d+)$", level)
    return int(m.group(1)) if m else 10**9


def discover_cases_and_levels(pred_root: str) -> Tuple[List[str], List[str], Dict[str, Set[str]]]:
    """
    Scan pred_root for files: <case>_pred_l<number>.vtp
    """
    case_to_levels: Dict[str, Set[str]] = {}
    levels: Set[str] = set()
    for fn in os.listdir(pred_root):
        m = _PRED_PAT.match(fn)
        if not m:
            continue
        case, level = m.group(1), m.group(2)
        levels.add(level)
        case_to_levels.setdefault(case, set()).add(level)
    cases_sorted = sorted(case_to_levels.keys())
    levels_sorted = sorted(list(levels), key=_level_key)
    return cases_sorted, levels_sorted, case_to_levels


def parse_which(which: str, available_levels: List[str]) -> List[str]:
    w = which.strip().lower()
    if w == "all":
        return list(available_levels)
    parts = which.split(",")
    out: List[str] = []
    for p in parts:
        p2 = p.strip().lower()
        if not p2:
            continue
        if not re.match(r"^l\d+$", p2):
            raise ValueError(f"Invalid level in --which: '{p}'. Expected like l1,l2,... or all.")
        out.append(p2)
    if not out:
        raise ValueError(f"Invalid --which: {which}")
    return out


def parse_chambers(chambers: str) -> List[ChamberSpec]:
    s = chambers.strip().lower()
    if s == "all":
        return ["all"]
    parts = chambers.split(",")
    out: List[ChamberSpec] = []
    for p in parts:
        t = p.strip().lower()
        if not t:
            continue
        if t == "all":
            out.append("all")
            continue
        if not re.match(r"^\d+$", t):
            raise ValueError(f"Invalid chamber id '{p}'. Use like 4,5,6 or all.")
        out.append(int(t))
    if not out:
        raise ValueError(f"Invalid --chambers: {chambers}")
    return out


# -----------------------------
# Summary helpers
# -----------------------------

def mean_ignore_nan(vals: List[float]) -> float:
    good = [float(x) for x in vals if not math.isnan(float(x))]
    return float("nan") if len(good) == 0 else float(np.mean(good))


def dataset_means(rows: List[Dict[str, float]]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k in _METRIC_KEYS:
        out[k] = mean_ignore_nan([float(r.get(k, float("nan"))) for r in rows])
    return out


def group_means(rows: List[Dict[str, float]], group_key: str) -> Dict[str, Dict[str, float]]:
    groups: Dict[str, List[Dict[str, float]]] = {}
    for r in rows:
        g = str(r.get(group_key, ""))
        groups.setdefault(g, []).append(r)

    out: Dict[str, Dict[str, float]] = {}
    for g, rs in groups.items():
        out[g] = {}
        for k in _METRIC_KEYS:
            out[g][k] = mean_ignore_nan([float(r.get(k, float("nan"))) for r in rs])
    return out


def write_summary_txt(path: str,
                      args: argparse.Namespace,
                      rows_ok: List[Dict[str, float]],
                      n_tasks: int,
                      n_errors: int) -> None:
    overall = dataset_means(rows_ok)
    by_level = group_means(rows_ok, "level")
    by_ch = group_means(rows_ok, "chamber")

    lines: List[str] = []
    lines.append("Bi-PT evaluation summary (VERTEX-BASED; NO surface sampling)")
    lines.append("")
    lines.append(f"pred-root      : {args.pred_root}")
    lines.append(f"gt-root        : {args.gt_root}")
    lines.append(f"which          : {args.which}")
    lines.append(f"chambers       : {args.chambers}")
    lines.append(f"label-array    : {args.label_array}")
    lines.append(f"label-on       : {args.label_on}")
    lines.append(f"cd-sqrt        : {args.cd_sqrt}")
    lines.append(f"max-points     : {args.max_points}")
    lines.append(f"max-p2f-points : {args.max_p2f_points}")
    lines.append("")
    lines.append(f"tasks          : {n_tasks}  (ok={len(rows_ok)}, failed={n_errors})")
    lines.append("")
    lines.append("Parallel:")
    lines.append(f"  backend : {args.backend}")
    lines.append(f"  workers : {args.workers}")
    lines.append("")
    lines.append("=== Overall mean (all evaluated rows) ===")
    for k in _METRIC_KEYS:
        lines.append(f"{k}: {overall[k]:.6g}")
    lines.append("")
    lines.append("=== Mean by level ===")
    for lvl in sorted(by_level.keys(), key=_level_key):
        lines.append(f"[{lvl}] " + "  ".join([f"{k}={by_level[lvl][k]:.6g}" for k in _METRIC_KEYS]))
    lines.append("")
    lines.append("=== Mean by chamber ===")
    ch_keys = list(by_ch.keys())
    if "all" in ch_keys:
        ch_keys.remove("all")
        ch_keys = ["all"] + sorted(ch_keys, key=lambda x: int(x) if x.isdigit() else 10**9)
    else:
        ch_keys = sorted(ch_keys, key=lambda x: int(x) if x.isdigit() else 10**9)

    for ch in ch_keys:
        lines.append(f"[ch={ch}] " + "  ".join([f"{k}={by_ch[ch][k]:.6g}" for k in _METRIC_KEYS]))

    out_dir = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(out_dir, exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


# -----------------------------
# Main
# -----------------------------

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--pred-root", required=True, help="Folder with <case>_pred_l*.vtp")
    ap.add_argument("--gt-root", required=True, help="Folder with GT .vtp meshes")
    ap.add_argument("--gt-suffix", default=".vtp", help="GT filename suffix (default .vtp)")

    ap.add_argument("--which", default="l2",
                    help="Prediction levels: l2 | l1,l2,l3 | all (default l2)")
    ap.add_argument("--chambers", default="all",
                    help="Chambers to evaluate: 4,5,6 or all or all,4,5 (default all)")

    ap.add_argument("--label-array", default="VertexBoundaryLabels",
                    help="Label array name in .vtp (default VertexLabels3/VertexBoundaryLabels)")
    ap.add_argument("--label-on", choices=["point", "cell"], default="point",
                    help="Where labels are stored: point or cell (default point)")

    ap.add_argument("--seed", type=int, default=42)

    # Vertex-based controls
    ap.add_argument("--cd-sqrt", action="store_true",
                    help="If set, CD becomes RMSE(A->B)+RMSE(B->A) instead of mean-squared sum.")
    ap.add_argument("--max-points", type=int, default=None,
                    help="Max vertices used for CD/NC (deterministic subsample). Default: use all.")
    ap.add_argument("--max-p2f-points", type=int, default=None,
                    help="Max vertices used for P2F (deterministic subsample). Default: use all.")

    ap.add_argument("--emd-samples", type=int, default=500,
                    help="Max vertices for EMD per task (equal-sized subsample; default 500)")
    ap.add_argument("--no-emd", action="store_true", help="Skip EMD (much faster)")

    ap.add_argument("--workers", type=int, default=8, help="Parallel workers (default 8)")
    ap.add_argument("--backend", choices=["thread", "process"], default="process",
                    help="Parallel backend: thread or process (default process)")

    ap.add_argument("--compute-si", action="store_true",
                    help="Compute SI with BVH (requires CUDA + mesh_intersection)")
    ap.add_argument("--force-si", action="store_true",
                    help="Force SI even with process workers>1 (not recommended)")
    ap.add_argument("--max-collisions", type=int, default=8,
                    help="BVH max_collisions per triangle for the SI computation (default 8).")
    ap.add_argument("--no-nmve", action="store_true",
                    help="Skip the Open3D non-manifold vertex/edge metrics (NM-V/NM-E -> NaN). "
                         "That native call can hard-crash the worker on some Open3D builds "
                         "(e.g. certain HPC/cluster installs); use this to run everything else.")

    ap.add_argument("--out-csv", required=True, help="Output CSV path")
    ap.add_argument("--out-summary", default=None,
                    help="Summary txt path (default: result.txt next to out-csv)")

    args = ap.parse_args()

    cases, avail_levels, case_to_levels = discover_cases_and_levels(args.pred_root)
    if len(cases) == 0:
        raise RuntimeError(f"No prediction files found in {args.pred_root} matching '*_pred_l*.vtp'")

    selected_levels = [l for l in parse_which(args.which, avail_levels) if l in avail_levels]
    if len(selected_levels) == 0:
        raise RuntimeError(f"--which resolved to no available levels. Available: {avail_levels}")

    selected_chambers = parse_chambers(args.chambers)

    # Build tasks ONLY for requested chambers (no implicit "all")
    tasks: List[Tuple[str, str, ChamberSpec]] = []
    for c in cases:
        lvls = case_to_levels.get(c, set())
        for l in selected_levels:
            if l not in lvls:
                continue
            for ch in selected_chambers:
                tasks.append((c, l, ch))

    if len(tasks) == 0:
        raise RuntimeError("No tasks to run. Check --which/--chambers and pred filenames.")

    # SI safety: default-disable SI for multiprocessing with workers>1
    compute_si = bool(args.compute_si)
    if compute_si and args.backend == "process" and args.workers > 1 and not args.force_si:
        print("[INFO] Disabling SI because --backend process with --workers>1 can conflict with CUDA.")
        print("       If you really want SI in parallel, rerun with --force-si (not recommended).")
        compute_si = False

    Executor = ThreadPoolExecutor if args.backend == "thread" else ProcessPoolExecutor

    common_kwargs = dict(
        pred_root=args.pred_root,
        gt_root=args.gt_root,
        gt_suffix=args.gt_suffix,
        seed=int(args.seed),
        emd_samples=int(args.emd_samples),
        compute_emd_flag=(not args.no_emd),
        compute_si=compute_si,
        label_array=args.label_array,
        label_on=args.label_on,
        cd_sqrt=bool(args.cd_sqrt),
        max_points=(None if args.max_points is None else int(args.max_points)),
        max_p2f_points=(None if args.max_p2f_points is None else int(args.max_p2f_points)),
        max_collisions=int(args.max_collisions),
        compute_nmve=(not args.no_nmve),
    )

    print(f"[INFO] Cases: {len(cases)} | Available levels: {avail_levels}")
    print(f"[INFO] Selected levels  : {selected_levels}")
    print(f"[INFO] Selected chambers: {selected_chambers}  (NO implicit 'all')")
    print(f"[INFO] Tasks: {len(tasks)} | backend={args.backend} | workers={args.workers}")
    print("[INFO] Metrics are VERTEX-BASED (NO surface sampling).")
    print(f"[INFO] CD sqrt mode: {args.cd_sqrt}  | max_points={args.max_points}  | max_p2f_points={args.max_p2f_points}")
    if args.no_emd:
        print("[INFO] EMD disabled.")
    print(f"[INFO] SI {'enabled' if compute_si else 'disabled'} (SI will be NaN if disabled).")

    rows: List[Dict[str, float]] = []
    errors = 0

    with Executor(max_workers=int(args.workers)) as ex:
        futs = []
        for case, level, chamber in tasks:
            futs.append(ex.submit(evaluate_case_level_chamber, case, level, chamber, **common_kwargs))

        for fut in as_completed(futs):
            try:
                r = fut.result()
                rows.append(r)
                print(f"[OK] {r['case']} {r['level']} ch={r['chamber']}: "
                      f"CD={r['CD']:.6g} EMD={r['EMD']:.6g} P2F={r['P2F']:.6g} NC={r['NC']:.6g} "
                      f"NM-V={r['NM-V']:.6g} NM-E={r['NM-E']:.6g} NM-F={r['NM-F']:.6g} SI={r['SI']:.6g}")
            except Exception as e:
                errors += 1
                print(f"[WARN] task failed: {e}")

    # Write CSV
    fieldnames = ["case", "level", "chamber"] + _METRIC_KEYS
    out_dir = os.path.dirname(os.path.abspath(args.out_csv)) or "."
    os.makedirs(out_dir, exist_ok=True)

    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            row_out = {k: r.get(k, "") for k in fieldnames}
            w.writerow(row_out)

    # Print overall means (over evaluated rows only)
    overall = dataset_means(rows)
    print("\n=== Dataset mean (overall; evaluated rows only) ===")
    for k in _METRIC_KEYS:
        print(f"{k}: {overall[k]:.6g}")

    # Write summary.txt next to CSV (unless user overrides)
    summary_path = args.out_summary
    if summary_path is None:
        summary_path = os.path.join(out_dir, "result.txt")
    write_summary_txt(summary_path, args, rows, len(tasks), errors)

    print(f"\n[OK] Wrote CSV    : {args.out_csv}")
    print(f"[OK] Wrote summary: {summary_path}")
    if errors:
        print(f"[INFO] {errors} tasks failed (see warnings above).")


if __name__ == "__main__":
    main()