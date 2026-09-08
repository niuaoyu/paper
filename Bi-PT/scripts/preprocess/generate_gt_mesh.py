#!/usr/bin/env python3
# Copyright (C) 2022 Fanwei Kong, Shawn C. Shadden, University of California, Berkeley
# Licensed under the Apache License, Version 2.0 (the "License");
#
# NOTE (Bi-PT integration): this ground-truth mesh generator depends on external
# utilities from the SurfaceNets / HeartDeformNet codebase (Kong & Shadden):
#   vtk_utils.vtk_utils, utils, pre_process
# Point BIPT_MESH_EXTERNAL / BIPT_MESH_SRC at those folders, or add them to
# PYTHONPATH. The original ../external and ../src relative paths are kept as a
# fallback. See scripts/preprocess/README.md for details.

import os
import sys

# Optional explicit locations for the external modules (override if needed).
_ext = os.environ.get("BIPT_MESH_EXTERNAL")
_src = os.environ.get("BIPT_MESH_SRC")
if _ext:
    sys.path.append(_ext)
if _src:
    sys.path.append(_src)

# Vendored dependencies used by this script (all present under thirdparty/):
# thirdparty/pre_process.py, thirdparty/utils.py, thirdparty/vtk_utils/vtk_utils.py.
# See thirdparty/README.md.
sys.path.append(os.path.join(os.path.dirname(__file__), 'thirdparty'))

# Original relative fallbacks (kept for compatibility with the source layout).
sys.path.append(os.path.join(os.path.dirname(__file__), '../external'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

import vtk
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
import numpy as np
import argparse
import SimpleITK as sitk

from vtk_utils.vtk_utils import *
from utils import *
from pre_process import resample_spacing


# -----------------------------
# args
# -----------------------------

def parse():
    p = argparse.ArgumentParser()
    p.add_argument('--seg_fn', required=True)
    p.add_argument('--target_node_num', default=1000000, type=int)
    p.add_argument('--output', required=True)
    p.add_argument('--keep_labels', nargs='+', type=int, default=[1,2,3,4,5])
    p.add_argument('--bg_id', type=int, default=0)

    p.add_argument('--eps_vox', type=float, default=0.2)
    p.add_argument('--eps_max', type=float, default=1.4)
    p.add_argument('--eps_step', type=float, default=0.2)

    p.add_argument('--eps_classify', type=float, default=0.5)

    p.add_argument(
        '--forbidden_pair',
        action='append',
        nargs=2,
        type=int,
        default=None,
        help='Forbidden label pairs (used for pre-separation + march freeze + repair). Repeatable.'
    )

    # NEW: post-contour only (no seg erosion / no pre-separation / no march freeze)
    p.add_argument(
        '--forbidden_pair_no_erosion',
        action='append',
        nargs=2,
        type=int,
        default=None,
        help='Forbidden label pairs applied only AFTER initial contour labeling (repair stage only). Repeatable.'
    )

    p.add_argument('--forbidden_band_vox', type=int, default=2)
    p.add_argument('--forbidden_roi_pad', type=int, default=3)
    p.add_argument('--forbidden_verbose', action='store_true')

    args = p.parse_args()

    if args.forbidden_pair is None:
        args.forbidden_pair = [[2, 3]]

    if args.forbidden_pair_no_erosion is None:
        args.forbidden_pair_no_erosion = []

    return args


# -----------------------------
# forbidden pairs helpers (NEW, minimal)
# -----------------------------

def normalize_forbidden_pairs(forbidden_pairs):
    """
    Normalize forbidden pairs into a unique list of (a,b) tuples with a<b.

    Accepts:
      - None
      - [a,b]  (flat single pair, backward compatibility)
      - (a,b)
      - [[a,b], [c,d], ...]
      - [(a,b), (c,d), ...]
    """
    if forbidden_pairs is None:
        return []

    # Backward-compat: a flat [a,b] pair
    if isinstance(forbidden_pairs, (list, tuple, np.ndarray)):
        try:
            if len(forbidden_pairs) == 2 and all(
                isinstance(x, (int, np.integer)) for x in forbidden_pairs
            ):
                forbidden_pairs = [forbidden_pairs]
        except Exception:
            pass

    out = []
    for p in forbidden_pairs:
        if p is None:
            continue
        if not isinstance(p, (list, tuple, np.ndarray)) or len(p) != 2:
            raise ValueError(f"Each forbidden pair must be two ints. Got: {p}")
        a, b = int(p[0]), int(p[1])
        if a == b:
            continue
        out.append((min(a, b), max(a, b)))

    # unique, stable order
    seen = set()
    uniq = []
    for ab in out:
        if ab not in seen:
            uniq.append(ab)
            seen.add(ab)
    return uniq


# -----------------------------
# seg helpers
# -----------------------------

def filter_labels(seg: sitk.Image, keep_labels, bg_id=0) -> sitk.Image:
    arr = sitk.GetArrayFromImage(seg).astype(np.int32)  # (z, y, x)
    keep = np.isin(arr, np.array(keep_labels, dtype=np.int32))
    arr[~keep] = int(bg_id)
    out = sitk.GetImageFromArray(arr.astype(np.int32))
    out.CopyInformation(seg)
    return out


def seg_to_mesh_array(seg_img: sitk.Image) -> np.ndarray:
    arr = sitk.GetArrayFromImage(seg_img).astype(np.int32)  # (z, y, x)
    arr = arr[:, ::-1, ::-1]  # flip last two axes (your original convention)
    return arr


def sample_nn(arr_mesh: np.ndarray, pts_xyz: np.ndarray, bg_id: int):
    idx = np.rint(pts_xyz).astype(np.int64)
    x, y, z = idx[:, 0], idx[:, 1], idx[:, 2]

    nz, nx, ny = arr_mesh.shape
    inside = (x >= 0) & (x < nx) & (y >= 0) & (y < ny) & (z >= 0) & (z < nz)

    out = np.full(pts_xyz.shape[0], int(bg_id), dtype=np.int32)
    out[inside] = arr_mesh[z[inside], x[inside], y[inside]]
    return out


def clean_bg_pairs(pairs: np.ndarray, bg_id: int) -> np.ndarray:
    pairs = np.asarray(pairs).astype(np.int32).copy()
    pairs.sort(axis=1)
    m = (pairs[:, 0] == bg_id) & (pairs[:, 1] != bg_id)
    pairs[m, 0] = pairs[m, 1]
    return pairs


def eps_schedule(eps0: float, eps_max: float, eps_step: float):
    if eps_step <= 0:
        raise ValueError("eps_step must be > 0")
    if eps_max < eps0:
        raise ValueError("eps_max must be >= eps0")

    eps = [float(eps0)]
    e = float(eps0 + eps_step)
    while e <= float(eps_max) + 1e-9:
        eps.append(float(e))
        e += float(eps_step)
    return eps


# -----------------------------
# fast forbidden pair separation (ROI + early-stop dilation, no full distance map)
# -----------------------------

def _bbox_union_for_labels(seg: sitk.Image, labels, pad: int):
    """
    Union bbox for labels in seg, padded by `pad` voxels. Returns (index, size) or None.
    LabelShapeStatisticsImageFilter bbox is in index space (x,y,z,sx,sy,sz).
    """
    ls = sitk.LabelShapeStatisticsImageFilter()
    ls.Execute(seg)

    bbs = []
    for lab in labels:
        if ls.HasLabel(int(lab)):
            bbs.append(ls.GetBoundingBox(int(lab)))

    if not bbs:
        return None

    x0 = min(bb[0] for bb in bbs)
    y0 = min(bb[1] for bb in bbs)
    z0 = min(bb[2] for bb in bbs)

    x1 = max(bb[0] + bb[3] for bb in bbs)
    y1 = max(bb[1] + bb[4] for bb in bbs)
    z1 = max(bb[2] + bb[5] for bb in bbs)

    sx, sy, sz = seg.GetSize()
    x0 = max(0, x0 - pad); y0 = max(0, y0 - pad); z0 = max(0, z0 - pad)
    x1 = min(sx, x1 + pad); y1 = min(sy, y1 + pad); z1 = min(sz, z1 + pad)

    size = (max(0, x1 - x0), max(0, y1 - y0), max(0, z1 - z0))
    if size[0] == 0 or size[1] == 0 or size[2] == 0:
        return None
    return (x0, y0, z0), size


def _binary_dilate_ball(mask: sitk.Image, radius_vox: int) -> sitk.Image:
    if radius_vox <= 0:
        return mask
    return sitk.BinaryDilate(mask, [int(radius_vox)] * 3, sitk.sitkBall)


def separate_forbidden_pair_fast(seg: sitk.Image,
                                 a: int, b: int,
                                 band_vox: int,
                                 bg_id: int = 0,
                                 roi_pad: int = 3,
                                 verbose: bool = False) -> sitk.Image:
    """
    If labels a and b are closer than or equal to band_vox (in voxel units),
    carve a thin gap band near their interface (set to bg_id) to prevent SurfaceNets bridging.

    This avoids computing a full distance transform:
      - crops to a tight ROI around labels
      - estimates min distance by progressive dilation with early stopping
      - if too close, removes only (A within band of B) and (B within band of A) in the ROI
    """
    band_vox = int(band_vox)
    if band_vox <= 0:
        return seg

    bbox = _bbox_union_for_labels(seg, [a, b], pad=band_vox + int(roi_pad))
    if bbox is None:
        if verbose:
            print("[forbidden-sep] One/both labels missing. Skip.")
        return seg

    idx0, size = bbox
    roi = sitk.RegionOfInterest(seg, size=size, index=idx0)

    mA = sitk.Equal(roi, int(a))
    mB = sitk.Equal(roi, int(b))

    stats = sitk.StatisticsImageFilter()
    stats.Execute(mA)
    if stats.GetSum() <= 0:
        if verbose:
            print("[forbidden-sep] Label A empty in ROI. Skip.")
        return seg
    stats.Execute(mB)
    if stats.GetSum() <= 0:
        if verbose:
            print("[forbidden-sep] Label B empty in ROI. Skip.")
        return seg

    # 1) early-stop estimate of min distance <= band_vox
    min_d = None
    for r in range(0, band_vox + 1):
        dA = _binary_dilate_ball(mA, r)
        inter = sitk.And(dA, mB)
        stats.Execute(inter)
        if stats.GetSum() > 0:
            min_d = r
            break

    if min_d is None:
        if verbose:
            print(f"[forbidden-sep] min distance > {band_vox} vox. No action.")
        return seg

    if verbose:
        print(f"[forbidden-sep] min distance({a},{b}) ≈ {min_d} vox <= band {band_vox}: carve gap band.")

    # 2) carve the interface band (within band_vox)
    dA_band = _binary_dilate_ball(mA, band_vox)
    dB_band = _binary_dilate_ball(mB, band_vox)

    cutA = sitk.And(mA, dB_band)
    cutB = sitk.And(mB, dA_band)
    cut  = sitk.Or(cutA, cutB)

    roi_i32 = sitk.Cast(roi, sitk.sitkInt32)
    roi_out = sitk.Mask(roi_i32, sitk.Not(cut), outsideValue=int(bg_id))

    # paste back
    out = sitk.Image(seg)
    out = sitk.Paste(out, roi_out, roi_out.GetSize(),
                     destinationIndex=idx0, sourceIndex=(0, 0, 0))
    return out


# -----------------------------
# mesh helpers
# -----------------------------

def decimation(poly, rate):
    decimate = vtk.vtkQuadricDecimation()
    decimate.SetInputData(poly)
    decimate.AttributeErrorMetricOn()
    decimate.ScalarsAttributeOn()
    decimate.SetTargetReduction(rate)
    decimate.VolumePreservationOff()
    decimate.Update()
    return decimate.GetOutput()


def vtk_surface_nets_multi(vtkLabel, bg_id=0):
    ids = np.unique(vtk_to_numpy(vtkLabel.GetPointData().GetScalars()))
    ids = ids[ids != bg_id]

    sn = vtk.vtkSurfaceNets3D()
    sn.SetInputData(vtkLabel)
    sn.SetBackgroundLabel(float(bg_id))

    for idx, lab in enumerate(ids):
        sn.SetValue(idx, float(lab))

    sn.SetOutputMeshTypeToTriangles()
    sn.SmoothingOff()
    sn.Update()
    return sn.GetOutput()


def convert_to_surfs(seg, new_spacing=(0.3, 0.3, 0.3), target_node_num=1000000, bg_id=0,
                     forbidden_pair=None, forbidden_band_vox=0, forbidden_roi_pad=3, forbidden_verbose=False):
    # keep your original boundary erase (minimal change)
    py_seg = sitk.GetArrayFromImage(seg)
    py_seg = eraseBoundary(py_seg, 1, bg_id)

    seg2 = sitk.GetImageFromArray(py_seg.astype(np.int32))
    seg2.CopyInformation(seg)

    # CHANGED (minimal): apply pre-separation for ALL forbidden pairs
    forb_list = normalize_forbidden_pairs(forbidden_pair)
    if forb_list and int(forbidden_band_vox) > 0:
        for (a, b) in forb_list:
            seg2 = separate_forbidden_pair_fast(
                seg2, a=a, b=b,
                band_vox=int(forbidden_band_vox),
                bg_id=int(bg_id),
                roi_pad=int(forbidden_roi_pad),
                verbose=bool(forbidden_verbose),
            )

    seg_vtk, _ = exportSitk2VTK(seg2)
    seg_vtk = vtkImageResample(seg_vtk, list(new_spacing), 'NN')

    poly = vtk_surface_nets_multi(seg_vtk, bg_id=bg_id)
    poly = smooth_polydata(poly, iteration=50)

    rate = max(0.0, 1.0 - float(target_node_num) / float(max(1, poly.GetNumberOfPoints())))
    poly = decimation(poly, rate)
    poly = cleanPolyData(poly, 0.0)
    return poly


def create_tmplt(seg, target_num, bg_id=0, forbidden_pair=None, forbidden_band_vox=0, forbidden_roi_pad=3, forbidden_verbose=False):
    template = convert_to_surfs(
        seg,
        new_spacing=(1.0, 1.0, 1.0),
        target_node_num=target_num,
        bg_id=bg_id,
        forbidden_pair=forbidden_pair,
        forbidden_band_vox=forbidden_band_vox,
        forbidden_roi_pad=forbidden_roi_pad,
        forbidden_verbose=forbidden_verbose,
    )
    template = smooth_polydata(template, 25)
    return create_tmplt_mesh(template, seg)


def create_tmplt_mesh(mesh, ref):
    SIZE = (256, 256, 256)
    img_center = np.array(ref.TransformContinuousIndexToPhysicalPoint(np.array(ref.GetSize()) / 2.0), dtype=np.float64)

    seg_r = resample_spacing(ref, template_size=SIZE, order=0)[0]
    img_center2 = np.array(seg_r.TransformContinuousIndexToPhysicalPoint(np.array(seg_r.GetSize()) / 2.0), dtype=np.float64)

    transform = build_transform_matrix(seg_r)
    mesh = transform_polydata(mesh, img_center2 - img_center, transform, SIZE)
    return mesh, seg_r


# -----------------------------
# endo / epi (option 1)
# -----------------------------

def classify_endo_epi(arr_mesh, centers, normals, bg_id, lv_id=4, myo_id=5, eps_classify=0.6):
    pts_p = centers + float(eps_classify) * normals
    pts_m = centers - float(eps_classify) * normals

    s_p = sample_nn(arr_mesh, pts_p, bg_id=bg_id).astype(np.int32)
    s_m = sample_nn(arr_mesh, pts_m, bg_id=bg_id).astype(np.int32)

    a = np.minimum(s_p, s_m)
    b = np.maximum(s_p, s_m)

    endo = (a == min(lv_id, myo_id)) & (b == max(lv_id, myo_id))
    epi = (a == bg_id) & (b == myo_id)
    return endo, epi


# -----------------------------
# dynamic march (keep your freeze rules)
# -----------------------------

def march_labels(arr_mesh, centers, normals, bg_id, eps0, eps_max, eps_step, sign, forbidden_pair=None):
    eps_list = eps_schedule(eps0, eps_max, eps_step)

    # CHANGED (minimal): multiple forbidden pairs
    forb_list = normalize_forbidden_pairs(forbidden_pair)

    pts0 = centers + (sign * eps_list[0]) * normals
    labels = sample_nn(arr_mesh, pts0, bg_id=bg_id).astype(np.int32)

    frozen = np.zeros(labels.shape[0], dtype=bool)

    for eps in eps_list[1:]:
        idx = np.where(~frozen)[0]
        if idx.size == 0:
            break

        pts = centers[idx] + (sign * eps) * normals[idx]
        new = sample_nn(arr_mesh, pts, bg_id=bg_id).astype(np.int32)
        cur = labels[idx]

        revert_to_bg = (cur != bg_id) & (new == bg_id)

        forbidden_hit = np.zeros(idx.size, dtype=bool)
        if forb_list:
            for (a, b) in forb_list:
                forbidden_hit |= ((cur == a) & (new == b)) | ((cur == b) & (new == a))

        freeze_now = revert_to_bg | forbidden_hit
        if np.any(freeze_now):
            frozen[idx[freeze_now]] = True

        ok = ~freeze_now
        if np.any(ok):
            labels[idx[ok]] = new[ok]

    return labels


# -----------------------------
# repair bad cells by neighbor vote (no deletion)
# -----------------------------

def build_neighbors(poly_tri: vtk.vtkPolyData):
    n_cells = poly_tri.GetNumberOfCells()
    neighbors = [set() for _ in range(n_cells)]

    polys = poly_tri.GetPolys()
    polys.InitTraversal()
    idlist = vtk.vtkIdList()

    def ekey(u, v):
        return (u, v) if u < v else (v, u)

    edge2cells = {}
    cid = 0
    while polys.GetNextCell(idlist):
        a = idlist.GetId(0); b = idlist.GetId(1); c = idlist.GetId(2)
        for e in (ekey(a, b), ekey(b, c), ekey(c, a)):
            edge2cells.setdefault(e, []).append(cid)
        cid += 1

    for cids in edge2cells.values():
        if len(cids) < 2:
            continue
        for i in range(len(cids)):
            for j in range(i + 1, len(cids)):
                u, v = cids[i], cids[j]
                neighbors[u].add(v)
                neighbors[v].add(u)

    return [list(s) for s in neighbors]


def bad_cells(pairs, bg_id, forbidden_pair=None):
    pairs = np.asarray(pairs, dtype=np.int32)
    bg_bg = (pairs[:, 0] == bg_id) & (pairs[:, 1] == bg_id)

    # CHANGED (minimal): multiple forbidden pairs
    forb_list = normalize_forbidden_pairs(forbidden_pair)
    if not forb_list:
        return bg_bg

    forb_mask = np.zeros(pairs.shape[0], dtype=bool)
    for (a, b) in forb_list:
        forb_mask |= (pairs[:, 0] == a) & (pairs[:, 1] == b)

    return bg_bg | forb_mask


def repair_pairs(poly_tri, pairs, bg_id, forbidden_pair=None, iters=10):
    pairs = np.asarray(pairs, dtype=np.int32).copy()
    pairs.sort(axis=1)

    nbrs = build_neighbors(poly_tri)

    for _ in range(int(iters)):
        bad = bad_cells(pairs, bg_id=bg_id, forbidden_pair=forbidden_pair)
        if not np.any(bad):
            break

        out = pairs.copy()
        for cid in np.where(bad)[0]:
            cand = []
            for nb in nbrs[cid]:
                if not bad[nb]:
                    cand.append((int(pairs[nb, 0]), int(pairs[nb, 1])))
            if not cand:
                continue
            cand = np.array(cand, dtype=np.int32)
            uniq, cnt = np.unique(cand, axis=0, return_counts=True)
            best = uniq[np.argmax(cnt)]
            maxc = cnt.max()
            tied = uniq[cnt == maxc]
            if tied.shape[0] > 1:
                tied = tied[np.lexsort((tied[:, 1], tied[:, 0]))]
                best = tied[0]
            out[cid] = best

        pairs = out
        pairs.sort(axis=1)

    pairs = clean_bg_pairs(pairs, bg_id=bg_id)
    pairs.sort(axis=1)
    return pairs


# -----------------------------
# epi rule: do not allow LV on epi
# -----------------------------

def fix_epi_no_lv(pairs, epi_mask, lv_id, myo_id):
    pairs = np.asarray(pairs, dtype=np.int32).copy()
    pairs.sort(axis=1)

    epi_mask = np.asarray(epi_mask, dtype=bool)
    a = pairs[:, 0]
    b = pairs[:, 1]

    has_lv = (a == lv_id) | (b == lv_id)
    tgt = epi_mask & has_lv
    if not np.any(tgt):
        return pairs

    other = np.where(a == lv_id, b, a)

    mixed = tgt & (other != lv_id)
    if np.any(mixed):
        pairs[mixed, 0] = other[mixed]
        pairs[mixed, 1] = other[mixed]

    dbl = tgt & (a == lv_id) & (b == lv_id)
    if np.any(dbl):
        pairs[dbl, 0] = myo_id
        pairs[dbl, 1] = myo_id

    pairs.sort(axis=1)
    return pairs


# -----------------------------
# point labels from cell labels
# -----------------------------

def point_pairs_from_cells(poly: vtk.vtkPolyData, cell_pairs: np.ndarray, out_name="VertexBoundaryLabels"):
    n_cells = poly.GetNumberOfCells()
    n_pts = poly.GetNumberOfPoints()

    cell_pairs = np.asarray(cell_pairs, dtype=np.int32)
    if cell_pairs.shape != (n_cells, 2):
        raise ValueError(f"cell_pairs must be (n_cells,2). got {cell_pairs.shape}, n_cells={n_cells}")

    cp = cell_pairs.copy()
    cp.sort(axis=1)

    poly.BuildLinks()
    idlist = vtk.vtkIdList()

    out = np.zeros((n_pts, 2), dtype=np.int32)

    for pid in range(n_pts):
        poly.GetPointCells(pid, idlist)
        k = idlist.GetNumberOfIds()
        if k == 0:
            out[pid] = cp[0]
            continue

        pairs = np.empty((k, 2), dtype=np.int32)
        for j in range(k):
            pairs[j] = cp[idlist.GetId(j)]

        uniq, cnt = np.unique(pairs, axis=0, return_counts=True)
        best = uniq[np.argmax(cnt)]
        maxc = cnt.max()
        tied = uniq[cnt == maxc]
        if tied.shape[0] > 1:
            tied = tied[np.lexsort((tied[:, 1], tied[:, 0]))]
            best = tied[0]

        out[pid] = best

    arr = numpy_to_vtk(out, deep=True, array_type=vtk.VTK_INT)
    arr.SetNumberOfComponents(2)
    arr.SetName(out_name)

    poly.GetPointData().RemoveArray(out_name)
    poly.GetPointData().AddArray(arr)
    poly.GetPointData().SetActiveScalars(out_name)
    return poly

def merge_forbidden_pairs(*pair_lists):
    """
    Merge multiple forbidden pair specs into a single unique, canonicalized list of (a,b) with a<b.
    Each input can be None, [a,b], [[a,b],...], [(a,b),...], etc.
    """
    merged = []
    for pl in pair_lists:
        merged.extend(normalize_forbidden_pairs(pl))
    # normalize again to unique + stable order
    return normalize_forbidden_pairs(merged)

# -----------------------------
# main labeling
# -----------------------------
def label_from_seg(seg_img, poly, bg_id=0,
                   eps_vox=0.5, eps_max=1.5, eps_step=0.2,
                   eps_classify=0.6,
                   out_name="BoundaryLabels",
                   forbidden_pair=None,
                   forbidden_pair_no_erosion=None):
    if poly.GetNumberOfCells() == 0:
        return poly

    arr_mesh = seg_to_mesh_array(seg_img)

    tri = vtk.vtkTriangleFilter()
    tri.SetInputData(poly)
    tri.Update()
    poly_t = tri.GetOutput()

    nrm = vtk.vtkPolyDataNormals()
    nrm.SetInputData(poly_t)
    nrm.ComputeCellNormalsOn()
    nrm.ComputePointNormalsOff()
    nrm.SplittingOff()
    nrm.ConsistencyOff()
    nrm.Update()
    poly_n = nrm.GetOutput()

    normals_vtk = poly_n.GetCellData().GetArray("Normals")
    if normals_vtk is None:
        raise RuntimeError("Normals missing after vtkPolyDataNormals.")
    normals = vtk_to_numpy(normals_vtk).astype(np.float64)

    cc = vtk.vtkCellCenters()
    cc.SetInputData(poly_n)
    cc.Update()
    centers = vtk_to_numpy(cc.GetOutput().GetPoints().GetData()).astype(np.float64)

    endo_mask, epi_mask = classify_endo_epi(
        arr_mesh=arr_mesh,
        centers=centers,
        normals=normals,
        bg_id=bg_id,
        lv_id=5,
        myo_id=4,
        eps_classify=float(eps_classify)
    )

    # march_labels uses ONLY forbidden_pair (the one that matches your erosion/pre-sep policy)
    lab_p = march_labels(arr_mesh, centers, normals,
                         bg_id=bg_id, eps0=float(eps_vox),
                         eps_max=float(eps_max), eps_step=float(eps_step),
                         sign=+1.0, forbidden_pair=forbidden_pair)

    lab_m = march_labels(arr_mesh, centers, normals,
                         bg_id=bg_id, eps0=float(eps_vox),
                         eps_max=float(eps_max), eps_step=float(eps_step),
                         sign=-1.0, forbidden_pair=forbidden_pair)

    pairs = np.stack([lab_p, lab_m], axis=1).astype(np.int32)
    pairs.sort(axis=1)

    pairs = clean_bg_pairs(pairs, bg_id=bg_id)
    pairs.sort(axis=1)

    forbidden_post = merge_forbidden_pairs(forbidden_pair, forbidden_pair_no_erosion)

    # repair considers BOTH sets (so new pairs are treated as "bad" and voted away)
    pairs = repair_pairs(poly_n, pairs, bg_id=bg_id, forbidden_pair=forbidden_post, iters=10)

    # epi cannot contain LV
    pairs = fix_epi_no_lv(pairs, epi_mask=epi_mask, lv_id=5, myo_id=4)
    pairs = clean_bg_pairs(pairs, bg_id=bg_id)
    pairs.sort(axis=1)

    # endo fixed to (4,5)
    if np.any(endo_mask):
        pairs[endo_mask, 0] = 4
        pairs[endo_mask, 1] = 5
        pairs.sort(axis=1)
        pairs = clean_bg_pairs(pairs, bg_id=bg_id)
        pairs.sort(axis=1)

    # cell array
    cell_arr = numpy_to_vtk(pairs, deep=True, array_type=vtk.VTK_INT)
    cell_arr.SetName(out_name)
    poly_n.GetCellData().RemoveArray(out_name)
    poly_n.GetCellData().AddArray(cell_arr)
    poly_n.GetCellData().SetActiveScalars(out_name)

    # point array (pair)
    poly_n = point_pairs_from_cells(poly_n, pairs, out_name="VertexBoundaryLabels")

    # prints
    uniq, cnt = np.unique(pairs, axis=0, return_counts=True)
    print("\n[BoundaryLabels] pairs -> count:")
    for (a, b), c in sorted(zip(uniq.tolist(), cnt.tolist()), key=lambda t: (-t[1], t[0])):
        print(f"  ({a},{b}) -> {c}")

    bb = pairs.copy()
    bb.sort(axis=1)
    bg_bg = (bb[:, 0] == bg_id) & (bb[:, 1] == bg_id)
    print(f"[BoundaryLabels] (bg,bg): {int(bg_bg.sum())} / {pairs.shape[0]}")
    print(f"[VertexBoundaryLabels] points: {poly_n.GetNumberOfPoints()}")

    return poly_n


# -----------------------------
# main
# -----------------------------

if __name__ == '__main__':
    args = parse()

    seg = sitk.ReadImage(args.seg_fn)
    seg = filter_labels(seg, keep_labels=args.keep_labels, bg_id=args.bg_id)

    # Minimal-change integration:
    # - We keep all your labeling logic unchanged.
    # - We only add a PREVENTION step before SurfaceNets inside create_tmplt -> convert_to_surfs.
    tmplt, seg_r = create_tmplt(
        seg,
        args.target_node_num,
        bg_id=args.bg_id,
        forbidden_pair=args.forbidden_pair,
        forbidden_band_vox=args.forbidden_band_vox,
        forbidden_roi_pad=args.forbidden_roi_pad,
        forbidden_verbose=args.forbidden_verbose,
    )

    tmplt = label_from_seg(
        seg_img=seg_r,
        poly=tmplt,
        bg_id=args.bg_id,
        eps_vox=args.eps_vox,
        eps_max=args.eps_max,
        eps_step=args.eps_step,
        eps_classify=args.eps_classify,
        out_name="BoundaryLabels",
        forbidden_pair=args.forbidden_pair,
        forbidden_pair_no_erosion=args.forbidden_pair_no_erosion,
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    write_vtk_polydata(tmplt, args.output)
    print(f"[OK] Wrote: {args.output}")
