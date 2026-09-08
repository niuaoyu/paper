#!/usr/bin/env python3
# Convert to sparse cloud points, combining 4 chambers
# (with forbidden-pair separation + bg-pair cleanup + OPTIONAL post-contour forbidden-pair collapse)

import argparse, os, random, hashlib
import SimpleITK as sitk
import numpy as np, scipy.ndimage as ndi
from scipy.interpolate import RegularGridInterpolator
import traceback

# -----------------------------
# Utility: forbidden pairs + bg cleanup
# -----------------------------

def _norm_pair(p):
    a, b = int(p[0]), int(p[1])
    return (a, b) if a <= b else (b, a)

def _norm_pairs_list(pairs):
    if not pairs:
        return []
    return [_norm_pair(p) for p in pairs]

def _stable_case_seed(global_seed: int, key: str) -> int:
    """
    Deterministic per-case seed derived from (global_seed, key).
    Uses md5 for stable hashing across Python runs.
    """
    h = hashlib.md5(key.encode("utf-8")).hexdigest()
    # take 32 bits
    hv = int(h[:8], 16)
    return (int(global_seed) + hv) & 0xFFFFFFFF

def _pick_one_pair_deterministic(pairs, seed: int, key: str):
    """
    Pick exactly one pair from `pairs` deterministically given (seed, key).
    Returns [] if input empty.
    """
    if not pairs:
        return []
    s = _stable_case_seed(seed, key)
    rng = np.random.default_rng(s)
    idx = int(rng.integers(0, len(pairs)))
    return [pairs[idx]]

def clean_bg_pairs_np(pairs: np.ndarray, bg_id: int) -> np.ndarray:
    """
    Same semantics as your mesh code:
      (bg, x) -> (x, x), after sorting.
    """
    pairs = np.asarray(pairs, dtype=np.int32).copy()
    pairs.sort(axis=1)
    m = (pairs[:, 0] == int(bg_id)) & (pairs[:, 1] != int(bg_id))
    pairs[m, 0] = pairs[m, 1]
    return pairs

def collapse_forbidden_pairs_np(pairs: np.ndarray, forbidden_pairs, bg_id: int) -> np.ndarray:
    """
    Extra safety: if any forbidden (a,b) appears, collapse to (min(a,b), min(a,b)).
    After this, it is no longer a forbidden boundary pair.
    """
    if not forbidden_pairs:
        pairs = clean_bg_pairs_np(pairs, bg_id=bg_id)
        pairs.sort(axis=1)
        return pairs

    forb = set(_norm_pairs_list(forbidden_pairs))
    pairs = np.asarray(pairs, dtype=np.int32).copy()
    pairs.sort(axis=1)

    # membership test (O(N)); OK for typical boundary counts
    mask = np.fromiter((tuple(p) in forb for p in pairs), dtype=bool, count=pairs.shape[0])
    if np.any(mask):
        pairs[mask, 1] = pairs[mask, 0]

    pairs = clean_bg_pairs_np(pairs, bg_id=bg_id)
    pairs.sort(axis=1)
    return pairs


# -----------------------------
# SAME erosion/band-carve technique in 3D (SimpleITK ROI + early-stop)
# -----------------------------

def _bbox_union_for_labels(seg: sitk.Image, labels, pad: int):
    """
    Union bbox for labels in seg, padded by `pad` voxels.
    Returns (index(x,y,z), size(x,y,z)) or None.
    """
    ls = sitk.LabelShapeStatisticsImageFilter()
    ls.Execute(seg)

    bbs = []
    for lab in labels:
        lab = int(lab)
        if ls.HasLabel(lab):
            bbs.append(ls.GetBoundingBox(lab))  # (x,y,z,sx,sy,sz)

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

def separate_forbidden_pair_fast_3d(seg: sitk.Image,
                                   a: int, b: int,
                                   band_vox: int,
                                   bg_id: int = 0,
                                   roi_pad: int = 3,
                                   verbose: bool = False) -> sitk.Image:
    """
    If labels a and b are within <= band_vox, carve a thin gap band to bg_id (in a tight ROI).
    """
    band_vox = int(band_vox)
    if band_vox <= 0:
        return seg

    bbox = _bbox_union_for_labels(seg, [a, b], pad=band_vox + int(roi_pad))
    if bbox is None:
        if verbose:
            print(f"[forbidden-3d] Missing labels in seg for pair ({a},{b}). Skip.")
        return seg

    idx0, size = bbox
    roi = sitk.RegionOfInterest(seg, size=size, index=idx0)

    mA = sitk.Equal(roi, int(a))
    mB = sitk.Equal(roi, int(b))

    stats = sitk.StatisticsImageFilter()
    stats.Execute(mA)
    if stats.GetSum() <= 0:
        return seg
    stats.Execute(mB)
    if stats.GetSum() <= 0:
        return seg

    # early-stop min distance <= band_vox
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
            print(f"[forbidden-3d] min distance > {band_vox} for ({a},{b}). No carve.")
        return seg

    if verbose:
        print(f"[forbidden-3d] Pair ({a},{b}) min distance ≈ {min_d} <= {band_vox}. Carve band.")

    dA_band = _binary_dilate_ball(mA, band_vox)
    dB_band = _binary_dilate_ball(mB, band_vox)

    cutA = sitk.And(mA, dB_band)
    cutB = sitk.And(mB, dA_band)
    cut  = sitk.Or(cutA, cutB)

    roi_i32 = sitk.Cast(roi, sitk.sitkInt32)
    roi_out = sitk.Mask(roi_i32, sitk.Not(cut), outsideValue=int(bg_id))

    out = sitk.Image(seg)
    out = sitk.Paste(out, roi_out, roi_out.GetSize(),
                     destinationIndex=idx0, sourceIndex=(0, 0, 0))
    return out

def separate_forbidden_pairs_fast_3d(seg: sitk.Image,
                                    forbidden_pairs,
                                    band_vox: int,
                                    bg_id: int = 0,
                                    roi_pad: int = 3,
                                    verbose: bool = False) -> sitk.Image:
    if not forbidden_pairs:
        return seg
    out = seg
    for a, b in forbidden_pairs:
        out = separate_forbidden_pair_fast_3d(out, int(a), int(b),
                                              band_vox=band_vox,
                                              bg_id=int(bg_id),
                                              roi_pad=roi_pad,
                                              verbose=verbose)
    return out


# -----------------------------
# SAME technique in 2D (for SAX slices + LAX resampled planes)
# -----------------------------

def separate_forbidden_pairs_fast_2d(labels2d: np.ndarray,
                                     forbidden_pairs,
                                     band_px: int,
                                     bg_id: int = 0) -> np.ndarray:
    """
    2D analog of the carve-band idea.
    """
    if not forbidden_pairs or int(band_px) <= 0:
        return labels2d

    out = labels2d.astype(np.int32).copy()
    structure = np.ones((3, 3), dtype=bool)

    for a, b in forbidden_pairs:
        a = int(a); b = int(b)
        mA = (out == a)
        mB = (out == b)
        if not mA.any() or not mB.any():
            continue

        min_d = None
        for r in range(0, int(band_px) + 1):
            dA = mA if r == 0 else ndi.binary_dilation(mA, structure=structure, iterations=r)
            if np.any(dA & mB):
                min_d = r
                break
        if min_d is None:
            continue

        dA_band = mA if band_px == 0 else ndi.binary_dilation(mA, structure=structure, iterations=int(band_px))
        dB_band = mB if band_px == 0 else ndi.binary_dilation(mB, structure=structure, iterations=int(band_px))

        cut = (mA & dB_band) | (mB & dA_band)
        out[cut] = int(bg_id)

    return out.astype(labels2d.dtype, copy=False)


# -----------------------------
# args helpers
# -----------------------------

def find_corner_index(line):
    n = len(line)
    down = next((i for i in range(n-1)
                 if line[i] > 0.9 and line[i+1] > 0.9), 0)
    up = next((i for i in range(n-1, 0, -1)
               if line[i] > 0.9 and line[i-1] > 0.9), n-1)
    return down, up

def fps_2d(points: np.ndarray, k: int, seed: int | None = None) -> np.ndarray:
    points = np.asarray(points, float)
    n = points.shape[0]
    if n == 0 or k <= 0:
        return np.zeros((0,), dtype=int)
    out_k = k if k <= n else n
    rng = np.random.default_rng(seed)
    idx = np.empty(out_k, dtype=int)
    d2  = np.full(n, np.inf)
    far = int(rng.integers(0, n))
    for i in range(out_k):
        idx[i] = far
        diff = points - points[far]
        d2   = np.minimum(d2, np.einsum('ij,ij->i', diff, diff))
        far  = int(np.argmax(d2))
    if k > n:
        return np.resize(idx, k)
    return idx


def _edge_midpoints_subpixel(
    labels2d: np.ndarray,
    return_pairs: bool = False,
    bg_id: int = 0,
    forbidden_pairs=None,                 # used for 2D carve
    forbidden_band_px: int = 0,
    forbidden_pairs_collapse=None,        # used ONLY for pair collapse
):
    """
    Midpoints for every grid edge where labels differ + optional label-pair output.

    - forbidden_pairs + forbidden_band_px: 2D carve on labels2d (pre-boundary extraction)
    - forbidden_pairs_collapse: post-boundary pair collapse (no carve needed)
    """
    # 2D carve
    if forbidden_pairs and int(forbidden_band_px) > 0:
        labels2d = separate_forbidden_pairs_fast_2d(labels2d, forbidden_pairs,
                                                    band_px=int(forbidden_band_px),
                                                    bg_id=int(bg_id))

    lab = labels2d.astype(np.int32)
    H, W = lab.shape

    pts_list = []
    pair_list = []

    # Horizontal edges
    if W > 1:
        diff_h = lab[:, 1:] != lab[:, :-1]
        r, c = np.where(diff_h)
        if r.size > 0:
            pts_h = np.column_stack([r.astype(np.float32), (c + 0.5).astype(np.float32)])
            pts_list.append(pts_h)
            if return_pairs:
                a = lab[r, c]
                b = lab[r, c + 1]
                pairs_h = np.stack([np.minimum(a, b), np.maximum(a, b)], axis=1).astype(np.int32)
                pair_list.append(pairs_h)

    # Vertical edges
    if H > 1:
        diff_v = lab[1:, :] != lab[:-1, :]
        r, c = np.where(diff_v)
        if r.size > 0:
            pts_v = np.column_stack([(r + 0.5).astype(np.float32), c.astype(np.float32)])
            pts_list.append(pts_v)
            if return_pairs:
                a = lab[r, c]
                b = lab[r + 1, c]
                pairs_v = np.stack([np.minimum(a, b), np.maximum(a, b)], axis=1).astype(np.int32)
                pair_list.append(pairs_v)

    if not pts_list:
        coords = np.empty((0, 2), dtype=np.float32)
        if return_pairs:
            pairs = np.empty((0, 2), dtype=np.int32)
            return coords, pairs
        return coords

    coords = np.vstack(pts_list).astype(np.float32)
    if not return_pairs:
        return coords

    pairs = np.vstack(pair_list).astype(np.int32)

    # post pair cleanup (no-erosion / post-contour path)
    collapse_list = forbidden_pairs_collapse if forbidden_pairs_collapse is not None else forbidden_pairs

    pairs = clean_bg_pairs_np(pairs, bg_id=int(bg_id))
    pairs = collapse_forbidden_pairs_np(pairs, forbidden_pairs=collapse_list, bg_id=int(bg_id))
    pairs.sort(axis=1)
    return coords, pairs


# ---------- landmarks ----------
def compute_landmarks_cardiac(dense_label: np.ndarray, lv_id, rv_id, la_id, ra_id):
    seg_LV = (dense_label == lv_id).astype(np.uint8)
    seg_RV = (dense_label == rv_id).astype(np.uint8)
    seg_LA = (dense_label == la_id).astype(np.uint8)
    seg_RA = (dense_label == ra_id).astype(np.uint8)
    seg_bin = (dense_label != 0).astype(np.uint8)

    seg_LA_dil = ndi.binary_dilation(seg_LA, iterations=1).astype(np.uint8)
    mvc = np.mean(np.where((seg_LV * seg_LA_dil) == 1), axis=1)

    seg_RA_dil = ndi.binary_dilation(seg_RA, iterations=1).astype(np.uint8)
    tvc = np.mean(np.where((seg_RV * seg_RA_dil) == 1), axis=1)

    LV_pts = np.array(np.where(seg_LV == 1)).T
    if LV_pts.size == 0:
        apex = np.array(mvc)
    else:
        d = np.linalg.norm(LV_pts - mvc[None, :], axis=1)
        apex = LV_pts[np.argmax(d)].astype(float)

    coh = np.mean(np.where(seg_bin == 1), axis=1)
    lvc = np.mean(np.where(seg_LV == 1), axis=1) if seg_LV.sum() else apex
    rvc = np.mean(np.where(seg_RV == 1), axis=1) if seg_RV.sum() else tvc

    return dict(mvc=mvc, tvc=tvc, apex=apex, coh=coh, lvc=lvc, rvc=rvc)

def compute_avc_cardiac(dense_label: np.ndarray, ao_label, lv_id):
    seg_LV_o = (dense_label == lv_id).astype(np.uint8)
    seg_Ao_o = (dense_label == ao_label).astype(np.uint8)
    seg_Ao_d = ndi.binary_dilation(seg_Ao_o, iterations=1).astype(np.uint8)
    avc = np.mean(np.where((seg_LV_o * seg_Ao_d) == 1), axis=1)
    return avc

def plane_from_two_axes(iop_one, iop_two, mvc, coh):
    iop_one = iop_one / np.linalg.norm(iop_one)
    iop_two = iop_two / np.linalg.norm(iop_two)
    n = np.cross(iop_one, iop_two); n /= np.linalg.norm(n)

    vec = coh - mvc
    d = float(np.dot(vec, n))
    ipp_center = coh - d * n
    return iop_one, iop_two, n, ipp_center


# -----------------------------
# LAX
# -----------------------------

def sparse_pc_from_LAX_plane(dense_label, iop_one, iop_two, ipp_center,
                            num_points, seed=None,
                            bg_id: int = 0,
                            forbidden_pairs=None,
                            forbidden_band_px: int = 0,
                            forbidden_pairs_collapse=None):
    Z, Y, X = dense_label.shape
    z_ax = np.arange(Z, dtype=np.float32)
    y_ax = np.arange(Y, dtype=np.float32)
    x_ax = np.arange(X, dtype=np.float32)
    f = RegularGridInterpolator(
        (z_ax, y_ax, x_ax),
        dense_label.astype(np.int32),
        method="nearest",
        bounds_error=False,
        fill_value=int(bg_id)
    )

    size = max(Y, X)
    half = size // 2
    us = np.arange(-half, half, 1, dtype=np.float32)
    vs = np.arange(-half, half, 1, dtype=np.float32)
    U, V = np.meshgrid(us, vs, indexing="ij")

    P = ipp_center[None, None, :] + U[..., None] * iop_one[None, None, :] + V[..., None] * iop_two[None, None, :]
    Q = P.reshape(-1, 3).astype(np.float32)
    vals = f(Q).reshape(size, size)

    coords_uv, pairs_uv = _edge_midpoints_subpixel(
        vals,
        return_pairs=True,
        bg_id=int(bg_id),
        forbidden_pairs=forbidden_pairs,
        forbidden_band_px=int(forbidden_band_px),
        forbidden_pairs_collapse=forbidden_pairs_collapse,
    )

    out_pts   = np.zeros((num_points, 3), dtype=np.float32)
    out_pairs = np.zeros((num_points, 2), dtype=np.int32)

    if coords_uv.shape[0] == 0:
        return out_pts, out_pairs

    k = min(num_points, coords_uv.shape[0])
    sel = fps_2d(coords_uv, k, seed=None if seed is None else seed + 17)

    chosen_uv = coords_uv[sel]
    chosen_pairs = pairs_uv[sel]

    u = (chosen_uv[:, 0] - float(half))
    v = (chosen_uv[:, 1] - float(half))
    pts = ipp_center[None, :] + u[:, None] * iop_one[None, :] + v[:, None] * iop_two[None, :]

    # Store as (y,x,z) to match SAX
    out_pts[:k, 0] = pts[:, 1]
    out_pts[:k, 1] = pts[:, 2]
    out_pts[:k, 2] = pts[:, 0]

    # post-sample collapse (extra safety)
    collapse_list = forbidden_pairs_collapse if forbidden_pairs_collapse is not None else forbidden_pairs
    chosen_pairs = clean_bg_pairs_np(chosen_pairs, bg_id=int(bg_id))
    chosen_pairs = collapse_forbidden_pairs_np(chosen_pairs, forbidden_pairs=collapse_list, bg_id=int(bg_id))
    chosen_pairs.sort(axis=1)
    out_pairs[:k] = chosen_pairs.astype(np.int32)

    if k < num_points:
        rng = np.random.default_rng(seed)
        extra = rng.integers(0, k, size=(num_points - k,))
        out_pts[k:]   = out_pts[extra]
        out_pairs[k:] = out_pairs[extra]

    return out_pts, out_pairs

def _mask_keep_ids(dense_label: np.ndarray, keep_ids, bg_id: int):
    keep_ids = list(map(int, keep_ids))
    return np.where(np.isin(dense_label, keep_ids), dense_label, int(bg_id)).astype(dense_label.dtype, copy=False)


def pc_4ch_from_landmarks(dense_label, apex, mvc, tvc, coh, num_points, seed=None,
                          bg_id=0, forbidden_pairs=None, forbidden_band_px=0, forbidden_pairs_collapse=None):
    vec_one = apex - mvc
    vec_two_tmp = tvc - mvc
    vec_three = np.cross(vec_one, vec_two_tmp)
    vec_two = np.cross(vec_three, vec_one)
    iop_one, iop_two, n, ipp = plane_from_two_axes(vec_one, vec_two, mvc, coh)
    return sparse_pc_from_LAX_plane(
        dense_label, iop_one, iop_two, ipp, num_points,
        seed=seed, bg_id=bg_id,
        forbidden_pairs=forbidden_pairs, forbidden_band_px=forbidden_band_px,
        forbidden_pairs_collapse=forbidden_pairs_collapse
    )

def pc_3ch_from_landmarks(dense_label, apex, mvc, avc_cardiac, coh, num_points, seed=None,
                          bg_id=0, forbidden_pairs=None, forbidden_band_px=0, forbidden_pairs_collapse=None,
                          la_id=2, myo_id=4, lv_id=5, rv_id=6):
    # 3CH should only have LA, MYO, LV, RV (NO RA)
    dense_label_view = _mask_keep_ids(dense_label, keep_ids=[la_id, myo_id, lv_id, rv_id], bg_id=bg_id)

    vec_one_tmp = apex - mvc
    vec_two = avc_cardiac - mvc
    vec_three = np.cross(vec_one_tmp, vec_two)
    vec_one = np.cross(vec_two, vec_three)
    iop_one, iop_two, n, ipp = plane_from_two_axes(vec_one, vec_two, mvc, coh)

    theta = random.choice([-1, 1]) * (np.pi / random.randint(60, 180))
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(theta), -np.sin(theta)],
                   [0, np.sin(theta),  np.cos(theta)]])
    Ry = np.array([[ np.cos(theta), 0, np.sin(theta)],
                   [0, 1, 0],
                   [-np.sin(theta), 0, np.cos(theta)]])

    A_LAX = np.zeros((3, 3))
    third_col = np.cross(iop_one, iop_two)
    third_col = third_col / np.linalg.norm(third_col)
    A_LAX[:, 0] = iop_one
    A_LAX[:, 1] = iop_two
    A_LAX[:, 2] = third_col

    if random.randint(-1, 8) > 0:
        A_LAX = np.matmul(A_LAX, Rx)
    elif random.randint(-1, 8) > 0:
        A_LAX = np.matmul(A_LAX, Ry)

    iop_one = A_LAX[:, 0]
    iop_two = A_LAX[:, 1]

    return sparse_pc_from_LAX_plane(
        dense_label_view, iop_one, iop_two, ipp, num_points,
        seed=seed, bg_id=bg_id,
        forbidden_pairs=forbidden_pairs, forbidden_band_px=forbidden_band_px,
        forbidden_pairs_collapse=forbidden_pairs_collapse
    )

def pc_2ch_from_landmarks(dense_label, apex, mvc, rvc, coh, num_points, seed=None,
                          bg_id=0, forbidden_pairs=None, forbidden_band_px=0, forbidden_pairs_collapse=None,
                          la_id=2, myo_id=4, lv_id=5):
    # 2CH should only have LA, MYO, LV (NO RA, NO RV)
    dense_label_view = _mask_keep_ids(dense_label, keep_ids=[la_id, myo_id, lv_id], bg_id=bg_id)

    vec_one = apex - mvc
    vec_three = rvc - mvc
    vec_two = np.cross(vec_three, vec_one)
    iop_one, iop_two, n, ipp = plane_from_two_axes(vec_one, vec_two, mvc, coh)

    return sparse_pc_from_LAX_plane(
        dense_label_view, iop_one, iop_two, ipp, num_points,
        seed=seed, bg_id=bg_id,
        forbidden_pairs=forbidden_pairs, forbidden_band_px=forbidden_band_px,
        forbidden_pairs_collapse=forbidden_pairs_collapse
    )


# -----------------------------
# SAX
# -----------------------------

def SAX_plane_pc(seg_arr, num_points, seed, myo_id, rv_id, lv_id, ra_id, la_id,
                 bg_id: int = 0,
                 forbidden_pairs=None,
                 forbidden_band_px: int = 0,
                 forbidden_pairs_collapse=None):
    rng = np.random.default_rng(seed)
    seg_arr = np.where(
        (seg_arr == myo_id) | (seg_arr == rv_id) | (seg_arr == lv_id) | (seg_arr == ra_id) | (seg_arr == la_id),
        seg_arr, int(bg_id)
    ).astype(np.uint8)
    Z, Y, X = seg_arr.shape

    lv_rv_array = ((seg_arr == myo_id) | (seg_arr == rv_id)).astype(np.uint8)
    z_line = lv_rv_array.sum(axis=(1, 2))
    z_1, z_2 = find_corner_index(z_line)

    m = 10
    step = (z_2 * 0.95 - z_1) / (m + 1) if z_2 > z_1 else 0.0
    z_slices = np.array([int(z_1 + (s + 1) * step) for s in range(m + 1)], dtype=int)
    z_slices = np.clip(z_slices, 0, Z - 1)

    pcs   = np.zeros((m + 1, num_points, 3), dtype=np.float32)
    pairs = np.zeros((m + 1, num_points, 2), dtype=np.int32)

    for s, z in enumerate(z_slices):
        labels2d = seg_arr[z]

        coords, lpairs = _edge_midpoints_subpixel(
            labels2d,
            return_pairs=True,
            bg_id=int(bg_id),
            forbidden_pairs=forbidden_pairs,
            forbidden_band_px=int(forbidden_band_px),
            forbidden_pairs_collapse=forbidden_pairs_collapse,
        )

        if coords.shape[0] == 0:
            pcs[s, :, 2] = float(z)
            pairs[s, :, :] = int(bg_id)
            continue

        k = min(num_points, coords.shape[0])
        sel = fps_2d(coords, k, seed=None if seed is None else seed + s)

        sampled = coords[sel]
        spairs  = lpairs[sel]

        if k < num_points:
            extra = rng.integers(0, k, size=(num_points - k,))
            sampled = np.vstack([sampled, sampled[extra]])
            spairs  = np.vstack([spairs,  spairs[extra]])

        block = np.zeros((num_points, 3), dtype=np.float32)
        block[:, :2] = sampled[:num_points]
        block[:, 2]  = float(z)

        block[:, 0] = np.clip(block[:, 0], 0, Y - 1)
        block[:, 1] = np.clip(block[:, 1], 0, X - 1)

        # post-sample collapse (extra safety)
        collapse_list = forbidden_pairs_collapse if forbidden_pairs_collapse is not None else forbidden_pairs
        spairs = clean_bg_pairs_np(spairs, bg_id=int(bg_id))
        spairs = collapse_forbidden_pairs_np(spairs, forbidden_pairs=collapse_list, bg_id=int(bg_id))
        spairs.sort(axis=1)

        pcs[s] = block
        pairs[s] = spairs[:num_points].astype(np.int32)

    return pcs, pairs


# -----------------------------
# main
# -----------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate sparse point clouds from 4-chamber label maps.")
    parser.add_argument("--src-root", type=str, required=True, help="Root with patient 4-chamber label subfolders.")
    parser.add_argument("--dst-root", type=str, required=True, help="Where to write outputs.")
    parser.add_argument("--name-contains", type=str, default="merge", help="Substring that must appear in filename (case-insensitive).")
    parser.add_argument("--num-points", type=int, default=400, help="Number of points to sample per slice maximum.")
    parser.add_argument("--rv-id", type=int, default=6, help="Label ID for RV.")
    parser.add_argument("--ra-id", type=int, default=3, help="Label ID for RA.")
    parser.add_argument("--la-id", type=int, default=2, help="Label ID for LA.")
    parser.add_argument("--myo-id", type=int, default=4, help="Label ID for MYO.")
    parser.add_argument("--lv-id", type=int, default=5, help="Label ID for LV.")
    parser.add_argument("--ao-id", type=int, default=1, help="Label ID for Aorta.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for point sampling.")

    # forbidden-pair controls
    parser.add_argument("--bg-id", type=int, default=0, help="Background label id (default 0).")
    parser.add_argument(
        "--forbidden_pair",
        action="append",
        nargs=2,
        type=int,
        default=None,
        help="Forbidden label pairs (two ints). Repeat flag for multiple pairs. (Used for 3D carve + 2D carve + collapse.)"
    )
    parser.add_argument(
        "--forbidden_pair_no_erosion",
        action="append",
        nargs=2,
        type=int,
        default=None,
        help="Forbidden label pairs applied ONLY post-contour via pair-collapse; per case, one pair is chosen at random from this list."
    )
    parser.add_argument("--forbidden_band_vox", type=int, default=2,
                        help="Band size (voxels) for 3D carve and 2D carve (default 2).")
    parser.add_argument("--forbidden_roi_pad", type=int, default=3,
                        help="Extra ROI padding for 3D carve (default 3).")
    parser.add_argument("--forbidden_verbose", action="store_true",
                        help="Verbose forbidden carving logs.")

    args = parser.parse_args()

    if not os.path.exists(args.src_root):
        raise FileNotFoundError(f"Source root not found: {args.src_root}")
    os.makedirs(args.dst_root, exist_ok=True)

    # Normalize lists
    forbidden_pairs_carve = _norm_pairs_list(args.forbidden_pair) if args.forbidden_pair else []
    forbidden_pairs_no_erosion_all = _norm_pairs_list(args.forbidden_pair_no_erosion) if args.forbidden_pair_no_erosion else []

    for subroot, dirs, files in os.walk(args.src_root):
        for file in files:
            if args.name_contains.lower() not in file.lower():
                continue
            if not (file.endswith(".nii") or file.endswith(".nii.gz")):
                continue

            input_path = os.path.join(subroot, file)
            base = file.split("_")[0]
            output_path_npz = os.path.join(args.dst_root, f"{base}.npz")

            try:
                # Choose ONE post-contour forbidden pair for this case (deterministic random)
                chosen_no_erosion = _pick_one_pair_deterministic(
                    forbidden_pairs_no_erosion_all,
                    seed=int(args.seed),
                    key=str(base)
                )
                # Collapse list = (carve list) + (one chosen no-erosion pair)
                forbidden_pairs_collapse = (forbidden_pairs_carve or []) + (chosen_no_erosion or [])

                if args.forbidden_verbose and chosen_no_erosion:
                    print(f"[no-erosion] case={base} chosen forbidden_pair_no_erosion={chosen_no_erosion[0]}")

                label_img = sitk.ReadImage(input_path)
                label_arr = sitk.GetArrayFromImage(label_img)  # (Z,Y,X)

                # landmarks from original (needs Ao for AVC)
                lm = compute_landmarks_cardiac(label_arr, args.lv_id, args.rv_id, args.la_id, args.ra_id)
                apex = lm['apex']; mvc = lm['mvc']; tvc = lm['tvc']; coh = lm['coh']; rvc = lm['rvc']
                avc_cardiac = compute_avc_cardiac(label_arr, args.ao_id, args.lv_id)

                # chamber-only segmentation
                valid_ids = {args.rv_id, args.la_id, args.ra_id, args.myo_id, args.lv_id}
                ch_label_arr = np.where(np.isin(label_arr, list(valid_ids)), label_arr, int(args.bg_id)).astype(np.int32)

                # 3D forbidden carve: ONLY uses forbidden_pairs_carve (not no_erosion)
                if forbidden_pairs_carve and int(args.forbidden_band_vox) > 0:
                    seg_keep_img = sitk.GetImageFromArray(ch_label_arr.astype(np.int32))
                    seg_keep_img.CopyInformation(label_img)
                    seg_keep_img = separate_forbidden_pairs_fast_3d(
                        seg_keep_img,
                        forbidden_pairs=forbidden_pairs_carve,
                        band_vox=int(args.forbidden_band_vox),
                        bg_id=int(args.bg_id),
                        roi_pad=int(args.forbidden_roi_pad),
                        verbose=bool(args.forbidden_verbose),
                    )
                    ch_label_arr = sitk.GetArrayFromImage(seg_keep_img).astype(np.int32)

                ch_label_arr = ch_label_arr.astype(np.uint8)

                # LAX: 2D carve uses forbidden_pairs_carve; post-contour collapse uses forbidden_pairs_collapse
                pc4_pts, pc4_pairs = pc_4ch_from_landmarks(
                    ch_label_arr, apex, mvc, tvc, coh, args.num_points, seed=args.seed + 1,
                    bg_id=int(args.bg_id),
                    forbidden_pairs=forbidden_pairs_carve,
                    forbidden_band_px=int(args.forbidden_band_vox),
                    forbidden_pairs_collapse=forbidden_pairs_collapse,
                )
                pc3_pts, pc3_pairs = pc_3ch_from_landmarks(
                    ch_label_arr, apex, mvc, avc_cardiac, coh, args.num_points, seed=args.seed + 2,
                    bg_id=int(args.bg_id),
                    forbidden_pairs=forbidden_pairs_carve,
                    forbidden_band_px=int(args.forbidden_band_vox),
                    forbidden_pairs_collapse=forbidden_pairs_collapse,
                )
                pc2_pts, pc2_pairs = pc_2ch_from_landmarks(
                    ch_label_arr, apex, mvc, rvc, coh, args.num_points, seed=args.seed + 3,
                    bg_id=int(args.bg_id),
                    forbidden_pairs=forbidden_pairs_carve,
                    forbidden_band_px=int(args.forbidden_band_vox),
                    forbidden_pairs_collapse=forbidden_pairs_collapse,
                )

                lax_pcs   = np.stack([pc4_pts, pc3_pts, pc2_pts], axis=0).astype(np.float32)       # (3,P,3)
                lax_pairs = np.stack([pc4_pairs, pc3_pairs, pc2_pairs], axis=0).astype(np.int32)   # (3,P,2)

                # SAX: 2D carve uses forbidden_pairs_carve; post-contour collapse uses forbidden_pairs_collapse
                sax_pcs, sax_pairs = SAX_plane_pc(
                    ch_label_arr, args.num_points, args.seed,
                    args.myo_id, args.rv_id, args.lv_id, args.ra_id, args.la_id,
                    bg_id=int(args.bg_id),
                    forbidden_pairs=forbidden_pairs_carve,
                    forbidden_band_px=int(args.forbidden_band_vox),
                    forbidden_pairs_collapse=forbidden_pairs_collapse,
                )

                pcs_all   = np.concatenate([sax_pcs, lax_pcs], axis=0).astype(np.float32)     # (S,P,3)
                pairs_all = np.concatenate([sax_pairs, lax_pairs], axis=0).astype(np.int32)   # (S,P,2)

                np.savez_compressed(output_path_npz, pcs=pcs_all, pairs=pairs_all)

                total_points = int(pcs_all.shape[0] * pcs_all.shape[1])
                print(f"Processed {input_path} -> {output_path_npz}, total points: {total_points}")

            except Exception as e:
                print(f"[WARN] Skipping {input_path} due to error: {e}")
                print("----- TRACEBACK -----")
                print(traceback.format_exc())
