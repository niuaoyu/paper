#!/usr/bin/env python3
"""Downsample SPCs and GT meshes to a fixed vertex count (the final preprocessing step).

`generate_spc.py` and `generate_gt_mesh.py` produce a *variable* number of points per
case. Bi-PT trains on tensors with a fixed size (`N0 = M0 = 5632` in the paper), so this
script resamples every case to exactly ``--target-n`` points — hence the ``spc_ds`` /
``gt_ds_5632_vtp`` ("ds" = down-sampled) folder names.

Two modes:

* ``--mode sparse`` — farthest-point-sample the ``.npz`` point cloud (``pcs`` + ``pairs``)
  down to ``target-n``. Cases with fewer points are skipped and logged (no padding).
* ``--mode gt`` — farthest-point-sample the mesh vertices, then **remap the triangles onto
  the kept vertices with a label-constrained nearest-neighbor search** so the boundary
  labels stay intact: (1) exact boundary-pair match, (2) share-one-label fallback,
  (3) global nearest neighbor. ``VertexBoundaryLabels`` is required; ``VertexLabels3`` is
  preserved when present (3-label variant).

Because the GT path needs *more* than ``target-n`` vertices to sample down from, run
`generate_gt_mesh.py` with a higher ``--target_node_num`` (e.g. ~8000) and let this step
bring it to exactly 5632. Likewise pick ``generate_spc.py --num-points`` so ``14 * P`` >
``target-n``.

Self-contained: needs only ``vtk``, ``numpy``, ``scipy``.
"""

import os
import glob
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
from scipy.spatial import cKDTree


class UndersampledError(ValueError):
    pass


# ----------------------------- FPS -----------------------------
def farthest_point_sample(xyz: np.ndarray, npoint: int, seed: int | None = None) -> np.ndarray:
    N = xyz.shape[0]
    rng = np.random.default_rng(seed)
    centroids = np.zeros(npoint, dtype=np.int64)
    distance = np.full(N, 1e10, dtype=np.float32)
    farthest = int(rng.integers(0, N))
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest]
        dist = ((xyz - centroid) ** 2).sum(axis=1)
        distance = np.minimum(distance, dist)
        farthest = int(distance.argmax())
    return centroids


def _file_seed(base_seed: int, name: str) -> int:
    return (int(base_seed) + (hash(name) & 0xFFFF)) & 0x7FFFFFFF


# ----------------------------- sparse (.npz) -----------------------------
def load_sparse_npz(npz_path: str):
    npz = np.load(npz_path, allow_pickle=False)
    if "pcs" not in npz:
        raise KeyError(f"{npz_path}: missing key 'pcs'")
    pcs = np.asarray(npz["pcs"])
    if pcs.ndim == 3 and pcs.shape[-1] == 3:
        pcs = pcs.reshape(-1, 3)
    if pcs.ndim != 2 or pcs.shape[1] != 3:
        raise ValueError(f"{npz_path}: pcs expected (N,3) or (S,P,3), got {pcs.shape}")
    pcs = pcs.astype(np.float32)

    pairs = None
    if "pairs" in npz:
        pairs = np.asarray(npz["pairs"])
        if pairs.ndim == 3 and pairs.shape[-1] == 2:
            pairs = pairs.reshape(-1, 2)
        if pairs.ndim != 2 or pairs.shape[1] < 2:
            raise ValueError(f"{npz_path}: pairs expected (N,2) or (S,P,2), got {pairs.shape}")
        pairs = pairs[:, :2].astype(np.int32)
        if pairs.shape[0] != pcs.shape[0]:
            raise ValueError(f"{npz_path}: pcs N={pcs.shape[0]} != pairs N={pairs.shape[0]}")
    return pcs, pairs


def _process_sparse_file(sp_file: str, target_n: int, out_dir: str, seed: int):
    try:
        pcs, pairs = load_sparse_npz(sp_file)
        base = os.path.basename(sp_file)
        if pcs.shape[0] < target_n:
            return ("undersampled", sp_file, f"[SPARSE] Skipping {base}: {pcs.shape[0]} < target {target_n}.")

        idx = farthest_point_sample(pcs, target_n, seed=_file_seed(seed, base))
        out = {"pcs": pcs[idx].astype(np.float32)}
        if pairs is not None:
            out["pairs"] = pairs[idx].astype(np.int32)

        os.makedirs(out_dir, exist_ok=True)
        np.savez_compressed(os.path.join(out_dir, base), **out)
        return ("ok", sp_file, f"[SPARSE] {base}: {pcs.shape[0]} -> {target_n} (pairs={'yes' if pairs is not None else 'no'})")
    except Exception as e:
        return ("error", sp_file, f"[SPARSE] Error {sp_file}: {e}")


# ----------------------------- GT mesh (.vtp) -----------------------------
def _read_vtp(path: str) -> vtk.vtkPolyData:
    r = vtk.vtkXMLPolyDataReader()
    r.SetFileName(path)
    r.Update()
    poly = r.GetOutput()
    if poly is None or poly.GetNumberOfPoints() == 0:
        raise RuntimeError(f"No points in VTP: {path}")
    return poly


def _write_vtp(poly: vtk.vtkPolyData, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    w = vtk.vtkXMLPolyDataWriter()
    w.SetFileName(path)
    w.SetInputData(poly)
    w.Write()


def _get_array(poly: vtk.vtkPolyData, name: str, required: bool):
    arr = poly.GetPointData().GetArray(name)
    if arr is None:
        if required:
            names = [poly.GetPointData().GetArrayName(i) for i in range(poly.GetPointData().GetNumberOfArrays())]
            raise RuntimeError(f"Missing required point array '{name}'. Available: {names}")
        return None
    out = vtk_to_numpy(arr)
    return out.reshape(-1, 1) if out.ndim == 1 else out


def _attach_array(poly, name, mat, active=False):
    mat = np.asarray(mat)
    if mat.ndim == 1:
        mat = mat.reshape(-1, 1)
    v = numpy_to_vtk(mat.astype(np.int32), deep=1)
    v.SetName(name)
    v.SetNumberOfComponents(mat.shape[1])
    poly.GetPointData().AddArray(v)
    if active:
        poly.GetPointData().SetActiveScalars(name)


def _tri_cells(poly: vtk.vtkPolyData) -> np.ndarray:
    tri = vtk.vtkTriangleFilter()
    tri.SetInputData(poly)
    tri.Update()
    polys = tri.GetOutput().GetPolys()
    polys.InitTraversal()
    idlist = vtk.vtkIdList()
    out = []
    while polys.GetNextCell(idlist):
        if idlist.GetNumberOfIds() == 3:
            out.append([idlist.GetId(0), idlist.GetId(1), idlist.GetId(2)])
    return np.asarray(out, dtype=np.int64)


def label_constrained_downsample(poly, target_n, pair_array, vert3_array, seed):
    """FPS the vertices to target_n, remap triangles with a label-constrained NN search."""
    pts = vtk_to_numpy(poly.GetPoints().GetData()).astype(np.float32)
    N = pts.shape[0]
    if N == target_n:
        return poly
    if N < target_n:
        raise UndersampledError(f"{N} points < target {target_n} (over-generate the mesh first).")

    pair_lbl = _get_array(poly, pair_array, required=True).astype(np.int32)
    pair_canon = np.sort(pair_lbl, axis=1) if pair_lbl.shape[1] >= 2 else pair_lbl
    vert3_lbl = _get_array(poly, vert3_array, required=False)
    if vert3_lbl is not None:
        vert3_lbl = vert3_lbl.astype(np.int32)

    tris = _tri_cells(poly)

    idx_kept = farthest_point_sample(pts, target_n, seed=seed)
    kept_pts = pts[idx_kept]
    kept_pair = pair_lbl[idx_kept]
    kept_pair_canon = pair_canon[idx_kept]
    kept_vert3 = vert3_lbl[idx_kept] if vert3_lbl is not None else None
    K = kept_pts.shape[0]

    rep = np.full(N, -1, dtype=np.int64)

    # Pass 1: exact boundary-pair match
    key_to_kept = {}
    for ki in range(K):
        key_to_kept.setdefault(tuple(kept_pair_canon[ki].tolist()), []).append(ki)
    for key, kept_list in key_to_kept.items():
        kept_list = np.asarray(kept_list, dtype=np.int64)
        mask = np.all(pair_canon == np.asarray(key, dtype=np.int32).reshape(1, -1), axis=1)
        if not np.any(mask):
            continue
        _, nn = cKDTree(kept_pts[kept_list]).query(pts[mask], k=1)
        rep[mask] = kept_list[nn]

    # Pass 2: share-one-label fallback
    unmapped = rep == -1
    if np.any(unmapped):
        label_to_tree = {}
        label_to_kept = {}
        for ki in range(K):
            for v in np.unique(kept_pair_canon[ki]).tolist():
                label_to_kept.setdefault(int(v), []).append(ki)
        for v, idxs in label_to_kept.items():
            idxs = np.asarray(idxs, dtype=np.int64)
            label_to_tree[v] = (cKDTree(kept_pts[idxs]), idxs)
        for oi in np.where(unmapped)[0]:
            best_d, best_k = None, None
            for v in np.unique(pair_canon[oi]).tolist():
                v = int(v)
                if v not in label_to_tree:
                    continue
                tree, idxs = label_to_tree[v]
                d, nn = tree.query(pts[oi], k=1)
                if best_d is None or float(d) < best_d:
                    best_d, best_k = float(d), int(idxs[int(nn)])
            if best_k is not None:
                rep[oi] = best_k

    # Pass 3: global NN fallback
    unmapped = rep == -1
    if np.any(unmapped):
        _, nn = cKDTree(kept_pts).query(pts[unmapped], k=1)
        rep[unmapped] = nn

    # remap + drop degenerate/unmapped triangles
    tm = rep[tris]
    keep = ((tm != -1).all(axis=1)
            & (tm[:, 0] != tm[:, 1]) & (tm[:, 1] != tm[:, 2]) & (tm[:, 0] != tm[:, 2]))
    tm = tm[keep]

    out_pts = vtk.vtkPoints()
    out_pts.SetData(numpy_to_vtk(kept_pts, deep=1))
    cells = vtk.vtkCellArray()
    for a, b, c in tm:
        cells.InsertNextCell(3)
        cells.InsertCellPoint(int(a)); cells.InsertCellPoint(int(b)); cells.InsertCellPoint(int(c))
    out = vtk.vtkPolyData()
    out.SetPoints(out_pts)
    out.SetPolys(cells)

    has_v3 = kept_vert3 is not None
    _attach_array(out, pair_array, kept_pair, active=not has_v3)
    if has_v3:
        _attach_array(out, vert3_array, kept_vert3, active=True)

    print(f"[GT] {N} -> {out.GetNumberOfPoints()} verts, {out.GetNumberOfCells()} tris "
          f"(kept arrays: {pair_array}{'+' + vert3_array if has_v3 else ''})")
    return out


def _process_gt_file(gt_file, target_n, out_dir, pair_array, vert3_array, seed):
    try:
        poly = label_constrained_downsample(
            _read_vtp(gt_file), target_n, pair_array, vert3_array,
            seed=_file_seed(seed, os.path.basename(gt_file)),
        )
        base = os.path.splitext(os.path.basename(gt_file))[0]
        out_path = os.path.join(out_dir, base + ".vtp")
        _write_vtp(poly, out_path)
        return ("ok", gt_file, f"[GT] {base}: -> {out_path} ({poly.GetNumberOfPoints()} verts)")
    except UndersampledError as ue:
        return ("undersampled", gt_file, f"[GT] Skipping {gt_file}: {ue}")
    except Exception as e:
        return ("error", gt_file, f"[GT] Error {gt_file}: {e}")


def _run_dir(files, worker, out_dir, workers, log_name):
    os.makedirs(out_dir, exist_ok=True)
    if not files:
        print(f"  (no input files)")
        return
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(*w) for w in worker]
        for fut in as_completed(futs):
            status, path, msg = fut.result()
            print(msg)
            if status == "undersampled":
                with open(os.path.join(out_dir, log_name), "a") as f:
                    f.write(path + "\n")


def main():
    p = argparse.ArgumentParser(description="Downsample SPCs / GT meshes to a fixed point count.")
    p.add_argument("--mode", required=True, choices=["sparse", "gt", "both"])
    p.add_argument("--sparse-in", type=str, default=None, help="dir of raw .npz SPCs")
    p.add_argument("--sparse-out", type=str, default=None, help="dir for downsampled .npz")
    p.add_argument("--gt-in", type=str, default=None, help="dir of raw labeled .vtp meshes")
    p.add_argument("--gt-out", type=str, default=None, help="dir for downsampled .vtp")
    p.add_argument("--target-n", type=int, default=5632, help="target point/vertex count (default 5632)")
    p.add_argument("--pair-array", type=str, default="VertexBoundaryLabels")
    p.add_argument("--vert3-array", type=str, default="VertexLabels3", help="optional 3-label array (preserved if present)")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if args.mode in ("sparse", "both"):
        assert args.sparse_in and args.sparse_out, "--sparse-in/--sparse-out required for sparse mode"
        files = sorted(glob.glob(os.path.join(args.sparse_in, "*.npz")))
        print(f"=== SPARSE: {len(files)} files, {args.sparse_in} -> {args.sparse_out} (N={args.target_n}) ===")
        _run_dir(files,
                 [( _process_sparse_file, f, args.target_n, args.sparse_out, args.seed) for f in files],
                 args.sparse_out, args.workers, "undersampled_files.txt")

    if args.mode in ("gt", "both"):
        assert args.gt_in and args.gt_out, "--gt-in/--gt-out required for gt mode"
        files = sorted(glob.glob(os.path.join(args.gt_in, "*.vtp")))
        print(f"=== GT: {len(files)} files, {args.gt_in} -> {args.gt_out} (N={args.target_n}) ===")
        _run_dir(files,
                 [(_process_gt_file, f, args.target_n, args.gt_out, args.pair_array, args.vert3_array, args.seed) for f in files],
                 args.gt_out, args.workers, "undersampled_files.txt")

    print("Done.")


if __name__ == "__main__":
    main()
