#!/usr/bin/env python3
"""Derive the 3-label vertex array (``VertexLabels3``) for the 3-label variant.

The 2-label ground truth (`generate_gt_mesh.py`) tags every vertex with its boundary
*pair* — the two structures it sits between (``VertexBoundaryLabels``, 2 slots). At
triple junctions three structures meet, which a pair cannot express. This script builds
a 3-slot per-vertex array from the mesh's **cell** boundary labels:

  For each vertex:
    1. union the label components of every incident triangle's ``--cell_array``
       (``BoundaryLabels``, the 2-vector cell array written by generate_gt_mesh);
    2. drop background (``--bg_id``) and negative labels;
    3. sort the unique labels ascending;
    4. write the first three as ``VertexLabels3 = (L0, L1, L2)`` (padded with -1),
       plus ``VertexLabelCount`` = the true unique count (may exceed 3).

No voting — it is a pure set union, so a vertex touching LA/LV/MYO gets all three.
Run this on the **raw** mesh (which still carries the ``BoundaryLabels`` cell array)
*before* ``downsample.py`` (which then preserves ``VertexLabels3``). The result feeds
the 3-label training/eval path (``--label-dim 3`` / ``VertexLabels3``) and the
``atlas_3lab.vtp`` template.

Self-contained: needs only ``vtk`` and ``numpy``.
"""

import os
import argparse
import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk


def read_polydata(fn: str) -> vtk.vtkPolyData:
    ext = os.path.splitext(fn)[1].lower()
    if ext == ".vtp":
        r = vtk.vtkXMLPolyDataReader()
    elif ext == ".vtk":
        r = vtk.vtkPolyDataReader()
    else:
        raise ValueError(f"Unsupported input extension: {ext} (use .vtp or .vtk)")
    r.SetFileName(fn)
    r.Update()
    poly = r.GetOutput()
    if poly is None or poly.GetNumberOfPoints() == 0:
        raise RuntimeError(f"Failed to read polydata or empty mesh: {fn}")
    return poly


def write_polydata(poly: vtk.vtkPolyData, fn: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(fn)), exist_ok=True)
    ext = os.path.splitext(fn)[1].lower()
    if ext == ".vtp":
        w = vtk.vtkXMLPolyDataWriter()
    elif ext == ".vtk":
        w = vtk.vtkPolyDataWriter()
    else:
        raise ValueError(f"Unsupported output extension: {ext} (use .vtp or .vtk)")
    w.SetFileName(fn)
    w.SetInputData(poly)
    if w.Write() != 1:
        raise RuntimeError(f"Failed to write: {fn}")


def get_cell_labels_matrix(poly: vtk.vtkPolyData, cell_array: str) -> np.ndarray:
    arr = poly.GetCellData().GetArray(cell_array)
    if arr is None:
        names = [poly.GetCellData().GetArrayName(i) for i in range(poly.GetCellData().GetNumberOfArrays())]
        raise RuntimeError(f"CellData array not found: {cell_array}. Available: {names}")
    n_cells = poly.GetNumberOfCells()
    ncomp = arr.GetNumberOfComponents()
    return vtk_to_numpy(arr).astype(np.int32).reshape((n_cells, ncomp))


def assign_vertex_labels_3slots_union(
    poly: vtk.vtkPolyData,
    cell_labels: np.ndarray,               # (n_cells, ncomp)
    out_name: str = "VertexLabels3",
    out_count_name: str = "VertexLabelCount",
    bg_id: int = 0,
    keep_bg: bool = False,
    keep_negative: bool = False,
    pad_value: int = -1,
    progress_every: int = 0,
) -> vtk.vtkPolyData:
    n_cells = poly.GetNumberOfCells()
    n_pts = poly.GetNumberOfPoints()

    cell_labels = np.asarray(cell_labels, dtype=np.int32)
    if cell_labels.shape[0] != n_cells:
        raise ValueError(f"cell_labels first dim must be n_cells={n_cells}, got {cell_labels.shape}")

    poly.BuildLinks()
    idlist = vtk.vtkIdList()

    out = np.full((n_pts, 3), int(pad_value), dtype=np.int32)
    out_count = np.zeros((n_pts,), dtype=np.int32)

    for pid in range(n_pts):
        if progress_every and pid > 0 and (pid % progress_every == 0):
            print(f"[progress] {pid}/{n_pts} points")

        poly.GetPointCells(pid, idlist)
        k = idlist.GetNumberOfIds()
        if k == 0:
            continue

        # Union labels over incident cells
        labs = np.concatenate([cell_labels[idlist.GetId(j)] for j in range(k)]).astype(np.int32, copy=False)

        if not keep_negative:
            labs = labs[labs >= 0]
        if not keep_bg:
            labs = labs[labs != int(bg_id)]

        if labs.size == 0:
            continue

        uniq = np.unique(labs)  # sorted ascending
        out_count[pid] = int(uniq.size)
        take = min(3, int(uniq.size))
        out[pid, :take] = uniq[:take].astype(np.int32)

    pd = poly.GetPointData()

    vtk_out = numpy_to_vtk(out, deep=True, array_type=vtk.VTK_INT)
    vtk_out.SetName(out_name)
    vtk_out.SetNumberOfComponents(3)

    vtk_cnt = numpy_to_vtk(out_count, deep=True, array_type=vtk.VTK_INT)
    vtk_cnt.SetName(out_count_name)
    vtk_cnt.SetNumberOfComponents(1)

    pd.RemoveArray(out_name)
    pd.RemoveArray(out_count_name)
    pd.AddArray(vtk_out)
    pd.AddArray(vtk_cnt)
    pd.SetActiveScalars(out_count_name)

    return poly


def main():
    ap = argparse.ArgumentParser(description="Add a 3-slot VertexLabels3 array from cell BoundaryLabels.")
    ap.add_argument("--in_mesh", required=True, help="Input mesh (.vtp or .vtk) with a CellData label array")
    ap.add_argument("--out_mesh", required=True, help="Output mesh (.vtp or .vtk)")
    ap.add_argument("--cell_array", default="BoundaryLabels", help="CellData label array name (from generate_gt_mesh.py)")
    ap.add_argument("--out_array", default="VertexLabels3", help="PointData output array name (3 components)")
    ap.add_argument("--count_array", default="VertexLabelCount", help="PointData count array name")
    ap.add_argument("--bg_id", type=int, default=0)
    ap.add_argument("--keep_bg", action="store_true")
    ap.add_argument("--keep_negative", action="store_true")
    ap.add_argument("--pad_value", type=int, default=-1)
    ap.add_argument("--progress_every", type=int, default=0)
    args = ap.parse_args()

    poly = read_polydata(args.in_mesh)
    cell_mat = get_cell_labels_matrix(poly, args.cell_array)

    poly = assign_vertex_labels_3slots_union(
        poly=poly,
        cell_labels=cell_mat,
        out_name=args.out_array,
        out_count_name=args.count_array,
        bg_id=int(args.bg_id),
        keep_bg=bool(args.keep_bg),
        keep_negative=bool(args.keep_negative),
        pad_value=int(args.pad_value),
        progress_every=int(args.progress_every),
    )

    write_polydata(poly, args.out_mesh)
    print(f"[OK] wrote: {args.out_mesh}")
    print(f"     Cell array:  {args.cell_array}")
    print(f"     Point array: {args.out_array} (3 slots, padded with {args.pad_value})")
    print(f"     Count array: {args.count_array} (true unique count, may be >3)")


if __name__ == "__main__":
    main()
