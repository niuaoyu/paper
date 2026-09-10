#!/usr/bin/env python3
"""Convert one DeepSDF ``3d_data`` case to the NPZ layout expected by NDM.

The public NDM code expects, for every case and phase, four unnamed NPZ arrays:

    HR_<PHASE>_pc.npz        (5600, 3)
    HR_<PHASE>_epi_myo.npz   (3000, 3)
    HR_<PHASE>_endo_myo.npz  (3000, 3)
    HR_<PHASE>_rv.npz        (3000, 3)

DeepSDF ``3d_data`` contains only two dense mesh labels (LV and RV), not the
three dense NDM targets (LV epicardium, LV endocardium, RV).  Consequently,
this converter refuses to invent the missing LV surface by default.  The
``--approximate-missing-lv`` option exists only to make a pipeline smoke test;
its normal-offset surface is not a scientifically valid replacement for a
segmented LV epicardial/endocardial surface.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import trimesh


def load_npz_first(path: Path) -> np.ndarray:
    with np.load(path) as data:
        if not data.files:
            raise ValueError(f"empty NPZ: {path}")
        return np.asarray(data[data.files[0]])


def load_point_or_mesh(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        array = np.load(path)
    elif suffix == ".npz":
        array = load_npz_first(path)
    elif suffix == ".ply":
        geometry = trimesh.load(path, process=False)
        if isinstance(geometry, trimesh.Scene):
            geometry = geometry.dump(concatenate=True)
        array = np.asarray(geometry.vertices)
    else:
        raise ValueError(f"unsupported surface format: {path}")
    array = np.asarray(array, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] < 3:
        raise ValueError(f"surface must be (N, >=3), got {array.shape}: {path}")
    return array[:, :3]


def load_labeled_mesh(path: Path) -> tuple[trimesh.Trimesh, np.ndarray]:
    mesh = trimesh.load(path, process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"expected a triangular mesh: {path}")
    raw = mesh.metadata.get("_ply_raw", {}).get("vertex", {}).get("data")
    if raw is None or "label" not in raw.dtype.names:
        raise ValueError(f"PLY has no per-vertex 'label' property: {path}")
    labels = np.asarray(raw["label"])
    if len(labels) != len(mesh.vertices):
        raise ValueError(f"label/vertex count mismatch: {path}")
    return mesh, labels


def mesh_for_label(mesh: trimesh.Trimesh, labels: np.ndarray, label: int) -> trimesh.Trimesh:
    face_labels = labels[np.asarray(mesh.faces)]
    face_mask = np.all(face_labels == label, axis=1)
    if not np.any(face_mask):
        raise ValueError(f"mesh contains no all-{label} faces")
    global_faces = np.asarray(mesh.faces)[face_mask]
    vertices_used, inverse = np.unique(global_faces.reshape(-1), return_inverse=True)
    local_faces = inverse.reshape(-1, 3)
    return trimesh.Trimesh(
        vertices=np.asarray(mesh.vertices)[vertices_used],
        faces=local_faces,
        process=False,
    )


def sample_surface(mesh: trimesh.Trimesh, count: int, rng: np.random.Generator) -> np.ndarray:
    triangles = np.asarray(mesh.vertices)[np.asarray(mesh.faces)]
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    valid = np.isfinite(areas) & (areas > 0)
    if not np.any(valid):
        raise ValueError("mesh has no finite non-degenerate triangles")
    valid_indices = np.flatnonzero(valid)
    probabilities = areas[valid] / areas[valid].sum()
    chosen = rng.choice(valid_indices, size=count, replace=True, p=probabilities)
    tri = triangles[chosen]
    r1 = np.sqrt(rng.random(count))
    r2 = rng.random(count)
    bary = np.column_stack((1.0 - r1, r1 * (1.0 - r2), r1 * r2))
    return np.einsum("ni,nij->nj", bary, tri)


def offset_mesh(mesh: trimesh.Trimesh, distance: float) -> trimesh.Trimesh:
    result = mesh.copy()
    # A watertight positive-volume component gives outward-facing normals.
    result.fix_normals(multibody=True)
    result.vertices = np.asarray(result.vertices) + distance * np.asarray(result.vertex_normals)
    return result


def raw_points_to_canonical(points: np.ndarray, case_dir: Path, norm_kind: str) -> np.ndarray:
    pose = np.load(case_dir / "pose.npy")
    params = np.load(case_dir / f"{norm_kind}.npz")
    offset = np.asarray(params["offset"], dtype=np.float64)
    scale = float(params["scale"])
    homogeneous = np.column_stack((points, np.ones(len(points))))
    posed = (pose @ homogeneous.T).T[:, :3]
    return (posed + offset) * scale


def fit_ndm_axis_normalization(input_points: np.ndarray) -> dict[str, np.ndarray]:
    centroid = input_points.mean(axis=0)
    centered = input_points - centroid
    minimum = centered.min(axis=0)
    maximum = centered.max(axis=0)
    extent = maximum - minimum
    if np.any(extent <= 1e-12):
        raise ValueError(f"degenerate input extent: {extent}")
    return {
        "centroid": centroid,
        "minimum": minimum,
        "maximum": maximum,
        "axis_scale": 1.7 / extent,
    }


def apply_ndm_axis_normalization(points: np.ndarray, params: dict[str, np.ndarray]) -> np.ndarray:
    centered = points - params["centroid"]
    return (centered - params["minimum"]) * params["axis_scale"] - 0.85


def rotation_lv_minus_rv_to_positive_y(lv: np.ndarray, rv: np.ndarray) -> tuple[np.ndarray, float]:
    vector = lv.mean(axis=0) - rv.mean(axis=0)
    if np.linalg.norm(vector[:2]) <= 1e-12:
        return np.eye(3), 0.0
    angle = float(np.arctan2(vector[0], vector[1]))
    c, s = np.cos(angle), np.sin(angle)
    rotation = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return rotation, angle


def apply_rotation(points: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    return np.asarray(points) @ rotation.T


def resize_input(points: np.ndarray, count: int, rng: np.random.Generator) -> np.ndarray:
    if len(points) == 0:
        raise ValueError("input point cloud is empty")
    if len(points) >= count:
        return points[rng.choice(len(points), size=count, replace=False)]
    # Point Transformer is permutation invariant; retain every observed point and
    # repeat a shuffled subset, matching the public NDM helper's N < npoint behavior.
    order = rng.permutation(len(points))
    repeats = np.resize(order, count)
    return points[repeats]


def jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def save_unnamed_npz(path: Path, array: np.ndarray) -> None:
    array = np.asarray(array, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != 3 or not np.isfinite(array).all():
        raise ValueError(f"invalid output {path.name}: shape={array.shape}")
    np.savez_compressed(path, array)


def convert_phase(args: argparse.Namespace, phase: str, output_case: Path) -> dict[str, Any]:
    root = args.input_root
    rng = np.random.default_rng(args.seed + (0 if phase == "ED" else 1))

    combined, labels = load_labeled_mesh(root / "mesh" / args.case / f"{phase}.ply")
    lv_mesh = mesh_for_label(combined, labels, args.lv_label)
    rv_mesh = mesh_for_label(combined, labels, args.rv_label)

    lv_surface = sample_surface(lv_mesh, args.target_points, rng)
    rv_surface = sample_surface(rv_mesh, args.target_points, rng)

    if args.other_lv_surface is not None:
        other_path = Path(str(args.other_lv_surface).format(case=args.case, phase=phase))
        other_points = load_point_or_mesh(other_path)
        if len(other_points) >= args.target_points:
            other_surface = other_points[rng.choice(len(other_points), args.target_points, replace=False)]
        else:
            other_surface = other_points[np.resize(rng.permutation(len(other_points)), args.target_points)]
        missing_source = str(other_path)
    elif args.approximate_missing_lv:
        sign = 1.0 if args.lv_role == "endo" else -1.0
        other_surface = sample_surface(offset_mesh(lv_mesh, sign * args.wall_offset), args.target_points, rng)
        missing_source = f"DEBUG normal offset {sign * args.wall_offset:+g} in canonical coordinates"
    else:
        missing = "epicardial" if args.lv_role == "endo" else "endocardial"
        raise RuntimeError(
            f"DeepSDF label {args.lv_label} supplies only one LV surface; NDM also requires an LV {missing} "
            "surface. Pass --other-lv-surface or, for smoke testing only, --approximate-missing-lv."
        )

    if args.lv_role == "endo":
        endo_surface, epi_surface = lv_surface, other_surface
    else:
        epi_surface, endo_surface = lv_surface, other_surface

    sparse_geometry = trimesh.load(root / "points" / args.case / f"{phase}.ply", process=False)
    if isinstance(sparse_geometry, trimesh.Scene):
        sparse_geometry = sparse_geometry.dump(concatenate=True)
    sparse_raw = np.asarray(sparse_geometry.vertices, dtype=np.float64)
    sparse = raw_points_to_canonical(sparse_raw, root / "norm" / args.case, args.norm_kind)
    sparse = resize_input(sparse, args.input_points, rng)

    surfaces = {"pc": sparse, "epi": epi_surface, "endo": endo_surface, "rv": rv_surface}

    first_norm = fit_ndm_axis_normalization(surfaces["pc"])
    surfaces = {name: apply_ndm_axis_normalization(data, first_norm) for name, data in surfaces.items()}

    rotation = np.eye(3)
    angle = 0.0
    second_norm = None
    if args.align_lv_rv_y:
        rotation, angle = rotation_lv_minus_rv_to_positive_y(surfaces[args.lv_role], surfaces["rv"])
        surfaces = {name: apply_rotation(data, rotation) for name, data in surfaces.items()}
        # The released preprocessing script normalizes once before and once after z rotation.
        second_norm = fit_ndm_axis_normalization(surfaces["pc"])
        surfaces = {name: apply_ndm_axis_normalization(data, second_norm) for name, data in surfaces.items()}

    prefix = f"HR_{phase}"
    save_unnamed_npz(output_case / f"{prefix}_pc.npz", surfaces["pc"])
    save_unnamed_npz(output_case / f"{prefix}_epi_myo.npz", surfaces["epi"])
    save_unnamed_npz(output_case / f"{prefix}_endo_myo.npz", surfaces["endo"])
    save_unnamed_npz(output_case / f"{prefix}_rv.npz", surfaces["rv"])

    face_labels = labels[np.asarray(combined.faces)]
    report = {
        "phase": phase,
        "source": {
            "mesh": str(root / "mesh" / args.case / f"{phase}.ply"),
            "points": str(root / "points" / args.case / f"{phase}.ply"),
            "mesh_vertex_count": len(combined.vertices),
            "mesh_face_count": len(combined.faces),
            "label_counts": {str(int(v)): int(np.sum(labels == v)) for v in np.unique(labels)},
            "cross_label_face_count": int(np.sum(np.any(face_labels != face_labels[:, :1], axis=1))),
            "raw_sparse_count": len(sparse_raw),
        },
        "mapping": {
            "deep_sdf_lv_label": args.lv_label,
            "deep_sdf_lv_assumed_role": args.lv_role,
            "deep_sdf_rv_label": args.rv_label,
            "missing_lv_surface_source": missing_source,
        },
        "normalization": {
            "deep_sdf_norm_kind": args.norm_kind,
            "first_ndm_axis_normalization": first_norm,
            "align_lv_minus_rv_to_positive_y": args.align_lv_rv_y,
            "z_rotation_radians": angle,
            "rotation": rotation,
            "second_ndm_axis_normalization": second_norm,
        },
        "outputs": {
            name: {
                "shape": list(data.shape),
                "dtype": "float32",
                "minimum": data.min(axis=0),
                "maximum": data.max(axis=0),
            }
            for name, data in surfaces.items()
        },
        "warning": (
            "The generated missing LV surface is an artificial normal offset and must not be used for scientific training or evaluation."
            if args.approximate_missing_lv and args.other_lv_surface is None
            else None
        ),
    }
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=Path("/home/nay/github/DeepSDF/3d_data"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--case", default="0")
    parser.add_argument("--phases", nargs="+", choices=("ED", "ES"), default=("ED", "ES"))
    parser.add_argument("--lv-label", type=int, default=1)
    parser.add_argument("--rv-label", type=int, default=2)
    parser.add_argument("--lv-role", choices=("endo", "epi"), default="endo")
    parser.add_argument(
        "--other-lv-surface",
        help="PLY/NPY/NPZ path for the missing LV surface; supports {case} and {phase} placeholders and must already use mesh canonical coordinates.",
    )
    parser.add_argument("--approximate-missing-lv", action="store_true", help="DEBUG ONLY: synthesize the missing LV surface by normal offset")
    parser.add_argument("--wall-offset", type=float, default=0.08, help="canonical-coordinate offset used only with --approximate-missing-lv")
    parser.add_argument("--norm-kind", choices=("unit_sphere", "aabb"), default="unit_sphere")
    parser.add_argument("--input-points", type=int, default=5600)
    parser.add_argument("--target-points", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=20230717)
    parser.add_argument("--align-lv-rv-y", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_case = args.output_root / args.case
    output_case.mkdir(parents=True, exist_ok=True)
    reports = [convert_phase(args, phase, output_case) for phase in args.phases]
    report_path = output_case / "conversion_report.json"
    report_path.write_text(json.dumps(jsonable({"case": args.case, "phases": reports}), indent=2), encoding="utf-8")
    print(f"wrote NDM-layout data to {output_case}")
    print(f"wrote conversion report to {report_path}")
    for report in reports:
        print(report["phase"], {name: item["shape"] for name, item in report["outputs"].items()})
        if report["warning"]:
            print("WARNING:", report["warning"])


if __name__ == "__main__":
    main()
