#!/usr/bin/env python3
"""Create a corrected DeepSDF data root from existing labeled sparse PLY files.

The legacy NIfTI acquisition inputs are not part of ``3d_data``.  This utility
therefore does not rerun contour extraction.  It repairs the existing derived
``points`` files by aligning their encoded LV-minus-RV direction with the
labeled canonical mesh, while preserving every PLY label and color byte.

``mesh``, ``norm`` and ``sdf`` are linked read-only into a new data root.  The
source data is never overwritten.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np


POINT_DTYPE = np.dtype([
    ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
    ("label", "u1"), ("red", "u1"), ("green", "u1"), ("blue", "u1"),
])


def read_labeled_ply(path: Path):
    with path.open("rb") as stream:
        header_lines = []
        vertex_count = None
        while True:
            line = stream.readline()
            if not line:
                raise ValueError(f"incomplete PLY header: {path}")
            header_lines.append(line)
            text = line.decode("ascii").strip()
            if text.startswith("element vertex "):
                vertex_count = int(text.split()[-1])
            if text == "end_header":
                break
        if vertex_count is None:
            raise ValueError(f"PLY has no vertex count: {path}")
        records = np.fromfile(stream, dtype=POINT_DTYPE, count=vertex_count)
        trailing = stream.read()
    if len(records) != vertex_count:
        raise ValueError(f"short PLY vertex payload: {path}")
    points = np.column_stack((records["x"], records["y"], records["z"])).astype(np.float64)
    return b"".join(header_lines), records, trailing, points


def write_labeled_ply(path: Path, header: bytes, records, trailing: bytes, points):
    output = records.copy()
    output["x"] = points[:, 0]
    output["y"] = points[:, 1]
    output["z"] = points[:, 2]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        stream.write(header)
        output.tofile(stream)
        stream.write(trailing)


def to_canonical(points, pose, offset, scale):
    homogeneous = np.column_stack((points, np.ones(len(points))))
    posed = (pose @ homogeneous.T).T[:, :3]
    return scale * (posed + offset)


def from_canonical(points, pose, offset, scale):
    posed = points / scale - offset
    homogeneous = np.column_stack((posed, np.ones(len(posed))))
    raw = (np.linalg.inv(pose) @ homogeneous.T).T
    return raw[:, :3] / raw[:, 3:4]


def wrap_angle(angle):
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def encoded_input_groups(labels):
    labels = np.asarray(labels, dtype=np.int64)
    hundreds = labels // 100
    if np.any(hundreds == 1) and np.any(hundreds == 2):
        return hundreds
    if np.any(labels == 1) and np.any(labels == 2):
        return labels
    raise ValueError(f"cannot derive LV/RV groups from labels {np.unique(labels).tolist()}")


def rotation_z(angle):
    cosine, sine = np.cos(angle), np.sin(angle)
    return np.array([
        [cosine, -sine, 0.0],
        [sine, cosine, 0.0],
        [0.0, 0.0, 1.0],
    ])


def direction_angle_degrees(first, second):
    first = np.asarray(first)[:2]
    second = np.asarray(second)[:2]
    cosine = np.dot(first, second) / (np.linalg.norm(first) * np.linalg.norm(second))
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def ensure_shared_directories(source_root: Path, output_root: Path):
    output_root.mkdir(parents=True, exist_ok=True)
    for name in ("mesh", "norm", "sdf"):
        source = (source_root / name).resolve()
        target = output_root / name
        if not source.exists() or target.exists() or target.is_symlink():
            continue
        target.symlink_to(source, target_is_directory=True)


def correct_phase(source_root, output_root, case_id, phase, center_mode):
    input_path = source_root / "points" / case_id / f"{phase}.ply"
    mesh_path = source_root / "mesh" / case_id / f"{phase}.ply"
    norm_path = source_root / "norm" / case_id
    input_header, input_records, input_trailing, raw_input = read_labeled_ply(input_path)
    _, mesh_records, _, mesh_points = read_labeled_ply(mesh_path)

    pose = np.load(norm_path / "pose.npy")
    with np.load(norm_path / "unit_sphere.npz") as params:
        offset = np.asarray(params["offset"], dtype=np.float64)
        scale = float(params["scale"])
    canonical_input = to_canonical(raw_input, pose, offset, scale)

    groups = encoded_input_groups(input_records["label"])
    input_vector = (
        canonical_input[groups == 1].mean(axis=0)
        - canonical_input[groups == 2].mean(axis=0)
    )
    mesh_labels = np.asarray(mesh_records["label"], dtype=np.int64)
    gt_vector = (
        mesh_points[mesh_labels == 1].mean(axis=0)
        - mesh_points[mesh_labels == 2].mean(axis=0)
    )
    before_angle = direction_angle_degrees(input_vector, gt_vector)
    correction = wrap_angle(
        np.arctan2(gt_vector[1], gt_vector[0])
        - np.arctan2(input_vector[1], input_vector[0])
    )
    rotation = rotation_z(correction)
    center = canonical_input.mean(axis=0) if center_mode == "input" else np.zeros(3)
    corrected_canonical = (canonical_input - center) @ rotation.T + center
    corrected_raw = from_canonical(corrected_canonical, pose, offset, scale)

    output_path = output_root / "points" / case_id / f"{phase}.ply"
    write_labeled_ply(
        output_path, input_header, input_records, input_trailing, corrected_raw
    )
    corrected_vector = (
        corrected_canonical[groups == 1].mean(axis=0)
        - corrected_canonical[groups == 2].mean(axis=0)
    )
    return {
        "case_id": case_id,
        "phase": phase,
        "source": str(input_path),
        "output": str(output_path),
        "center_mode": center_mode,
        "correction_degrees": float(np.degrees(correction)),
        "direction_angle_before_degrees": before_angle,
        "direction_angle_after_degrees": direction_angle_degrees(corrected_vector, gt_vector),
        "input_count": int(len(raw_input)),
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root", type=Path,
        default=Path("/home/nay/github/DeepSDF/3d_data"),
    )
    parser.add_argument(
        "--output-root", type=Path,
        default=Path("/home/nay/github/DeepSDF/3d_data_corrected"),
    )
    parser.add_argument("--case-ids", nargs="*", default=None)
    parser.add_argument("--phases", nargs="+", choices=("ED", "ES"), default=("ED", "ES"))
    parser.add_argument("--center", choices=("origin", "input"), default="origin")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    if source_root == output_root:
        raise ValueError("output root must differ from source root; source data is never overwritten")
    ensure_shared_directories(source_root, output_root)

    case_ids = args.case_ids
    if not case_ids:
        case_ids = sorted(
            (item.name for item in (source_root / "points").iterdir() if item.is_dir()),
            key=lambda value: int(value),
        )
    reports = []
    for case_id in case_ids:
        for phase in args.phases:
            source = source_root / "points" / case_id / f"{phase}.ply"
            output = output_root / "points" / case_id / f"{phase}.ply"
            if not source.exists():
                continue
            if output.exists() and not args.overwrite:
                print(f"skip existing: {output}")
                continue
            report = correct_phase(source_root, output_root, case_id, phase, args.center)
            reports.append(report)
            print(
                f"{case_id}/{phase}: {report['direction_angle_before_degrees']:.2f}° "
                f"-> {report['direction_angle_after_degrees']:.6f}° "
                f"(rotate {report['correction_degrees']:+.2f}°)"
            )

    report_path = output_root / "point_realign_report.json"
    report_path.write_text(json.dumps({
        "source_root": str(source_root),
        "output_root": str(output_root),
        "method": "per-case z rotation aligning encoded sparse LV-RV direction to labeled mesh",
        "reports": reports,
    }, indent=2), encoding="utf-8")
    print(f"wrote {len(reports)} corrected point files")
    print(f"report: {report_path}")


if __name__ == "__main__":
    main()
