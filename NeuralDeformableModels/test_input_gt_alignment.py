#!/usr/bin/env python3
"""Audit sparse input acquisition and its alignment with labeled LV/RV GT meshes.

This test deliberately runs before training.  It verifies the acquisition
(row, column) convention, reports per-case geometric evidence, and exits with a
non-zero status when the input/GT LV-minus-RV direction is inconsistent.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from NeuralDeformableModel.dataset.acquisition_geometry import (
    contour_yx_to_xyz,
    rotation_lv_minus_rv_to_positive_y,
)
from NeuralDeformableModel.dataset.heart_dataset import HeartDataset


def angle_degrees(first, second):
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    denominator = np.linalg.norm(first) * np.linalg.norm(second)
    if denominator <= 1e-12:
        return float("nan")
    cosine = np.clip(np.dot(first, second) / denominator, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def principal_axes(points):
    _, _, axes = np.linalg.svd(points - points.mean(axis=0), full_matrices=False)
    return axes


def acquisition_label_group(labels):
    """Map encoded labels such as 110/150 and 212/252 to LV=1 and RV=2."""
    labels = np.asarray(labels, dtype=np.int64)
    groups = labels // 100
    if np.any(groups == 1) and np.any(groups == 2):
        return groups
    if np.any(labels == 1) and np.any(labels == 2):
        return labels
    return None


def export_cloud(path, point_sets):
    """Write colored point sets to one ASCII PLY file for visual inspection."""
    points = np.concatenate([item[0] for item in point_sets], axis=0)
    colors = np.concatenate([
        np.repeat(np.asarray(item[1], dtype=np.uint8)[None, :], len(item[0]), axis=0)
        for item in point_sets
    ], axis=0)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii") as stream:
        stream.write("ply\nformat ascii 1.0\n")
        stream.write(f"element vertex {len(points)}\n")
        stream.write("property float x\nproperty float y\nproperty float z\n")
        stream.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        stream.write("end_header\n")
        for point, color in zip(points, colors):
            stream.write(
                f"{point[0]:.8g} {point[1]:.8g} {point[2]:.8g} "
                f"{color[0]} {color[1]} {color[2]}\n"
            )


def export_sample_preview(dataset, case_id, phase, output_dir):
    """Export input and GT in the same canonical coordinate system."""
    point_path = dataset.root / "points" / str(case_id) / f"{phase}.ply"
    mesh_path = dataset.root / "mesh" / str(case_id) / f"{phase}.ply"
    raw_input, input_labels = dataset._load_ply_with_labels(point_path)
    raw_mesh, mesh_labels = dataset._load_ply_with_labels(mesh_path)

    norm_params = dataset._load_normalization(case_id)
    input_points = dataset._apply_normalization(raw_input, norm_params)
    mesh_points = dataset._apply_normalization(raw_mesh, norm_params)
    input_groups = acquisition_label_group(input_labels)
    if input_groups is None:
        raise AssertionError(f"{case_id}/{phase}: cannot color input LV/RV groups")

    input_lv = input_points[input_groups == 1]
    input_rv = input_points[input_groups == 2]
    gt_lv = mesh_points[mesh_labels == 1]
    gt_rv = mesh_points[mesh_labels == 2]
    stem = f"case_{case_id}_{phase}"
    export_cloud(
        output_dir / f"{stem}_input.ply",
        [(input_lv, (255, 80, 80)), (input_rv, (80, 140, 255))],
    )
    export_cloud(
        output_dir / f"{stem}_gt.ply",
        [(gt_lv, (80, 255, 80)), (gt_rv, (0, 220, 220))],
    )
    export_cloud(
        output_dir / f"{stem}_overlay.ply",
        [
            (input_lv, (255, 80, 80)),
            (input_rv, (80, 140, 255)),
            (gt_lv, (80, 255, 80)),
            (gt_rv, (0, 220, 220)),
        ],
    )


def audit_sample(dataset, case_id, phase, max_direction_angle):
    point_path = dataset.root / "points" / str(case_id) / f"{phase}.ply"
    mesh_path = dataset.root / "mesh" / str(case_id) / f"{phase}.ply"
    raw_input, input_labels = dataset._load_ply_with_labels(point_path)
    raw_mesh, mesh_labels = dataset._load_ply_with_labels(mesh_path)

    norm_params = dataset._load_normalization(case_id)
    input_points = dataset._apply_normalization(raw_input, norm_params)
    mesh_points = dataset._apply_normalization(raw_mesh, norm_params)

    gt_lv = mesh_points[mesh_labels == 1]
    gt_rv = mesh_points[mesh_labels == 2]
    if not len(gt_lv) or not len(gt_rv):
        raise AssertionError(f"{case_id}/{phase}: mesh must contain labels 1 and 2")

    input_groups = acquisition_label_group(input_labels)
    if input_groups is None:
        raise AssertionError(
            f"{case_id}/{phase}: input labels cannot identify LV/RV acquisition groups; "
            f"found {np.unique(input_labels).tolist()}"
        )
    input_lv = input_points[input_groups == 1]
    input_rv = input_points[input_groups == 2]

    input_vector = input_lv.mean(axis=0) - input_rv.mean(axis=0)
    gt_vector = gt_lv.mean(axis=0) - gt_rv.mean(axis=0)
    angle_xy = angle_degrees(input_vector[:2], gt_vector[:2])
    angle_3d = angle_degrees(input_vector, gt_vector)

    gt_tree = cKDTree(mesh_points)
    input_tree = cKDTree(input_points)
    input_to_gt = float(gt_tree.query(input_points, k=1)[0].mean())
    gt_to_input = float(input_tree.query(mesh_points, k=1)[0].mean())

    failures = []
    if not np.isfinite(angle_xy) or angle_xy > max_direction_angle:
        failures.append(
            f"LV-RV XY direction angle {angle_xy:.2f} deg exceeds "
            f"{max_direction_angle:.2f} deg"
        )

    return {
        "case_id": int(case_id),
        "phase": phase,
        "status": "fail" if failures else "pass",
        "failures": failures,
        "input_count": int(len(input_points)),
        "input_label_counts": {
            str(int(label)): int(np.sum(input_labels == label))
            for label in np.unique(input_labels)
        },
        "input_bbox_min": input_points.min(axis=0).tolist(),
        "input_bbox_max": input_points.max(axis=0).tolist(),
        "gt_bbox_min": mesh_points.min(axis=0).tolist(),
        "gt_bbox_max": mesh_points.max(axis=0).tolist(),
        "input_centroid": input_points.mean(axis=0).tolist(),
        "gt_centroid": mesh_points.mean(axis=0).tolist(),
        "input_lv_minus_rv": input_vector.tolist(),
        "gt_lv_minus_rv": gt_vector.tolist(),
        "direction_angle_xy_degrees": angle_xy,
        "direction_angle_3d_degrees": angle_3d,
        "input_principal_axes": principal_axes(input_points).tolist(),
        "gt_principal_axes": principal_axes(mesh_points).tolist(),
        "mean_input_to_gt_distance": input_to_gt,
        "mean_gt_to_input_distance": gt_to_input,
    }


def test_coordinate_helpers():
    contour = np.array([[10.0, 20.0], [30.0, 40.0]])
    expected = np.array([[20.0, 10.0, 7.0], [40.0, 30.0, 7.0]])
    np.testing.assert_allclose(contour_yx_to_xyz(contour, 7.0), expected)

    angle = rotation_lv_minus_rv_to_positive_y([10.0, 20.0], [10.0, 0.0])
    c, s = np.cos(angle), np.sin(angle)
    rotation = np.array([[c, -s], [s, c]])
    rotated = rotation @ np.array([20.0, 0.0])
    np.testing.assert_allclose(rotated, [0.0, 20.0], atol=1e-10)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root", type=Path,
        default=Path("/media/nay/f6b53612-1834-4e4e-aed5-19b00ed6cfc3/nay/3d-data"),
    )
    parser.add_argument("--case-ids", nargs="+", type=int, default=[16])
    parser.add_argument("--phases", nargs="+", choices=("ED", "ES"), default=["ED", "ES"])
    parser.add_argument("--max-direction-angle", type=float, default=20.0)
    parser.add_argument("--report", type=Path, default=Path("output/alignment_audit/report.json"))
    parser.add_argument(
        "--export-dir", type=Path, default=None,
        help="write canonical input/GT/overlay PLY files for visual inspection",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    test_coordinate_helpers()
    dataset = HeartDataset(
        root=args.data_root,
        case_ids=args.case_ids,
        phases=args.phases,
        deterministic=True,
    )
    reports = [
        audit_sample(dataset, case_id, phase, args.max_direction_angle)
        for case_id, phase in dataset.samples
    ]
    if args.export_dir is not None:
        for case_id, phase in dataset.samples:
            export_sample_preview(dataset, case_id, phase, args.export_dir)
    payload = {
        "status": "fail" if any(item["failures"] for item in reports) else "pass",
        "max_direction_angle_degrees": args.max_direction_angle,
        "samples": reports,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    for item in reports:
        print(
            f"[{item['status'].upper()}] {item['case_id']}/{item['phase']} "
            f"direction_xy={item['direction_angle_xy_degrees']:.2f} deg, "
            f"input->GT={item['mean_input_to_gt_distance']:.4f}, "
            f"GT->input={item['mean_gt_to_input_distance']:.4f}"
        )
        for failure in item["failures"]:
            print(f"  - {failure}")
    print(f"report: {args.report}")
    if args.export_dir is not None:
        print(f"PLY previews: {args.export_dir}")
    return 1 if payload["status"] == "fail" else 0


if __name__ == "__main__":
    sys.exit(main())
