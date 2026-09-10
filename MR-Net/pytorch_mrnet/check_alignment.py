from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree

from .data import load_vertices_and_labels, points_to_canonical
from .io import export_alignment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check contour/mesh canonical alignment")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--cases", nargs="+", default=["0"])
    parser.add_argument("--output", type=Path, default=Path("output/pytorch_mrnet/alignment"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    report = []
    for case_id in args.cases:
        for phase in ("ED", "ES"):
            point_path = args.data_root / "points" / case_id / f"{phase}.ply"
            mesh_path = args.data_root / "mesh" / case_id / f"{phase}.ply"
            points, _ = load_vertices_and_labels(point_path)
            points = points_to_canonical(points, args.data_root / "norm" / case_id)
            mesh = trimesh.load(mesh_path, process=False)
            vertices = np.asarray(mesh.vertices)
            point_to_mesh = cKDTree(vertices).query(points)[0]
            sampled_vertices = vertices[
                np.linspace(0, len(vertices) - 1, min(10000, len(vertices)), dtype=int)
            ]
            mesh_to_point = cKDTree(points).query(sampled_vertices)[0]
            item = {
                "case_id": case_id,
                "phase": phase,
                "point_bbox": [points.min(0).tolist(), points.max(0).tolist()],
                "mesh_bbox": [vertices.min(0).tolist(), vertices.max(0).tolist()],
                "point_to_mesh_mean": float(point_to_mesh.mean()),
                "mesh_to_point_mean": float(mesh_to_point.mean()),
                "symmetric_mean": float(0.5 * (point_to_mesh.mean() + mesh_to_point.mean())),
            }
            report.append(item)
            export_alignment(points, mesh_path, args.output / f"case_{case_id}_{phase}.glb")
            print(json.dumps(item, ensure_ascii=False))
    (args.output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
