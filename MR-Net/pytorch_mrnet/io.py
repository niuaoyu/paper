from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh


def export_mesh(vertices: np.ndarray, faces: np.ndarray, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    trimesh.Trimesh(vertices=vertices, faces=faces, process=False).export(path)


def export_alignment(points: np.ndarray, mesh_path: str | Path, output: str | Path) -> None:
    mesh = trimesh.load(mesh_path, process=False)
    point_cloud = trimesh.points.PointCloud(points, colors=[255, 40, 40, 255])
    scene = trimesh.Scene([mesh, point_cloud])
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    scene.export(output)
