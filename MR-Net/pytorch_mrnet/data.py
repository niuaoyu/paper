from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import trimesh
from torch.utils.data import Dataset


PHASES = ("ED", "ES")


def _load_ply(path: Path) -> trimesh.Trimesh | trimesh.points.PointCloud:
    geometry = trimesh.load(path, process=False)
    if isinstance(geometry, trimesh.Scene):
        geometry = geometry.to_geometry()
    return geometry


def load_vertices_and_labels(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    geometry = _load_ply(path)
    vertices = np.asarray(geometry.vertices, dtype=np.float32)
    labels = None
    raw = geometry.metadata.get("_ply_raw", {}).get("vertex", {}).get("data")
    if raw is not None and raw.dtype.names and "label" in raw.dtype.names:
        labels = np.asarray(raw["label"], dtype=np.int64)
    return vertices, labels


def points_to_canonical(points: np.ndarray, norm_dir: Path) -> np.ndarray:
    """Apply the DeepSDF pose and unit-sphere transform to physical-space points."""
    pose = np.load(norm_dir / "pose.npy", allow_pickle=False)
    with np.load(norm_dir / "unit_sphere.npz", allow_pickle=False) as normalization:
        offset = np.asarray(normalization["offset"], dtype=np.float64)
        scale = float(normalization["scale"])
    homogeneous = np.concatenate(
        [points.astype(np.float64), np.ones((len(points), 1), dtype=np.float64)], axis=1
    )
    posed = (homogeneous @ pose.T)[:, :3]
    return ((posed + offset) * scale).astype(np.float32)


def deterministic_sample(values: np.ndarray, count: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    replace = len(values) < count
    indices = rng.choice(len(values), size=count, replace=replace)
    return values[indices]


def sample_mesh_surface(mesh_path: Path, count: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    mesh = _load_ply(mesh_path)
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Expected a triangle mesh: {mesh_path}")
    rng_state = np.random.get_state()
    np.random.seed(seed)
    try:
        points, face_indices = trimesh.sample.sample_surface(mesh, count)
    finally:
        np.random.set_state(rng_state)

    vertex_labels = None
    raw = mesh.metadata.get("_ply_raw", {}).get("vertex", {}).get("data")
    if raw is not None and raw.dtype.names and "label" in raw.dtype.names:
        vertex_labels = np.asarray(raw["label"], dtype=np.int64)
    if vertex_labels is None:
        labels = np.zeros(count, dtype=np.int64)
    else:
        face_labels = vertex_labels[np.asarray(mesh.faces)[face_indices]]
        labels = np.asarray(
            [np.bincount(row).argmax() for row in face_labels], dtype=np.int64
        )
    return points.astype(np.float32), labels


@dataclass(frozen=True)
class HeartSample:
    case_id: str
    phase: str


class DeepSDFHeartDataset(Dataset[dict[str, Any]]):
    """Read DeepSDF contour points and variable-topology target meshes."""

    def __init__(
        self,
        root: str | Path,
        samples: list[HeartSample],
        input_points: int = 3000,
        target_points: int = 2048,
        cache: bool = True,
    ) -> None:
        self.root = Path(root)
        self.samples = samples
        self.input_points = input_points
        self.target_points = target_points
        self.cache = cache
        self._cache: dict[int, dict[str, Any]] = {}

    def __len__(self) -> int:
        return len(self.samples)

    def _load(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        seed = int(sample.case_id) * 17 + (0 if sample.phase == "ED" else 1) + 1024
        point_path = self.root / "points" / sample.case_id / f"{sample.phase}.ply"
        mesh_path = self.root / "mesh" / sample.case_id / f"{sample.phase}.ply"
        norm_dir = self.root / "norm" / sample.case_id

        points, point_labels = load_vertices_and_labels(point_path)
        points = points_to_canonical(points, norm_dir)
        sampled_indices = deterministic_sample(np.arange(len(points)), self.input_points, seed)
        points = points[sampled_indices]
        if point_labels is None:
            point_labels = np.zeros(self.input_points, dtype=np.int64)
        else:
            point_labels = point_labels[sampled_indices]

        target, target_labels = sample_mesh_surface(mesh_path, self.target_points, seed)
        return {
            "input_points": torch.from_numpy(points),
            "input_labels": torch.from_numpy(point_labels),
            "target_points": torch.from_numpy(target),
            "target_labels": torch.from_numpy(target_labels),
            "case_id": sample.case_id,
            "phase": sample.phase,
        }

    def __getitem__(self, index: int) -> dict[str, Any]:
        if self.cache and index in self._cache:
            return self._cache[index]
        item = self._load(index)
        if self.cache:
            self._cache[index] = item
        return item


def case_samples(case_ids: list[str], phases: tuple[str, ...] = PHASES) -> list[HeartSample]:
    return [HeartSample(case_id, phase) for case_id in case_ids for phase in phases]
