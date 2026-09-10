from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import trimesh

from .data import load_vertices_and_labels


def faces_to_edges(faces: np.ndarray) -> np.ndarray:
    edges = np.concatenate(
        [faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0
    )
    edges = np.sort(edges, axis=1)
    return np.unique(edges, axis=0).astype(np.int64)


def create_two_structure_template(
    mesh_path: str | Path,
    subdivisions: int = 2,
) -> dict[str, np.ndarray]:
    """Create two ellipsoidal components fitted to mesh labels 1 and 2."""
    vertices, labels = load_vertices_and_labels(Path(mesh_path))
    if labels is None or not {1, 2}.issubset(set(np.unique(labels).tolist())):
        raise ValueError("Template source mesh must contain vertex labels 1 and 2")

    all_vertices: list[np.ndarray] = []
    all_faces: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    vertex_offset = 0
    for label in (1, 2):
        component = vertices[labels == label]
        center = component.mean(axis=0)
        lower, upper = np.quantile(component, [0.02, 0.98], axis=0)
        radius = np.maximum((upper - lower) * 0.5, 0.05)

        sphere = trimesh.creation.icosphere(subdivisions=subdivisions, radius=1.0)
        fitted = np.asarray(sphere.vertices, dtype=np.float32) * radius + center
        faces = np.asarray(sphere.faces, dtype=np.int64) + vertex_offset
        all_vertices.append(fitted.astype(np.float32))
        all_faces.append(faces)
        all_labels.append(np.full(len(fitted), label, dtype=np.int64))
        vertex_offset += len(fitted)

    template_vertices = np.concatenate(all_vertices, axis=0)
    template_faces = np.concatenate(all_faces, axis=0)
    return {
        "vertices": template_vertices,
        "faces": template_faces,
        "edges": faces_to_edges(template_faces),
        "labels": np.concatenate(all_labels, axis=0),
    }


def save_template(template: dict[str, np.ndarray], output: str | Path) -> None:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **template)


def load_template(path: str | Path, device: torch.device) -> dict[str, torch.Tensor]:
    with np.load(path, allow_pickle=False) as data:
        return {
            "vertices": torch.as_tensor(data["vertices"], dtype=torch.float32, device=device),
            "faces": torch.as_tensor(data["faces"], dtype=torch.long, device=device),
            "edges": torch.as_tensor(data["edges"], dtype=torch.long, device=device),
            "labels": torch.as_tensor(data["labels"], dtype=torch.long, device=device),
        }
