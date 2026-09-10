from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import trimesh
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader

from .data import DeepSDFHeartDataset, HeartSample, case_samples
from .io import export_mesh
from .losses import chamfer_distance, multi_stage_mesh_loss
from .model import MinimalMRNet
from .template import create_two_structure_template, load_template, save_template


@dataclass(frozen=True)
class Split:
    name: str
    case_ids: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate a case-disjoint MinimalMRNet baseline"
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("output/pytorch_mrnet/baseline"))
    parser.add_argument("--train-cases", type=int, default=80)
    parser.add_argument("--validation-cases", type=int, default=10)
    parser.add_argument("--test-cases", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--target-points", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1024, 2048, 4096])
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def discover_case_ids(data_root: Path) -> list[str]:
    case_ids = []
    for path in (data_root / "points").iterdir():
        if path.is_dir() and (path / "ED.ply").exists() and (path / "ES.ply").exists():
            case_ids.append(path.name)
    return sorted(case_ids, key=lambda value: int(value))


def make_splits(args: argparse.Namespace) -> list[Split]:
    required = args.train_cases + args.validation_cases + args.test_cases
    case_ids = discover_case_ids(args.data_root)
    if len(case_ids) < required:
        raise ValueError(f"Need {required} complete cases, found {len(case_ids)}")
    train_end = args.train_cases
    validation_end = train_end + args.validation_cases
    return [
        Split("train", case_ids[:train_end]),
        Split("validation", case_ids[train_end:validation_end]),
        Split("test", case_ids[validation_end:required]),
    ]


def make_dataset(args: argparse.Namespace, split: Split) -> DeepSDFHeartDataset:
    return DeepSDFHeartDataset(
        args.data_root,
        case_samples(split.case_ids),
        target_points=args.target_points,
    )


def mean_chamfer(
    model: MinimalMRNet,
    loader: DataLoader[dict[str, Any]],
    template: dict[str, torch.Tensor],
    device: torch.device,
) -> float:
    total = 0.0
    count = 0
    model.eval()
    with torch.no_grad():
        for batch in loader:
            inputs = batch["input_points"].to(device)
            target = batch["target_points"].to(device)
            prediction = model(inputs, template["vertices"], template["edges"])
            for batch_index in range(len(inputs)):
                total += float(
                    chamfer_distance(
                        prediction[batch_index : batch_index + 1],
                        target[batch_index : batch_index + 1],
                    )
                )
                count += 1
    return total / max(count, 1)


def template_chamfer(
    loader: DataLoader[dict[str, Any]],
    template_vertices: torch.Tensor,
    device: torch.device,
) -> float:
    total = 0.0
    count = 0
    prediction = template_vertices.unsqueeze(0)
    with torch.no_grad():
        for batch in loader:
            target = batch["target_points"].to(device)
            expanded = prediction.expand(len(target), -1, -1)
            for batch_index in range(len(target)):
                total += float(
                    chamfer_distance(
                        expanded[batch_index : batch_index + 1],
                        target[batch_index : batch_index + 1],
                    )
                )
                count += 1
    return total / max(count, 1)


def enclosed_volume(vertices: np.ndarray, faces: np.ndarray) -> float:
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    components = mesh.split(only_watertight=False)
    if not components:
        return float(abs(mesh.volume))
    return float(sum(abs(component.volume) for component in components))


def mesh_quality(
    vertices: np.ndarray,
    template_vertices: np.ndarray,
    faces: np.ndarray,
    edges: np.ndarray,
) -> dict[str, float | bool]:
    triangles = vertices[faces]
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    template_triangles = template_vertices[faces]
    template_cross = np.cross(
        template_triangles[:, 1] - template_triangles[:, 0],
        template_triangles[:, 2] - template_triangles[:, 0],
    )
    orientation_dot = np.sum(cross * template_cross, axis=1)
    lengths = np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1)
    template_lengths = np.linalg.norm(
        template_vertices[edges[:, 0]] - template_vertices[edges[:, 1]], axis=1
    )
    stretch = lengths / np.maximum(template_lengths, 1e-8)
    return {
        "finite": bool(np.isfinite(vertices).all()),
        "maximum_absolute_coordinate": float(np.max(np.abs(vertices))),
        "degenerate_face_fraction": float(np.mean(areas < 1e-8)),
        "flipped_face_fraction": float(np.mean(orientation_dot < 0.0)),
        "edge_stretch_p05": float(np.quantile(stretch, 0.05)),
        "edge_stretch_p95": float(np.quantile(stretch, 0.95)),
    }


def save_projection(
    input_points: np.ndarray,
    target_points: np.ndarray,
    prediction: np.ndarray,
    edges: np.ndarray,
    path: Path,
) -> None:
    width, height = 1200, 400
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image, "RGBA")
    views = ((0, 1, "XY"), (0, 2, "XZ"), (1, 2, "YZ"))
    combined = np.concatenate([input_points, target_points, prediction], axis=0)
    for view_index, (axis_x, axis_y, name) in enumerate(views):
        left = view_index * 400
        values = combined[:, [axis_x, axis_y]]
        lower = values.min(axis=0)
        upper = values.max(axis=0)
        center = 0.5 * (lower + upper)
        scale = 350.0 / max(float(np.max(upper - lower)), 1e-6)

        def project(points: np.ndarray) -> np.ndarray:
            projected = (points[:, [axis_x, axis_y]] - center) * scale
            projected[:, 0] += left + 200
            projected[:, 1] = 200 - projected[:, 1]
            return projected

        target_2d = project(target_points[:: max(1, len(target_points) // 1000)])
        input_2d = project(input_points[:: max(1, len(input_points) // 1000)])
        prediction_2d = project(prediction)
        for x, y in target_2d:
            draw.ellipse((x - 1, y - 1, x + 1, y + 1), fill=(110, 110, 110, 100))
        for source, target in edges:
            draw.line(
                (*prediction_2d[source], *prediction_2d[target]),
                fill=(20, 80, 220, 180),
                width=1,
            )
        for x, y in input_2d:
            draw.ellipse((x - 1, y - 1, x + 1, y + 1), fill=(230, 30, 30, 150))
        draw.text((left + 8, 8), f"{name}: input red, target gray, prediction blue", fill="black")
        if view_index:
            draw.line((left, 0, left, height), fill=(180, 180, 180, 255), width=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def evaluate_test(
    model: MinimalMRNet,
    dataset: DeepSDFHeartDataset,
    template: dict[str, torch.Tensor],
    args: argparse.Namespace,
    seed_dir: Path,
    device: torch.device,
) -> dict[str, Any]:
    faces = template["faces"].cpu().numpy()
    edges = template["edges"].cpu().numpy()
    template_vertices = template["vertices"].cpu().numpy()
    predictions: list[torch.Tensor] = []
    per_sample: list[dict[str, Any]] = []
    by_case: dict[str, dict[str, dict[str, float]]] = {}
    quality_passes = []
    model.eval()
    with torch.no_grad():
        for index, sample in enumerate(dataset.samples):
            item = dataset[index]
            target = item["target_points"].unsqueeze(0).to(device)
            prediction = model(
                item["input_points"].unsqueeze(0).to(device),
                template["vertices"],
                template["edges"],
            )[0]
            prediction_np = prediction.cpu().numpy()
            predictions.append(prediction.cpu())
            prediction_chamfer = float(chamfer_distance(prediction.unsqueeze(0), target))
            initial_chamfer = float(
                chamfer_distance(template["vertices"].unsqueeze(0), target)
            )
            predicted_volume = enclosed_volume(prediction_np, faces)
            target_mesh = trimesh.load(
                args.data_root / "mesh" / sample.case_id / f"{sample.phase}.ply",
                process=False,
            )
            target_volume = enclosed_volume(
                np.asarray(target_mesh.vertices), np.asarray(target_mesh.faces)
            )
            quality = mesh_quality(prediction_np, template_vertices, faces, edges)
            quality_pass = bool(
                quality["finite"]
                and quality["maximum_absolute_coordinate"] < 2.0
                and quality["degenerate_face_fraction"] < 0.001
                and quality["flipped_face_fraction"] < 0.05
                and quality["edge_stretch_p05"] > 0.2
                and quality["edge_stretch_p95"] < 5.0
            )
            quality_passes.append(quality_pass)
            per_sample.append(
                {
                    "case_id": sample.case_id,
                    "phase": sample.phase,
                    "chamfer": prediction_chamfer,
                    "template_chamfer": initial_chamfer,
                    "predicted_volume": predicted_volume,
                    "target_volume": target_volume,
                    "mesh_quality": quality,
                    "mesh_quality_pass": quality_pass,
                }
            )
            by_case.setdefault(sample.case_id, {})[sample.phase] = {
                "predicted_volume": predicted_volume,
                "target_volume": target_volume,
            }
            export_mesh(
                prediction_np,
                faces,
                seed_dir / "test_predictions" / f"case_{sample.case_id}_{sample.phase}.ply",
            )
            save_projection(
                item["input_points"].numpy(),
                item["target_points"].numpy(),
                prediction_np,
                edges,
                seed_dir / "visualizations" / f"case_{sample.case_id}_{sample.phase}.png",
            )

    stacked = torch.stack(predictions)
    pairwise = torch.cdist(stacked.flatten(1), stacked.flatten(1)) / np.sqrt(stacked.shape[1])
    nonzero = pairwise[pairwise > 0]
    direction_results = []
    for case_id, phases in by_case.items():
        if "ED" not in phases or "ES" not in phases:
            continue
        predicted_delta = phases["ED"]["predicted_volume"] - phases["ES"]["predicted_volume"]
        target_delta = phases["ED"]["target_volume"] - phases["ES"]["target_volume"]
        direction_results.append(
            {
                "case_id": case_id,
                "predicted_ed_minus_es_volume": predicted_delta,
                "target_ed_minus_es_volume": target_delta,
                "direction_match": bool(np.sign(predicted_delta) == np.sign(target_delta)),
            }
        )
    direction_accuracy = float(np.mean([item["direction_match"] for item in direction_results]))
    return {
        "mean_chamfer": float(np.mean([item["chamfer"] for item in per_sample])),
        "mean_template_chamfer": float(
            np.mean([item["template_chamfer"] for item in per_sample])
        ),
        "unique_outputs_at_1e-5": int(
            len(torch.unique(torch.round(stacked.flatten(1) / 1e-5), dim=0))
        ),
        "minimum_output_rms_difference": float(nonzero.min()),
        "ed_es_volume_direction_accuracy": direction_accuracy,
        "all_mesh_quality_pass": bool(all(quality_passes)),
        "direction_results": direction_results,
        "per_sample": per_sample,
    }


def run_seed(
    args: argparse.Namespace,
    seed: int,
    splits: list[Split],
    datasets: dict[str, DeepSDFHeartDataset],
    template: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, Any]:
    seed_everything(seed)
    seed_dir = args.output / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        datasets["train"],
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=args.num_workers,
    )
    evaluation_loaders = {
        name: DataLoader(
            datasets[name],
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )
        for name in ("train", "validation", "test")
    }
    validation_template_chamfer = template_chamfer(
        evaluation_loaders["validation"], template["vertices"], device
    )
    test_template_chamfer = template_chamfer(
        evaluation_loaders["test"], template["vertices"], device
    )

    model = MinimalMRNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    best_validation = float("inf")
    best_epoch = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        sample_count = 0
        for batch in train_loader:
            inputs = batch["input_points"].to(device)
            target = batch["target_points"].to(device)
            predictions = model(
                inputs, template["vertices"], template["edges"], return_stages=True
            )
            loss, _ = multi_stage_mesh_loss(
                predictions, target, template["vertices"], template["edges"]
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.detach()) * len(inputs)
            sample_count += len(inputs)
        train_loss = running_loss / max(sample_count, 1)
        validation = mean_chamfer(
            model, evaluation_loaders["validation"], template, device
        )
        item = {"epoch": epoch, "train_loss": train_loss, "validation_chamfer": validation}
        history.append(item)
        print(json.dumps({"seed": seed, **item}))
        if validation < best_validation:
            best_validation = validation
            best_epoch = epoch
            torch.save(
                {"model": model.state_dict(), "epoch": epoch, "validation_chamfer": validation},
                seed_dir / "best_checkpoint.pt",
            )

    checkpoint = torch.load(seed_dir / "best_checkpoint.pt", map_location=device)
    model.load_state_dict(checkpoint["model"])
    train_chamfer = mean_chamfer(model, evaluation_loaders["train"], template, device)
    validation_chamfer = mean_chamfer(
        model, evaluation_loaders["validation"], template, device
    )
    test_chamfer = mean_chamfer(model, evaluation_loaders["test"], template, device)
    test_details = evaluate_test(model, datasets["test"], template, args, seed_dir, device)
    validation_improvement = 1.0 - validation_chamfer / validation_template_chamfer
    test_validation_gap = abs(test_chamfer - validation_chamfer) / max(validation_chamfer, 1e-12)
    summary = {
        "seed": seed,
        "best_epoch": best_epoch,
        "train_chamfer": train_chamfer,
        "validation_chamfer": validation_chamfer,
        "test_chamfer": test_chamfer,
        "validation_template_chamfer": validation_template_chamfer,
        "test_template_chamfer": test_template_chamfer,
        "validation_improvement_over_template": validation_improvement,
        "test_validation_relative_gap": test_validation_gap,
        "test": test_details,
        "checks": {
            "validation_better_than_template_by_10_percent": validation_improvement > 0.10,
            "test_within_25_percent_of_validation": test_validation_gap < 0.25,
            "all_test_outputs_unique": test_details["unique_outputs_at_1e-5"]
            == len(datasets["test"]),
            "ed_es_direction_accuracy_at_least_80_percent": test_details[
                "ed_es_volume_direction_accuracy"
            ]
            >= 0.80,
            "mesh_quality_pass": test_details["all_mesh_quality_pass"],
        },
    }
    summary["seed_pass"] = bool(all(summary["checks"].values()))
    (seed_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    (seed_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> int:
    args = parse_args()
    device = torch.device(args.device)
    args.output.mkdir(parents=True, exist_ok=True)
    splits = make_splits(args)
    split_dict = {split.name: split.case_ids for split in splits}
    (args.output / "splits.json").write_text(json.dumps(split_dict, indent=2) + "\n")
    print(json.dumps({"device": str(device), "splits": split_dict}))

    template_path = args.output / "template_case0_ed.npz"
    if not template_path.exists():
        save_template(
            create_two_structure_template(args.data_root / "mesh" / "0" / "ED.ply"),
            template_path,
        )
    template = load_template(template_path, device)
    datasets = {split.name: make_dataset(args, split) for split in splits}

    seed_summaries = [
        run_seed(args, seed, splits, datasets, template, device) for seed in args.seeds
    ]
    validation_improvements = [
        item["validation_improvement_over_template"] for item in seed_summaries
    ]
    test_gaps = [item["test_validation_relative_gap"] for item in seed_summaries]
    aggregate = {
        "configuration": {
            "train_cases": args.train_cases,
            "validation_cases": args.validation_cases,
            "test_cases": args.test_cases,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "target_points": args.target_points,
            "seeds": args.seeds,
        },
        "seed_summaries": seed_summaries,
        "mean_validation_improvement_over_template": float(np.mean(validation_improvements)),
        "mean_test_validation_relative_gap": float(np.mean(test_gaps)),
        "all_seeds_improve_validation_template": bool(
            all(value > 0.10 for value in validation_improvements)
        ),
        "all_seed_trends_consistent": bool(all(item["seed_pass"] for item in seed_summaries)),
    }
    aggregate["baseline_pass"] = bool(
        aggregate["all_seeds_improve_validation_template"]
        and aggregate["all_seed_trends_consistent"]
    )
    (args.output / "summary.json").write_text(json.dumps(aggregate, indent=2) + "\n")
    print(json.dumps(aggregate, indent=2))
    return 0 if aggregate["baseline_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())