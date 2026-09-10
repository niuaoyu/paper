from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import DeepSDFHeartDataset, HeartSample, case_samples
from .io import export_mesh
from .losses import chamfer_distance, multi_stage_mesh_loss
from .model import MinimalMRNet
from .template import create_two_structure_template, load_template, save_template


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the minimal PyTorch MR-Net feasibility test")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--mode", choices=("single", "five-cases"), default="single")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--output", type=Path, default=Path("output/pytorch_mrnet"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def seed_everything(seed: int = 1024) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> int:
    args = parse_args()
    seed_everything()
    device = torch.device(args.device)
    run_dir = args.output / args.mode
    run_dir.mkdir(parents=True, exist_ok=True)

    template_path = args.output / "template_case0_ed.npz"
    if not template_path.exists():
        template = create_two_structure_template(args.data_root / "mesh" / "0" / "ED.ply")
        save_template(template, template_path)
    template = load_template(template_path, device)

    samples = (
        [HeartSample("0", "ED")]
        if args.mode == "single"
        else case_samples(["0", "1", "2", "3", "4"])
    )
    dataset = DeepSDFHeartDataset(args.data_root, samples)
    loader = DataLoader(dataset, batch_size=1 if args.mode == "single" else 2, shuffle=True)
    batches = list(loader)

    model = MinimalMRNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    history: list[dict[str, float]] = []
    initial_loss = None
    model.train()
    for step in range(args.steps):
        batch = batches[step % len(batches)]
        inputs = batch["input_points"].to(device)
        target = batch["target_points"].to(device)
        predictions = model(
            inputs, template["vertices"], template["edges"], return_stages=True
        )
        loss, metrics = multi_stage_mesh_loss(
            predictions, target, template["vertices"], template["edges"]
        )
        if initial_loss is None:
            initial_loss = float(loss.detach())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step == 0 or (step + 1) % max(1, args.steps // 10) == 0:
            item = {
                "step": step + 1,
                "loss": float(loss.detach()),
                "chamfer": float(metrics["stage3_chamfer"].detach()),
                "edge": float(metrics["stage3_edge"].detach()),
                "laplacian": float(metrics["stage3_laplacian"].detach()),
            }
            history.append(item)
            print(json.dumps(item))

    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "steps": args.steps,
        "initial_loss": initial_loss,
        "final_loss": history[-1]["loss"],
    }
    torch.save(checkpoint, run_dir / "checkpoint.pt")
    (run_dir / "history.json").write_text(
        json.dumps(history, indent=2) + "\n", encoding="utf-8"
    )

    model.eval()
    predictions = []
    per_sample = []
    with torch.no_grad():
        for sample_index, sample in enumerate(samples):
            item = dataset[sample_index]
            prediction = model(
                item["input_points"].unsqueeze(0).to(device),
                template["vertices"],
                template["edges"],
            )[0]
            sample_chamfer = chamfer_distance(
                prediction.unsqueeze(0),
                item["target_points"].unsqueeze(0).to(device),
            )
            predictions.append(prediction.cpu())
            per_sample.append(
                {
                    "case_id": sample.case_id,
                    "phase": sample.phase,
                    "chamfer": float(sample_chamfer),
                }
            )
            export_mesh(
                prediction.cpu().numpy(),
                template["faces"].cpu().numpy(),
                run_dir / f"case_{sample.case_id}_{sample.phase}_prediction.ply",
            )

    ratio = history[-1]["loss"] / max(initial_loss or 1.0, 1e-12)
    stacked_predictions = torch.stack(predictions)
    if len(predictions) > 1:
        pairwise = torch.cdist(
            stacked_predictions.flatten(1), stacked_predictions.flatten(1)
        ) / np.sqrt(stacked_predictions.shape[1])
        nonzero_pairwise = pairwise[pairwise > 0]
        minimum_output_difference = float(nonzero_pairwise.min())
        unique_outputs = int(
            len(torch.unique(torch.round(stacked_predictions.flatten(1) / 1e-5), dim=0))
        )
    else:
        minimum_output_difference = 0.0
        unique_outputs = 1
    summary = {
        "mode": args.mode,
        "device": str(device),
        "sample_count": len(samples),
        "template_vertices": int(len(template["vertices"])),
        "initial_loss": initial_loss,
        "final_loss": history[-1]["loss"],
        "final_to_initial_ratio": ratio,
        "mean_sample_chamfer": float(np.mean([item["chamfer"] for item in per_sample])),
        "minimum_output_rms_difference": minimum_output_difference,
        "unique_outputs_at_1e-5": unique_outputs,
        "per_sample": per_sample,
        "feasibility_pass": ratio < 0.5 and unique_outputs == len(samples),
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0 if summary["feasibility_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
