#!/usr/bin/env python3
"""Run the released NDM core model on one converted input without a checkpoint.

This verifies imports, tensor shape compatibility, Point Transformer encoding,
global parameter heads, primitive construction, and optionally a small NODE
local-flow subset.  Outputs are random because the public repository contains
no pretrained checkpoint.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "NeuralDeformableModel"))

from model.model import NeuralDeformableModel  # noqa: E402


def first_npz(path: Path) -> np.ndarray:
    with np.load(path) as data:
        return np.asarray(data[data.files[0]])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="HR_ED_pc.npz or HR_ES_pc.npz")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--node-points", type=int, default=0, help="also run branch-1 NODE flow on this many primitive points")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    points = first_npz(args.input).astype(np.float32, copy=False)
    if points.shape != (5600, 3):
        raise ValueError(f"NDM input must be (5600, 3), got {points.shape}")
    if not np.isfinite(points).all():
        raise ValueError("input contains NaN/Inf")

    device = torch.device(args.device)
    model = NeuralDeformableModel(zdim=512).to(device).float().eval()
    tensor = torch.from_numpy(points).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        code1, code2, code3 = outputs[:3]
        primitive = outputs[-1]
        node_shape = None
        if args.node_points:
            flowed, _ = model.db1(code1, primitive[:, : args.node_points], y=None)
            node_shape = list(flowed.shape)

    summary = {
        "status": "ok",
        "checkpoint_loaded": False,
        "warning": "Randomly initialized weights: this is a compatibility smoke test, not meaningful reconstruction.",
        "input_shape": list(tensor.shape),
        "input_dtype": str(tensor.dtype),
        "device": str(device),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "output_count": len(outputs),
        "latent_shapes": [list(code1.shape), list(code2.shape), list(code3.shape)],
        "primitive_shape": list(primitive.shape),
        "node_subset_output_shape": node_shape,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
