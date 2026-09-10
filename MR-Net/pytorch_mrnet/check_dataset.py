from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .data import DeepSDFHeartDataset, case_samples


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the PyTorch heart dataset")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--cases", nargs="+", default=["0", "1", "2", "3", "4"])
    args = parser.parse_args()
    dataset = DeepSDFHeartDataset(args.data_root, case_samples(args.cases))
    batch = next(iter(DataLoader(dataset, batch_size=min(2, len(dataset)), shuffle=False)))
    for key in ("input_points", "input_labels", "target_points", "target_labels"):
        value = batch[key]
        print(f"{key}: shape={tuple(value.shape)}, dtype={value.dtype}")
        if not torch.isfinite(value).all():
            raise RuntimeError(f"Non-finite values in {key}")
    print(f"input range: {batch['input_points'].amin().item():.4f} .. {batch['input_points'].amax().item():.4f}")
    print(f"target range: {batch['target_points'].amin().item():.4f} .. {batch['target_points'].amax().item():.4f}")
    print(f"samples: {list(zip(batch['case_id'], batch['phase']))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
