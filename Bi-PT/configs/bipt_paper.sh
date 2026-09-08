#!/usr/bin/env bash
# Bi-PT (full bidirectional model) — paper training configuration.
# See the paper Sec. 3.2 for hyperparameters. Run from the repository root.
#
#   bash configs/bipt_paper.sh
#
# Note the architectural constraint: num_hidden == 32 * 2**ca_nblocks (=> 512 with
# ca_nblocks=4) and latent_len == 32. See the README.
set -euo pipefail

python scripts/train.py \
  --ca-type double \
  --label-dim 2 \
  --cd-mode both \
  --sparse-dir   data/train/spc_ds \
  --gt-dir-2     data/train/gt_ds_5632_vtp \
  --atlas-path-2 data/atlas.vtp \
  --epochs 300 \
  --batch-size 5 \
  --lr 5e-4 \
  --weight-decay 1e-4 \
  --grad-clip 0.1 \
  --l1-loss-weight 0.3 \
  --lap-weight 1.0 \
  --num-hidden 512 \
  --latent-len 32 \
  --ca-nblocks 4 \
  --ca-nneighbor 16 \
  --norm-type in \
  --ckpt-dir ./checkpoints
