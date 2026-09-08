<div align="center">

# Bi-PT: Bidirectional Cross-Attention Point Transformers for Four-Chamber Heart Reconstruction from Sparse Cardiac MRI Data

Chenchuhui Hu¹, Shaoming Pan¹, Leon Axel², Meng Ye¹

¹ Department of Computer Science and Engineering, University of Texas at Arlington
² NYU Grossman School of Medicine

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)

_STACOM @ MICCAI 2026_

</div>

---

## Overview

**Bi-PT** reconstructs a dense 3D **four-chamber heart mesh** (RA / LA / RV / LV / MYO)
from a **sparse point cloud (SPC)** of the kind that can be extracted from the few
long-axis (LAX) and short-axis (SAX) views used in routine clinical CMR protocols.

Given a shared **atlas** mesh and a subject's SPC, Bi-PT learns an atlas→target
deformation in four stages:

1. **Point feature encoding** — a Point Transformer encoder processes the atlas and
   the SPC independently, summarizing each point's local neighborhood.
2. **Bidirectional cross-attention** — a core layer exchanges information in *both*
   directions (atlas→SPC for local target cues, SPC→atlas for a global target-shape
   descriptor via global average pooling), putting the two point sets in correspondence.
3. **Point feature decoding** — Point Transformer decoders upsample the fused features
   back to full resolution.
4. **Locally Affine Diffeomorphic Deformation (LADD)** — two Neural ODE (NODE) blocks
   integrate a per-point affine + translation velocity field to deform the atlas into an
   intermediate and then the final mesh, guaranteeing a topology-preserving deformation.

Training uses a **semantic-aware Chamfer distance** (nearest-neighbor search restricted
by per-point chamber labels) with **deep supervision** on both NODE outputs, plus a
**Laplacian regularization** to keep the deformation smooth.

> Paper method reference: encoders/decoders (Sec. 2.1), bidirectional cross-attention
> (Sec. 2.2), LADD dynamics (Sec. 2.3), losses (Sec. 2.4).

## Highlights

- A point transformer that captures **local and global** cues via **bidirectional cross-attention** between an atlas and the SPC.
- A **locally affine diffeomorphic deformation** (LADD) that models complex, large deformation while preserving atlas topology (zero self-intersections in practice).
- **Label-aware reconstruction**: semantic labels folded into the Chamfer loss for chamber-consistent matching.
- A **Laplacian regularization** that stabilizes and smooths the learned deformation.

## Results

Quantitative comparison on the 100-case test set (mean over the five heart structures).
CD / EMD / P2F in mm; **lower is better** except NC (higher is better). Full details and
per-method discussion are in the paper (Table 1).

| Method        | CD ↓     | EMD ↓    | P2F ↓    | NC ↑     | ENF ↓    | SI (×10⁻⁵) ↓ |
|---------------|----------|----------|----------|----------|----------|--------------|
| CPD           | 5.16     | 12.74    | 4.56     | 0.58     | 0.34     | 0            |
| NMF           | 2.99     | 13.28    | 2.19     | 0.57     | 0.34     | 0            |
| MR-Net        | 4.90     | 12.21    | 4.28     | 0.58     | 0.34     | 79.4         |
| NDM           | 3.01     | 12.51    | 2.07     | 0.04     | 0.01     | 868.6        |
| LTN           | 2.34     | 6.36     | 1.41     | 0.61     | 0.35     | 0.56         |
| LTN-DSTN      | 2.62     | 6.96     | 1.77     | 0.60     | 0.36     | 1.17         |
| **Bi-PT (ours)** | **2.28** | **6.54** | **1.41** | **0.66** | **0.34** | **0**    |

## Repository structure

```
Bi-PT/
├── bipt/                          # importable package
│   ├── models/
│   │   ├── deform.py              # DeformBlockConcat (Bi-PT), DeformBlockConcatSingleCA (sCA ablation)
│   │   ├── node.py               # ODEFuncConcat / ODEFuncConcatTranslation / NODEBlockConcat (LADD)
│   │   ├── cross_attention.py    # Single / Double (bidirectional) cross-attention modules
│   │   └── point_transformer/    # Point Transformer encoder / decoder / cross-attention core
│   ├── losses/
│   │   └── chamfer.py            # semantic-aware Chamfer distance
│   └── data/
│       └── dataset.py            # labeled dataset + VTK/npz IO helpers
├── scripts/
│   ├── train.py                  # training entry point
│   ├── infer.py                  # deform atlas -> predicted meshes (.vtp) + Chamfer eval
│   ├── evaluate.py               # geometric + mesh-quality metrics (CD/EMD/P2F/NC/NM/SI)
│   └── preprocess/               # data generation
│       ├── generate_spc.py       # sparse point clouds (.npz)
│       ├── generate_gt_mesh.py   # labeled GT meshes (.vtp)
│       └── thirdparty/           # vendored mesh deps (Apache-2.0 + MIT)
├── configs/                      # example run configurations (bipt_paper.sh)
├── data/                         # (git-ignored) datasets — see data/README.md
├── docs/                         # additional documentation
├── requirements.txt / environment.yml / requirements-preprocess.txt
├── pyproject.toml
└── LICENSE
```

## Installation

```bash
git clone https://github.com/Chenchuhui/Bi-PT.git
cd Bi-PT

# 1) Create the environment (edit CUDA versions to match your machine)
conda env create -f environment.yml
conda activate bipt

# 2) PyTorch3D must match your torch/CUDA build — see
#    https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md

# 3) Install this package (editable)
pip install -e .
```

See `requirements.txt` / `environment.yml` for details. Bi-PT was trained on
NVIDIA A100 (80 GB) GPUs.

## Data

Full datasets are not distributed with the repo, but the shared **atlas** meshes and
a small **toy sample** (5 real cases) *are* bundled under `data/` so you can smoke-test
the whole pipeline immediately — see the quick-start in [`data/README.md`](data/README.md).
That file also documents the expected directory layout and formats (sparse `.npz` point
clouds and labeled `.vtp` meshes) for pointing the scripts at your own prepared data.

## Preprocessing

Two generators build the training inputs from labeled 4-chamber segmentations
(see [`scripts/preprocess/README.md`](scripts/preprocess/README.md) for full docs,
the label convention, and forbidden-pair handling):

- `scripts/preprocess/generate_spc.py` — sparse point clouds (`.npz` with `pcs` /
  `pairs`), from 11 short-axis + 3 long-axis planes. Self-contained
  (`SimpleITK`, `scipy`, `numpy`).
- `scripts/preprocess/generate_gt_mesh.py` — labeled ground-truth meshes (`.vtp`
  with `VertexBoundaryLabels`) via SurfaceNets + normal-marching labels. Requires
  external modules from the SurfaceNets / HeartDeformNet codebase (see the docs).

```bash
pip install -r requirements-preprocess.txt
python scripts/preprocess/generate_spc.py --src-root <labels> --dst-root data/train/spc_ds --forbidden_pair 2 3
```

## Training

Bi-PT (full bidirectional model), matching the paper's settings:

```bash
python scripts/train.py \
  --ca-type double \
  --label-dim 2 \
  --cd-mode both \
  --sparse-dir data/train/spc_ds \
  --gt-dir-2   data/train/gt_ds_5632_vtp \
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
  --ckpt-dir ./checkpoints
```

This exact command is also provided as a runnable launcher: [`configs/bipt_paper.sh`](configs/bipt_paper.sh).

Paper hyperparameters: AdamW, lr `5e-4`, weight decay `1e-4`, 300 epochs with cosine
annealing, gradient clipping (max norm `0.1`), batch size `5`; loss weights
`λ_plain = 0.5`, `λ_sa = 0.5` (via `--cd-mode both`), `λ₁ = 0.3`, `λ₂ = 0.7`
(via `--l1-loss-weight 0.3`), `λ_lap = 1.0`; `N₀ = M₀ = 5632`, `C = d = 512`.

> **Architectural constraint.** The cross-attention width equals the Point Transformer
> bottleneck width, so `--num-hidden` must equal `32 · 2^(--ca-nblocks)` (the paper's
> `512 = 32·2⁴` with `--ca-nblocks 4`), and `--latent-len` must be `32` (the decoder's
> finest channel width). If you change `--ca-nblocks`, set `--num-hidden` to match
> (e.g. `ca-nblocks 3 → num-hidden 256`).

### Reproducing the ablations (paper Table 2)

| Variant                       | Change to the command above          |
|-------------------------------|--------------------------------------|
| Plain CD (no semantic labels) | `--cd-mode plain`                    |
| Single cross-attention (sCA)  | `--ca-type single`                   |
| Translation-only (no affine)  | `--no-affine-dynamics`               |
| Without Laplacian             | `--lap-weight 0`                     |
| 3-label SA-CD                 | `--label-dim 3` (with `--gt-dir-3` / `--atlas-path-3`) |

## Evaluation

Evaluation is two stages: **infer** deforms the atlas toward each test SPC and writes
predicted meshes, then **evaluate** scores those meshes against the ground truth.

```bash
# extra deps for the metric suite (scipy / trimesh / open3d; SI needs torch-mesh-isect)
pip install -r requirements-eval.txt
```

**1. Inference** — `scripts/infer.py` runs the model, denormalizes predictions back to
world coordinates, and writes `<case>_pred_l1.vtp` / `<case>_pred_l2.vtp` (with the atlas
label arrays). Architecture is auto-loaded from the checkpoint (`train.py` stores it), so no
model flags are needed. Only for **checkpoints trained before the architecture was saved**
does inference fall back to the paper defaults — for those, pass any non-default
`--num-hidden` / `--ca-nblocks` / `--ca-nneighbor` explicitly. It can also report Chamfer
distance directly.

```bash
python scripts/infer.py \
  --ckpt checkpoints/recon_model_lab2_q8_..._latest.pt \
  --spc-dir data/test/spc_ds \
  --gt-dir  data/test/gt_ds_5632_vtp \
  --atlas-path-2 data/atlas.vtp \
  --eval-cd both \
  --out-dir outputs/infer \
  --save-metrics outputs/infer/cd_metrics.csv
```

**2. Metrics** — `scripts/evaluate.py` consumes the predicted meshes and reports the paper's
geometric metrics (Chamfer distance, Earth Mover's Distance, point-to-surface) and
mesh-quality metrics (normal consistency, non-manifold vertex/edge/face ratios — NM-F is
the paper's ENF — and the self-intersection ratio SI). Metrics are vertex-based and can be
computed on the full mesh (`--chambers all`) or per chamber (`--chambers 4,5,6`).

```bash
python scripts/evaluate.py \
  --pred-root outputs/infer \
  --gt-root   data/test/gt_ds_5632_vtp \
  --which l2 --chambers all \
  --label-array VertexBoundaryLabels --label-on point \
  --compute-si --workers 8 --backend process \
  --out-csv outputs/eval/metrics.csv
```

> **SI** (`--compute-si`) uses the torch-mesh-isect CUDA extension; without it (or without a
> GPU) SI is reported as `NaN` and every other metric still runs. The non-manifold
> vertex/edge metrics (NM-V/NM-E) call Open3D, whose native routine can hard-crash (SIGSEGV)
> on some cluster Open3D builds — that takes the worker down (`process ... terminated
> abruptly`). Pass **`--no-nmve`** to skip just NM-V/NM-E (reported as `NaN`) and compute
> everything else (CD/EMD/P2F/NC/NM-F/SI) anywhere.

## Qualitative results

![Bi-PT qualitative reconstructions (paper Fig. 3)](assets/fig3.png)

Predicted four-chamber meshes overlaid on the input sparse point cloud, colored by
per-vertex distance to the ground truth (paper Fig. 3).

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{hu2026bipt,
  title     = {Bi-PT: Bidirectional Cross-Attention Point Transformers for
               Four-Chamber Heart Reconstruction from Sparse Cardiac MRI Data},
  author    = {Hu, Chenchuhui and Pan, Shaoming and Axel, Leon and Ye, Meng},
  booktitle = {Statistical Atlases and Computational Models of the Heart (STACOM),
               MICCAI Workshop},
  year      = {2026}
}
```
<!-- Update venue/pages/publisher once the proceedings are finalized. -->

## Acknowledgements

The Point Transformer backbone and the Neural ODE deformation build on prior work
cited in the paper (Point Transformer; Neural ODEs; NDM; Neural Mesh Flow, and others).

## License

Released under the [MIT License](LICENSE).
