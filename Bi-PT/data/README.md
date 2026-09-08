# Data

Full datasets are **not** committed to this repository (see `.gitignore`). What
*is* bundled: the shared **atlas** meshes (`atlas.vtp`, `atlas_3lab.vtp`) and a
tiny **toy sample** of a few real cases under `data/sample/`, enough to smoke-test
the pipeline end to end. This document describes the directory layout and file
formats the code expects, so you can point the scripts at your own prepared data.

## Bundled toy sample

`data/sample/` contains 5 real cases (SPC + labeled GT mesh) — not a meaningful
training set, just enough to check that training, inference, and evaluation run.
Because there is no separate split, the same folder is used as train and test here:

```
data/
├── atlas.vtp
├── atlas_3lab.vtp
└── sample/
    ├── labels_id/         # 101,102 _merge_4ch_stack_id_affine.nii.gz  (raw 4-chamber label maps — preprocessing input)
    ├── spc_ds/            # 101.npz … 105.npz  (sparse point clouds)
    └── gt_ds_5632_vtp/    # 101.vtp … 105.vtp  (labeled GT meshes)
```

`labels_id/` holds two raw segmentation label maps (voxel IDs 1–6) so you can also
run the **preprocessing** (`scripts/preprocess/`) end to end — see that folder's
README for the verified commands that turn a label map into `spc_ds/*.npz` and
`gt_ds_5632_vtp/*.vtp`.

```bash
# 1) train a few steps on the sample (won't converge — smoke test only)
python scripts/train.py --ca-type double --label-dim 2 --cd-mode both \
  --sparse-dir data/sample/spc_ds --gt-dir-2 data/sample/gt_ds_5632_vtp \
  --atlas-path-2 data/atlas.vtp --epochs 3 --batch-size 2 \
  --num-hidden 512 --ca-nblocks 4 --ca-nneighbor 16 --norm-type in \
  --ckpt-dir ./checkpoints_sample

# 2) inference (auto-resolves architecture from the checkpoint)
python scripts/infer.py --ckpt ./checkpoints_sample/*_latest.pt \
  --spc-dir data/sample/spc_ds --gt-dir data/sample/gt_ds_5632_vtp \
  --atlas-path-2 data/atlas.vtp --eval-cd both --out-dir ./outputs_sample

# 3) metrics (--no-nmve avoids an Open3D crash on some cluster builds)
python scripts/evaluate.py --pred-root ./outputs_sample \
  --gt-root data/sample/gt_ds_5632_vtp --which l1,l2 --chambers all \
  --label-array VertexBoundaryLabels --no-nmve --out-csv ./outputs_sample/metrics.csv
```

To reproduce paper numbers, use a pretrained checkpoint and the full test set in
place of the sample.

## Source data

Bi-PT was developed on a public cardiac CT collection segmented with
[TotalSegmentator V2](https://github.com/wasserth/TotalSegmentator) into the
four cardiac chambers plus myocardium (RA, LA, RV, LV, MYO) and major vessels.
Each case is standardized from the patient coordinate space to a cardiac
coordinate space, and a **sparse point cloud (SPC)** with semantic labels is
generated to mimic the slices acquired in a routine CMR protocol. Labeled
ground-truth meshes are constructed from the segmentation masks.

> See the paper (Sec. 3.1) for the full preprocessing description. The scripts
> that produce the files below live in [`../scripts/preprocess/`](../scripts/preprocess/).

## Expected layout

```
data/
├── atlas.vtp                     # shared atlas mesh (2-label variant), VertexBoundaryLabels   [bundled]
├── atlas_3lab.vtp                # optional: atlas for the 3-label ablation, VertexLabels3      [bundled]
├── sample/                       # small toy sample (5 real cases)                             [bundled]
│   ├── spc_ds/                   #   101.npz … 105.npz
│   └── gt_ds_5632_vtp/           #   101.vtp … 105.vtp
├── train/                        # your full data (git-ignored)
│   ├── spc_ds/                   # sparse point clouds (one .npz per case)
│   │   ├── 101.npz
│   │   └── ...
│   └── gt_ds_5632_vtp/           # ground-truth meshes (one .vtp per case)
│       ├── 101.vtp
│       └── ...
├── val/
│   ├── spc_ds/
│   └── gt_ds_5632_vtp/
└── test/
    ├── spc_ds/
    └── gt_ds_5632_vtp/
```

The subject-level split used in the paper is 800 / 100 / 100 (train / val / test).

## File formats

### Sparse point cloud — `.npz`
Each file contains a sampled sparse point cloud for one case:

| key      | shape    | dtype   | meaning                                             |
|----------|----------|---------|-----------------------------------------------------|
| `pcs`    | `(N, 3)` | float   | 3D point coordinates                                |
| `pairs`  | `(N, 2)` | int     | per-point **boundary-pair** semantic label `(a, b)` |

(The default key names `pcs` / `pairs` can be overridden with
`--sparse-pcs-key` / `--sparse-pairs-key`.)

### Ground-truth / atlas mesh — `.vtp`
VTK PolyData with triangular faces and per-point label arrays:

| array name             | components | used for                                  |
|------------------------|------------|-------------------------------------------|
| `VertexBoundaryLabels` | 2          | 2-label semantic-aware Chamfer (default)  |
| `VertexLabels3`        | 3          | 3-label variant (ablation, `--label-dim 3`) |

Labels follow the chamber id convention used in the code (`LABEL_IDS = [2, 3, 4, 5, 6]`).

## Atlas

The atlas is a single heart mesh randomly selected from the training set
(`N0 = 5632` vertices in the paper). It is shared across all subjects and is
deformed toward each target. Provide it at `data/atlas.vtp` (and
`data/atlas_3lab.vtp` for the 3-label variant).
