# Preprocessing

Turn labeled 4-chamber segmentations into the fixed-size inputs Bi-PT trains on.
The full pipeline is **three steps** — two generators that produce *variable-size*
outputs, then a downsampler that pins everything to exactly `N` points/vertices
(`N = 5632` in the paper, hence the `spc_ds` / `gt_ds_5632_vtp` — "ds" = down-sampled
— folder names in [`../../data/README.md`](../../data/README.md)).

```
                                            ┌─ generate_spc.py ──▶ raw SPC .npz  (14·P points) ─┐
 4-chamber label map (.nii.gz, IDs 1–6) ────┤                                                   ├──▶ downsample.py ──▶ spc_ds/*.npz          (exactly N)
                                            └─ generate_gt_mesh.py ─▶ raw GT .vtp (~variable) ──┘                     gt_ds_5632_vtp/*.vtp   (exactly N)
```

| # | Script | Input | Output | Consumed as |
|---|--------|-------|--------|-------------|
| 1 | `generate_spc.py`     | 4-chamber label map (`.nii`/`.nii.gz`) | raw `.npz` (`pcs (14,P,3)`, `pairs (14,P,2)`) | → step 3 |
| 2 | `generate_gt_mesh.py` | 4-chamber label map (`.nii`/`.nii.gz`) | raw labeled `.vtp` (`VertexBoundaryLabels` + `BoundaryLabels` cells)  | → step 3 |
| 2b | `modify_vertex_label.py` *(3-label variant only)* | raw `.vtp` (`BoundaryLabels` cell array) | adds `VertexLabels3 (N,3)` | → step 3 |
| 3 | `downsample.py`       | raw `.npz` and/or raw `.vtp`            | fixed-`N` `.npz` / `.vtp`                     | `data/*/spc_ds/`, `data/*/gt_ds_5632_vtp/`, and the atlas |

> Because step 3 samples *down*, steps 1–2 must produce **more** than `N` points:
> use `generate_spc.py --num-points P` with `14·P > N` (e.g. `--num-points 500 → 7000`)
> and `generate_gt_mesh.py --target_node_num` above `N` (e.g. `8000`).

## Label convention

Background `0`, and (matching the training `LABEL_IDS = [2,3,4,5,6]`):

| id | 1 | 2 | 3 | 4 | 5 | 6 |
|----|---|---|---|---|---|---|
| structure | Aorta | LA | RA | MYO | LV | RV |

**Forbidden pairs** are label pairs that must not share a boundary (e.g. LA–RA
`2 3`). The generators can (a) *carve* a thin gap band between them in voxel space
before boundary extraction, and (b) *collapse* the pair if it still appears after
labeling. `--forbidden_pair` is repeatable.

## 1. Sparse point clouds — `generate_spc.py`

Self-contained (`SimpleITK`, `numpy`, `scipy`). Slices each label map into 11
short-axis (SAX) + 3 long-axis (LAX: 4CH/3CH/2CH) planes derived from cardiac
landmarks, extracts sub-pixel boundary midpoints, and farthest-point samples
`--num-points` per plane. Output per case: `pcs (S,P,3)` and `pairs (S,P,2)`, `S = 14`.

```bash
python scripts/preprocess/generate_spc.py \
  --src-root /path/to/label_maps --dst-root data/train/spc_raw \
  --name-contains merge --num-points 500 \
  --forbidden_pair 2 3 --forbidden_band_vox 2 --seed 42
```

Recurses `--src-root` for files whose name contains `--name-contains` (default
`merge`) ending in `.nii`/`.nii.gz`; output name is the filename's first
`_`-separated token (`case001_merge.nii.gz → case001.npz`). Label IDs are
configurable via `--rv-id/--ra-id/--la-id/--myo-id/--lv-id/--ao-id`.

## 2. Ground-truth meshes — `generate_gt_mesh.py`

Runs SurfaceNets on the segmentation, smooths/decimates to a triangle mesh, then
labels every vertex with its boundary pair (`VertexBoundaryLabels`, a 2-vector)
via normal-marching + neighbor-vote (endo fixed to MYO–LV `(4,5)`, epi forbidden
from carrying LV). Needs **VTK ≥ 9.3** (`vtkSurfaceNets3D`).

**Dependencies (vendored).** Reuses helpers by Fanwei Kong under
[`thirdparty/`](thirdparty/): `pre_process.py` + `utils.py` (Apache-2.0) and
`vtk_utils/vtk_utils.py` (MIT); added to `sys.path` automatically — just
`pip install vtk SimpleITK scipy`. (Set `BIPT_MESH_EXTERNAL` / `BIPT_MESH_SRC`
to use an external HeartDeformNet checkout instead.)

```bash
python scripts/preprocess/generate_gt_mesh.py \
  --seg_fn /path/to/case001_merge.nii.gz \
  --output data/train/gt_raw/case001.vtp \
  --keep_labels 2 3 4 5 6 --target_node_num 8000 \
  --forbidden_pair 2 3 --forbidden_pair 2 4 --forbidden_pair 3 4 --forbidden_pair 5 6 \
  --forbidden_band_vox 2
```

`--keep_labels 2 3 4 5 6` excludes the aorta (label 1); the four forbidden pairs
(LA–RA, LA–MYO, RA–MYO, LV–RV) are the config that reproduces the committed reference
GT — see the correctness check at the bottom of this file.

The **atlas** is one such mesh (see `../../data/README.md`) — generate it the same
way, downsample it (step 3), and place it at `data/atlas.vtp`.

## 3. Downsample to a fixed size — `downsample.py`

Resamples every case to exactly `--target-n` points/vertices. Self-contained
(`vtk`, `numpy`, `scipy`). Two modes:

* `--mode sparse` — farthest-point-samples the `.npz` cloud (`pcs` + `pairs`).
  Cases with fewer than `target-n` points are skipped and logged (no padding).
* `--mode gt` — farthest-point-samples the mesh vertices, then remaps the
  triangles onto the kept vertices with a **label-constrained** nearest-neighbor
  search (exact boundary-pair → share-one-label → global NN) so boundary labels
  stay intact. `VertexBoundaryLabels` is required; `VertexLabels3` is preserved
  when present (3-label variant).

```bash
python scripts/preprocess/downsample.py --mode both \
  --sparse-in data/train/spc_raw --sparse-out data/train/spc_ds \
  --gt-in     data/train/gt_raw  --gt-out     data/train/gt_ds_5632_vtp \
  --target-n 5632 --pair-array VertexBoundaryLabels --seed 42
```

## 3-label variant — `modify_vertex_label.py` (optional)

The default GT tags each vertex with its boundary **pair** (`VertexBoundaryLabels`,
2 slots) — the two structures it lies between. Where **three** structures meet (a
triple junction) a pair can't represent all of them, so the 3-label ablation
(`--label-dim 3`, array `VertexLabels3`) uses a 3-slot vertex array. This script
builds it from the mesh's **cell** boundary labels (the `BoundaryLabels` CellData
array that `generate_gt_mesh.py` writes):

> **Conversion (pure set-union, no voting).** For each vertex:
> 1. gather the label components of every incident triangle's `BoundaryLabels`;
> 2. drop background (`--bg_id 0`) and negatives;
> 3. sort the unique labels ascending;
> 4. write the first three as `VertexLabels3 = (L0, L1, L2)`, padding with `-1` when
>    fewer than three; also write `VertexLabelCount` (the true unique count, may be >3).
>
> So a vertex between just LA–LV becomes `(2, 5, -1)`, while one where LA, MYO and
> LV meet becomes `(2, 4, 5)`.

Run it on the **raw** mesh (which still has the `BoundaryLabels` cell array) *before*
`downsample.py`; downsample then carries `VertexLabels3` through with
`--vert3-array VertexLabels3`:

```bash
python scripts/preprocess/modify_vertex_label.py \
  --in_mesh data/train/gt_raw/case001.vtp \
  --out_mesh data/train/gt_raw_3lab/case001.vtp \
  --cell_array BoundaryLabels --out_array VertexLabels3

python scripts/preprocess/downsample.py --mode gt \
  --gt-in data/train/gt_raw_3lab --gt-out data/train/gt_ds_3lab_vtp \
  --target-n 5632 --pair-array VertexBoundaryLabels --vert3-array VertexLabels3
```

The result has both `VertexBoundaryLabels (N,2)` and `VertexLabels3 (N,3)` — the
inputs for `--label-dim 3` and the `atlas_3lab.vtp` template.

## End-to-end verification (bundled sample)

Verified on the label maps in [`../../data/sample/labels_id/`](../../data/sample/labels_id/)
(env: VTK 9.5). These are the **exact parameters that reproduce the committed reference
GT** (`../../data/sample/gt_ds_5632_vtp/`): labels `2 3 4 5 6` (no aorta) and forbidden
pairs LA–RA, LA–MYO, RA–MYO, LV–RV.

```bash
# 1) raw SPC (14×500 = 7000 pts) and raw GT mesh (~7.8k verts)
python scripts/preprocess/generate_spc.py --src-root data/sample/labels_id \
  --dst-root /tmp/pp/spc_raw --num-points 500 --forbidden_pair 2 3 --forbidden_band_vox 2
python scripts/preprocess/generate_gt_mesh.py \
  --seg_fn data/sample/labels_id/101_merge_4ch_stack_id_affine.nii.gz \
  --output /tmp/pp/gt_raw/101.vtp --keep_labels 2 3 4 5 6 --target_node_num 8000 \
  --forbidden_pair 2 3 --forbidden_pair 2 4 --forbidden_pair 3 4 --forbidden_pair 5 6 \
  --forbidden_band_vox 2

# 2) downsample both to exactly 5632
python scripts/preprocess/downsample.py --mode both \
  --sparse-in /tmp/pp/spc_raw --sparse-out /tmp/pp/spc_ds \
  --gt-in /tmp/pp/gt_raw --gt-out /tmp/pp/gt_ds --target-n 5632
```

**Correctness check (vs. the committed GT).** Comparing the regenerated `gt_ds/101.vtp`
against `data/sample/gt_ds_5632_vtp/101.vtp`:

- symmetric vertex **Chamfer distance = 1.5 mm** — about half the ~3.0 mm mean vertex
  spacing, i.e. the floor from independently FPS-resampling the *same* surface to 5632
  vertices (not error); centroids agree to sub-mm;
- the `VertexBoundaryLabels` boundary-pair set matches **exactly** (9 / 9, none extra,
  none missing).

Including the aorta (`--keep_labels 1 …`) instead raises CD to ~4 mm with a ~58 mm
outlier tail (the aorta extends well beyond the chambers) — a useful sanity signal that
the metric and frame are wired correctly.

Result: `spc_ds/101.npz` → `pcs (5632,3)` / `pairs (5632,2)`; `gt_ds/101.vtp` →
5632 verts with `VertexBoundaryLabels (5632,2)`, and the forbidden `(2,3)` pair
absent throughout.

## Dependencies

See [`../../requirements-preprocess.txt`](../../requirements-preprocess.txt):
`SimpleITK`, `scipy`, `numpy`, `vtk`. The mesh generator additionally uses the
vendored Berkeley helpers under `thirdparty/`.
