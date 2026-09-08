# Third-party dependencies (vendored)

The ground-truth mesh generator [`../generate_gt_mesh.py`](../generate_gt_mesh.py)
reuses a small subset of two projects by Fanwei Kong. Only the files actually
imported by the mesh generator are vendored here; the rest of the HeartDeformNet
project (its TensorFlow model, losses, and data pipeline, and the `tf_nndistance`
CUDA op) is **not** included, as Bi-PT does not use it.

## Contents

| File | Provides | License |
|------|----------|---------|
| `pre_process.py` | `resample_spacing` (image resampling/centering) | Apache-2.0 (HeartDeformNet) — see [`LICENSE`](LICENSE) |
| `utils.py` | misc helpers; imported via `from utils import *` (no function called; its TensorFlow / `dataset` imports are guarded, so TensorFlow is **not** required) | Apache-2.0 (HeartDeformNet) — see [`LICENSE`](LICENSE) |
| `vtk_utils/vtk_utils.py` | `eraseBoundary`, `exportSitk2VTK`, `vtkImageResample`, `smooth_polydata`, `cleanPolyData`, `build_transform_matrix`, `transform_polydata`, `write_vtk_polydata` | MIT (Copyright 2020 Fanwei Kong) — see [`vtk_utils/LICENSE`](vtk_utils/LICENSE) |

> Copyright (C) 2022 Fanwei Kong, Shawn C. Shadden, University of California,
> Berkeley — `pre_process.py` / `utils.py` under Apache-2.0. `vtk_utils.py` is
> under the MIT License (Copyright 2020 Fanwei Kong).

## How it's wired

`generate_gt_mesh.py` adds this `thirdparty/` folder to `sys.path`, so
`import pre_process`, `import utils`, and `from vtk_utils.vtk_utils import *` all
resolve with no extra configuration (`vtk_utils/` is a Python 3 namespace
package — no `__init__.py` needed). To use an external HeartDeformNet checkout
instead, point `BIPT_MESH_SRC` / `BIPT_MESH_EXTERNAL` at it (env-var block at the
top of the script).

Validated end-to-end in this repo: `generate_gt_mesh.py` runs against a synthetic
4-chamber volume and writes a `.vtp` with the `VertexBoundaryLabels (N,2)` point
array the training loader expects.

## Local modifications

None. `pre_process.py`, `utils.py`, and `vtk_utils/vtk_utils.py` are byte-for-byte
copies of the originals.
