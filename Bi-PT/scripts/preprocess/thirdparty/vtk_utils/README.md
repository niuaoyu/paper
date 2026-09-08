# vtk_utils (vendored)

`vtk_utils.py` provides the VTK mesh/image helpers used by
[`../../generate_gt_mesh.py`](../../generate_gt_mesh.py): `eraseBoundary`,
`exportSitk2VTK`, `vtkImageResample`, `smooth_polydata`, `cleanPolyData`,
`build_transform_matrix`, `transform_polydata`, `write_vtk_polydata`, and more.

> Vendored from Fanwei Kong's `vtk_utils` — MIT License (Copyright (c) 2020
> Fanwei Kong); see [`LICENSE`](LICENSE). Byte-for-byte copy, no modifications.

`generate_gt_mesh.py` adds the parent `thirdparty/` folder to `sys.path`, so
`from vtk_utils.vtk_utils import *` resolves with no extra configuration
(`vtk_utils/` is imported as a Python 3 namespace package — no `__init__.py`
needed). Requires the `vtk` package.
