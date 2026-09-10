"""Coordinate helpers shared by SAX/LAX sparse-point acquisition code."""

import numpy as np


def contour_yx_to_xyz(contour_yx, z):
    """Convert scikit-image ``(row=y, column=x)`` points to Cartesian XYZ."""
    contour_yx = np.asarray(contour_yx, dtype=np.float64)
    if contour_yx.ndim != 2 or contour_yx.shape[1] != 2:
        raise ValueError(f"contour must have shape (N, 2), got {contour_yx.shape}")
    xyz = np.empty((len(contour_yx), 3), dtype=np.float64)
    xyz[:, 0] = contour_yx[:, 1]
    xyz[:, 1] = contour_yx[:, 0]
    xyz[:, 2] = z
    return xyz


def rotation_lv_minus_rv_to_positive_y(lv_yx, rv_yx):
    """Return a z angle mapping the LV-minus-RV image vector to positive Y.

    Both centroids are supplied in image ``(row=y, column=x)`` order.  The
    returned angle follows the Cartesian right-handed convention used by
    ``scipy.spatial.transform.Rotation``.
    """
    vector_yx = np.asarray(lv_yx, dtype=np.float64) - np.asarray(rv_yx, dtype=np.float64)
    if vector_yx.shape != (2,):
        raise ValueError(f"centroids must each contain two coordinates, got {vector_yx.shape}")
    vector_xy = vector_yx[::-1]
    if np.linalg.norm(vector_xy) <= 1e-12:
        return 0.0
    return float(np.arctan2(vector_xy[0], vector_xy[1]))
