"""Lift 2D CoTracker3 tracks → 3D world-space track positions (spec/06 §5 step 4).

Given per-frame 2D pixel tracks ``(u_px, v_px)`` (on the **processed 518-width
grid** — grid-consistency, spec/06 §5 step 4), VGGT per-frame depth, and **pixel**
intrinsics, unproject each sample into the OpenCV camera frame and transform it into
**mayavius world space** (spec/05 §2 / spec/06 §4.1a):

    OpenCV unprojection (z-along-axis depth ``D`` at pixel ``(u_px, v_px)``):
        x_cam = (u_px - cx) * D / fx
        y_cam = (v_px - cy) * D / fy
        z_cam = D
    camera->world:   p_world_opencv = R_c2w · p_cam + t_c2w
    OpenCV->mayavius axis flip (F = diag(1, -1, -1)):
        p_may = F · p_world_opencv = (x, -y, -z)

The flip and the cam->world transform MUST be applied identically to camera poses
(done in the adapter) so points and cameras stay consistent. Depth is sampled at
``(round(v_px), round(u_px))``; samples landing on an invalid/hole depth (≤0 or
non-finite) or out of bounds are marked **invisible** (a ribbon gap).

Pure numpy — NO torch, NO opencv, NO fastapi.
"""

from __future__ import annotations

import numpy as np

# OpenCV -> mayavius axis flip (spec/06 §4.1a). F == F^-1.
_F = np.array([1.0, -1.0, -1.0], dtype=np.float32)


def _normalize_intrinsics(intrinsics: np.ndarray, T: int) -> np.ndarray:
    """Coerce per-frame intrinsics to ``(T, 4)`` float32 ``(fx, fy, cx, cy)`` (pixels).

    Accepts either ``(T, 4)`` already, a single ``(4,)`` broadcast across frames, a
    ``(T, 3, 3)`` pixel K-matrix stack, or a single ``(3, 3)`` K broadcast.
    """
    arr = np.asarray(intrinsics, dtype=np.float32)
    if arr.ndim == 1 and arr.shape == (4,):
        return np.broadcast_to(arr, (T, 4)).astype(np.float32).copy()
    if arr.ndim == 2 and arr.shape == (T, 4):
        return arr.copy()
    if arr.ndim == 2 and arr.shape == (3, 3):
        fx, fy, cx, cy = arr[0, 0], arr[1, 1], arr[0, 2], arr[1, 2]
        row = np.array([fx, fy, cx, cy], dtype=np.float32)
        return np.broadcast_to(row, (T, 4)).astype(np.float32).copy()
    if arr.ndim == 3 and arr.shape == (T, 3, 3):
        out = np.empty((T, 4), dtype=np.float32)
        out[:, 0] = arr[:, 0, 0]
        out[:, 1] = arr[:, 1, 1]
        out[:, 2] = arr[:, 0, 2]
        out[:, 3] = arr[:, 1, 2]
        return out
    raise ValueError(
        f"intrinsics must be (T,4), (4,), (T,3,3) or (3,3); got shape {arr.shape} for T={T}"
    )


def _normalize_c2w(c2w: np.ndarray, T: int) -> np.ndarray:
    """Coerce camera->world transforms to ``(T, 4, 4)`` float32.
