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

