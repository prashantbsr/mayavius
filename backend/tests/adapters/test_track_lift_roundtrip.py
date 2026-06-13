"""Round-trip guard for the CoTracker3 2D->3D track lift (spec/06 §5 step 4 / §4.1a).

VggtAdapter emits a MAYAVIUS ``CameraTrack``; ``run_tracks`` must feed the lift the
native OPENCV c2w (the lift applies the single OpenCV->mayavius flip itself). The
earlier impl fed the mayavius c2w, applying the flip TWICE and detaching the ribbons
from the VGGT cloud — a silent failure the smoke test (which only checks track
count/shape) could not catch. This projects a KNOWN mayavius world point to a pixel,
lifts it back through ``_camera_to_opencv_c2w_stack`` + ``lift_tracks_to_3d``, and
asserts recovery; a second test confirms the double-flip path would be wrong (teeth).

Torch-free (numpy lift + the camera helper) — runs in the no-ML CI gate.
"""

from __future__ import annotations

import math

import numpy as np

from app.adapters.cotracker3_adapter import (
    _camera_to_opencv_c2w_stack,
    _quat_xyzw_to_rotmat,
)
from app.core.domain.models import CameraTrack
from app.pipeline.lift import lift_tracks_to_3d

# OpenCV<->mayavius axis flip (spec/06 §4.1a), as a diagonal matrix for the test math.
_F = np.diag([1.0, -1.0, -1.0]).astype(np.float32)


def _scene():
    """A non-trivial mayavius camera + a known camera-space point projecting to
    INTEGER pixels (so depth sampling at round(v),round(u) is exact)."""
    a = math.radians(40.0)  # 40deg about +Y -> a non-identity rotation
    q_may = np.array([0.0, math.sin(a / 2), 0.0, math.cos(a / 2)], dtype=np.float32)
    t_may = np.array([0.5, -0.3, 1.2], dtype=np.float32)
    R_may = _quat_xyzw_to_rotmat(q_may)

    # Native OpenCV c2w (ground truth) via the F-involution VggtAdapter inverts.
    R_c2w = _F @ R_may @ _F
    t_c2w = _F @ t_may

    fx = fy = 500.0
    cx = cy = 259.0  # 518-wide processed grid
    H = W = 518
    xc, yc, zc = 0.8, -0.4, 4.0  # OpenCV camera-space point (z>0, in front)
    u = fx * xc / zc + cx        # 359.0 (integer)
    v = fy * yc / zc + cy        # 209.0 (integer)
    D = zc
    p_cam = np.array([xc, yc, zc], dtype=np.float32)

    p_world_ocv = R_c2w @ p_cam + t_c2w
    p_may_true = (_F @ p_world_ocv).astype(np.float32)

    depth = np.zeros((1, H, W), dtype=np.float32)
    depth[0, int(round(v)), int(round(u))] = D

    cam = CameraTrack(
        poses=np.array([[*q_may, *t_may]], dtype=np.float32),
        intrinsics=np.array([[fx / W, fy / H, cx / W, cy / H]], dtype=np.float32),
