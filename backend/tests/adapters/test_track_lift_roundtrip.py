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


