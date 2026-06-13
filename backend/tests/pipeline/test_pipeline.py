"""W3.T1 — pure-numpy unit tests for the model-agnostic pipeline utils.

Covers (spec/06 §1/§5 steps 1,4,5; spec/05 §2/§4):
  (a) lift_tracks_to_3d — synthetic depth + known pixel + identity-ish camera
      unprojects to the expected mayavius world point (verifies x=(u-cx)D/fx etc.
      AND the F=diag(1,-1,-1) OpenCV->mayavius flip);
  (b) quantize_positions — matches the encoder/decoder inverse within tolerance +
      degenerate axis -> 0;
  (c) assemble_scene4d — a synthetic GeometryResult+TrackResult with one obviously
      moving track lands the moving region in dynamic_positions and the rest in
      static_positions; tracks/cameras populated; AABB spans all; and the scene is
      RAW (enforce_caps called SEPARATELY confirms assemble did not cap);
  (d) decode_and_subsample — cv2-gated (pytest.importorskip) on a real backend
      tiny.mp4: asserts [S,3,~518,...] shape + S <= max_frames; skips without cv2.

Pure numpy / opencv-only — NO torch, NO fastapi. Runs in `make test` (no-ML).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from app.core.domain.models import CameraTrack, ReconstructionRequest, Scene4D
from app.core.services.reconstruction_service import enforce_caps
from app.pipeline.assemble import (
    GeometryResult,
    TrackResult,
    assemble_scene4d,
)
from app.pipeline.lift import lift_tracks_to_3d
from app.pipeline.quantize import compute_aabb, quantize_positions

_QMAX = 65535


# ---------------------------------------------------------------------------
# (a) lift correctness — unprojection formula + OpenCV->mayavius axis flip
# ---------------------------------------------------------------------------

def test_lift_unprojects_with_identity_camera_and_axis_flip() -> None:
    """A known pixel + depth + identity c2w lands at the expected mayavius point.

    With c2w = identity, the OpenCV camera point is the world point, and the only
    transform is the F = diag(1,-1,-1) flip. Pixel intrinsics fx,fy,cx,cy and depth
    D give: x_cam=(u-cx)D/fx, y_cam=(v-cy)D/fy, z_cam=D; mayavius=(x,-y,-z).
    """
    H, W = 40, 60
    fx, fy, cx, cy = 50.0, 50.0, 30.0, 20.0
    u, v, D = 45.0, 35.0, 2.0

    depth = np.zeros((1, H, W), dtype=np.float32)
    depth[0, int(round(v)), int(round(u))] = D  # depth sampled at (round(v), round(u))

    tracks_2d = np.array([[[u, v]]], dtype=np.float32)   # (M=1, T=1, 2)
    visibility = np.array([[True]], dtype=bool)
    intrinsics = np.array([fx, fy, cx, cy], dtype=np.float32)  # (4,) broadcast
    c2w = np.eye(4, dtype=np.float32)                          # identity
