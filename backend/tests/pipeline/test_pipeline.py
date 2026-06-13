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

    positions, vis_out = lift_tracks_to_3d(tracks_2d, visibility, depth, intrinsics, c2w)

    x_cam = (u - cx) * D / fx
    y_cam = (v - cy) * D / fy
    z_cam = D
    expected = np.array([x_cam, -y_cam, -z_cam], dtype=np.float32)  # F flip

    assert vis_out[0, 0]
    assert np.allclose(positions[0, 0], expected, atol=1e-5), (positions[0, 0], expected)


def test_lift_axis_flip_is_diag_1_neg1_neg1_under_rotated_camera() -> None:
    """With a non-trivial c2w the flip + cam->world compose correctly.

    Build c2w with a known rotation R and translation t; verify
    p_may = F · (R · p_cam + t).
    """
    H, W = 30, 30
    fx = fy = 25.0
    cx = cy = 15.0
    u, v, D = 20.0, 10.0, 3.0

    depth = np.zeros((1, H, W), dtype=np.float32)
    depth[0, int(round(v)), int(round(u))] = D

    # 90-degree rotation about world Z, translation (1,2,3).
    theta = np.pi / 2
    R = np.array(
        [[np.cos(theta), -np.sin(theta), 0.0],
         [np.sin(theta), np.cos(theta), 0.0],
         [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    t = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = R
    c2w[:3, 3] = t

    tracks_2d = np.array([[[u, v]]], dtype=np.float32)
    visibility = np.array([[True]], dtype=bool)
    intrinsics = np.array([fx, fy, cx, cy], dtype=np.float32)

    positions, vis_out = lift_tracks_to_3d(tracks_2d, visibility, depth, intrinsics, c2w)

    p_cam = np.array([(u - cx) * D / fx, (v - cy) * D / fy, D], dtype=np.float32)
    p_world = R @ p_cam + t
    F = np.array([1.0, -1.0, -1.0], dtype=np.float32)
    expected = (p_world * F).astype(np.float32)

    assert vis_out[0, 0]
    assert np.allclose(positions[0, 0], expected, atol=1e-4), (positions[0, 0], expected)


def test_lift_marks_depth_hole_invisible() -> None:
    """A sample on a zero/invalid depth pixel is marked invisible (ribbon gap)."""
    H, W = 20, 20
    depth = np.zeros((1, H, W), dtype=np.float32)  # all holes
    tracks_2d = np.array([[[10.0, 10.0]]], dtype=np.float32)
    visibility = np.array([[True]], dtype=bool)
    intrinsics = np.array([10.0, 10.0, 10.0, 10.0], dtype=np.float32)
    c2w = np.eye(4, dtype=np.float32)

    _, vis_out = lift_tracks_to_3d(tracks_2d, visibility, depth, intrinsics, c2w)
    assert not vis_out[0, 0]


def test_lift_out_of_bounds_pixel_invisible() -> None:
    """A pixel outside the depth grid is invisible (no out-of-range crash)."""
    H, W = 16, 16
    depth = np.ones((1, H, W), dtype=np.float32) * 2.0
    tracks_2d = np.array([[[100.0, 100.0]]], dtype=np.float32)  # OOB
    visibility = np.array([[True]], dtype=bool)
    intrinsics = np.array([10.0, 10.0, 8.0, 8.0], dtype=np.float32)
    c2w = np.eye(4, dtype=np.float32)
    _, vis_out = lift_tracks_to_3d(tracks_2d, visibility, depth, intrinsics, c2w)
    assert not vis_out[0, 0]


# ---------------------------------------------------------------------------
# (b) quantize_positions — encoder/decoder inverse + degenerate axis
# ---------------------------------------------------------------------------

def _dequantize(q: np.ndarray, amin: np.ndarray, amax: np.ndarray) -> np.ndarray:
    span = (amax - amin).astype(np.float32)
    return (amin + q.astype(np.float32) / np.float32(_QMAX) * span).astype(np.float32)


def test_quantize_roundtrip_within_one_lsb() -> None:
    """quantize -> dequantize reconstructs each position within one quantization step."""
