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
    rng = np.random.default_rng(7)
    pts = (rng.random((1000, 3)) * 10.0 - 5.0).astype(np.float32)
    amin, amax = compute_aabb(pts)
    q = quantize_positions(pts, amin, amax)
    assert q.dtype == np.uint16
    deq = _dequantize(q, amin, amax)

    step = (amax - amin).astype(np.float32) / np.float32(_QMAX)
    # Reconstruction error <= half a step + f32 slack on every axis.
    err = np.abs(deq - pts)
    tol = 0.5 * step[None, :] + 1e-4 * np.maximum(np.abs(amax), np.abs(amin))[None, :] + 1e-5
    assert np.all(err <= tol), float(err.max())


def test_quantize_matches_encoder_quantize_exactly() -> None:
    """pipeline.quantize_positions is byte-identical to the encoder's _quantize."""
    from app.wire.encoder import _quantize as enc_quantize

    rng = np.random.default_rng(11)
    pts = (rng.random((500, 3)) * 4.0 - 2.0).astype(np.float32)
    amin, amax = compute_aabb(pts)
    q_pipeline = quantize_positions(pts, amin, amax)
    q_encoder = enc_quantize(pts, amin, amax)
    assert np.array_equal(q_pipeline, q_encoder)


def test_quantize_degenerate_axis_is_zero() -> None:
    """A degenerate axis (aabb_max == aabb_min) quantizes that axis to 0 for all points."""
    pts = np.array(
        [[0.0, 5.0, 1.0], [0.0, 5.0, 2.0], [0.0, 5.0, 3.0]], dtype=np.float32
    )  # x and y constant -> degenerate
    amin, amax = compute_aabb(pts)
    assert amin[0] == amax[0] and amin[1] == amax[1]  # degenerate x, y
    q = quantize_positions(pts, amin, amax)
    assert np.all(q[:, 0] == 0)
    assert np.all(q[:, 1] == 0)
    # z is non-degenerate: endpoints map to 0 and 65535.
    assert q[0, 2] == 0
    assert q[-1, 2] == _QMAX


def test_quantize_clips_out_of_aabb_points() -> None:
    """Points outside the supplied AABB are defensively clipped to [0,65535]."""
    amin = np.zeros(3, dtype=np.float32)
    amax = np.ones(3, dtype=np.float32)
    pts = np.array([[-1.0, 2.0, 0.5]], dtype=np.float32)  # below min, above max
    q = quantize_positions(pts, amin, amax)
    assert q[0, 0] == 0
    assert q[0, 1] == _QMAX
    assert 0 <= q[0, 2] <= _QMAX


def test_compute_aabb_over_multiple_sets() -> None:
    """compute_aabb spans the union of all passed point sets."""
    a = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    b = np.array([[1.0, -2.0, 3.0]], dtype=np.float32)
    c = np.array([[-4.0, 5.0, 1.0]], dtype=np.float32)
    amin, amax = compute_aabb(a, b, c)
    assert np.array_equal(amin, np.array([-4.0, -2.0, 0.0], dtype=np.float32))
    assert np.array_equal(amax, np.array([1.0, 5.0, 3.0], dtype=np.float32))


# ---------------------------------------------------------------------------
# (c) assemble_scene4d — static/dynamic split sanity + RAW (no caps)
# ---------------------------------------------------------------------------

def _synthetic_geo_and_tracks(seed: int = 3):
    """Build a synthetic GeometryResult + TrackResult with one obviously-moving track.

    Geometry: S=4 frames of a flat static plane of points (z=0), PLUS a tight cluster
    of points around a moving location that translates frame-to-frame. One CoTracker
    track follows the moving cluster (large inter-frame displacement); a second track
    is stationary (and so must NOT mark the static plane dynamic).
    """
    rng = np.random.default_rng(seed)
    S = 4
    H, W = 8, 8  # 64 plane points/frame

    # Static plane: a fixed grid on z=0 in [0,4]^2, identical every frame.
    gx, gy = np.meshgrid(np.linspace(0.0, 4.0, W), np.linspace(0.0, 4.0, H))
    plane = np.stack([gx, gy, np.zeros_like(gx)], axis=-1).astype(np.float32)  # (H,W,3)

    # Moving cluster center translates +x each frame, far from the plane (z high).
    centers = [np.array([10.0 + 3.0 * t, 10.0, 5.0], dtype=np.float32) for t in range(S)]

    world_points = np.zeros((S, H, W, 3), dtype=np.float32)
    colors = np.zeros((S, H, W, 3), dtype=np.uint8)
    conf = np.zeros((S, H, W), dtype=np.float32)
    for t in range(S):
        wp = plane.copy()
        # Overwrite the last row of the grid with the moving cluster so each frame
        # has a dense moving blob co-located with the moving track sample.
        blob = centers[t][None, None, :] + (rng.random((1, W, 3)).astype(np.float32) - 0.5) * 0.1
        wp[-1, :, :] = blob[0]
        world_points[t] = wp
        # Color: plane gray, blob red — so we can confirm color follows the split.
        colors[t, :, :, :] = 100
        colors[t, -1, :, 0] = 255
        colors[t, -1, :, 1] = 0
        colors[t, -1, :, 2] = 0
        conf[t] = rng.random((H, W)).astype(np.float32)

    depth = np.ones((S, H, W), dtype=np.float32)
    depth_conf = np.ones((S, H, W), dtype=np.float32)
    poses = np.tile(np.array([0, 0, 0, 1, 0, 0, 0], np.float32), (S, 1))
    intr = np.tile(np.array([1.0, 1.0, 0.5, 0.5], np.float32), (S, 1))
    camera = CameraTrack(poses=poses, intrinsics=intr)

    geo = GeometryResult(
        world_points=world_points,
        world_points_conf=conf,
        depth=depth,
        depth_conf=depth_conf,
        camera=camera,
    )
    # attach colors so the split carries them (the real combo does this).
    geo.colors = colors  # type: ignore[attr-defined]

    # Tracks: M=2. Track 0 follows the moving cluster center (big displacement);
    # track 1 sits on a fixed plane point (no motion).
    M = 2
    tpos = np.zeros((M, S, 3), dtype=np.float32)
    for t in range(S):
        tpos[0, t] = centers[t]            # moving
        tpos[1, t] = np.array([2.0, 2.0, 0.0], dtype=np.float32)  # stationary plane pt
    tvis = np.ones((M, S), dtype=bool)
    tcol = np.array([[255, 0, 0], [100, 100, 100]], dtype=np.uint8)
    tr = TrackResult(positions=tpos, visibility=tvis, colors=tcol)
    return geo, tr, S, centers


def test_assemble_splits_moving_into_dynamic_rest_into_static() -> None:
    """The moving cluster lands in dynamic_positions; the static plane in static_positions."""
    geo, tr, S, centers = _synthetic_geo_and_tracks()
    request = ReconstructionRequest(video_path="/tmp/x.mp4", max_frames=24, target_fps=12.0)

    scene = assemble_scene4d(geo, tr, request, motion_thresh=0.95)

    assert isinstance(scene, Scene4D)
    assert scene.frame_count == S
    assert len(scene.dynamic_positions) == S
    assert len(scene.dynamic_colors) == S

    # Every frame's moving cluster (8 points near its center) is in dynamic_positions.
    for t in range(S):
        dyn = scene.dynamic_positions[t]
        assert dyn.shape[0] > 0, f"frame {t} has no dynamic points"
        # Dynamic points are near the moving center, not on the z=0 plane.
        d_to_center = np.linalg.norm(dyn - centers[t][None, :], axis=1)
        assert np.all(d_to_center < 1.0), (t, d_to_center.max())
        assert np.all(dyn[:, 2] > 1.0)  # well above the z=0 plane
        # Their colors are red (the blob color), aligned to positions.
        col = scene.dynamic_colors[t]
        assert col.shape[0] == dyn.shape[0]
        assert np.all(col[:, 0] >= 200) and np.all(col[:, 1] <= 50)

    # Static points exist, all on (or near) the z=0 plane, NONE in the moving region.
    assert scene.static_positions.shape[0] > 0
    assert np.all(np.abs(scene.static_positions[:, 2]) < 1.0)
    for t in range(S):
        d = np.linalg.norm(scene.static_positions - centers[t][None, :], axis=1)
        assert np.all(d > 1.0), f"a static point is in the moving cluster at frame {t}"

    # static_conf present (u8) and aligned.
    assert scene.static_conf is not None
    assert scene.static_conf.dtype == np.uint8
    assert scene.static_conf.shape[0] == scene.static_positions.shape[0]

    # tracks + cameras populated.
    assert scene.tracks is not None
    assert scene.tracks.positions.shape == (2, S, 3)
    assert scene.cameras is not None
    assert scene.cameras.poses.shape == (S, 7)

    # AABB spans ALL positions (static ∪ dynamic ∪ tracks).
    all_pts = [scene.static_positions]
    all_pts += [p for p in scene.dynamic_positions if p.size]
    all_pts.append(scene.tracks.positions.reshape(-1, 3))
    allp = np.concatenate(all_pts, axis=0)
    assert np.all(scene.aabb_min <= allp.min(axis=0) + 1e-5)
    assert np.all(scene.aabb_max >= allp.max(axis=0) - 1e-5)


def test_assemble_returns_raw_scene_does_not_cap() -> None:
    """assemble returns a RAW Scene4D — enforce_caps (called separately) is the capper.

    Build an over-cap geometry (a frame with > 20k dynamic points). assemble must NOT
    cap it (so a frame exceeds the per-frame dynamic cap); enforce_caps THEN brings it
    under cap — proving the two stages are distinct and assemble is raw.
    """
    S = 2
    n_dyn = 25_000  # > 20k dynamic cap
    rng = np.random.default_rng(5)

    # All-moving cluster: every VGGT point sits near a moving track sample, so the
    # split puts (essentially) all of them in dynamic_positions, exceeding the cap.
    centers = [np.array([0.0, 0.0, 0.0], np.float32), np.array([50.0, 0.0, 0.0], np.float32)]
    world_points = np.zeros((S, 1, n_dyn, 3), dtype=np.float32)
    colors = np.zeros((S, 1, n_dyn, 3), dtype=np.uint8)
    conf = rng.random((S, 1, n_dyn)).astype(np.float32)
    for t in range(S):
        world_points[t, 0] = centers[t][None, :] + (rng.random((n_dyn, 3)).astype(np.float32) - 0.5) * 0.2
        colors[t, 0] = 200
    depth = np.ones((S, 1, n_dyn), np.float32)
    geo = GeometryResult(
        world_points=world_points,
        world_points_conf=conf,
        depth=depth,
        depth_conf=depth,
        camera=CameraTrack(
            poses=np.tile(np.array([0, 0, 0, 1, 0, 0, 0], np.float32), (S, 1)),
            intrinsics=np.tile(np.array([1.0, 1.0, 0.5, 0.5], np.float32), (S, 1)),
        ),
    )
    geo.colors = colors  # type: ignore[attr-defined]

    # One moving track following the centers (drives the dynamic classification).
    tpos = np.stack(centers, axis=0)[None, :, :].astype(np.float32)  # (1,S,3)
    tr = TrackResult(
        positions=tpos,
        visibility=np.ones((1, S), bool),
        colors=np.array([[200, 200, 200]], np.uint8),
    )
    request = ReconstructionRequest(video_path="/tmp/x.mp4", max_frames=24, target_fps=12.0)

    raw = assemble_scene4d(geo, tr, request)

    # RAW: at least one frame is OVER the per-frame dynamic cap (assemble did not cap).
    max_dyn = max(p.shape[0] for p in raw.dynamic_positions)
    assert max_dyn > 20_000, max_dyn

    # enforce_caps (the SEPARATE core step) brings every frame under the cap.
    capped = enforce_caps(raw)
    for p in capped.dynamic_positions:
        assert p.shape[0] <= 20_000
    assert max(p.shape[0] for p in capped.dynamic_positions) == 20_000


def test_assemble_fallback_to_sparse_when_vggt_empty() -> None:
    """If VGGT per-frame maps are unusable, dynamic = sparse moving track points only."""
    S = 3
    # All-NaN VGGT world points -> unusable -> fallback path.
    world_points = np.full((S, 2, 2, 3), np.nan, dtype=np.float32)
    conf = np.zeros((S, 2, 2), dtype=np.float32)
    depth = np.ones((S, 2, 2), np.float32)
    geo = GeometryResult(
        world_points=world_points,
        world_points_conf=conf,
        depth=depth,
        depth_conf=depth,
        camera=CameraTrack(
            poses=np.tile(np.array([0, 0, 0, 1, 0, 0, 0], np.float32), (S, 1)),
            intrinsics=np.tile(np.array([1.0, 1.0, 0.5, 0.5], np.float32), (S, 1)),
        ),
    )
    # A clearly-moving track.
    tpos = np.zeros((1, S, 3), dtype=np.float32)
    for t in range(S):
        tpos[0, t] = np.array([10.0 * t, 0.0, 0.0], dtype=np.float32)
    tr = TrackResult(
        positions=tpos,
        visibility=np.ones((1, S), bool),
        colors=np.array([[1, 2, 3]], np.uint8),
    )
    request = ReconstructionRequest(video_path="/tmp/x.mp4", max_frames=24, target_fps=12.0)
    scene = assemble_scene4d(geo, tr, request)

    # Sparse dynamic: at least one frame has the moving track point.
    total_dyn = sum(p.shape[0] for p in scene.dynamic_positions)
    assert total_dyn >= 1
    # Static is empty (no usable VGGT points).
    assert scene.static_positions.shape[0] == 0
    assert scene.tracks is not None


# ---------------------------------------------------------------------------
# (d) decode_and_subsample — cv2-gated; skips cleanly without opencv
# ---------------------------------------------------------------------------

def test_decode_and_subsample_shape_and_cap() -> None:
    """decode_and_subsample -> [S,3,~518,W] uint8 RGB, S <= max_frames (cv2-gated)."""
    pytest.importorskip("cv2")
    from tests.pipeline._gen_tiny_mp4 import ensure_tiny_mp4
    from app.pipeline.decode import decode_and_subsample

    fixture = Path(__file__).resolve().parent.parent / "fixtures" / "tiny_real.mp4"
    path = ensure_tiny_mp4(fixture)
    if path is None:
        pytest.skip("cv2 cannot write/read mp4 in this environment")

    max_frames = 6
    request = ReconstructionRequest(
        video_path=str(path), max_frames=max_frames, target_fps=12.0
    )
    frames = decode_and_subsample(request)

    assert frames.dtype == np.uint8
    assert frames.ndim == 4
    S, C, Hh, Ww = frames.shape
    assert C == 3
