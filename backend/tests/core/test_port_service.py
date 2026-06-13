"""T-120 / T-121 / T-122 — port, service, and FakeAdapter contract (spec/10 §1.2).

These exercise the pure core with a deterministic, torch-free `FakeAdapter` — no
weights, no ML deps. The encoder is deliberately NOT imported here (that chains into
T-100, the wire-format suite).
"""

from __future__ import annotations

import numpy as np

from app.core.domain.models import ReconstructionRequest, Scene4D, Tracks
from app.core.ports.reconstruction_port import ReconstructionPort
from app.core.services.reconstruction_service import ReconstructionService

from tests.fakes.fake_adapter import FakeAdapter

# MV4D caps (spec/05 §4) — assert the service's output is within them.
_MAX_STATIC = 150_000
_MAX_DYNAMIC_PER_FRAME = 20_000
_MAX_TRACKS = 4_096
_MAX_FRAMES = 64


def _request() -> ReconstructionRequest:
    return ReconstructionRequest(video_path="/tmp/fake.mp4", max_frames=24, target_fps=12.0)


def test_service_delegates_to_port() -> None:
    """T-120 — the service runs the adapter then applies core post-processing."""
    scene = ReconstructionService(FakeAdapter()).run(_request())

    assert isinstance(scene, Scene4D)

    # Post-processing applied: still has static + tracks, within caps.
    assert scene.static_positions.shape[0] > 0
    assert scene.tracks is not None
    assert scene.tracks.positions.shape[0] >= 2

    # Within MV4D caps (spec/05 §4).
    assert scene.frame_count <= _MAX_FRAMES
    assert scene.static_positions.shape[0] <= _MAX_STATIC
    assert all(p.shape[0] <= _MAX_DYNAMIC_PER_FRAME for p in scene.dynamic_positions)
    assert scene.tracks.positions.shape[0] <= _MAX_TRACKS

    # Provenance stamped from the adapter's info.
    assert scene.adapter_id == "fake"
    assert scene.weights_license == "cc-by-nc-4.0"

    # Track positions are SMOOTHED — do NOT assert exact positions here.
    assert scene.tracks.positions.dtype == np.float32


def test_fake_adapter_satisfies_port() -> None:
    """T-121 — FakeAdapter is a concrete ReconstructionPort returning a valid Scene4D."""
    adapter = FakeAdapter()
    assert isinstance(adapter, ReconstructionPort)

    scene = adapter.reconstruct(_request())
    assert isinstance(scene, Scene4D)

    # Structurally valid per spec/05 §5.1 (shapes / dtypes).
    T = scene.frame_count
    assert T == 3
    assert isinstance(scene.fps, float)

    assert scene.aabb_min.shape == (3,) and scene.aabb_min.dtype == np.float32
    assert scene.aabb_max.shape == (3,) and scene.aabb_max.dtype == np.float32

    # Static section.
    n_s = scene.static_positions.shape[0]
    assert scene.static_positions.shape == (n_s, 3)
    assert scene.static_positions.dtype == np.float32
    assert scene.static_colors.shape == (n_s, 3)
    assert scene.static_colors.dtype == np.uint8
    assert scene.static_conf is not None
    assert scene.static_conf.shape == (n_s,)
    assert scene.static_conf.dtype == np.uint8

    # Dynamic section: list of length T, one (incl. an empty) per frame.
    assert len(scene.dynamic_positions) == T
    assert len(scene.dynamic_colors) == T
    assert any(p.shape[0] == 0 for p in scene.dynamic_positions)  # an empty frame
    for pos, col in zip(scene.dynamic_positions, scene.dynamic_colors):
        assert pos.ndim == 2 and pos.shape[1] == 3 and pos.dtype == np.float32
        assert col.ndim == 2 and col.shape[1] == 3 and col.dtype == np.uint8
        assert pos.shape[0] == col.shape[0]

    # Tracks section: (M, T, 3) f32 positions, (M, T) bool visibility, (M, 3) u8 colors.
    assert isinstance(scene.tracks, Tracks)
