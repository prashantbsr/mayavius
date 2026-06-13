"""MV4D v1 encoder + reference decoder tests (W0.T2).

The wire bytes are owned by spec/05-data-contract.md §3; these tests pin the
encoder/decoder against that contract — round-trip exactness, quantization
tolerance, header/directory invariants, the all-section AABB, optional-section
omission, the empty dynamic frame, and the version constant.

T-104 (caps enforcement) is W1's `enforce_caps` test and is deliberately NOT
here (the encoder assumes an already-capped scene, spec/05 §4).
"""

from __future__ import annotations

import struct

import numpy as np
import pytest

from app.core.domain.models import CameraTrack, Scene4D, Tracks
from app.wire.decoder import decode
from app.wire.encoder import MV4D_VERSION, encode_reconstruction

_HEADER_BYTES = 24
_AABB_BYTES = 24
_DIR_ENTRY_BYTES = 16
_DIR_OFFSET = _HEADER_BYTES + _AABB_BYTES

_KIND_STATIC = 1
_KIND_DYNAMIC = 2
_KIND_TRACKS = 3
_KIND_CAMERAS = 4

_FLAG_HAS_STATIC = 1 << 0
_FLAG_HAS_DYNAMIC = 1 << 1
_FLAG_HAS_TRACKS = 1 << 2
_FLAG_HAS_CAMERAS = 1 << 3
_FLAG_HAS_STATIC_CONF = 1 << 4
_FLAG_HAS_TRACK_COLOR = 1 << 5


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _parse_header(buf: bytes):
    (
        magic,
        version,
        flags,
        pos_bits,
        section_count,
        frame_count,
        reserved0,
        fps,
        reserved1,
        reserved2,
    ) = struct.unpack_from("<4sBBBBHHfII", buf, 0)
    return {
        "magic": magic,
        "version": version,
        "flags": flags,
        "pos_bits": pos_bits,
        "section_count": section_count,
        "frame_count": frame_count,
        "reserved0": reserved0,
        "fps": fps,
        "reserved1": reserved1,
        "reserved2": reserved2,
    }


def _parse_directory(buf: bytes, section_count: int):
    entries = []
    for i in range(section_count):
        kind, off, length, count = struct.unpack_from(
            "<IIII", buf, _DIR_OFFSET + i * _DIR_ENTRY_BYTES
        )
        entries.append({"kind": kind, "off": off, "length": length, "count": count})
    return entries


def _build_representative_scene(seed: int = 7) -> Scene4D:
    """static + dynamic (incl an empty frame) + tracks + cameras, T=4."""
    rng = np.random.default_rng(seed)
    T = 4
    aabb_min = np.array([-1.0, -2.0, 0.0], dtype=np.float32)
    aabb_max = np.array([3.0, 2.0, 5.0], dtype=np.float32)

    def rand_pts(n):
        return (rng.random((n, 3), dtype=np.float32) * (aabb_max - aabb_min) + aabb_min).astype(
            np.float32
        )

    static_positions = rand_pts(40)
    static_colors = rng.integers(0, 256, size=(40, 3), dtype=np.uint8)
    static_conf = rng.integers(0, 256, size=(40,), dtype=np.uint8)

    # one empty dynamic frame (index 2)
    counts = [5, 3, 0, 7]
    dynamic_positions = [rand_pts(c) for c in counts]
    dynamic_colors = [rng.integers(0, 256, size=(c, 3), dtype=np.uint8) for c in counts]

    M = 6
    track_positions = rand_pts(M * T).reshape(M, T, 3)
    visibility = rng.integers(0, 2, size=(M, T), dtype=np.uint8).astype(bool)
    track_colors = rng.integers(0, 256, size=(M, 3), dtype=np.uint8)
    tracks = Tracks(positions=track_positions, visibility=visibility, colors=track_colors)

    poses = rng.random((T, 7), dtype=np.float32).astype(np.float32)
    intrinsics = rng.random((T, 4), dtype=np.float32).astype(np.float32)
    cameras = CameraTrack(poses=poses, intrinsics=intrinsics)

    return Scene4D(
        frame_count=T,
        fps=24.0,
        aabb_min=aabb_min,
        aabb_max=aabb_max,
        static_positions=static_positions,
        static_colors=static_colors,
        static_conf=static_conf,
        dynamic_positions=dynamic_positions,
        dynamic_colors=dynamic_colors,
        tracks=tracks,
        cameras=cameras,
    )


# --------------------------------------------------------------------------- #
# T-100 — full round-trip
# --------------------------------------------------------------------------- #
def test_t100_roundtrip_all_sections():
    scene = _build_representative_scene()
    buf = encode_reconstruction(scene)
    out = decode(buf)

    # counts exact
    assert out.frame_count == scene.frame_count
    assert out.fps == pytest.approx(scene.fps)
    assert out.static_positions.shape == scene.static_positions.shape
    assert len(out.dynamic_positions) == scene.frame_count
    for t in range(scene.frame_count):
        assert out.dynamic_positions[t].shape == scene.dynamic_positions[t].shape
    assert out.tracks is not None
    assert out.tracks.positions.shape == scene.tracks.positions.shape
    assert out.cameras is not None

    # colors u8 exact
    np.testing.assert_array_equal(out.static_colors, scene.static_colors)
    for t in range(scene.frame_count):
        np.testing.assert_array_equal(out.dynamic_colors[t], scene.dynamic_colors[t])
    np.testing.assert_array_equal(out.tracks.colors, scene.tracks.colors)
    np.testing.assert_array_equal(out.static_conf, scene.static_conf)
    assert out.static_colors.dtype == np.uint8
    assert out.tracks.colors.dtype == np.uint8

    # visibility bitmask exact
    np.testing.assert_array_equal(out.tracks.visibility, scene.tracks.visibility)
    assert out.tracks.visibility.dtype == bool

    # cameras within f32
    np.testing.assert_allclose(out.cameras.poses, scene.cameras.poses, rtol=0, atol=1e-6)
    np.testing.assert_allclose(out.cameras.intrinsics, scene.cameras.intrinsics, rtol=0, atol=1e-6)

    # positions within quantization tolerance (per-axis, see T-101)
    span = (out.aabb_max - out.aabb_min).astype(np.float32)
    tol = span / 65535.0
    np.testing.assert_array_less(
        np.abs(out.static_positions - scene.static_positions).max(axis=0), tol + 1e-6
    )
    for t in range(scene.frame_count):
        if scene.dynamic_positions[t].shape[0]:
            err = np.abs(out.dynamic_positions[t] - scene.dynamic_positions[t]).max(axis=0)
            np.testing.assert_array_less(err, tol + 1e-6)
    track_err = np.abs(out.tracks.positions - scene.tracks.positions).reshape(-1, 3).max(axis=0)
    np.testing.assert_array_less(track_err, tol + 1e-6)


# --------------------------------------------------------------------------- #
# T-101 — quantization tolerance + degenerate axis
# --------------------------------------------------------------------------- #
def test_t101_quantization_tolerance():
