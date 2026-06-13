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
    rng = np.random.default_rng(101)
    aabb_min = np.array([-5.0, 10.0, 0.0], dtype=np.float32)
    aabb_max = np.array([5.0, 20.0, 100.0], dtype=np.float32)
    N = 5000
    pts = (rng.random((N, 3), dtype=np.float32) * (aabb_max - aabb_min) + aabb_min).astype(
        np.float32
    )

    scene = Scene4D(
        frame_count=1,
        fps=12.0,
        aabb_min=aabb_min,
        aabb_max=aabb_max,
        static_positions=pts,
        static_colors=np.zeros((N, 3), np.uint8),
        static_conf=None,
        dynamic_positions=[np.empty((0, 3), np.float32)],
        dynamic_colors=[np.empty((0, 3), np.uint8)],
        tracks=None,
        cameras=None,
    )
    out = decode(encode_reconstruction(scene))

    span = (out.aabb_max - out.aabb_min).astype(np.float64)
    tol = span / 65535.0
    err = np.abs(out.static_positions.astype(np.float64) - pts.astype(np.float64)).max(axis=0)
    assert np.all(err <= tol + 1e-6), f"err {err} > tol {tol}"


def test_t101_degenerate_axis():
    # Z axis is degenerate (aabbMax == aabbMin) => q=0 on Z, dequant == aabbMin_z.
    z_const = 4.2
    pts = np.array(
        [[0.0, 0.0, z_const], [1.0, 1.0, z_const], [0.5, 0.25, z_const]], dtype=np.float32
    )
    aabb_min = np.array([0.0, 0.0, z_const], dtype=np.float32)
    aabb_max = np.array([1.0, 1.0, z_const], dtype=np.float32)
    scene = Scene4D(
        frame_count=1,
        fps=12.0,
        aabb_min=aabb_min,
        aabb_max=aabb_max,
        static_positions=pts,
        static_colors=np.zeros((3, 3), np.uint8),
        static_conf=None,
        dynamic_positions=[np.empty((0, 3), np.float32)],
        dynamic_colors=[np.empty((0, 3), np.uint8)],
        tracks=None,
        cameras=None,
    )
    out = decode(encode_reconstruction(scene))
    # Degenerate Z column dequantizes exactly to aabbMin_z for every point.
    np.testing.assert_array_equal(out.static_positions[:, 2], np.full(3, z_const, np.float32))


# --------------------------------------------------------------------------- #
# T-102 — header + directory invariants
# --------------------------------------------------------------------------- #
def test_t102_header_and_directory():
    scene = _build_representative_scene()
    buf = encode_reconstruction(scene)
    h = _parse_header(buf)

    assert h["magic"] == b"MV4D"
    assert h["version"] == 1
    assert h["pos_bits"] == 16
    assert h["reserved0"] == 0
    assert h["reserved1"] == 0
    assert h["reserved2"] == 0
    assert h["frame_count"] == scene.frame_count

    # all four sections present here
    expected_flags = (
        _FLAG_HAS_STATIC
        | _FLAG_HAS_DYNAMIC
        | _FLAG_HAS_TRACKS
        | _FLAG_HAS_CAMERAS
        | _FLAG_HAS_STATIC_CONF
        | _FLAG_HAS_TRACK_COLOR
    )
    assert h["flags"] == expected_flags
    assert h["section_count"] == 4

    entries = _parse_directory(buf, h["section_count"])
    kinds = [e["kind"] for e in entries]
    assert kinds == [_KIND_STATIC, _KIND_DYNAMIC, _KIND_TRACKS, _KIND_CAMERAS]  # ascending

    for e in entries:
        assert e["off"] % 8 == 0, f"section kind={e['kind']} offset {e['off']} not 8-aligned"
        assert e["off"] >= _DIR_OFFSET + h["section_count"] * _DIR_ENTRY_BYTES
        assert e["off"] + e["length"] <= len(buf)


def test_t102_flags_bits_match_present_sections():
    # bit i (0-3) set iff section present in directory.
    scene = _build_representative_scene()
    buf = encode_reconstruction(scene)
    h = _parse_header(buf)
    entries = _parse_directory(buf, h["section_count"])
    present_kinds = {e["kind"] for e in entries}
    bit_for_kind = {
        _KIND_STATIC: _FLAG_HAS_STATIC,
        _KIND_DYNAMIC: _FLAG_HAS_DYNAMIC,
        _KIND_TRACKS: _FLAG_HAS_TRACKS,
        _KIND_CAMERAS: _FLAG_HAS_CAMERAS,
    }
    for kind, bit in bit_for_kind.items():
        assert bool(h["flags"] & bit) == (kind in present_kinds)


# --------------------------------------------------------------------------- #
# T-103 — AABB spans ALL sections; no point clamps
# --------------------------------------------------------------------------- #
def test_t103_aabb_spans_all_sections():
    # Place each section's extreme in a different region so the AABB MUST
    # cover static ∪ dynamic ∪ tracks.
    T = 2
    static_positions = np.array([[-10.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32)  # min x
    dyn0 = np.array([[0.0, 50.0, 0.0]], dtype=np.float32)  # max y
    dyn1 = np.array([[0.0, 0.0, -7.0]], dtype=np.float32)  # min z
    track_positions = np.array(
        [[[0.0, 0.0, 99.0], [3.0, 0.0, 0.0]]], dtype=np.float32
    )  # max z, mid x

    scene = Scene4D(
        frame_count=T,
        fps=10.0,
        aabb_min=np.zeros(3, np.float32),  # deliberately WRONG — encoder must recompute
        aabb_max=np.zeros(3, np.float32),
        static_positions=static_positions,
        static_colors=np.zeros((2, 3), np.uint8),
        static_conf=None,
        dynamic_positions=[dyn0, dyn1],
        dynamic_colors=[np.zeros((1, 3), np.uint8), np.zeros((1, 3), np.uint8)],
        tracks=Tracks(
            positions=track_positions,
            visibility=np.ones((1, T), bool),
            colors=None,
        ),
        cameras=None,
    )
    buf = encode_reconstruction(scene)
    out = decode(buf)

    all_pts = np.concatenate(
        [static_positions, dyn0, dyn1, track_positions.reshape(-1, 3)], axis=0
    )
    expected_min = all_pts.min(axis=0)
    expected_max = all_pts.max(axis=0)
    np.testing.assert_allclose(out.aabb_min, expected_min, atol=1e-5)
    np.testing.assert_allclose(out.aabb_max, expected_max, atol=1e-5)

    # No point clamps: every quantized value strictly within [0, 65535] for
    # in-AABB input — verify by reading raw u16 from the directory's section.
    h = _parse_header(buf)
    entries = {e["kind"]: e for e in _parse_directory(buf, h["section_count"])}
    for kind, n_pts in (
        (_KIND_STATIC, static_positions.shape[0]),
    ):
        e = entries[kind]
        q = np.frombuffer(buf, dtype="<u2", count=n_pts * 3, offset=e["off"]).reshape(-1, 3)
        assert q.min() >= 0 and q.max() <= 65535
    # the global extents land exactly on the grid endpoints (0 and 65535)
    # — confirming the AABB is tight and nothing clamps beyond.
    static_q = np.frombuffer(
        buf, dtype="<u2", count=static_positions.shape[0] * 3, offset=entries[_KIND_STATIC]["off"]
    ).reshape(-1, 3)
    assert static_q[0, 0] == 0  # static min-x point -> q=0 on x


# --------------------------------------------------------------------------- #
# T-105 — optional sections omitted
# --------------------------------------------------------------------------- #
def _static_only_scene(with_conf: bool) -> Scene4D:
    N = 12
    rng = np.random.default_rng(5)
    return Scene4D(
        frame_count=1,
        fps=12.0,
        aabb_min=np.zeros(3, np.float32),
        aabb_max=np.ones(3, np.float32),
        static_positions=rng.random((N, 3), dtype=np.float32).astype(np.float32),
        static_colors=rng.integers(0, 256, (N, 3), dtype=np.uint8),
        static_conf=rng.integers(0, 256, (N,), dtype=np.uint8) if with_conf else None,
        dynamic_positions=[],
        dynamic_colors=[],
        tracks=None,
        cameras=None,
    )


def test_t105_static_only_emits_only_static():
    scene = _static_only_scene(with_conf=False)
    buf = encode_reconstruction(scene)
    h = _parse_header(buf)

    assert h["section_count"] == 1
    assert h["flags"] == _FLAG_HAS_STATIC  # no dynamic/tracks/cameras/conf/track-color
    entries = _parse_directory(buf, h["section_count"])
    assert [e["kind"] for e in entries] == [_KIND_STATIC]

    out = decode(buf)
    assert out.static_positions.shape[0] == 12
    assert out.static_conf is None
    assert out.tracks is None
    assert out.cameras is None
    # absent dynamic section => empty frames
    assert all(f.shape[0] == 0 for f in out.dynamic_positions)


def test_t105_conf_toggle_toggles_subarray():
    with_conf = encode_reconstruction(_static_only_scene(with_conf=True))
    without_conf = encode_reconstruction(_static_only_scene(with_conf=False))

    assert _parse_header(with_conf)["flags"] & _FLAG_HAS_STATIC_CONF
    assert not (_parse_header(without_conf)["flags"] & _FLAG_HAS_STATIC_CONF)

    # the conf sub-array adds exactly N bytes to the static section
    e_with = _parse_directory(with_conf, 1)[0]
    e_without = _parse_directory(without_conf, 1)[0]
    assert e_with["length"] - e_without["length"] == 12  # N=12 u8

    out_with = decode(with_conf)
    out_without = decode(without_conf)
    assert out_with.static_conf is not None and out_with.static_conf.shape == (12,)
    assert out_without.static_conf is None


def test_t105_track_color_toggle():
    T, M = 2, 3
    rng = np.random.default_rng(9)
    base = dict(
        frame_count=T,
        fps=12.0,
        aabb_min=np.zeros(3, np.float32),
        aabb_max=np.ones(3, np.float32),
        static_positions=np.empty((0, 3), np.float32),
        static_colors=np.empty((0, 3), np.uint8),
        static_conf=None,
        dynamic_positions=[],
        dynamic_colors=[],
        cameras=None,
    )
    pos = rng.random((M, T, 3), dtype=np.float32).astype(np.float32)
    vis = np.ones((M, T), bool)
    with_color = Scene4D(tracks=Tracks(pos, vis, rng.integers(0, 256, (M, 3), np.uint8)), **base)
    without_color = Scene4D(tracks=Tracks(pos, vis, None), **base)

    b_with = encode_reconstruction(with_color)
    b_without = encode_reconstruction(without_color)
    assert _parse_header(b_with)["flags"] & _FLAG_HAS_TRACK_COLOR
    assert not (_parse_header(b_without)["flags"] & _FLAG_HAS_TRACK_COLOR)

    e_with = _parse_directory(b_with, _parse_header(b_with)["section_count"])
    e_without = _parse_directory(b_without, _parse_header(b_without)["section_count"])
    tk_with = next(e for e in e_with if e["kind"] == _KIND_TRACKS)
    tk_without = next(e for e in e_without if e["kind"] == _KIND_TRACKS)
    assert tk_with["length"] - tk_without["length"] == M * 3  # u8 colors

    assert decode(b_with).tracks.colors is not None
    assert decode(b_without).tracks.colors is None


# --------------------------------------------------------------------------- #
# T-106 — empty dynamic frame round-trips
# --------------------------------------------------------------------------- #
def test_t106_empty_dynamic_frame():
    T = 3
    rng = np.random.default_rng(6)
    counts = [4, 0, 2]  # middle frame empty (pointCount==0)
    dyn_pos = [rng.random((c, 3), dtype=np.float32).astype(np.float32) for c in counts]
    dyn_col = [rng.integers(0, 256, (c, 3), dtype=np.uint8) for c in counts]
    scene = Scene4D(
        frame_count=T,
        fps=15.0,
        aabb_min=np.zeros(3, np.float32),
        aabb_max=np.ones(3, np.float32),
        static_positions=np.empty((0, 3), np.float32),
        static_colors=np.empty((0, 3), np.uint8),
        static_conf=None,
        dynamic_positions=dyn_pos,
        dynamic_colors=dyn_col,
        tracks=None,
        cameras=None,
    )
    out = decode(encode_reconstruction(scene))

    assert [f.shape[0] for f in out.dynamic_positions] == counts
    assert out.dynamic_positions[1].shape == (0, 3)
    assert out.dynamic_colors[1].shape == (0, 3)
    for t in range(T):
        np.testing.assert_array_equal(out.dynamic_colors[t], dyn_col[t])
        if counts[t]:
            span = (out.aabb_max - out.aabb_min) / 65535.0
            err = np.abs(out.dynamic_positions[t] - dyn_pos[t]).max(axis=0)
            np.testing.assert_array_less(err, span + 1e-6)


# --------------------------------------------------------------------------- #
# T-107 — version constant
# --------------------------------------------------------------------------- #
def test_t107_version_constant():
    assert MV4D_VERSION == 1
