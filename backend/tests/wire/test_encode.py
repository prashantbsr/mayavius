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
