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
