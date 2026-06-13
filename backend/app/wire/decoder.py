"""Python reference decoder for the MV4D v1 wire format — the exact inverse of
``app/wire/encoder.py``.

The byte layout is owned by spec/05-data-contract.md §3 and is NEVER redefined
here. This decoder mirrors §3 exactly: it validates the header, reads the AABB +
section directory, and reconstructs a ``Scene4D`` with **dequantized float32**
positions. It is the inverse used by the wire round-trip tests (T-100) and later
by ``FixtureAdapter`` (spec/06 §4.6) to load a committed MV4D blob back into a
``Scene4D``.

Unknown section kinds are skipped (forward compatibility, spec/05 §1). numpy
only — NO torch, NO fastapi.
"""

from __future__ import annotations

import struct

import numpy as np

from app.core.domain.models import CameraTrack, Scene4D, Tracks
from app.wire.encoder import (
    MV4D_VERSION,
    _AABB_BYTES,
    _DIR_ENTRY_BYTES,
    _FLAG_HAS_STATIC_CONF,
    _FLAG_HAS_TRACK_COLOR,
    _HEADER_BYTES,
    _KIND_CAMERAS,
    _KIND_DYNAMIC,
    _KIND_STATIC,
    _KIND_TRACKS,
    _MAGIC,
    _POS_BITS,
    _QMAX,
)

__all__ = ["decode", "MV4D_VERSION"]


def _dequantize(q: np.ndarray, aabb_min: np.ndarray, aabb_max: np.ndarray) -> np.ndarray:
    """Inverse of the encoder quantization (spec/05 §2).

    ``p = aabbMin + q/65535 * (aabbMax - aabbMin)`` in float32. ``q`` is (N, 3)
    uint16; returns (N, 3) float32. Degenerate axes (max == min) yield ``aabbMin``
    because every ``q`` there is 0.
    """
    span = (aabb_max - aabb_min).astype(np.float32)
    qf = q.astype(np.float32)
    return (aabb_min + qf / np.float32(_QMAX) * span).astype(np.float32)


def decode(buffer: bytes) -> Scene4D:
    """Decode an MV4D v1 buffer back into a ``Scene4D`` (spec/05 §3).

    Validates ``magic == "MV4D"``, ``version == 1``, ``posBits == 16`` (raises
    ``ValueError`` otherwise). Uses the section directory offsets (order-agnostic)
    and skips unknown kinds. Optional sub-arrays (static conf, track colors) are
    gated by flags bits 4/5.
    """
    buf = bytes(buffer)
    if len(buf) < _HEADER_BYTES + _AABB_BYTES:
        raise ValueError("MV4D buffer too short for header + AABB")

    # ---- header (spec/05 §3.1) ----
    (
        magic,
        version,
        flags,
        pos_bits,
        section_count,
        frame_count,
        _reserved0,
        fps,
        _reserved1,
        _reserved2,
    ) = struct.unpack_from("<4sBBBBHHfII", buf, 0)

    if magic != _MAGIC:
        raise ValueError(f"bad MV4D magic: {magic!r} (expected {_MAGIC!r})")
    if version != MV4D_VERSION:
        raise ValueError(f"unsupported MV4D version: {version} (expected {MV4D_VERSION})")
    if pos_bits != _POS_BITS:
        raise ValueError(f"unsupported posBits: {pos_bits} (expected {_POS_BITS})")

    frame_count = int(frame_count)

    # ---- AABB block (spec/05 §3.2) ----
    aabb = struct.unpack_from("<6f", buf, _HEADER_BYTES)
    aabb_min = np.array(aabb[0:3], dtype=np.float32)
    aabb_max = np.array(aabb[3:6], dtype=np.float32)

    # ---- section directory (spec/05 §3.3) ----
    dir_offset = _HEADER_BYTES + _AABB_BYTES
    entries: list[tuple[int, int, int, int]] = []
    for i in range(section_count):
        kind, byte_offset, byte_length, count = struct.unpack_from(
            "<IIII", buf, dir_offset + i * _DIR_ENTRY_BYTES
        )
        if byte_offset + byte_length > len(buf):
            raise ValueError(
                f"section kind={kind} exceeds buffer: "
                f"offset={byte_offset} length={byte_length} buflen={len(buf)}"
            )
        entries.append((int(kind), int(byte_offset), int(byte_length), int(count)))

    has_static_conf = bool(flags & _FLAG_HAS_STATIC_CONF)
    has_track_color = bool(flags & _FLAG_HAS_TRACK_COLOR)

    static_positions = np.empty((0, 3), dtype=np.float32)
    static_colors = np.empty((0, 3), dtype=np.uint8)
    static_conf: np.ndarray | None = None
    dynamic_positions: list[np.ndarray] = [np.empty((0, 3), np.float32) for _ in range(frame_count)]
    dynamic_colors: list[np.ndarray] = [np.empty((0, 3), np.uint8) for _ in range(frame_count)]
    tracks: Tracks | None = None
    cameras: CameraTrack | None = None

    for kind, off, length, count in entries:
        if kind == _KIND_STATIC:
            n = count
