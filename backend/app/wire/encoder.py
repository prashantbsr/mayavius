"""Binary wire-format encoder (backend side) — MV4D v1.

JSON is forbidden for point payloads (handover §4.5) — it is the difference
between a ~2s and a ~40s load and it gates shareable result links. This encoder
and the frontend decoder (frontend/src/lib/wire/decoder.ts) are two
implementations of ONE format whose single source of truth is
spec/05-data-contract.md — header, version byte, dtypes, quantization ranges,
track indexing, visibility encoding. They MUST stay byte-for-byte compatible.

The byte layout below is owned by spec/05-data-contract.md §3 and is NEVER
redefined here — this module merely renders it. The Python reference decoder
(`app/wire/decoder.py`) is the exact inverse.

numpy only — NO torch, NO fastapi (this sits adjacent to the pure core).
"""

from __future__ import annotations

import logging
import struct

import numpy as np

from app.core.domain.models import Scene4D

logger = logging.getLogger(__name__)

# Exported version constant — the decoder mirrors this; tests assert they match
# (spec/05 §7).
MV4D_VERSION = 1

_MAGIC = b"MV4D"
_POS_BITS = 16

# flags bits (spec/05 §3.1)
_FLAG_HAS_STATIC = 1 << 0
_FLAG_HAS_DYNAMIC = 1 << 1
_FLAG_HAS_TRACKS = 1 << 2
_FLAG_HAS_CAMERAS = 1 << 3
_FLAG_HAS_STATIC_CONF = 1 << 4
_FLAG_HAS_TRACK_COLOR = 1 << 5

# section kinds (spec/05 §3.3)
_KIND_STATIC = 1
_KIND_DYNAMIC = 2
_KIND_TRACKS = 3
_KIND_CAMERAS = 4

_HEADER_BYTES = 24
_AABB_BYTES = 24
_DIR_ENTRY_BYTES = 16
_QMAX = 65535


def _align8(offset: int) -> int:
    """Round ``offset`` up to the next multiple of 8 (spec/05 §2 alignment)."""
    return (offset + 7) & ~7


def _quantize(positions: np.ndarray, aabb_min: np.ndarray, aabb_max: np.ndarray) -> np.ndarray:
    """16-bit quantize float32 positions against the f32 AABB (spec/05 §2).

    ``q = clip(rint((p_f32 - min) / (max - min) * 65535), 0, 65535)`` with
    round-half-to-even (``numpy.rint``). Degenerate axes (``max == min``) yield 0.
    ``positions`` is (N, 3); returns uint16 (N, 3). All math is float32 so the
    decoder reconstructs from the identical f32 range (byte-parity, spec/05 §2).
    """
    p = np.asarray(positions, dtype=np.float32).reshape(-1, 3)
    span = (aabb_max - aabb_min).astype(np.float32)  # (3,) f32
    # Avoid divide-by-zero on degenerate axes; those columns are forced to 0 below.
    safe_span = np.where(span == 0, np.float32(1.0), span)
    norm = ((p - aabb_min) / safe_span).astype(np.float32)
    norm = np.where(span == 0, np.float32(0.0), norm)
    q = np.rint(norm * np.float32(_QMAX))
    # Defensive clip to [0, 65535] (spec/05 §4 — encoder clamps defensively).
    q = np.clip(q, 0, _QMAX)
    return q.astype(np.uint16)


def _gather_all_positions(scene: Scene4D) -> list[np.ndarray]:
    """All world positions (static ∪ dynamic ∪ tracks), each (N, 3) float32.

    Used to recompute the AABB so every section shares one quantization range
    (spec/05 §5.1).
    """
    chunks: list[np.ndarray] = []
    if scene.static_positions is not None and scene.static_positions.size:
        chunks.append(np.asarray(scene.static_positions, dtype=np.float32).reshape(-1, 3))
    for frame in scene.dynamic_positions or []:
        arr = np.asarray(frame, dtype=np.float32).reshape(-1, 3)
        if arr.size:
            chunks.append(arr)
    if scene.tracks is not None and scene.tracks.positions is not None:
        tp = np.asarray(scene.tracks.positions, dtype=np.float32).reshape(-1, 3)
        if tp.size:
            chunks.append(tp)
    return chunks


def _compute_aabb(scene: Scene4D) -> tuple[np.ndarray, np.ndarray]:
    """Recompute the AABB over ALL positions in float32 (spec/05 §5.1).

    Falls back to the scene-provided AABB (then to a unit box) if there are no
    positions at all, so a tracks/cameras-only or empty scene still encodes.
    """
    chunks = _gather_all_positions(scene)
    if chunks:
        allp = np.concatenate(chunks, axis=0).astype(np.float32)
        return allp.min(axis=0).astype(np.float32), allp.max(axis=0).astype(np.float32)
    if scene.aabb_min is not None and scene.aabb_max is not None:
        return (
            np.asarray(scene.aabb_min, dtype=np.float32).reshape(3),
            np.asarray(scene.aabb_max, dtype=np.float32).reshape(3),
        )
    return np.zeros(3, dtype=np.float32), np.ones(3, dtype=np.float32)


def encode_reconstruction(scene: Scene4D) -> bytes:
    """Encode an already-capped ``Scene4D`` into an MV4D v1 buffer (spec/05 §3).

    The encoder does NOT cull (caps are enforced upstream by the service, spec/06
    §5 step 7); it recomputes the AABB over all positions in float32, quantizes
    against that same f32 range, writes sections in ascending-kind order at
    8-byte-aligned offsets with tight intra-section packing, and logs the final
    counts + total payload size. ``q`` is defensively clipped to ``[0, 65535]``.
    """
    aabb_min, aabb_max = _compute_aabb(scene)

    frame_count = int(scene.frame_count)

    # ---- determine which sections are present (directory is authoritative) ----
    has_static = scene.static_positions is not None and int(np.asarray(scene.static_positions).reshape(-1, 3).shape[0]) > 0
    has_dynamic = bool(scene.dynamic_positions) and frame_count > 0
    has_tracks = scene.tracks is not None and int(np.asarray(scene.tracks.positions).reshape(-1, 3).shape[0] // max(frame_count, 1)) > 0 if scene.tracks is not None else False
    has_cameras = scene.cameras is not None

    has_static_conf = has_static and scene.static_conf is not None
    has_track_color = has_tracks and scene.tracks is not None and scene.tracks.colors is not None

    flags = 0
    if has_static:
        flags |= _FLAG_HAS_STATIC
    if has_dynamic:
        flags |= _FLAG_HAS_DYNAMIC
    if has_tracks:
        flags |= _FLAG_HAS_TRACKS
    if has_cameras:
        flags |= _FLAG_HAS_CAMERAS
    if has_static_conf:
        flags |= _FLAG_HAS_STATIC_CONF
    if has_track_color:
        flags |= _FLAG_HAS_TRACK_COLOR

    # ---- build each present section payload (ascending kind order) ----
    # Each entry: (kind, count, payload_bytes)
    sections: list[tuple[int, int, bytes]] = []

    n_static = 0
    n_dynamic_total = 0
    n_tracks = 0

    if has_static:
        sp = np.asarray(scene.static_positions, dtype=np.float32).reshape(-1, 3)
        n_static = sp.shape[0]
        pos_q = _quantize(sp, aabb_min, aabb_max)  # (N_s, 3) u16
        colors = np.asarray(scene.static_colors, dtype=np.uint8).reshape(-1, 3)
        parts = [pos_q.tobytes(), colors.tobytes()]
        if has_static_conf:
            conf = np.asarray(scene.static_conf, dtype=np.uint8).reshape(-1)
            parts.append(conf.tobytes())
        sections.append((_KIND_STATIC, n_static, b"".join(parts)))

    if has_dynamic:
        # frameDir: u32[T*2] {startPoint, pointCount} cumulative.
        frame_dir = np.zeros((frame_count, 2), dtype=np.uint32)
        pos_chunks: list[np.ndarray] = []
        col_chunks: list[np.ndarray] = []
        start = 0
        frames_pos = scene.dynamic_positions or []
        frames_col = scene.dynamic_colors or []
        for t in range(frame_count):
            fp = np.asarray(frames_pos[t], dtype=np.float32).reshape(-1, 3) if t < len(frames_pos) else np.empty((0, 3), np.float32)
            count_t = fp.shape[0]
            frame_dir[t, 0] = start
            frame_dir[t, 1] = count_t
            if count_t:
                pos_chunks.append(_quantize(fp, aabb_min, aabb_max))
                fc = np.asarray(frames_col[t], dtype=np.uint8).reshape(-1, 3) if t < len(frames_col) else np.zeros((count_t, 3), np.uint8)
                col_chunks.append(fc)
            start += count_t
        n_dynamic_total = start
        pos_all = np.concatenate(pos_chunks, axis=0) if pos_chunks else np.empty((0, 3), np.uint16)
        col_all = np.concatenate(col_chunks, axis=0) if col_chunks else np.empty((0, 3), np.uint8)
        payload = b"".join([frame_dir.tobytes(), pos_all.tobytes(), col_all.tobytes()])
        sections.append((_KIND_DYNAMIC, frame_count, payload))

    if has_tracks:
        tracks = scene.tracks
        tp = np.asarray(tracks.positions, dtype=np.float32).reshape(-1, frame_count, 3)
        n_tracks = tp.shape[0]
        m_t = n_tracks * frame_count
        pos_q = _quantize(tp.reshape(-1, 3), aabb_min, aabb_max).reshape(m_t, 3)
        # visibility: u8[ceil(M*T/8)] LSB-first bitmask; bit i = m*T + t.
        vis = np.asarray(tracks.visibility, dtype=bool).reshape(m_t)
        packed = np.packbits(vis, bitorder="little")  # length ceil(M*T/8)
        parts = [pos_q.tobytes(), packed.tobytes()]
        if has_track_color:
            tcol = np.asarray(tracks.colors, dtype=np.uint8).reshape(-1, 3)
            parts.append(tcol.tobytes())
        sections.append((_KIND_TRACKS, n_tracks, b"".join(parts)))

    if has_cameras:
        cam = scene.cameras
        poses = np.asarray(cam.poses, dtype=np.float32).reshape(frame_count, 7)
        intr = np.asarray(cam.intrinsics, dtype=np.float32).reshape(frame_count, 4)
        payload = b"".join([poses.tobytes(), intr.tobytes()])
        sections.append((_KIND_CAMERAS, frame_count, payload))

    section_count = len(sections)

    # ---- compute absolute offsets: header + aabb + directory, then payloads ----
    dir_offset = _HEADER_BYTES + _AABB_BYTES
    payload_cursor = _align8(dir_offset + section_count * _DIR_ENTRY_BYTES)

    placed: list[tuple[int, int, int, int, bytes]] = []  # (kind, byteOffset, byteLength, count, payload)
    for kind, count, payload in sections:
        off = _align8(payload_cursor)
        placed.append((kind, off, len(payload), count, payload))
        payload_cursor = off + len(payload)

    total_len = payload_cursor

    buf = bytearray(total_len)

    # ---- header (spec/05 §3.1) ----
    struct.pack_into(
        "<4sBBBBHHfII",
        buf,
        0,
        _MAGIC,
        MV4D_VERSION,
        flags,
        _POS_BITS,
        section_count,
        frame_count,
        0,  # reserved0
        float(scene.fps),
        0,  # reserved1
        0,  # reserved2
    )

    # ---- AABB block (spec/05 §3.2) ----
    struct.pack_into("<6f", buf, _HEADER_BYTES, *aabb_min.astype(np.float32).tolist(), *aabb_max.astype(np.float32).tolist())

    # ---- section directory + payloads (spec/05 §3.3) ----
    dir_cursor = dir_offset
    for kind, off, length, count, payload in placed:
        struct.pack_into("<IIII", buf, dir_cursor, kind, off, length, count)
        dir_cursor += _DIR_ENTRY_BYTES
        buf[off:off + length] = payload

    logger.info(
        "MV4D v1 encoded: T=%d static=%d dynamic_total=%d tracks=%d cameras=%s "
        "sections=%d payload_bytes=%d",
        frame_count,
        n_static,
        n_dynamic_total,
        n_tracks,
        has_cameras,
        section_count,
        total_len,
    )

    return bytes(buf)
