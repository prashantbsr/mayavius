"""AABB + 16-bit quantization helpers for the MV4D encoder (pipeline-side).

This module factors out the **exact** quantization math the encoder
(`app/wire/encoder.py`) performs, so the pipeline can compute a shared AABB and
pre-quantize positions consistently with the wire format. The byte-level
quantization contract is owned by spec/05-data-contract.md §2 and is NEVER
redefined here — these helpers merely render it:

    q = clip(rint((p_f32 - aabb_min_f32) / (aabb_max_f32 - aabb_min_f32) * 65535),
             0, 65535)         # round-half-to-even (numpy.rint)
    degenerate axis (aabb_max == aabb_min) -> q = 0

Working precision is **float32 end-to-end** (positions AND the AABB cast to f32,
using the same f32 range as the divisor) so an independent encoder/decoder
reconstructs the identical f32 value — byte parity (spec/05 §2, T-203). The encoder
in `app/wire/encoder.py` MUST stay consistent with this math (same constants,
same rounding, same degenerate-axis rule).

numpy only — NO torch, NO fastapi, NO opencv. Part of the model-agnostic pipeline.
"""

from __future__ import annotations

import numpy as np

# Quantization range (spec/05 §2 / §3.1 posBits=16). Mirrors encoder._QMAX.
_QMAX = 65535


def compute_aabb(*pointsets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Axis-aligned bounding box over ALL given point arrays, in float32.

    Each argument is an ``(N_i, 3)`` (or reshapeable-to ``(-1, 3)``) array of world
    positions; empty arrays are ignored. Returns ``(aabb_min, aabb_max)``, both
    ``(3,)`` float32. The AABB is computed in **float32** so it matches the f32
    header range the encoder/decoder use as the quantization divisor (spec/05 §2,
    §5.1: the AABB spans static ∪ dynamic ∪ tracks).

    Falls back to the unit box ``([0,0,0], [1,1,1])`` if there are no points at all,
    so a degenerate/empty scene still yields a usable (non-NaN) range.
    """
    chunks: list[np.ndarray] = []
    for ps in pointsets:
        if ps is None:
            continue
        arr = np.asarray(ps, dtype=np.float32).reshape(-1, 3)
        if arr.size:
            chunks.append(arr)
    if not chunks:
        return np.zeros(3, dtype=np.float32), np.ones(3, dtype=np.float32)
    allp = np.concatenate(chunks, axis=0).astype(np.float32)
    return allp.min(axis=0).astype(np.float32), allp.max(axis=0).astype(np.float32)


def quantize_positions(
    points_f32: np.ndarray,
    aabb_min: np.ndarray,
    aabb_max: np.ndarray,
) -> np.ndarray:
    """16-bit quantize float32 positions against the f32 AABB (spec/05 §2).
