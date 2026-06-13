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
