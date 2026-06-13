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
