"""One-off generator for the committed wire fixtures (W0.T4).

Run from `backend/` with the venv python:

    ./.venv/bin/python -m tests.wire._gen_fixtures

Produces the three committed artifacts whose *bytes* are the source of truth:
  1. backend/tests/fixtures/golden_scene.mv4d
       = encode_reconstruction(golden_scene())  (T-200 byte-stability anchor)
  2. backend/tests/fixtures/tiny.mv4d
       = the hand-laid TINY REVERSE VECTOR (spec/05 §6), built byte-by-byte with
         struct/bytearray — INDEPENDENT of encode_reconstruction (T-203 guards the
         reverse decode direction; spec/05 §10 permits hand-laid bytes).
  3. frontend/src/lib/wire/__fixtures__/tiny.mv4d  (a copy of #2; the spec's
       __fixtures__ location read by the TS round-trip).

The byte layout is owned by spec/05-data-contract.md §3 and is NEVER redefined
elsewhere — this script renders §3.1/§3.2/§3.3/§3.5 for the tiny vector exactly.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

from app.wire.encoder import encode_reconstruction
from tests.wire._golden import golden_scene

_HERE = Path(__file__).resolve()
_BACKEND = _HERE.parents[2]                      # .../backend
_REPO = _BACKEND.parent                          # .../mayavius
_BACKEND_FIXTURES = _BACKEND / "tests" / "fixtures"
_FRONTEND_FIXTURES = _REPO / "frontend" / "src" / "lib" / "wire" / "__fixtures__"

# flags bits (spec/05 §3.1)
_FLAG_HAS_DYNAMIC = 1 << 1
# section kind (spec/05 §3.3)
_KIND_DYNAMIC = 2
_QMAX = 65535


def _build_tiny_reverse_vector() -> bytes:
    """Hand-lay the TINY REVERSE VECTOR (spec/05 §6) byte-by-byte.

    Scene: T=2, NO static, one dynamic point per frame, NO tracks, NO cameras.
    flags = HAS_DYNAMIC (0x02), fps=24.0, aabbMin=(0,0,0), aabbMax=(1,1,1),
    sectionCount=1. This is built WITHOUT ``encode_reconstruction`` so it
    independently guards the reverse (TS->Py) decode direction (T-203).

    Total = header24 + aabb24 + directory16 + dynamic(frameDir16 + pos12 + col6
    = 34) = 98 bytes.
    """
    buf = bytearray()

    # ---- header — 24 bytes @0 (spec/05 §3.1) ----
    # magic, version=1, flags=0x02, posBits=16, sectionCount=1, frameCount=2,
    # reserved0=0, fps=24.0, reserved1=0, reserved2=0
    buf += struct.pack(
        "<4sBBBBHHfII",
        b"MV4D",
        1,                  # version
        _FLAG_HAS_DYNAMIC,  # flags
        16,                 # posBits
        1,                  # sectionCount
        2,                  # frameCount (T)
        0,                  # reserved0
        24.0,               # fps
        0,                  # reserved1
        0,                  # reserved2
    )
    assert len(buf) == 24

    # ---- AABB block — 24 bytes @24 (spec/05 §3.2): min (0,0,0), max (1,1,1) ----
    buf += struct.pack("<6f", 0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
    assert len(buf) == 48

    # ---- DYNAMIC payload bytes (built first to size the directory entry) ----
    # frameDir u32[T*2] {startPoint, pointCount} cumulative: [0,1, 1,1].
    frame_dir = struct.pack("<4I", 0, 1, 1, 1)
    # positions u16[2*3]. point0 world (0.25,0.25,0.25) -> q=rint(0.25*65535)=16384;
    # point1 world (0.5,0.5,0.5) -> q=rint(0.5*65535)=32768 (round-half-even).
    q0 = int(np.rint(0.25 * _QMAX))  # 16384
    q1 = int(np.rint(0.5 * _QMAX))   # 32768
    assert q0 == 16384 and q1 == 32768, (q0, q1)
    positions = struct.pack("<6H", q0, q0, q0, q1, q1, q1)
    # colors u8[2*3]: point0 (255,128,0), point1 (0,128,255).
    colors = struct.pack("<6B", 255, 128, 0, 0, 128, 255)
    dynamic_payload = frame_dir + positions + colors
