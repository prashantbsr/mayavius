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

