"""T-201 — MV4D version parity (spec/10 §2, spec/05 §7).

`encoder.py` (Python) and `decoder.ts` (TS) are two implementations of ONE wire
format; nothing else forces their `MV4D_VERSION` constants to agree. This test
imports the Python constant and extracts the TS literal from `decoder.ts` source
via regex, asserting both `== 1` and equal. If a future change bumps one and not
the other, this fails (the seam guard, spec/10 §2 "Parity mechanics").

This pytest lives backend-side because it reads `decoder.ts`; it is owned by the
frontend wire task (W0.T3/T4) per the build plan.
"""

from __future__ import annotations

import re
from pathlib import Path

from app.wire.encoder import MV4D_VERSION as PY_MV4D_VERSION


def _decoder_ts_path() -> Path:
    # tests/wire/test_version_parity.py → parents[3] == repo root.
    return (
        Path(__file__).resolve().parents[3]
        / "frontend"
        / "src"
        / "lib"
        / "wire"
        / "decoder.ts"
    )
