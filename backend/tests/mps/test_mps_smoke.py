"""On-device MPS smoke test (spec/10 §5, T-500/T-510/T-511) — the 36 GB-Mac gate.

The ONLY test that needs ``requirements-ml.txt`` (real torch + multi-GB weights). It
proves the real default combo (VGGT + CoTracker3, lift 2D→3D via VGGT depth) runs on
Apple-Silicon MPS and **records** its cost — the project's hard constraint made
testable.

Marked ``@pytest.mark.mps`` (whole module via ``pytestmark``). Skip logic (spec/10 §5):
skip unless ``torch.backends.mps.is_available()`` AND
``MAYAVIUS_RUN_MPS_SMOKE=1`` — so it NEVER runs in default ``pytest`` / default CI;
the skip reason states which precondition failed.

COLLECTION WITHOUT ML DEPS: ``torch`` / ``vggt`` / ``cotracker`` are imported INSIDE
the test functions (never at module top level), so pytest can COLLECT this file with
zero ML deps installed — only an actual run (gated) imports them.

VERIFY, DON'T ASSERT BLIND (decision-log §E,H): T-510 MEASURES and PRINTS wall-time +
peak memory; it NEVER pre-asserts a GB threshold — the numbers are *outputs* recorded
for spec/08 §5 / the README, not gates.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

pytestmark = pytest.mark.mps
