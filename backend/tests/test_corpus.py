"""T-600 — sample-video corpus presence + licensing (spec/10 §6, D10).

The C-1..C-4 corpus (`walking-person`/`street-vehicle`/`pet-motion`/`static-scene`)
ships as 3-4 short (<=3 s), CC-licensed clips committed under repo-root
``assets/samples/`` (spec/10 §6). Each clip is a preloaded example AND a test
fixture; each is seeded by the backend lifespan from ``assets/samples/<slug>.mv4d``
(spec/06 §6). This test makes the license discipline **testable, not trust-based**:

- each corpus clip ``<slug>.mp4`` EXISTS,
- it has a sidecar ``<slug>.json`` with a **non-empty** ``license`` + ``source_url``,
- it is **<= 3 s** (duration sub-check via OpenCV when probeable; importorskip
  otherwise — never silently skip the whole test),
- it is under the size cap.

**This test is RED until the Source phase lands the licensed clips — that is
correct (spec/10 §6).** It MUST fail *clearly* (naming the missing/unlicensed slug)
rather than pass vacuously: the corpus slugs are an explicit module-level list, and
absence is asserted per-slug. Do NOT weaken it to skip-on-missing — a green vacuous
pass would let an unlicensed or absent clip ship.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.config import SAMPLES_DIR

