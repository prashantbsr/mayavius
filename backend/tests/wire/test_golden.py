"""Cross-implementation seam guard — golden fixture + reverse vector (W0.T4).

This is the most load-bearing pair of tests in the repo: ``encoder.py`` and
``decoder.ts`` are two implementations of ONE format (spec/05-data-contract.md);
nothing else forces them to agree. We commit a conformance vector and prove it
decodes the same on both sides.

- **T-200** (`test_golden_fixture_is_canonical`): the committed
  ``golden_scene.mv4d`` is regenerated from THE golden ``Scene4D`` literal
  (`tests/wire/_golden.py`) and compared **byte-for-byte** to the on-disk file.
  If the encoder's output changes, this fails until the fixture (and
  ``decoder.ts``) are updated in the same commit — enforces spec/05 §7.
- **T-203** (`encoder.reverse_conformance`): the **hand-laid** TINY REVERSE
  VECTOR (`tests/fixtures/tiny.mv4d`, built byte-by-byte INDEPENDENT of
  ``encode_reconstruction``) is decoded by the Python reference decoder and must
  match — guarding the *reverse* (client-encoded -> Python-decoded) direction so
  the seam is symmetric (spec/10 §2).

The byte layout is owned by spec/05 §3 and is never redefined here. T-201
(version parity, reads ``decoder.ts``) is owned by the Frontend phase — the TS
decoder does not exist yet — and is deliberately NOT written here (spec/09 W0.T4).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from app.wire.decoder import decode
