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
from app.wire.encoder import encode_reconstruction
from tests.wire._golden import golden_scene

_FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"
_GOLDEN_PATH = _FIXTURES / "golden_scene.mv4d"
_TINY_PATH = _FIXTURES / "tiny.mv4d"


# --------------------------------------------------------------------------- #
# T-200 — the committed golden fixture is byte-for-byte canonical
# --------------------------------------------------------------------------- #
def test_golden_fixture_is_canonical():
    """T-200: encode(golden_scene()) == the committed golden_scene.mv4d bytes.

    Byte-stability anchor (spec/05 §7 / spec/10 §2): if the encoder ever changes
    its output, this fails until the committed fixture AND decoder.ts are updated
    together in the same commit.
    """
    assert _GOLDEN_PATH.exists(), (
        f"missing committed golden fixture: {_GOLDEN_PATH} "
        "(regenerate via tests.wire._gen_fixtures)"
    )
    on_disk = _GOLDEN_PATH.read_bytes()
    regenerated = encode_reconstruction(golden_scene())

    assert regenerated == on_disk, (
        "golden_scene.mv4d is stale: encode_reconstruction(golden_scene()) no "
        f"longer matches the committed fixture (regenerated {len(regenerated)} B "
        f"vs on-disk {len(on_disk)} B). Update the fixture AND decoder.ts in the "
        "same commit (spec/05 §7)."
