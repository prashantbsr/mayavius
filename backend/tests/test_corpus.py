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

# The four corpus slugs (C-1..C-4, spec/10 §6). The slug IS the file name:
# assets/samples/<slug>.{mp4,json,mv4d}. Explicit so a missing clip fails by NAME.
CORPUS_SLUGS: list[str] = [
    "walking-person",  # C-1 (the hero)
    "street-vehicle",  # C-2
    "pet-motion",      # C-3
    "static-scene",    # C-4 (the negative control)
]

# Caps (spec/10 §6): clips are re-encoded to <= 3 s / <= 540p / <= ~2 MB each.
# The byte cap is the curated-corpus ceiling; do NOT raise it to make a clip fit —
# re-encode the clip instead (HARD RULE: never lower a cap / never weaken a test).
MAX_DURATION_S: float = 3.0
MAX_SIZE_BYTES: int = 2 * 1024 * 1024  # ~2 MB per clip (spec/10 §6)


def _clip_path(slug: str) -> Path:
    return SAMPLES_DIR / f"{slug}.mp4"


def _sidecar_path(slug: str) -> Path:
    return SAMPLES_DIR / f"{slug}.json"


@pytest.mark.parametrize("slug", CORPUS_SLUGS)
def test_corpus_present_and_licensed(slug: str) -> None:
    """T-600 — each C-1..C-4 clip is present, licensed, short, and under the cap."""
    clip = _clip_path(slug)
    sidecar = _sidecar_path(slug)

    # --- presence (fail clearly by name; never vacuously pass) ---------------
    assert clip.is_file(), (
        f"corpus clip missing: {clip} — the Source phase must land a CC-licensed "
        f"clip for slug {slug!r} (spec/10 §6). RED until sourced; do not weaken."
    )
    assert sidecar.is_file(), (
        f"corpus sidecar missing: {sidecar} — every clip ships "
        f"{{source_url, license, attribution, duration_s, expected}} (spec/10 §6)."
    )

    # --- licensing discipline (non-empty license + source_url) ---------------
    meta = json.loads(sidecar.read_text(encoding="utf-8"))
    license_tag = str(meta.get("license", "")).strip()
    source_url = str(meta.get("source_url", "")).strip()
    assert license_tag, (
        f"{sidecar} has an empty/missing 'license' — a clip whose license is "
        f"unverified MUST NOT ship (spec/10 §6 sourcing rule). Slug {slug!r}."
    )
    assert source_url, (
        f"{sidecar} has an empty/missing 'source_url' — record the exact CC "
        f"source URL (spec/10 §6 sourcing rule). Slug {slug!r}."
    )

    # --- size cap (committed corpus is small; ~2 MB ceiling) -----------------
    size = clip.stat().st_size
    assert size <= MAX_SIZE_BYTES, (
        f"corpus clip {clip} is {size} bytes, over the {MAX_SIZE_BYTES}-byte cap "
        f"(spec/10 §6). Re-encode to <=540p/<=3 s — do NOT raise the cap."
    )

