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

    # --- duration <= 3 s (probe with OpenCV; importorskip only this sub-check) -
    # If cv2 is unavailable we skip ONLY the duration assertion (importorskip),
    # never the presence/license assertions above — those are non-negotiable.
    cv2 = pytest.importorskip(
        "cv2",
        reason="OpenCV (cv2) not installed; skipping only the duration sub-check.",
    )
    cap = cv2.VideoCapture(str(clip))
    try:
        assert cap.isOpened(), f"OpenCV could not open corpus clip {clip}"
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    finally:
        cap.release()

    if fps and fps > 0 and frame_count and frame_count > 0:
        duration_s = frame_count / fps
        assert duration_s <= MAX_DURATION_S + 1e-3, (
            f"corpus clip {clip} is {duration_s:.2f}s, over the "
            f"{MAX_DURATION_S}s cap (spec/10 §6). Re-encode shorter; do not relax."
        )
    else:
        # Metadata unreadable (fps/frame-count == 0): fall back to the sidecar's
        # declared duration_s if present so the cap is still enforced; otherwise
        # skip ONLY this sub-check (the clip still passed presence/license/size).
        declared = meta.get("duration_s")
        if isinstance(declared, (int, float)) and declared > 0:
            assert declared <= MAX_DURATION_S + 1e-3, (
                f"sidecar duration_s={declared} for {slug!r} exceeds the "
                f"{MAX_DURATION_S}s cap (spec/10 §6)."
            )
        else:
            pytest.skip(
                f"OpenCV returned no fps/frame-count for {clip} and the sidecar "
                f"has no usable duration_s; duration sub-check skipped (presence, "
                f"license, and size were still enforced)."
            )


# A healthy static cloud is at least this many points — the negative-control bug
# baked ~3 static points (a degenerate split), so any sane re-bake clears this by
# orders of magnitude. NOT a tunable cap; a sanity floor on a non-degenerate split.
_MIN_HEALTHY_STATIC: int = 1_000


def _blob_path(slug: str) -> Path:
    return SAMPLES_DIR / f"{slug}.mv4d"


def _max_dynamic_per_frame(scene) -> int:
    """Largest per-frame dynamic point count in a decoded ``Scene4D`` (0 if none)."""
    return max((int(p.shape[0]) for p in scene.dynamic_positions), default=0)


@pytest.mark.parametrize("slug", CORPUS_SLUGS)
def test_corpus_blob_static_dynamic_split_healthy(slug: str) -> None:
    """The committed ``<slug>.mv4d`` decodes to a HEALTHY static/dynamic split.

    Decodes the committed corpus blob with the backend reference wire decoder
    (``app.wire.decoder.decode``, the exact inverse of the encoder — spec/05 §3)
    and asserts the net-excursion split (``app/pipeline/assemble.py``) produced a
    sane result, not the degenerate one the STALE blobs encode:

    - **static-scene (C-4, the negative control):** static is substantial AND far
      exceeds the per-frame dynamic count. The stale-blob bug is the inverse —
      static ≈ 3, dynamic ≈ 160k per frame — i.e. a near-static mountain valley
      classified as nearly all motion. This is the assertion that catches the bug.
    - **the other three (C-1..C-3):** a sane static cloud exists (a stable
      background the moving subject animates over — spec/10 §6 expected behavior).

    **This test is EXPECTED-RED until ``make bake-corpus`` regenerates the stale
    blobs** (they were baked before the split fix). That is correct — same
    convention as the presence/license test above (RED until sourced). It MUST
    fail *clearly* (naming the slug + the degenerate counts) rather than pass
    vacuously; do NOT weaken it to skip-on-bad-split. If the blob is genuinely
    ABSENT we skip cleanly (a fresh clone may have no committed corpus), but when
    PRESENT we assert.
    """
    from app.wire.decoder import decode  # backend reference decoder (spec/05 §3)

    blob = _blob_path(slug)
    if not blob.is_file():
        pytest.skip(
            f"corpus blob absent: {blob} — bake it with `make bake-corpus` "
            f"(spec/10 §6). Skipped only because the file is missing entirely."
        )

    scene = decode(blob.read_bytes())
    n_static = int(scene.static_positions.shape[0])
    max_dyn = _max_dynamic_per_frame(scene)

    if slug == "static-scene":
        # Negative control: a substantial static cloud that DOMINATES the dynamic
        # count. The stale blob fails BOTH halves (static≈3 ≪ floor; static ≪ dyn).
        assert n_static >= _MIN_HEALTHY_STATIC, (
            f"negative control {slug!r}: static={n_static} is below the healthy "
            f"floor {_MIN_HEALTHY_STATIC} — the STALE blob bakes ~3 static points "
            f"(degenerate split). EXPECTED-RED until `make bake-corpus` regenerates "
            f"it with the net-excursion split (app/pipeline/assemble.py)."
        )
        assert n_static > max_dyn, (
            f"negative control {slug!r}: static={n_static} does NOT exceed the "
            f"per-frame dynamic max={max_dyn} — an almost-static scene was split as "
            f"mostly MOTION (the stale-blob bug: static≈3, dynamic≈160k/frame). "
            f"EXPECTED-RED until `make bake-corpus` re-bakes the split."
        )
    else:
        # C-1..C-3: a moving subject animates over a STABLE background, so a sane
        # static cloud must exist (spec/10 §6 expected behavior).
        assert n_static >= _MIN_HEALTHY_STATIC, (
            f"corpus blob {slug!r}: static={n_static} is below the healthy floor "
            f"{_MIN_HEALTHY_STATIC} — no stable background cloud. EXPECTED-RED until "
            f"`make bake-corpus` regenerates the stale blob (app/pipeline/assemble.py)."
        )
