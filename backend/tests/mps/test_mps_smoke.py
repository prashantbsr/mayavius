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

# The torch floor (spec/10 §5 T-500 / spec/08 §4.1): first stable MPS autocast.
_TORCH_FLOOR = (2, 5, 0)

# A bundled ≤3 s sample clip (copied from frontend/e2e/fixtures/tiny.mp4 at W3; the
# full corpus C-1..C-4 is W4). Lives next to this file so collection has no repo-root
# dependency.
_SAMPLE_CLIP = Path(__file__).resolve().parent / "fixtures" / "sample.mp4"


def _mps_available() -> bool:
    """True iff torch is importable AND reports an available MPS backend."""
    try:
        import torch
    except Exception:
        return False
    try:
        return bool(torch.backends.mps.is_available())
    except Exception:
        return False


def _skip_reason() -> str | None:
    """Return a skip reason if the smoke preconditions are unmet, else ``None``."""
    if os.environ.get("MAYAVIUS_RUN_MPS_SMOKE") != "1":
        return "opt-in flag off: set MAYAVIUS_RUN_MPS_SMOKE=1 (spec/10 §5)"
    if not _mps_available():
        return "no MPS backend (torch.backends.mps.is_available() is False) (spec/10 §5)"
    return None


# Evaluate the gate once at import; both skipif blocks below share it.
_SKIP = _skip_reason()


def _torch_version_tuple(version: str) -> tuple[int, int, int]:
    """Parse e.g. '2.12.0', '2.12.0+cpu', '2.5.0a0' → (2, 12, 0)."""
    head = version.split("+", 1)[0]
    parts: list[int] = []
    for piece in head.split(".")[:3]:
        num = ""
        for ch in piece:
            if ch.isdigit():
                num += ch
            else:
                break
        parts.append(int(num) if num else 0)
    while len(parts) < 3:
        parts.append(0)
    return (parts[0], parts[1], parts[2])


def _peak_mps_gb() -> float | None:
    """Best-effort MPS peak allocation in GiB (driver-allocated, else current).

    ``torch.mps.driver_allocated_memory`` / ``current_allocated_memory`` exist on
    recent torch; absent → ``None`` (recorded, not a gate).
    """
    try:
        import torch

        mps = getattr(torch, "mps", None)
        if mps is None:
            return None
        for fn_name in ("driver_allocated_memory", "current_allocated_memory"):
            fn = getattr(mps, fn_name, None)
            if callable(fn):
                try:
                    return float(fn()) / (1024.0**3)
                except Exception:
                    continue
        return None
    except Exception:
        return None


# --- T-500 ----------------------------------------------------------------------
@pytest.mark.skipif(_SKIP is not None, reason=_SKIP or "")
def test_mps_available() -> None:
    """``torch.backends.mps.is_available()`` is True and torch ≥ 2.5.0 floor (else skip)."""
    import torch  # imported INSIDE the test (collection needs no ML deps)

    assert torch.backends.mps.is_available(), "MPS backend unavailable on this machine"

    ver = _torch_version_tuple(torch.__version__)
    if ver < _TORCH_FLOOR:
        pytest.skip(
            f"torch {torch.__version__} < floor {'.'.join(map(str, _TORCH_FLOOR))} "
            "(spec/10 §5 / spec/08 §4.1)"
        )


# --- T-510 ----------------------------------------------------------------------
@pytest.mark.skipif(_SKIP is not None, reason=_SKIP or "")
def test_vggt_cotracker3_smoke(capsys) -> None:
    """Real combo on ONE bundled ≤3 s clip → valid Scene4D encoding within caps.

    Asserts: completes without raising; ≥1 static point AND ≥1 track; the Scene4D
    encodes to MV4D and the encoded scene honors the MV4D caps (chains
    ``encode_reconstruction`` + the cap constants). MEASURES + PRINTS (does NOT assert)
    wall-time + peak MPS memory — verify, don't assert blind (decision-log §E,H).
    """
    import torch  # noqa: F401  (imported INSIDE the test — collection needs no ML deps)

    # vggt is pulled lazily by the adapter on first reconstruct(); skip cleanly if the
    # ML deps are not installed (collection already succeeded without them).
    try:
        import vggt  # noqa: F401
    except Exception as exc:  # pragma: no cover - exercised only on-device
        pytest.skip(f"vggt not installed (requirements-ml.txt): {exc!r}")

    if not _SAMPLE_CLIP.exists():
        pytest.skip(f"no bundled sample clip at {_SAMPLE_CLIP}")

    from app.adapters.combo import VggtCoTracker3Adapter
    from app.config import settings
    from app.core.domain.models import Scene4D
    from app.core.services.reconstruction_service import enforce_caps
    from app.wire.decoder import decode
    from app.wire.encoder import encode_reconstruction

    # MV4D caps (spec/05 §4) — verified post-encode against the decoded scene.
    _MAX_STATIC = 150_000
    _MAX_DYNAMIC_PER_FRAME = 20_000
    _MAX_TRACKS = 4_096
    _MAX_FRAMES = 64

    from app.core.domain.models import ReconstructionRequest

    # device "mps" (MAYAVIUS_DEVICE=mps); a short clip so it fits the 36 GB Mac.
    req = ReconstructionRequest(
        video_path=str(_SAMPLE_CLIP),
        max_frames=min(getattr(settings, "max_clip_frames", 8), 8),
        target_fps=getattr(settings, "target_fps", 12.0),
        device="mps",
    )

    # Reset the MPS peak counter if the API exists (best-effort).
    mps = getattr(torch, "mps", None)
    if mps is not None and hasattr(mps, "empty_cache"):
        try:
            mps.empty_cache()
        except Exception:
            pass

    adapter = VggtCoTracker3Adapter(settings)
    assert adapter.info.weights_license  # D2 license surface

    t0 = time.perf_counter()
    scene = adapter.reconstruct(req)  # RAW Scene4D (combo returns RAW; we cap below)
    wall_s = time.perf_counter() - t0
    peak_gb = _peak_mps_gb()

    assert isinstance(scene, Scene4D), type(scene)

    # The combo returns a RAW scene; apply the cap step (the service does this in prod)
    # so we validate the SAME caps the encoder ships under (spec/06 §5 step 7).
    capped = enforce_caps(scene)

    # ≥1 static point AND ≥1 track (the "cloud + ribbons" wow, spec/10 §5 T-510).
    assert capped.static_positions.shape[0] >= 1, "expected ≥1 static point"
    assert capped.tracks is not None and capped.tracks.positions.shape[0] >= 1, (
        "expected ≥1 track"
    )

    # Encodes to MV4D within caps (chains T-100 / T-104): encode → decode → re-check.
    buf = encode_reconstruction(capped)
    assert buf[:4] == b"MV4D"
    decoded = decode(buf)

    assert 1 <= decoded.frame_count <= _MAX_FRAMES
    assert decoded.static_positions.shape[0] <= _MAX_STATIC
    assert len(decoded.dynamic_positions) == decoded.frame_count
    for frame in decoded.dynamic_positions:
        assert frame.shape[0] <= _MAX_DYNAMIC_PER_FRAME
    assert decoded.tracks is not None
    assert decoded.tracks.positions.shape[0] <= _MAX_TRACKS
    assert decoded.tracks.positions.shape[1] == decoded.frame_count

    # MEASURE + PRINT (do NOT assert a GB threshold — the numbers are outputs).
    peak_str = f"{peak_gb:.1f} GB peak" if peak_gb is not None else "peak n/a"
    line = (
        f"mps_smoke: {wall_s:.1f}s wall, {peak_str} | "
        f"static={capped.static_positions.shape[0]} tracks={capped.tracks.positions.shape[0]} "
        f"frames={capped.frame_count}"
    )
    with capsys.disabled():
        print("\n" + line)


# --- T-511 ----------------------------------------------------------------------
@pytest.mark.skipif(_SKIP is not None, reason=_SKIP or "")
def test_mps_fallback_documented(caplog) -> None:
    """Record which ops fall back to CPU under PYTORCH_ENABLE_MPS_FALLBACK=1.

    Captures warnings/logs emitted during a real combo run; records the fallback op
    list (recorded, not asserted). If an op FAILS *even with* fallback, the run raises
    and the test fails with a message pointing at the cloud-GPU path (spec/11) — this
    is how a missing-op dead end gets documented, not rediscovered (spec/10 §5).
    """
    import logging
    import warnings

    import torch  # noqa: F401  (imported INSIDE the test)

    try:
        import vggt  # noqa: F401
    except Exception as exc:  # pragma: no cover - exercised only on-device
        pytest.skip(f"vggt not installed (requirements-ml.txt): {exc!r}")

    if not _SAMPLE_CLIP.exists():
        pytest.skip(f"no bundled sample clip at {_SAMPLE_CLIP}")

    # The fallback env MUST be set before the adapter imports torch (spec/08 §5); the
    # adapters setdefault it, but assert it here so the test documents the contract.
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    assert os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1"

    from app.adapters.combo import VggtCoTracker3Adapter
    from app.config import settings
    from app.core.domain.models import ReconstructionRequest

    req = ReconstructionRequest(
        video_path=str(_SAMPLE_CLIP),
        max_frames=min(getattr(settings, "max_clip_frames", 8), 8),
        target_fps=getattr(settings, "target_fps", 12.0),
        device="mps",
    )
    adapter = VggtCoTracker3Adapter(settings)

    caplog.set_level(logging.INFO)
    fallback_ops: list[str] = []
    with warnings.catch_warnings(record=True) as wlist:
        warnings.simplefilter("always")
        try:
            adapter.reconstruct(req)
        except Exception as exc:  # op failed EVEN WITH fallback → a documented dead end
            pytest.fail(
                f"an op failed on MPS even with PYTORCH_ENABLE_MPS_FALLBACK=1 "
                f"({exc!r}); run this adapter on the cloud-GPU deploy (spec/11)."
            )
        for w in wlist:
            msg = str(w.message)
            if "fallback" in msg.lower() or "MPS" in msg:
                fallback_ops.append(msg)

    # Also scan captured log records for any fallback note the adapter logged.
    for rec in caplog.records:
        text = rec.getMessage()
        if "fallback" in text.lower():
            fallback_ops.append(text)

    # Record the list (recorded, not asserted — verify, don't assert blind).
    if fallback_ops:
        print("\nmps_fallback ops (recorded, not asserted):")
        for op in fallback_ops:
            print(f"  - {op}")
    else:
        print("\nmps_fallback: none observed under PYTORCH_ENABLE_MPS_FALLBACK=1")
