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
