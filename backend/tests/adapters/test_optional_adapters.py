"""Optional-adapter honest-stub contract (spec/06 §4.3/§4.4/§4.5/§4.7, spec/10 §3/§5).

The three optional adapters are CUDA/GPU-only or no-MPS DEAD ENDS on the Mac
(decision-log §D/E): SpatialTrackerV2 is CUDA-only (cu124 pin); Pi3 has no official
MPS (PR #153 unmerged); OpenD4RT's MPS path is unverified. Per spec/10 §5 negative
knowledge we do NOT write a green MPS test for them — their on-device contract test
is ``@pytest.mark.gpu`` and SKIPPED on the Mac with a reason NAMING the constraint
(documented as skipped, not silently absent).

TWO layers here:
  1. ``@pytest.mark.gpu`` tests — the on-(cloud)-device contract; skipped on the Mac
     with a constraint-naming reason (collected + SKIPPED, never errored).
  2. An UNMARKED test — proves the HONEST-STUB contract without any GPU/ML deps: each
     optional adapter raises ``UnsupportedDeviceError`` (NOT a silent fall-back) on
     device "mps", with a message naming the constraint + pointing at the cloud-GPU
     deploy (spec/11). This part runs in ``make test`` (no ML deps needed — the raise
     happens BEFORE any model load, so no torch import occurs).
"""

from __future__ import annotations

import pytest

from app.core.domain.errors import UnsupportedDeviceError
from app.core.domain.models import ReconstructionRequest

# (id, module, class, a constraint keyword that MUST appear in the raised message).
_OPTIONAL_ADAPTERS = [
    pytest.param(
        "app.adapters.spatialtracker_adapter",
        "SpatialTrackerV2Adapter",
        "CUDA",
        id="spatialtracker_v2",
    ),
    pytest.param(
        "app.adapters.pi3_adapter",
        "Pi3Adapter",
        "MPS",
        id="pi3",
    ),
    pytest.param(
        "app.adapters.open_d4rt_adapter",
        "OpenD4RTAdapter",
        "MPS",
        id="open_d4rt",
    ),
]


def _build(module_name: str, class_name: str):
    import importlib

    cls = getattr(importlib.import_module(module_name), class_name)
    return cls(None)


def _cuda_available() -> bool:
    """True iff torch is importable AND reports an available CUDA device."""
    try:
        import torch
    except Exception:
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


# --- UNMARKED: honest-stub contract (runs in `make test`; no GPU / no ML deps) -----
@pytest.mark.parametrize("module_name, class_name, constraint", _OPTIONAL_ADAPTERS)
def test_optional_adapter_raises_unsupported_device_on_mps(
    module_name, class_name, constraint
) -> None:
    """Each optional adapter raises ``UnsupportedDeviceError`` on device "mps".

    It does NOT silently fall back; the message names the constraint and points at the
    cloud-GPU deploy (spec/11). This proves the honest-stub contract WITHOUT any GPU —
    the raise precedes any model load, so no torch import happens.
    """
    adapter = _build(module_name, class_name)

    # ``info`` is cheap + license-tagged even though reconstruct refuses MPS (D2).
    assert adapter.info.weights_license, "weights_license must be populated (D2)"
    assert adapter.info.mps_capable is False, "optional adapters are not MPS-capable"

    req = ReconstructionRequest(video_path="(ignored)", max_frames=4, device="mps")
    with pytest.raises(UnsupportedDeviceError) as exc_info:
        adapter.reconstruct(req)

    msg = str(exc_info.value)
    assert constraint in msg, (
        f"{class_name} message must name the constraint {constraint!r}: {msg!r}"
    )
    # Points at the cloud-GPU deploy (spec/11) — the documented remedy.
    assert "spec/11" in msg, f"{class_name} message must point at spec/11: {msg!r}"
    # The stable error code is surfaced to the API (spec/06 §2.2).
    assert exc_info.value.code == "unsupported_device"


# --- UNMARKED: also refuse "cpu" (the other Mac-local device) ----------------------
@pytest.mark.parametrize("module_name, class_name, constraint", _OPTIONAL_ADAPTERS)
def test_optional_adapter_raises_unsupported_device_on_cpu(
    module_name, class_name, constraint
) -> None:
    """Each optional adapter also raises ``UnsupportedDeviceError`` on device "cpu"."""
    adapter = _build(module_name, class_name)
    req = ReconstructionRequest(video_path="(ignored)", max_frames=4, device="cpu")
    with pytest.raises(UnsupportedDeviceError) as exc_info:
        adapter.reconstruct(req)
    assert "spec/11" in str(exc_info.value)


# --- @pytest.mark.gpu: the on-(cloud)-device contract — SKIPPED on the Mac ----------
# Negative knowledge (spec/10 §5): no green MPS test for these dead ends. The gpu
# marker means these are collected + SKIPPED on a Mac with a reason naming the
# constraint, never errored. On a real CUDA box (where the marker is selected) they
# would exercise the actual model path (a future cloud-deploy task).
# Per-adapter skip reason NAMING the constraint (spec/10 §3 negative-knowledge gate).
_GPU_SKIP_REASON = {
    "spatialtracker_v2": "SpatialTrackerV2 is CUDA-only (upstream cu124 pin) — needs a CUDA GPU (spec/11)",
