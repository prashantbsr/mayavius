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
