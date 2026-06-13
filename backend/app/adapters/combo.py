"""Default combo adapter ``vggt+cotracker3`` (spec/06 §4.6) — the MAYAVIUS_ADAPTER default.

A small COMPOSING adapter (not a sixth model file) that orchestrates ``VggtAdapter``
(static + depth/camera) and ``CoTracker3Adapter`` (tracks), running VGGT ONCE and
feeding its depth/intrinsics/pose to the CoTracker3 lift. ``reconstruct`` returns a
**RAW** ``Scene4D`` (the static/dynamic split is done by ``assemble_scene4d``); the
core ``ReconstructionService`` then smooths/culls/caps it (spec/06 §5 steps 6-7).

LAZY IMPORTS (hexagonal / T-130, spec/06 §4): no torch import lives here — the
sub-adapters import their SDKs lazily INSIDE their own ``run_geometry`` / ``run_tracks``
(spec/06 §4). Importing this MODULE (for the registry / ``info``) never imports torch,
so the API can advertise the combo's capabilities with zero ML deps installed. The
sub-adapters (``VggtAdapter`` / ``CoTracker3Adapter``) are constructed LAZILY on first
use so even constructing the combo stays torch-free.

MPS DISCIPLINE (spec/08 §5, C3): the sub-adapters set
``PYTORCH_ENABLE_MPS_FALLBACK=1`` before importing torch, take the device from
``request.device`` (default "mps"), force FP32 (no autocast), and run under
``torch.no_grad()``. No torch tensor crosses the port — the combo returns numpy /
Python only.
"""

from __future__ import annotations

from app.core.domain.models import Scene4D
from app.core.ports.reconstruction_port import (
    AdapterInfo,
    ProgressSink,
    ReconstructionPort,
)
