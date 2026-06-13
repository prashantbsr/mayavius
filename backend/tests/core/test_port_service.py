"""T-120 / T-121 / T-122 — port, service, and FakeAdapter contract (spec/10 §1.2).

These exercise the pure core with a deterministic, torch-free `FakeAdapter` — no
weights, no ML deps. The encoder is deliberately NOT imported here (that chains into
T-100, the wire-format suite).
"""

from __future__ import annotations

import numpy as np

from app.core.domain.models import ReconstructionRequest, Scene4D, Tracks
from app.core.ports.reconstruction_port import ReconstructionPort
from app.core.services.reconstruction_service import ReconstructionService

from tests.fakes.fake_adapter import FakeAdapter

# MV4D caps (spec/05 §4) — assert the service's output is within them.
_MAX_STATIC = 150_000
_MAX_DYNAMIC_PER_FRAME = 20_000
_MAX_TRACKS = 4_096
_MAX_FRAMES = 64


def _request() -> ReconstructionRequest:
    return ReconstructionRequest(video_path="/tmp/fake.mp4", max_frames=24, target_fps=12.0)


def test_service_delegates_to_port() -> None:
    """T-120 — the service runs the adapter then applies core post-processing."""
