"""A deterministic, torch-free `ReconstructionPort` for testing the core (spec/10 §1.2).

`FakeAdapter` returns a small, WITHIN-CAPS, ALREADY-SPLIT `Scene4D` exercising all
four sections — static + dynamic (incl. an empty frame) + tracks (mixed visibility)
+ cameras — structurally identical in spirit to THE GOLDEN SCENE (W0.T1 brief /
spec/10 §2) so it also encodes cleanly later (chains into T-100).

This is the unit-test fake (`backend/tests/fakes/`), distinct from the production
`FixtureAdapter` (`app/adapters/fixture_adapter.py`). NO torch, NO weights.
"""

from __future__ import annotations

import numpy as np

from app.core.domain.models import CameraTrack, Scene4D, Tracks
from app.core.ports.reconstruction_port import (
    AdapterInfo,
    ProgressSink,
    ReconstructionPort,
)


class FakeAdapter(ReconstructionPort):
    name = "fake"

    @property
    def info(self) -> AdapterInfo:
        return AdapterInfo(
            name="fake",
