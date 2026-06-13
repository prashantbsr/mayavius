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
            produces_tracks=True,
            dynamic=True,
            mps_capable=True,
            weights_license="cc-by-nc-4.0",
            default_weights="(fake)",
        )

    def reconstruct(self, request, progress: ProgressSink | None = None) -> Scene4D:
        if progress is not None:
            progress(0.25, "decode")
            progress(0.75, "assemble")

        # --- THE GOLDEN SCENE: T=3, fps=24 ---
        static_positions = np.array(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0.5, 0.5, 0.5]], dtype=np.float32
        )
        static_colors = np.array(
            [[255, 0, 0], [0, 255, 0], [0, 0, 255], [128, 128, 128]], dtype=np.uint8
        )
        # All conf > 127 so the default conf cull (thresh 0.5 -> 127.5) keeps every point.
        static_conf = np.array([200, 180, 160, 255], dtype=np.uint8)

        dynamic_positions = [
            np.array([[0.2, 0.2, 0.2], [0.8, 0.2, 0.2]], dtype=np.float32),  # frame 0
            np.zeros((0, 3), dtype=np.float32),                              # frame 1 (empty)
            np.array([[0.5, 0.9, 0.1]], dtype=np.float32),                   # frame 2
        ]
        dynamic_colors = [
            np.array([[10, 20, 30], [40, 50, 60]], dtype=np.uint8),
            np.zeros((0, 3), dtype=np.uint8),
