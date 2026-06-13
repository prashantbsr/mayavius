"""FixtureAdapter — the no-ML "fake" fixture mode (spec/06 §4.6).

A PRODUCTION (non-test) ``ReconstructionPort`` that imports **no torch** and
returns a deterministic, hand-authored ``Scene4D`` exercising all four sections —
static + dynamic (incl. an empty frame) + tracks (mixed visibility) + cameras — so
the full API + viewer are exercisable with zero ML deps. Waves 1-2 and all
Playwright e2e (spec/09, spec/10 §4) run with ``MAYAVIUS_ADAPTER=fake``.

This is DISTINCT from the unit-test ``FakeAdapter`` (``backend/tests/fakes/``):
``FakeAdapter`` is test-only; ``FixtureAdapter`` is importable in production.

It MUST emit ``progress(0.25, "decode")`` then ``progress(0.75, "assemble")``
BEFORE returning so the job deterministically passes through a non-terminal
``running`` state with monotonic progress (T-303), even though the work is
near-instant. In W1 there is no committed MV4D blob, so the scene is built inline
(no blob load).
"""

from __future__ import annotations

import os
import time

import numpy as np

from app.core.domain.models import CameraTrack, Scene4D, Tracks
from app.core.ports.reconstruction_port import (
    AdapterInfo,
    ProgressSink,
    ReconstructionPort,
