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
)

# A small default frame count; the honored T is min(request.max_frames, this) so a
# decoded result always reports frameCount <= 64 (T-306). Small + deterministic.
_DEFAULT_FRAMES = 6

# Per-step dwell (seconds) between the two progress emissions. The fixture work is
# near-instant; without a dwell a live UPLOAD job reaches `done` BEFORE a navigating
# browser opens its EventSource (measured subscribe latency ≈ 1s: hydrate +
# dynamic-import the R3F viewer bundle, then the useLoadScene effect opens the
# stream). Per the spec/06 §6 late-subscriber contract (already-terminal → emit the
# terminal event once + return), a job that finished first never replays its
# intermediate 0.25/0.75 progression, so the client's viewerStore.progress would
# only ever hold {0, 1} and e2e T-402 ("observed ≥1 value 0<p<1 over the run") is
# unsatisfiable.
#
# This dwell keeps a live job in `running` long enough (~2·delay) that a real client
# subscribes mid-run and the (now monotonic — see JobQueue.events) late-subscriber
# path delivers the running progression before `done`. It changes NO status
# semantics and NO terminal contract — it only SPACES the two progress emissions
# this adapter already promises, and runs inside the worker's thread-pool executor
# (spec/06 §6) so it never blocks the event loop or SSE delivery. ONLY the fake
# adapter path — the real model adapters are untouched. Default 0.6s → ~1.2s running
# window (> the ~1s subscribe latency, with margin); far inside every timeout
# (T-303 polls 10s, T-308 drains the stream, e2e polls 30s). It is paid only by an
# actual upload job (seeded examples are pre-baked and never call reconstruct), and
# the unit-test JobQueue path uses the test-only FakeAdapter (no dwell). Env-tunable
# via MAYAVIUS_FIXTURE_STEP_DELAY_S (set 0 to disable).
_STEP_DELAY_S = float(os.environ.get("MAYAVIUS_FIXTURE_STEP_DELAY_S", "0.6"))


class FixtureAdapter(ReconstructionPort):
    """Deterministic, torch-free fixture reconstructor (registry id ``fake``)."""

    name = "fake"

    def __init__(self, settings=None) -> None:
        # The registry factory passes ``settings``; the fixture ignores it (no ML).
        self._settings = settings

    @property
    def info(self) -> AdapterInfo:
        return AdapterInfo(
            name="fake",
            produces_tracks=True,
            dynamic=True,
            mps_capable=True,
            weights_license="none",
            default_weights="(fixture)",
        )

    def reconstruct(
        self, request, progress: ProgressSink | None = None
    ) -> Scene4D:
        # Emit a non-terminal running progression BEFORE returning (T-303), spaced
        # by a small dwell so a navigating browser subscribes mid-run and observes
        # 0<p<1 (e2e T-402). The dwell runs in the worker's executor thread
        # (spec/06 §6) — it never blocks the event loop or SSE delivery.
        if progress is not None:
            progress(0.25, "decode")
