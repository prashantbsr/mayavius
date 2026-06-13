"""Tests for the async job queue + background worker (spec/06 §6).

Drives the in-process `JobQueue` with `asyncio` (a fresh event loop per test, via
`run_until_complete`) over the torch-free `FakeAdapter` + the real
`ReconstructionService`, so the full lifecycle — submit → RUNNING → encode → write
the immutable MV4D blob → DONE — is exercised with zero ML deps.

Success is asserted by a BOUNDED poll/await loop (no fixed sleeps as the success
condition): we `await asyncio.sleep(0)` to yield to the worker and re-check, up to a
generous deadline, then fail loudly if the job never reaches a terminal state.
"""

from __future__ import annotations

import asyncio

import pytest

from app.core.domain.errors import InferenceError
from app.core.domain.models import (
    CameraTrack,
    ReconstructionRequest,
    Scene4D,
    Tracks,
)
from app.core.ports.reconstruction_port import AdapterInfo, ReconstructionPort
from app.core.services.reconstruction_service import ReconstructionService
from app.jobs.queue import Job, JobQueue, JobStatus, job_to_json
from tests.fakes.fake_adapter import FakeAdapter

import numpy as np

# MV4D v1 magic — the first 4 bytes of every encoded blob (spec/05 §3.1).
_MV4D_MAGIC = b"MV4D"

# Bounded-wait knobs: yield to the worker up to this many times before declaring
# the job hung. Each tick is an `await asyncio.sleep(0)` (a cooperative yield, NOT a
# wall-clock delay), so this is a step bound, not a timeout disguised as a sleep.
_MAX_TICKS = 2000


def _request() -> ReconstructionRequest:
    return ReconstructionRequest(video_path="/tmp/does-not-matter.mp4", max_frames=24)


async def _drive_to_terminal(queue: JobQueue, job_id: str) -> Job:
    """Yield to the worker until the job is terminal; fail (don't hang) if it never is."""
    for _ in range(_MAX_TICKS):
        job = queue.status(job_id)
        if job.status in (JobStatus.DONE, JobStatus.FAILED):
            return job
        await asyncio.sleep(0)  # cooperative yield, lets the worker + executor advance
    raise AssertionError(f"job {job_id} never reached a terminal state")


def _make_queue(tmp_path, adapter: ReconstructionPort | None = None) -> JobQueue:
    service = ReconstructionService(adapter or FakeAdapter())
    return JobQueue(service, str(tmp_path))


