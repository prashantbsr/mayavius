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


def test_submit_runs_to_done_and_writes_mv4d_blob(tmp_path):
    """submit → status reaches DONE with progress 1.0 and a written .mv4d (MV4D magic)."""
    queue = _make_queue(tmp_path)

    async def scenario() -> Job:
        job_id = await queue.submit("/tmp/clip.mp4", _request())
        assert isinstance(job_id, str) and job_id
        return await _drive_to_terminal(queue, job_id)

    job = asyncio.new_event_loop().run_until_complete(scenario())

    assert job.status is JobStatus.DONE
    assert job.progress == 1.0
    assert job.stage == "done"
    assert job.adapter_id == "fake"
    assert job.weights_license == "cc-by-nc-4.0"

    # The blob was written to result_dir/<id>.mv4d and starts with the MV4D magic.
    assert job.result_path is not None
    blob = (tmp_path / f"{job.id}.mv4d").read_bytes()
    assert blob[:4] == _MV4D_MAGIC
    assert job.bytes_len == len(blob)


def test_result_returns_the_written_bytes(tmp_path):
    """result() returns exactly the bytes written, whose first 4 bytes are MV4D."""
    queue = _make_queue(tmp_path)

    async def scenario() -> tuple[str, Job]:
        job_id = await queue.submit("/tmp/clip.mp4", _request())
