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

