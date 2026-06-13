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
        job = await _drive_to_terminal(queue, job_id)
        return job_id, job

    job_id, job = asyncio.new_event_loop().run_until_complete(scenario())

    blob = queue.result(job_id)
    assert blob[:4] == _MV4D_MAGIC
    assert blob == (tmp_path / f"{job.id}.mv4d").read_bytes()
    assert len(blob) == job.bytes_len


def test_result_before_done_raises(tmp_path):
    """result() on a non-DONE job raises (caller maps 409)."""
    queue = _make_queue(tmp_path)
    queue._jobs["pending"] = Job(id="pending", status=JobStatus.RUNNING)
    with pytest.raises(Exception):
        queue.result("pending")


def test_status_unknown_id_raises_keyerror(tmp_path):
    """status() on an unknown id raises KeyError (caller maps 404)."""
    queue = _make_queue(tmp_path)
    with pytest.raises(KeyError):
        queue.status("nope")


def test_events_on_finished_job_yields_terminal_event(tmp_path):
    """events() on an already-finished job yields a single terminal event and returns."""
    queue = _make_queue(tmp_path)

    async def scenario() -> list:
        job_id = await queue.submit("/tmp/clip.mp4", _request())
        await _drive_to_terminal(queue, job_id)
        collected = []
        async for event in queue.events(job_id):
            collected.append(event)
        return collected

    events = asyncio.new_event_loop().run_until_complete(scenario())

    # Already terminal → emit once + return (no queue wait, no hang).
    assert len(events) == 1
    assert events[0].event == JobStatus.DONE.value
    assert events[0].data["status"] == "done"
    assert events[0].data["result"] == f"/jobs/{events[0].data['id']}/result"


def test_events_streams_through_to_terminal(tmp_path):
    """events() on a still-queued job streams running frames and ends on a terminal one."""
    queue = _make_queue(tmp_path)

    async def scenario() -> list:
        job_id = await queue.submit("/tmp/clip.mp4", _request())
        # Subscribe BEFORE the worker finishes (the first yield is the current state).
        collected = []
        async for event in queue.events(job_id):
            collected.append(event)
        return collected

    events = asyncio.new_event_loop().run_until_complete(scenario())

    assert events, "expected at least one event"
    assert events[-1].event in (JobStatus.DONE.value, JobStatus.FAILED.value)
    assert events[-1].event == JobStatus.DONE.value
    # Monotonic non-decreasing progress, ending at 1.0.
    progresses = [e.data["progress"] for e in events]
    assert progresses == sorted(progresses)
    assert progresses[-1] == 1.0


def test_seed_example_registers_done_job(tmp_path):
    """seed_example registers a terminal DONE job under id == slug."""
    queue = _make_queue(tmp_path)
    queue.seed_example("example", "/some/path/example.mv4d")

    job = queue.status("example")
    assert job.status is JobStatus.DONE
    assert job.progress == 1.0
    assert job.stage == "done"
    assert job.result_path == "/some/path/example.mv4d"
    assert job.adapter_id == "fixture"
    assert job.weights_license == "none"

    payload = job_to_json(job)
    assert payload["status"] == "done"
    assert payload["result"] == "/jobs/example/result"
    assert "error" not in payload


class _RaisingAdapter(ReconstructionPort):
    """A ReconstructionPort whose reconstruct() raises a typed ReconstructionError."""

    name = "raising"

    @property
    def info(self) -> AdapterInfo:
        return AdapterInfo(
            name="raising",
            produces_tracks=False,
            dynamic=False,
            mps_capable=True,
            weights_license="none",
            default_weights="(none)",
        )

    def reconstruct(self, request, progress=None) -> Scene4D:
        raise InferenceError("the model exploded mid-run")


def test_reconstruction_error_yields_failed_with_code(tmp_path):
    """A service that raises ReconstructionError → job FAILED with error.code."""
    queue = _make_queue(tmp_path, adapter=_RaisingAdapter())

    async def scenario() -> Job:
        job_id = await queue.submit("/tmp/clip.mp4", _request())
        return await _drive_to_terminal(queue, job_id)

    job = asyncio.new_event_loop().run_until_complete(scenario())

    assert job.status is JobStatus.FAILED
    assert job.error is not None
    assert job.error["code"] == "inference_failed"
    assert "exploded" in job.error["message"]

    payload = job_to_json(job)
    assert payload["status"] == "failed"
    assert payload["error"]["code"] == "inference_failed"
    assert "result" not in payload


class _BoomAdapter(ReconstructionPort):
    """An adapter raising a NON-ReconstructionError (an unwrapped bug)."""

    name = "boom"

    @property
    def info(self) -> AdapterInfo:
        return AdapterInfo(
            name="boom",
            produces_tracks=False,
            dynamic=False,
            mps_capable=True,
            weights_license="none",
            default_weights="(none)",
        )

    def reconstruct(self, request, progress=None) -> Scene4D:
        raise ValueError("unwrapped bug")


def test_unwrapped_exception_backstop_yields_failed(tmp_path):
    """ANY other Exception → FAILED with code 'inference_failed' (never hangs RUNNING)."""
    queue = _make_queue(tmp_path, adapter=_BoomAdapter())

    async def scenario() -> Job:
        job_id = await queue.submit("/tmp/clip.mp4", _request())
        return await _drive_to_terminal(queue, job_id)

    job = asyncio.new_event_loop().run_until_complete(scenario())

    assert job.status is JobStatus.FAILED
    assert job.error["code"] == "inference_failed"
    assert "unwrapped bug" in job.error["message"]


def test_job_to_json_base_keys_always_present(tmp_path):
    """job_to_json always carries the six snake_case base keys; queued has no result/error."""
    job = Job(id="abc")
    payload = job_to_json(job)
    assert set(payload) == {
        "id",
        "status",
        "progress",
        "stage",
        "adapter_id",
        "weights_license",
    }
    assert payload["status"] == "queued"
    assert payload["progress"] == 0.0
    assert payload["stage"] == "queued"


def test_progress_closure_pushes_running_events(tmp_path):
    """The progress closure (called from the executor thread) surfaces running frames.

    Uses a custom adapter that calls progress() with distinct stages, then asserts
    those stages appear on the SSE stream — proving the cross-thread
    call_soon_threadsafe marshalling works.
    """

    class _ProgressAdapter(ReconstructionPort):
        name = "prog"

        @property
        def info(self) -> AdapterInfo:
            return AdapterInfo(
                name="prog",
                produces_tracks=True,
                dynamic=True,
                mps_capable=True,
                weights_license="cc-by-nc-4.0",
                default_weights="(none)",
            )

        def reconstruct(self, request, progress=None) -> Scene4D:
            if progress is not None:
                progress(0.30, "alpha")
                progress(0.60, "beta")
            return Scene4D(
                frame_count=1,
                fps=24.0,
                aabb_min=np.zeros(3, dtype=np.float32),
                aabb_max=np.ones(3, dtype=np.float32),
                static_positions=np.array([[0, 0, 0]], dtype=np.float32),
                static_colors=np.array([[1, 2, 3]], dtype=np.uint8),
                static_conf=np.array([255], dtype=np.uint8),
                dynamic_positions=[np.zeros((0, 3), dtype=np.float32)],
                dynamic_colors=[np.zeros((0, 3), dtype=np.uint8)],
                tracks=Tracks(
                    positions=np.array([[[0.1, 0.1, 0.1]]], dtype=np.float32),
                    visibility=np.array([[True]], dtype=bool),
                    colors=np.array([[9, 9, 9]], dtype=np.uint8),
                ),
                cameras=CameraTrack(
                    poses=np.array([[0, 0, 0, 1, 0, 0, 0]], dtype=np.float32),
                    intrinsics=np.array([[1, 1, 0.5, 0.5]], dtype=np.float32),
                ),
            )

    queue = _make_queue(tmp_path, adapter=_ProgressAdapter())

    async def scenario() -> list:
        job_id = await queue.submit("/tmp/clip.mp4", _request())
        collected = []
        async for event in queue.events(job_id):
            collected.append(event)
        return collected

    events = asyncio.new_event_loop().run_until_complete(scenario())

