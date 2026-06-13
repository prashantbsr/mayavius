"""Async job queue + in-process background worker (driving-side adapter — NOT core).

Upload returns a job id; the frontend polls/streams progress; the binary result is
fetched when ready (handover §4.4). MVP: an in-process `asyncio`-driven queue is
enough for local single-user dev; a durable/distributed queue (Redis/RQ) is an
optional deploy concern (spec/11) — the port boundary makes that swap
non-architectural. Finalized in spec/06-backend-spec.md §6.

DRIVING side: this module MAY import the core `ReconstructionService`, the wire
encoder, `fastapi.sse.ServerSentEvent`, asyncio, and uuid. The adapter is NOT
imported here — it is injected into the service (hexagonal boundary, T-130).

Worker flow (spec/06 §6): `submit` creates a `Job` and schedules `_run` on the
running loop. `_run` flips the job to RUNNING, builds a progress closure (called
FROM THE EXECUTOR THREAD — pushed onto the SSE queue race-free via
`loop.call_soon_threadsafe`), runs the synchronous/heavy `service.run` inside
`loop.run_in_executor` so the event loop (and SSE) stays responsive, then encodes +
writes the immutable MV4D blob (spec/05 §4) and pushes a TERMINAL event. A broad
`except Exception` backstop guarantees a fault never leaves the job hanging in
RUNNING (which would also hang an awaiting SSE client).
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from fastapi.sse import ServerSentEvent

from app.core.domain.errors import ReconstructionError
from app.core.domain.models import ReconstructionRequest
from app.core.services.reconstruction_service import ReconstructionService
from app.wire.encoder import encode_reconstruction


class JobStatus(str, Enum):
    """Lifecycle states of a reconstruction job (spec/06 §6)."""

    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


_TERMINAL = (JobStatus.DONE, JobStatus.FAILED)


@dataclass
class Job:
    """One reconstruction job. `_events` is the per-job SSE fan-out queue."""

    id: str
    status: JobStatus = JobStatus.QUEUED
    progress: float = 0.0                 # 0..1
    stage: str = "queued"
    result_path: str | None = None        # set when DONE
    bytes_len: int | None = None
    adapter_id: str = ""
    weights_license: str = ""
    error: dict | None = None             # {code, message} when FAILED
    _events: asyncio.Queue = field(default_factory=asyncio.Queue)  # for SSE


def job_to_json(job: Job) -> dict:
    """The wire JSON the frontend parses (from both poll and SSE `data`) — spec/06 §6.

    The six base keys are ALWAYS present (on `queued`: progress=0.0, stage="queued").
    `result` (a relative URL string) appears iff DONE; `error` ({code, message})
    appears iff FAILED. snake_case.
    """
    d = {
        "id": job.id,
        "status": job.status.value,
        "progress": job.progress,
        "stage": job.stage,
        "adapter_id": job.adapter_id,
        "weights_license": job.weights_license,
    }
    if job.status is JobStatus.DONE:
        d["result"] = f"/jobs/{job.id}/result"
    if job.status is JobStatus.FAILED:
        d["error"] = job.error
    return d


class JobQueue:
    """In-process async job queue + background worker (spec/06 §6)."""

    def __init__(self, service: ReconstructionService, result_dir: str) -> None:
        self._service = service
        self._result_dir = result_dir
        self._jobs: dict[str, Job] = {}

    async def submit(self, video_path: str, request: ReconstructionRequest) -> str:
        """Create a job and schedule the background worker on the running loop.

        Returns the new job id. The heavy work runs off the request path (the
        worker reschedules itself via `create_task` so `submit` returns instantly).
        """
        job = Job(id=uuid.uuid4().hex)
        self._jobs[job.id] = job
        loop = asyncio.get_running_loop()
        loop.create_task(self._run(job, video_path, request))
        return job.id

    async def _run(
        self, job: Job, video_path: str, request: ReconstructionRequest
    ) -> None:
        """Background worker: RUNNING → service.run (in executor) → DONE/FAILED.

        The broad `except Exception` is the backstop that GUARANTEES the job never
        hangs in RUNNING — even on an unwrapped torch/MPS fault — so an awaiting SSE
        client always sees a terminal event and closes.
        """
        loop = asyncio.get_running_loop()

        job.status = JobStatus.RUNNING
        job.progress = 0.0
        job.stage = "running"
        await job._events.put(
            ServerSentEvent(data=job_to_json(job), event=JobStatus.RUNNING.value)
        )

        def progress(p: float, stage: str) -> None:
            # CALLED FROM THE EXECUTOR THREAD. Mutating the job and putting the
            # event onto the asyncio.Queue must happen ON THE LOOP THREAD, so we
            # marshal back via call_soon_threadsafe (race-free; no cross-thread
            # asyncio.Queue mutation).
            def _apply() -> None:
                job.progress = p
                job.stage = stage
                job._events.put_nowait(
                    ServerSentEvent(
                        data=job_to_json(job), event=JobStatus.RUNNING.value
                    )
                )

            loop.call_soon_threadsafe(_apply)

        try:
            scene = await loop.run_in_executor(
                None, lambda: self._service.run(request, progress)
            )
        except ReconstructionError as err:
            job.status = JobStatus.FAILED
            job.error = {"code": err.code, "message": str(err)}
            await job._events.put(
                ServerSentEvent(data=job_to_json(job), event=JobStatus.FAILED.value)
            )
            return
        except Exception as exc:  # backstop: a bug / torch / MPS fault never hangs RUNNING
            job.status = JobStatus.FAILED
            job.error = {"code": "inference_failed", "message": str(exc)}
            await job._events.put(
                ServerSentEvent(data=job_to_json(job), event=JobStatus.FAILED.value)
            )
            return

        # --- success: encode + write the immutable MV4D blob (spec/05 §4) ---
        blob = encode_reconstruction(scene)
        result_path = str(Path(self._result_dir) / f"{job.id}.mv4d")
        Path(result_path).write_bytes(blob)

        job.result_path = result_path
        job.bytes_len = len(blob)
        job.adapter_id = scene.adapter_id
        job.weights_license = scene.weights_license
        job.status = JobStatus.DONE
        job.progress = 1.0
        job.stage = "done"
        await job._events.put(
            ServerSentEvent(data=job_to_json(job), event=JobStatus.DONE.value)
        )

    def status(self, job_id: str) -> Job:
        """Return the job; raises KeyError on an unknown id (caller maps 404)."""
        return self._jobs[job_id]

    def result(self, job_id: str) -> bytes:
        """Return the MV4D blob bytes; raises if the job is not DONE (caller maps 409)."""
        job = self._jobs[job_id]
        if job.status is not JobStatus.DONE or job.result_path is None:
            raise RuntimeError(f"job {job_id!r} is not done (status={job.status.value})")
        return Path(job.result_path).read_bytes()

    async def events(self, job_id: str):
        """Async generator of SSE events for `job_id` (spec/06 §6 late-subscriber).

        Emits the CURRENT job state once. If the job is ALREADY terminal
        (done/failed — e.g. a seeded example or a job that finished before the
        client connected) it yields that single terminal event and returns
        immediately (no queue wait, so a late subscriber never blocks). Otherwise it
        awaits `job._events`, yielding each event, until a terminal event is yielded,
        then returns.

        Monotonic coalescing (bug fix): because the generator yields the CURRENT
        state first and the per-job queue may still hold the EARLIER buffered
        running events (e.g. the initial `progress=0.0` pushed at RUNNING-start), a
        mid-run subscriber could otherwise see a stale lower progress replayed AFTER
        the higher current value — a non-monotonic stream (broke T-308 once the job
        ran long enough to be subscribed mid-flight). We drop any buffered
        non-terminal event whose progress is below the highest already-yielded
        progress, so the stream a client sees is always monotonically
        non-decreasing. Terminal events (done/failed) are NEVER dropped. This
        preserves the spec/06 §6 contract (emit current state, then drain to a
        terminal event) and is the SSE analogue of the poll loop's monotonicity.
        """
        job = self._jobs[job_id]

        yield ServerSentEvent(data=job_to_json(job), event=job.status.value)
        if job.status in _TERMINAL:
            return

        highest = job.progress  # progress already surfaced by the current-state yield
        while True:
            event = await job._events.get()
            is_terminal = event.event in (JobStatus.DONE.value, JobStatus.FAILED.value)
            # Coalesce stale buffered running events (keep the stream monotonic);
            # never drop a terminal event.
            if not is_terminal:
                p = event.data.get("progress") if isinstance(event.data, dict) else None
                if isinstance(p, (int, float)):
                    if p < highest:
                        continue
                    highest = p
            yield event
            if is_terminal:
                return

    def seed_example(self, slug: str, mv4d_path: str) -> None:
        """Register a pre-baked MV4D blob as a terminal DONE job under id == slug.

        The in-process queue is volatile, so example results are seeded at startup
        (never reconstructed at view time). The example `slug` shares the job-id
        space, so `GET /jobs/<slug>` + `/result` resolve identically after every boot
        (spec/06 §6).
        """
        self._jobs[slug] = Job(
            id=slug,
            status=JobStatus.DONE,
            progress=1.0,
            stage="done",
            result_path=mv4d_path,
            adapter_id="fixture",
            weights_license="none",
        )
