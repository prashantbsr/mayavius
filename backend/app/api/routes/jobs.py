"""Job endpoints (async reconstruction model) — spec/06 §7. DRIVING side.

Upload returns a job id; the frontend polls/streams progress; the binary result is
fetched when ready (handover §4.4). The /result payload uses the MV4D v1 binary wire
format (spec/05). The ``JobQueue`` is injected via ``Depends(get_queue)`` (built once
in the lifespan, spec/06 §7).

Endpoint contract:
- POST   /jobs            202 + {job_id, status, poll, stream, result}   (415, 413)
- GET    /jobs/{id}       200 JSON status                                (404)
- GET    /jobs/{id}/stream  200 text/event-stream (SSE; fastapi.sse, C7) (404)
- GET    /jobs/{id}/result  200 octet-stream MV4D                        (404, 409)
"""

from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Response, UploadFile, status
from fastapi.sse import EventSourceResponse

from app.api.deps import get_queue
from app.config import UPLOAD_DIR, settings
from app.core.domain.models import ReconstructionRequest
from app.jobs.queue import JobQueue, JobStatus, job_to_json

router = APIRouter(prefix="/jobs", tags=["jobs"])

# Stream uploads in bounded chunks; the running total is checked after every chunk
# so an oversize clip is aborted long before it is fully buffered (spec/06 §8).
_CHUNK_BYTES = 1 << 20  # 1 MiB


async def save_capped_upload(clip: UploadFile, max_upload_mb: int) -> str:
    """Stream ``clip`` into ``UPLOAD_DIR`` in chunks, aborting with 413 once the
    running byte total exceeds ``max_upload_mb`` (spec/06 §8).

    Does NOT trust ``Content-Length`` — the cap is enforced on the bytes actually
    read. Returns the absolute saved path.
    """
    limit = max_upload_mb * 1024 * 1024
    suffix = Path(clip.filename or "").suffix
    dest = UPLOAD_DIR / f"{uuid.uuid4().hex}{suffix}"

    total = 0
    with dest.open("wb") as out:
        while True:
            chunk = await clip.read(_CHUNK_BYTES)
            if not chunk:
                break
            total += len(chunk)
            if total > limit:
                out.close()
                dest.unlink(missing_ok=True)
                raise HTTPException(
                    413, f"upload exceeds MAX_UPLOAD_MB={max_upload_mb}"
                )
            out.write(chunk)

    return str(dest)


@router.post("", status_code=status.HTTP_202_ACCEPTED)
async def submit_job(
    clip: UploadFile, queue: JobQueue = Depends(get_queue)
) -> dict:
    """Accept a short ``video/*`` clip, enqueue a reconstruction job, return a job id."""
    if not (clip.content_type or "").startswith("video/"):
        raise HTTPException(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, "expected a video/* upload"
        )
    path = await save_capped_upload(clip, settings.max_upload_mb)
    req = ReconstructionRequest(
        video_path=path,
        max_frames=min(settings.max_clip_frames, 64),  # clamp at construction (06 §3)
        target_fps=settings.target_fps,
        device=settings.device,
    )
    job_id = await queue.submit(path, req)
    return {
        "job_id": job_id,
        "status": "queued",
        "poll": f"/jobs/{job_id}",
        "stream": f"/jobs/{job_id}/stream",
        "result": f"/jobs/{job_id}/result",
    }


@router.get("/{job_id}")
async def get_job(job_id: str, queue: JobQueue = Depends(get_queue)) -> dict:
    """Return job status/progress (polled by the frontend)."""
    try:
        return job_to_json(queue.status(job_id))
    except KeyError:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "unknown job id")


@router.get("/{job_id}/stream", response_class=EventSourceResponse)
async def stream_job(job_id: str, queue: JobQueue = Depends(get_queue)):
    """Server-Sent Events progress stream (spec/06 §7). 404 BEFORE streaming starts.

    A generator path operation: each yielded ``ServerSentEvent`` is encoded by
    ``EventSourceResponse``. The terminal event has ``event`` in {done, failed}.
    """
    try:
        queue.status(job_id)  # 404 before the stream starts
    except KeyError:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "unknown job id")
    async for event in queue.events(job_id):
        yield event


@router.get("/{job_id}/result")
async def get_result(
    job_id: str, queue: JobQueue = Depends(get_queue)
) -> Response:
    """Return the immutable MV4D blob (spec/05) when the job is DONE."""
    try:
        job = queue.status(job_id)
    except KeyError:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "unknown job id")

    if job.status is not JobStatus.DONE:
        if job.status is JobStatus.FAILED:
            code = (job.error or {}).get("code", "reconstruction_error")
            raise HTTPException(
                status.HTTP_409_CONFLICT, {"code": code, "error": job.error}
            )
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            {"code": "not_ready", "status": job.status.value},
        )

    return Response(
        content=queue.result(job_id),
        media_type="application/octet-stream",
        headers={
            "Cache-Control": "public, max-age=31536000, immutable",
            "ETag": chr(34) + job_id + chr(34),
            # spec/06 §7 headers table — inline, named after the job for shareable saves.
            "Content-Disposition": f'inline; filename="{job_id}.mv4d"',
        },
    )
