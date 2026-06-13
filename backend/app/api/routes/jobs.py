"""Job endpoints (async reconstruction model).

Upload returns a job id; the frontend polls/streams progress; the binary result
is fetched when ready (handover §4.4). All handlers are scaffolding stubs
returning 501 until implemented per spec/06-backend-spec.md. The /result payload
uses the binary wire format defined in spec/05-data-contract.md.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, UploadFile, status

router = APIRouter(prefix="/jobs", tags=["jobs"])

_NOT_IMPLEMENTED = "Not implemented — see spec/06-backend-spec.md"


@router.post("", status_code=status.HTTP_202_ACCEPTED)
async def submit_job(clip: UploadFile) -> dict[str, str]:
    """Accept a short clip, enqueue a reconstruction job, return a job id."""
    del clip  # consumed once implemented
    raise HTTPException(status.HTTP_501_NOT_IMPLEMENTED, _NOT_IMPLEMENTED)


@router.get("/{job_id}")
async def get_job(job_id: str) -> dict[str, str]:
    """Return job status/progress (polled or streamed by the frontend)."""
    del job_id
    raise HTTPException(status.HTTP_501_NOT_IMPLEMENTED, _NOT_IMPLEMENTED)


@router.get("/{job_id}/result")
async def get_result(job_id: str) -> dict[str, str]:
    """Return the binary reconstruction payload when ready (spec/05)."""
    del job_id
    raise HTTPException(status.HTTP_501_NOT_IMPLEMENTED, _NOT_IMPLEMENTED)
