"""Small SSE helpers (DRIVING side). spec/06 §7.

The streaming route uses ``fastapi.sse`` (C7 — NOT ``sse-starlette``): a generator
path operation declared with ``response_class=EventSourceResponse`` yields
``ServerSentEvent``s and FastAPI encodes them. The ``JobQueue`` already builds those
events (it owns the ``job_to_json`` payload + the ``event=<status>`` naming), so this
module just re-exports the markers for a single import site and offers a thin builder
for any ad-hoc event a route needs.
"""

from __future__ import annotations

from fastapi.sse import EventSourceResponse, ServerSentEvent

from app.jobs.queue import Job, job_to_json

__all__ = ["EventSourceResponse", "ServerSentEvent", "job_event"]


def job_event(job: Job) -> ServerSentEvent:
    """Build a ``ServerSentEvent`` for a job's current state.

    ``data`` is the poll JSON dict (fastapi.sse JSON-encodes it once — do not
    pre-``json.dumps``); ``event`` is the job's status name so the browser dispatches
    it to ``addEventListener(<status>, ...)`` (spec/06 §7).
    """
    return ServerSentEvent(data=job_to_json(job), event=job.status.value)
