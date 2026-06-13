"""FastAPI dependency wiring (DRIVING side).

The ``JobQueue`` is built once in the lifespan and stashed on ``app.state.queue``
(spec/06 §7). Handlers receive it via ``Depends(get_queue)`` so the route code never
reaches into app state directly.
"""

from __future__ import annotations

from fastapi import Request

from app.jobs.queue import JobQueue


def get_queue(request: Request) -> JobQueue:
    """Return the process-wide ``JobQueue`` built in the lifespan."""
    return request.app.state.queue
