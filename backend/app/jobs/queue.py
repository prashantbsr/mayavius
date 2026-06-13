"""Async job queue (driving-side adapter — NOT core).

Upload returns a job id; the frontend polls/streams progress; the binary result
is fetched when ready (handover §4.4). MVP: an in-process async queue is enough
for local single-user dev; a durable/distributed queue is an optional deployment
concern (spec/11-deployment-and-launch.md). Finalized in spec/06-backend-spec.md.
"""

from __future__ import annotations

# TODO(spec/06): JobQueue with submit()/status()/result() plus a background
# worker that runs ReconstructionService off the request thread and streams
# progress frames so the cloud appears progressively (handover §4.4).
