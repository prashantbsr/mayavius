"""Pure orchestration over the ReconstructionPort.

No FastAPI, no torch, no concrete-adapter imports here — that is the hexagonal
boundary (handover §3, §6). The adapter is injected.
"""

from __future__ import annotations

from app.core.domain.models import ReconstructionRequest, ReconstructionResult
from app.core.ports.reconstruction_port import ReconstructionPort


class ReconstructionService:
    def __init__(self, adapter: ReconstructionPort) -> None:
        self._adapter = adapter

    def run(self, request: ReconstructionRequest) -> ReconstructionResult:
        # TODO(spec/06): enforce clip caps, invoke adapter, post-process
        # (temporal smoothing, confidence culling, static/dynamic split).
        return self._adapter.reconstruct(request)
