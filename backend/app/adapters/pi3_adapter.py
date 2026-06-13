"""Pi3 / π³ adapter — ICLR 2026. Fast feedforward point cloud + camera, STATIC.

NEGATIVE KNOWLEDGE (handover §5): no official MPS path → unsuitable as the Mac
default; weights are non-commercial / research-only. Kept as a swappable adapter
for a CUDA/deploy path. Verify and pin in spec/08-dependencies-and-env.md.
"""

from __future__ import annotations

from app.core.domain.models import ReconstructionRequest, ReconstructionResult
from app.core.ports.reconstruction_port import ReconstructionPort


class Pi3Adapter(ReconstructionPort):
    name = "pi3"

    def reconstruct(self, request: ReconstructionRequest) -> ReconstructionResult:
        del request
        raise NotImplementedError("Pi3Adapter: see spec/06-backend-spec.md")
