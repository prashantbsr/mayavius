"""CoTracker3 adapter — point tracking, lifted to 3D with depth.

Pairs with a static reconstructor (VGGT/Pi3) to produce the D4RT-style
'cloud + coloured tracks' look. Part of the pragmatic MVP recipe (handover §5).
Pin repo IDs / weights in spec/08-dependencies-and-env.md.
"""

from __future__ import annotations

from app.core.domain.models import ReconstructionRequest, ReconstructionResult
from app.core.ports.reconstruction_port import ReconstructionPort


class CoTracker3Adapter(ReconstructionPort):
    name = "cotracker3"

    def reconstruct(self, request: ReconstructionRequest) -> ReconstructionResult:
        del request
        raise NotImplementedError("CoTracker3Adapter: see spec/06-backend-spec.md")
