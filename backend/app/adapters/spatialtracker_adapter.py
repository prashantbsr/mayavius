"""SpatialTrackerV2 adapter — produces dynamic 3D tracks + geometry in ONE model.

Slower (iterative refinement): fine for short-clip dev on Mac, heavy for
interactive use. The single-model route to the D4RT 'cloud + tracks' look (vs.
static-reconstructor + separate tracker). Has an HF demo. Pin in spec/08.
"""

from __future__ import annotations

from app.core.domain.models import ReconstructionRequest, ReconstructionResult
from app.core.ports.reconstruction_port import ReconstructionPort


class SpatialTrackerV2Adapter(ReconstructionPort):
    name = "spatialtracker_v2"

    def reconstruct(self, request: ReconstructionRequest) -> ReconstructionResult:
        del request
        raise NotImplementedError("SpatialTrackerV2Adapter: see spec/06-backend-spec.md")
