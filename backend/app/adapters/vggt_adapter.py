"""VGGT adapter — facebookresearch/vggt (CVPR 2025). Fast feedforward geometry,
STATIC (no native dynamics/tracks); pair with a tracker for the D4RT look.

MPS: a community float32 path runs on Apple Silicon (no fp16 autocast on MPS).
License: prefer the VGGT-1B-Commercial checkpoint for an open release. Repo IDs,
weights and tensor shapes are RE-VERIFIED and pinned in spec/08. Candidate
default for the Mac MPS demo path (handover §5).
"""

from __future__ import annotations

from app.core.domain.models import ReconstructionRequest, ReconstructionResult
from app.core.ports.reconstruction_port import ReconstructionPort


class VggtAdapter(ReconstructionPort):
    name = "vggt"

    def reconstruct(self, request: ReconstructionRequest) -> ReconstructionResult:
        del request
        raise NotImplementedError("VggtAdapter: see spec/06-backend-spec.md")
