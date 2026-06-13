"""FUTURE placeholder — OpenD4RT adapter.

D4RT (arXiv:2512.08924, Google DeepMind, Dec 2025) is UNRELEASED — no code or
weights (paper + project page + blog only). This adapter is the documented
drop-in point for a future open D4RT-style unified decoder, making the 'swap in
the real thing later' story concrete (handover §2, §8). It is NOT wired into the
MVP. See spec/03-decisions-locked.md and spec/12-risk-register.md.
"""

from __future__ import annotations

from app.core.domain.models import ReconstructionRequest, ReconstructionResult
from app.core.ports.reconstruction_port import ReconstructionPort


class OpenD4RTAdapter(ReconstructionPort):
    name = "open_d4rt"

    def reconstruct(self, request: ReconstructionRequest) -> ReconstructionResult:
        del request
        raise NotImplementedError("OpenD4RTAdapter: future direction — not part of the MVP")
