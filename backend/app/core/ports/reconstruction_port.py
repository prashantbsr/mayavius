"""The driven port the reconstruction core depends on (hexagonal architecture).

The core depends ONLY on this abstraction. Every model integration is an adapter
that implements it (app/adapters/*). Swapping models must not touch the core —
this is the project's core extensibility story (handover §3, §6).

The FULL method set, the canonical output structure (per-frame points, per-point
colour, 3D tracks, camera poses, confidence/visibility) and the error contract
are finalized in spec/06-backend-spec.md. The below is the scaffolding seam.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from app.core.domain.models import ReconstructionRequest, ReconstructionResult


class ReconstructionError(Exception):
    """Base error raised by adapters. Concrete error contract: spec/06."""


class ReconstructionPort(ABC):
    """Port implemented by every model adapter."""

    #: Short, stable adapter identifier (e.g. "vggt"). Set by each adapter.
    name: str

    @abstractmethod
    def reconstruct(self, request: ReconstructionRequest) -> ReconstructionResult:
        """Run feedforward 4D reconstruction on a short clip.

        Implementations MUST NOT assume a GPU; the MPS/CPU path must work for
        short clips on Apple Silicon (handover §3). Signature and options are
        finalized in spec/06-backend-spec.md.
        """
        raise NotImplementedError
