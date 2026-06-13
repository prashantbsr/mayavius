"""The driven port the reconstruction core depends on (hexagonal architecture).

The core depends ONLY on this abstraction (spec/06 §2). Every model integration is
an adapter that implements it (app/adapters/*); a model SDK / torch is imported
ONLY inside its adapter, never here. Swapping models must not touch the core
(handover §3, §6 / T-130).

Pure: imports only the stdlib + the domain models. No FastAPI, no torch, no
concrete-adapter imports.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Protocol

from app.core.domain.models import ReconstructionRequest, Scene4D


class ProgressSink(Protocol):
    """Adapters report fractional progress; the worker forwards it to /jobs/{id}.

    A pure callable so no FastAPI / queue type leaks into core (spec/06 §2.1).
    ``progress`` in [0, 1]; ``stage`` is a short label e.g. "vggt", "tracking".
    """

    def __call__(self, progress: float, stage: str) -> None: ...


@dataclass(frozen=True)
class AdapterInfo:
    """Static capability + provenance descriptor (spec/06 §2).

    Surfaced in job metadata so the API can label the active model's weight license
    (D2) and gate by capability. MUST be cheap to produce (no model load).
    """

    name: str                 # stable id, e.g. "vggt+cotracker3"
    produces_tracks: bool     # emits Tracks (the ribbons)?
    dynamic: bool             # emits per-frame dynamic foreground points?
    mps_capable: bool         # runs on the 36GB Apple-Silicon Mac via MPS (fp32)?
    weights_license: str      # SPDX-ish tag, e.g. "cc-by-nc-4.0" (D2 / spec/08 §7)
    default_weights: str      # HF repo id, e.g. "facebook/VGGT-1B"


class ReconstructionPort(ABC):
    """Implemented by every model adapter (app/adapters/*)."""

    name: str  # short, stable id; mirrors AdapterInfo.name

    @property
    @abstractmethod
    def info(self) -> AdapterInfo:
        """Capabilities + license. MUST be cheap (no model load)."""

    @abstractmethod
    def reconstruct(
        self,
        request: ReconstructionRequest,
        progress: ProgressSink | None = None,
    ) -> Scene4D:
        """Run feedforward 4D reconstruction on a short, already-capped clip.

        MUST NOT assume CUDA; the MPS/CPU path must work for short clips on Apple
        Silicon. Output is in mayavius world space (spec/05 §2): right-handed,
        +X right / +Y up / -Z forward — the ADAPTER transforms native output into
        this convention before returning. Raises ReconstructionError subclasses
        (spec/06 §2.2). Reports progress via ``progress`` if given.
        """
