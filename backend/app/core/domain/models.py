"""Canonical, model-agnostic domain types for the reconstruction core.

PLACEHOLDERS. The authoritative shapes — exact fields, dtypes, quantization and
the binary wire layout — are defined in spec/05-data-contract.md and
spec/06-backend-spec.md, and MUST match the frontend (frontend/src/types and
frontend/src/lib/wire). Do not treat these as final.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ReconstructionRequest:
    """Input to a reconstruction job."""

    video_path: str
    #: Short-clip cap — long video is the #1 scope risk (handover §4.6, §7).
    max_frames: int = 24


@dataclass
class ReconstructionResult:
    """Model-agnostic output, serialized via the binary wire format (app/wire).

    TODO(spec/05,06): per-frame points, per-point colour, 3D tracks, camera
    poses, confidence/visibility.
    """

    frame_count: int
