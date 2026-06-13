"""W3.T1 — pure-numpy unit tests for the model-agnostic pipeline utils.

Covers (spec/06 §1/§5 steps 1,4,5; spec/05 §2/§4):
  (a) lift_tracks_to_3d — synthetic depth + known pixel + identity-ish camera
      unprojects to the expected mayavius world point (verifies x=(u-cx)D/fx etc.
      AND the F=diag(1,-1,-1) OpenCV->mayavius flip);
  (b) quantize_positions — matches the encoder/decoder inverse within tolerance +
      degenerate axis -> 0;
  (c) assemble_scene4d — a synthetic GeometryResult+TrackResult with one obviously
      moving track lands the moving region in dynamic_positions and the rest in
      static_positions; tracks/cameras populated; AABB spans all; and the scene is
      RAW (enforce_caps called SEPARATELY confirms assemble did not cap);
  (d) decode_and_subsample — cv2-gated (pytest.importorskip) on a real backend
      tiny.mp4: asserts [S,3,~518,...] shape + S <= max_frames; skips without cv2.

Pure numpy / opencv-only — NO torch, NO fastapi. Runs in `make test` (no-ML).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from app.core.domain.models import CameraTrack, ReconstructionRequest, Scene4D
from app.core.services.reconstruction_service import enforce_caps
from app.pipeline.assemble import (
    GeometryResult,
    TrackResult,
