"""T-104 — rigorous MV4D cap enforcement (`enforce_caps`) + caps semantics.

W1.T1 acceptance gate (spec/10 §1.2 T-104). `enforce_caps` lives in
``app/core/services/reconstruction_service.py`` (the core service, NOT the encoder —
spec/06 §5 step 7). These tests build an OVER-CAP ``Scene4D`` and assert it is
culled/subsampled to within EVERY MV4D cap (spec/05 §4), in the documented order:

  - frames ``T`` -> 64 (uniform temporal subsample), run FIRST so the dynamic list,
    track T-axis, and camera arrays stay aligned to the new ``frame_count``;
  - static -> 150 000 (drop lowest ``static_conf`` first);
  - dynamic per-frame -> 20 000 (DETERMINISTIC fixed-seed uniform random subsample,
    confidence-free — dynamic frames carry no per-point conf in ``Scene4D``);
  - tracks ``M`` -> 4 096 (drop lowest mean-visibility first).

There is NO separate ">24 MB" exception path — escalating cull keeps the payload
under the ceiling (spec/05 §4, spec/06 §5 step 7); the ONLY encode-path raise is
``EmptyReconstructionError`` (when culling removes everything), and that fires from
the SERVICE ``run()`` emptiness guard, not from ``enforce_caps`` itself.

Pure numpy, no torch — keeps the hexagonal boundary (T-130) green.
"""

from __future__ import annotations

import inspect

import numpy as np

import app.core.services.reconstruction_service as svc_mod
from app.core.domain.errors import EmptyReconstructionError
