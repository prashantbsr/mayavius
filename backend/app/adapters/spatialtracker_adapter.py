"""SpatialTrackerV2 adapter — dynamic 3D tracks + geometry in ONE model (optional).

The single-model route to the D4RT 'cloud + tracks' look (vs. static-reconstructor +
separate tracker). Slower (iterative refinement). Has an HF demo. Repo IDs / weights /
licenses are pinned in spec/08 §4.5.

NEGATIVE KNOWLEDGE (spec/06 §4.3, spec/08 §4.5): **CUDA-only** — upstream pins
``torch==2.4.1+cu124`` and is NOT Mac-installable as-is. The code license is
**CC-BY-NC-4.0** (GitHub shows ``NOASSERTION`` — a CC-detection gap, NOT permissive).
So ``info.mps_capable=False``.

HONEST STUB (W3.T5, spec/06 §4.3/§4.7): this is a *production* adapter, not a test
fake. ``info`` is cheap + license-tagged (so the registry/API can advertise it without
ML deps). On ``device in {"mps","cpu"}`` ``reconstruct`` raises ``UnsupportedDeviceError``
NAMING the constraint and pointing at the cloud-GPU deploy (spec/11) — it does **not**
silently fall back. The raise happens BEFORE any model load, so no torch import is
needed (and none happens) — keeping this dead end documented, not rediscovered
(spec/10 §5 negative knowledge). On a real CUDA box (``device == "cuda"``) the model
path is a future cloud-deploy task.
"""

from __future__ import annotations

from app.core.domain.errors import UnsupportedDeviceError
from app.core.domain.models import ReconstructionRequest, Scene4D
from app.core.ports.reconstruction_port import (
    AdapterInfo,
    ProgressSink,
    ReconstructionPort,
)


class SpatialTrackerV2Adapter(ReconstructionPort):
    name = "spatialtracker_v2"

    def __init__(self, settings=None) -> None:
        self._settings = settings

    @property
    def info(self) -> AdapterInfo:
        return AdapterInfo(
            name="spatialtracker_v2",
            produces_tracks=True,
            dynamic=True,
            mps_capable=False,
            weights_license="cc-by-nc-4.0",
            default_weights="Yuxihenry/SpatialTrackerV2_Offline",
        )

    def reconstruct(
        self, request: ReconstructionRequest, progress: ProgressSink | None = None
    ) -> Scene4D:
        del progress
        device = getattr(request, "device", "mps")
        if device in {"mps", "cpu"}:
            raise UnsupportedDeviceError(
                "SpatialTrackerV2 is CUDA-only (upstream pins torch==2.4.1+cu124); "
                f"it cannot run on device={device!r}. Use the cloud-GPU deploy with "
                "MAYAVIUS_DEVICE=cuda (spec/11)."
            )
        # device == "cuda" (or another non-Mac device): the real model path is a
        # future cloud-deploy task — not part of the Mac MVP (spec/06 §4.3).
        raise NotImplementedError(
            "SpatialTrackerV2Adapter CUDA model path is a cloud-deploy task (spec/11)."
        )
