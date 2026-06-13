"""Pi3 / π³ adapter — fast feedforward point cloud + camera, STATIC (optional).

Like VGGT (a static reconstructor), kept as a CUDA/deploy-path swap-in for the static
layer — an alternative to VGGT. Repo IDs / weights / licenses are pinned in spec/08 §4.6.

NEGATIVE KNOWLEDGE (spec/06 §4.4, spec/08 §4.6): **no official MPS path** — PR #153 is
open/unmerged and ``demo_gradio.py`` hard-fails without CUDA. License is **code
BSD-3-Clause** (commercial OK, C2) but **weights CC-BY-NC-4.0** (HF inconsistently
tags ``bsd-2-clause`` → treat as NC). So ``info.mps_capable=False`` and it is NOT the
Mac default (handover §3 / hard constraint).

HONEST STUB (W3.T5, spec/06 §4.4/§4.7): a *production* adapter. ``info`` is cheap +
license-tagged. On ``device in {"mps","cpu"}`` ``reconstruct`` raises
``UnsupportedDeviceError`` NAMING the constraint and pointing at the cloud-GPU deploy
(spec/11) — it does **not** silently fall back. The raise happens BEFORE any model
load, so no torch import is needed (and none happens) — the dead end stays documented,
not rediscovered (spec/10 §5).
"""

from __future__ import annotations

from app.core.domain.errors import UnsupportedDeviceError
from app.core.domain.models import ReconstructionRequest, Scene4D
from app.core.ports.reconstruction_port import (
    AdapterInfo,
    ProgressSink,
    ReconstructionPort,
)


class Pi3Adapter(ReconstructionPort):
    name = "pi3"

    def __init__(self, settings=None) -> None:
        self._settings = settings

    @property
    def info(self) -> AdapterInfo:
        return AdapterInfo(
            name="pi3",
            produces_tracks=False,
            dynamic=False,
            mps_capable=False,
            # code BSD-3 / weights cc-by-nc-4.0 (treat as NC) — spec/08 §4.6.
            weights_license="code BSD-3 / weights cc-by-nc-4.0",
            default_weights="yyfz233/Pi3",
        )

    def reconstruct(
        self, request: ReconstructionRequest, progress: ProgressSink | None = None
    ) -> Scene4D:
        del progress
        device = getattr(request, "device", "mps")
        if device in {"mps", "cpu"}:
            raise UnsupportedDeviceError(
                "Pi3 has no official MPS path (PR #153 is open/unmerged; demo_gradio.py "
                f"hard-fails without CUDA); it cannot run on device={device!r}. Use the "
                "cloud-GPU deploy with MAYAVIUS_DEVICE=cuda (spec/11)."
            )
        # device == "cuda": the real CUDA static-reconstructor path is a future
        # cloud-deploy task — not part of the Mac MVP (spec/06 §4.4).
        raise NotImplementedError(
            "Pi3Adapter CUDA model path is a cloud-deploy task (spec/11)."
        )
