"""OpenD4RT adapter — the unofficial open D4RT (optional; GPU).

The concrete drop-in for the "swap in the real thing later" story (handover §2/§8, C5):
a unified static+dynamic decoder that, when it matures, can replace the entire
VGGT+CoTracker3 combo behind the SAME port — no core change (the architectural payoff,
spec/06 §4.5). Repo / weights / license are pinned in spec/08 §4.7.

NEGATIVE KNOWLEDGE (spec/06 §4.5, spec/08 §4.7): the official Google DeepMind D4RT
(arXiv:2512.08924, Dec 2025) is **still unreleased** (decision-log §F); the wrapped
candidate is the **unofficial** reimpl ``Lijiaxin0111/Open-d4rt`` (**Apache-2.0** — the
cleanest license in the set; weights ``Lijiaxin0111/OpenD4RT``). It is GPU/PyTorch-
oriented and its **MPS path is UNVERIFIED** — so ``info.mps_capable=False`` until
measured on the Mac (spec/10). Do NOT assume it runs on MPS.

HONEST STUB (W3.T5, spec/06 §4.5/§4.7): a *production* adapter. ``info`` is cheap +
license-tagged. On ``device in {"mps","cpu"}`` ``reconstruct`` raises
``UnsupportedDeviceError`` NAMING the constraint (MPS unverified) and pointing at the
cloud-GPU deploy (spec/11) — it does **not** silently fall back. The raise happens
BEFORE any model load, so no torch import is needed (and none happens) — the dead end
stays documented, not rediscovered (spec/10 §5).
"""

from __future__ import annotations

from app.core.domain.errors import UnsupportedDeviceError
from app.core.domain.models import ReconstructionRequest, Scene4D
from app.core.ports.reconstruction_port import (
    AdapterInfo,
    ProgressSink,
    ReconstructionPort,
)


class OpenD4RTAdapter(ReconstructionPort):
    name = "open_d4rt"

    def __init__(self, settings=None) -> None:
        self._settings = settings

    @property
    def info(self) -> AdapterInfo:
        return AdapterInfo(
            name="open_d4rt",
            produces_tracks=True,
            dynamic=True,
            mps_capable=False,
            weights_license="apache-2.0",
            default_weights="Lijiaxin0111/OpenD4RT",
        )

    def reconstruct(
        self, request: ReconstructionRequest, progress: ProgressSink | None = None
    ) -> Scene4D:
        del progress
        device = getattr(request, "device", "mps")
        if device in {"mps", "cpu"}:
            raise UnsupportedDeviceError(
                "OpenD4RT's MPS path is UNVERIFIED (GPU/PyTorch-oriented; not measured "
                f"on the Mac); it is not run on device={device!r}. Use the cloud-GPU "
                "deploy with MAYAVIUS_DEVICE=cuda (spec/11)."
            )
        # device == "cuda": the real unified-decoder path is a future cloud-deploy /
        # drop-in task (the architectural payoff) — not part of the Mac MVP (spec/06 §4.5).
        raise NotImplementedError(
            "OpenD4RTAdapter GPU model path is a future drop-in / cloud-deploy task "
            "(spec/06 §4.5, spec/11)."
        )
