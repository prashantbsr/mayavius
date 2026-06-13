"""Binary wire-format encoder (backend side) — MV4D v1.

JSON is forbidden for point payloads (handover §4.5) — it is the difference
between a ~2s and a ~40s load and it gates shareable result links. This encoder
and the frontend decoder (frontend/src/lib/wire/decoder.ts) are two
implementations of ONE format whose single source of truth is
spec/05-data-contract.md — header, version byte, dtypes, quantization ranges,
track indexing, visibility encoding. They MUST stay byte-for-byte compatible.
"""

from __future__ import annotations

from app.core.domain.models import ReconstructionResult


def encode_reconstruction(result: ReconstructionResult) -> bytes:
    del result  # placeholder
    raise NotImplementedError(
        "encode_reconstruction: not implemented (see spec/05-data-contract.md)"
    )
