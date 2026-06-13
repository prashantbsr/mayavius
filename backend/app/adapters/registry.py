"""Adapter registry — id -> factory; ``build_adapter(settings)`` (spec/06 §4.6).

Maps the ``MAYAVIUS_ADAPTER`` id to a factory that LAZILY imports its adapter class
INSIDE the factory. This is deliberate: importing ``registry`` must NOT import any
heavy or model-dependent adapter module (torch is never pulled in just to resolve a
``fake``/``vggt`` id). ``build_adapter`` is called ONCE at startup (the FastAPI
lifespan) and the instance is injected into ``ReconstructionService``.

DRIVING side: this module is wired by ``main.py`` / the worker. It does NOT belong
to the pure core (the core never knows concrete adapter classes — that is the whole
point of the port boundary, T-130).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:  # import only for type checkers — never at runtime (keeps imports lazy)
    from app.core.ports.reconstruction_port import ReconstructionPort


def _make_vggt_cotracker3(settings):
    from app.adapters.combo import VggtCoTracker3Adapter  # lazy

    return VggtCoTracker3Adapter(settings)


def _make_vggt(settings):
    from app.adapters.vggt_adapter import VggtAdapter  # lazy

