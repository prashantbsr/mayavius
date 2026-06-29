"""Re-bake the committed MV4D corpus blobs from their source mp4s (spec/10 §6, D10).

This regenerates ``assets/samples/<slug>.mv4d`` for the four corpus slugs
(C-1..C-4: ``walking-person`` / ``street-vehicle`` / ``pet-motion`` /
``static-scene``) by running the REAL default pipeline end to end — exactly what
``JobQueue._run`` does at request time (spec/06 §6): build the configured adapter,
wrap it in the pure ``ReconstructionService``, run each clip, then encode + write
the immutable MV4D v1 blob (spec/05 §4).

WHY THIS EXISTS — the committed blobs are STALE. They were baked BEFORE the
net-excursion static/dynamic split fix in ``app/pipeline/assemble.py``, so they
encode a degenerate split: the negative control ``static-scene`` decodes to ~3
static points and ~160k dynamic points per frame — i.e. the split classified an
almost-entirely-static mountain valley as nearly all *motion* (noise). Re-baking
with the current ``assemble_scene4d`` produces the healthy split the corpus
advertises (static-scene → a substantial static cloud, few/no dynamic points;
``test_corpus.py`` asserts this and is EXPECTED-RED until this script runs).

HEAVY + MANUAL — NOT run in CI. This drives the full ``vggt+cotracker3`` combo:
VGGT geometry + CoTracker3 tracking on MPS. It therefore REQUIRES the ML overlay
(``backend/requirements-ml.txt`` installed) and the multi-GB model weights already
cached locally (one-time download; never committed). It also OVERWRITES committed
binaries, so it is deferred and run by hand via ``make bake-corpus`` — never
automatically. Expect minutes-per-clip on a 36 GB Apple-Silicon Mac.

LAZY DISCIPLINE (T-130, spec/06 §4): no torch / no adapter import lives at module
top, so merely importing this file is torch-free and safe. The heavy imports
(``build_adapter`` resolves a torch-backed adapter; the service drives it) all
happen INSIDE ``main``, mirroring the registry's lazy-factory boundary.
"""

from __future__ import annotations

# Module-top imports stay torch-free (lazy discipline, T-130). The settings +
# request/encoder are numpy/pydantic only; the adapter (which DOES pull torch) is
# resolved lazily inside main() via build_adapter, exactly like the registry.
from app.config import SAMPLES_DIR, settings

# The four corpus slugs (C-1..C-4, spec/10 §6) in bake order. The slug IS the file
# name: assets/samples/<slug>.{mp4,mv4d}. Kept in lockstep with test_corpus.py.
CORPUS_SLUGS: list[str] = [
    "walking-person",  # C-1 (the hero)
    "street-vehicle",  # C-2
    "pet-motion",      # C-3
    "static-scene",    # C-4 (the negative control)
]

# MV4D hard frame ceiling (spec/05 §4); the bake clamps max_frames to it, matching
# the POST handler's ``max_frames=min(settings.max_clip_frames, 64)`` (spec/06 §5).
_MAX_FRAMES_HARD = 64


def _progress_print(slug: str):
    """Build a ``ProgressSink`` that prints the pipeline stage for one slug.

    Mirrors ``JobQueue._run``'s progress closure, but synchronous + to stdout
    (this is a CLI bake, not the async SSE worker — no loop / no thread marshal).
    """

    def progress(p: float, stage: str) -> None:
        print(f"  [{slug}] {p * 100:5.1f}%  {stage}")

    return progress


def _bake_one(service, slug: str) -> dict:
    """Run + encode + write one corpus slug; return a small count summary.

    Mirrors ``JobQueue._run`` (spec/06 §6): build the ``ReconstructionRequest``
    from Settings, ``service.run`` it (smooth/cull/caps applied in the service),
    ``encode_reconstruction`` the resulting ``Scene4D``, and write the immutable
    MV4D v1 blob to ``assets/samples/<slug>.mv4d``.
    """
    # Lazy heavy imports (kept out of module top — see module docstring): the wire
    # encoder is numpy-only, but the request/service path drives the torch adapter.
    from app.core.domain.models import ReconstructionRequest
    from app.wire.encoder import encode_reconstruction

    video_path = str(SAMPLES_DIR / f"{slug}.mp4")
    out_path = SAMPLES_DIR / f"{slug}.mv4d"

    request = ReconstructionRequest(
        video_path=video_path,
        max_frames=min(settings.max_clip_frames, _MAX_FRAMES_HARD),
        target_fps=settings.target_fps,
        device=settings.device,
    )

    print(f"[{slug}] reconstructing {video_path}")
    scene = service.run(request, _progress_print(slug))

    blob = encode_reconstruction(scene)
    out_path.write_bytes(blob)

    n_static = int(scene.static_positions.shape[0])
    n_dynamic_total = sum(int(p.shape[0]) for p in scene.dynamic_positions)
    n_tracks = 0 if scene.tracks is None else int(scene.tracks.positions.shape[0])
    summary = {
        "slug": slug,
        "frames": int(scene.frame_count),
        "static": n_static,
        "dynamic_total": n_dynamic_total,
        "tracks": n_tracks,
        "bytes": len(blob),
        "path": str(out_path),
    }
    print(
        f"[{slug}] wrote {out_path}  "
        f"(T={summary['frames']} static={n_static} "
        f"dynamic_total={n_dynamic_total} tracks={n_tracks} bytes={len(blob)})"
    )
    return summary


def main() -> None:
    """Re-bake all four corpus blobs with the configured default pipeline.

    Resolves the adapter from Settings (``MAYAVIUS_ADAPTER``, default
    ``vggt+cotracker3``) via the registry factory and injects it into a pure
    ``ReconstructionService`` (conf_thresh from Settings) — the SAME wiring
    ``app/main.py`` builds for the live worker (spec/06 §6). The torch-backed
    adapter is constructed HERE (not at module import), preserving the lazy
    boundary so importing this file stays torch-free.
    """
    # Heavy import deferred to call time (lazy discipline, T-130): build_adapter
    # resolves a torch-backed adapter (vggt+cotracker3) via its lazy factory.
    from app.adapters.registry import build_adapter
    from app.core.services.reconstruction_service import ReconstructionService

    print(
        "bake_corpus: re-baking the committed MV4D corpus with the current "
        "net-excursion static/dynamic split (app/pipeline/assemble.py)."
    )
    print(f"  adapter={settings.adapter!r} device={settings.device!r} "
          f"max_frames={min(settings.max_clip_frames, _MAX_FRAMES_HARD)} "
          f"target_fps={settings.target_fps} conf_thresh={settings.conf_thresh}")
    print(f"  samples_dir={SAMPLES_DIR}")

    adapter = build_adapter(settings)
    service = ReconstructionService(adapter, conf_thresh=settings.conf_thresh)

    summaries = [_bake_one(service, slug) for slug in CORPUS_SLUGS]

    print("\nbake_corpus: done. Re-commit the regenerated blobs:")
    for s in summaries:
        print(
            f"  {s['slug']:<16} T={s['frames']:<3} static={s['static']:<8} "
            f"dynamic_total={s['dynamic_total']:<8} tracks={s['tracks']:<5} "
            f"bytes={s['bytes']}"
        )


if __name__ == "__main__":
    main()
