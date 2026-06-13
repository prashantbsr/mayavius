# mayavius

> Drop in a video and orbit a live 4D reconstruction of the scene in your
> browser — moving objects and all.

Open-source, lightweight, interactive viewer for **feedforward 4D scene
reconstruction**. Upload a short clip; a backend runs a feedforward model
(colored 3D point cloud + 3D point tracks); the browser plays it back — orbit,
scrub the timeline, play/pause/loop, and a bullet-time freeze-and-orbit. **No GPU
required to view.**

> **Status: 🚧 scaffolding.** Project structure only — not yet functional.
> See [SPEC_BUILD_HANDOVER.md](SPEC_BUILD_HANDOVER.md) for the full brief and
> [CLAUDE.md](CLAUDE.md) for the architecture map.

## Stack

- **Frontend** — Next.js 16 (App Router) · React 19 · TypeScript · Tailwind v4 ·
  Three.js via react-three-fiber. SEO-first: static landing + rich share cards
  for result links.
- **Backend** — Python · FastAPI, hexagonal architecture (models behind a
  swappable `ReconstructionPort`), async job model. Runs on Apple-Silicon MPS for
  short clips; cloud GPU optional.

## Quickstart (Apple Silicon)

Prereqs: Node 20+, Python 3.10+.

```bash
make setup          # install frontend + backend deps (lightweight; no PyTorch)

# then, in two terminals:
make dev-backend    # FastAPI → http://localhost:8000/health
make dev-frontend   # Next.js → http://localhost:3000
```

Run `make help` for all targets. Per-side details:
[frontend/](frontend/) · [backend/README.md](backend/README.md).

## Layout

```
frontend/   Next.js app (viewer + SEO surfaces)
backend/    FastAPI service (core / ports / adapters / jobs / wire)
```

## License

Intended open-source; license to be finalized (MIT / Apache-2.0 candidates).
