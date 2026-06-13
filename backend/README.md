# mayavius backend

Python + FastAPI reconstruction service. **Hexagonal architecture**: a pure,
model-agnostic core depends only on `ReconstructionPort`; each model integration
is an adapter implementing that port. FastAPI and the job queue are driving-side
adapters. (See the root [CLAUDE.md](../CLAUDE.md) for the full map and rules.)

> Scaffolding only. Endpoints return `501` and adapters raise `NotImplementedError`
> until built per `spec/06-backend-spec.md`. ML deps are deferred — see
> `requirements-ml.txt`.

## Layout

```
app/
  main.py                       FastAPI entry (+ /health)
  config.py                     env-driven settings (MAYAVIUS_*)
  api/routes/jobs.py            upload / status / result endpoints (stubs)
  core/                         PURE core — no FastAPI, no torch, no adapters
    domain/models.py            canonical domain types (placeholders)
    ports/reconstruction_port.py  the ReconstructionPort interface
    services/reconstruction_service.py  orchestration over the port
  adapters/                     model integrations (implement the port)
    vggt_adapter.py  pi3_adapter.py  spatialtracker_adapter.py
    cotracker3_adapter.py  open_d4rt_adapter.py (future)
  jobs/queue.py                 async job queue (driving-side)
  wire/encoder.py               binary wire-format encoder (spec/05)
tests/test_health.py            smoke test
```

## Setup & run (Apple Silicon)

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt        # lightweight; no PyTorch
uvicorn app.main:app --reload --port 8000  # http://localhost:8000/health
pytest                                      # runs the smoke test
```

ML models (VGGT/CoTracker3/…) run on-device via **MPS** for short clips only
(fp32 — no fp16 autocast on MPS). Those deps and model repo IDs are installed
later from `requirements-ml.txt` once pinned in `spec/08-dependencies-and-env.md`.
