# 08 — Dependencies & Environment

Every version, repo ID, license, and install command here was **re-verified live
on 2026-06-13** against a primary source (npm registry, PyPI, GitHub, HuggingFace,
Apple). Evidence + dates: [decisions/decision-log.md](decisions/decision-log.md)
§B–§G. The executor must never need to look anything up.

Target dev machine: **MacBook Pro, Apple Silicon, 36 GB unified memory, macOS
14.0+**. The frontend runs fully locally; inference runs on **MPS** for short
clips. No cloud GPU is required for the MVP.

---

## 1. Toolchain prerequisites

| Tool | Version | Check |
|------|---------|-------|
| Node.js | ≥ 20 LTS (22 LTS recommended) | `node -v` |
| npm | ≥ 10 (ships with Node) | `npm -v` |
| Python | **3.12.x** (see D4 — supersedes CLAUDE.md's 3.10) | `python3.12 --version` |
| macOS | **14.0+** (Apple's floor for current torch MPS wheels) | `sw_vers` |
| git | any recent | `git --version` |

Install Python 3.12 via `brew install python@3.12` (or pyenv) if absent.

---

## 2. Frontend dependencies (pinned, verified, already installed)

`frontend/package.json` is correct as-is. Confirmed current `latest` on npm and
present in `frontend/node_modules`:

| Package | Pin | Installed / latest | License |
|---------|-----|--------------------|---------|
| `next` | `16.2.9` | 16.2.9 (== latest) | MIT |
| `react`, `react-dom` | `19.2.4` | 19.2.4 (latest 19.2.7) | MIT |
| `three` | `^0.184.0` | 0.184.0 | MIT |
| `@react-three/fiber` | `^9.6.1` | 9.6.1 — **peer `react >=19 <19.3`** | MIT |
| `@react-three/drei` | `^10.7.7` | 10.7.7 | MIT |
| `zustand` | `^5.0.14` | 5.0.14 | MIT |
| `tailwindcss` (+`@tailwindcss/postcss`) | `^4` | 4.3.1 | MIT |
| `typescript` (dev) | `^5` | 5.9.3 — **stay on 5.x** (TS6 exists; post-MVP bump) | Apache-2.0 |
| `@types/three` (dev) | `^0.184.1` | — | MIT |

**Install:** `cd frontend && npm install` (or `make setup-frontend`).
**Upgrade coupling (do not break):** bumping React to 19.3+ breaks `@react-three/fiber@9.6` (peer `<19.3`). Coordinate any React bump with an R3F bump (R3F 10.x is alpha — do not adopt for MVP).
**No new runtime deps for the Path-1 MVP.** `Line2`/`LineSegments2`/`LineGeometry`/`LineMaterial` ship inside `three` under `three/addons/lines/…` (import explicitly). Spark (`@sparkjsdev/spark`) is **Path-2, OUT of MVP** — do not add it.
**Dev/test deps added in spec/10** (frontend `devDependencies`; pin the major then
**resolve & freeze the exact version after first install**, same install-then-freeze
convention as `ruff`/ML — these are dev-only test tools): `vitest@^3`,
`@vitejs/plugin-react@^4`, `jsdom@^26` (use `jsdom`, **not** happy-dom — one engine),
`@playwright/test@^1`.

---

## 3. Backend — lightweight runtime (no ML)

`backend/requirements.txt` (pins verified on PyPI; all support Python 3.12 — only
the venv interpreter changes from 3.10 → 3.12, package pins are unchanged):

```
fastapi==0.136.3
uvicorn[standard]==0.49.0
python-multipart==0.0.32
pydantic==2.13.4
pydantic-settings==2.14.1
numpy>=2,<3          # core domain arrays + MV4D wire encoder — NOT ML. pin exact after first resolve.
```

- **`numpy` is in the lightweight tier, not `requirements-ml.txt`.** The
  model-agnostic core (`core/domain/models.py` `Scene4D` arrays, the
  `wire/encoder.py` quantizer, the pure-numpy service helpers) imports numpy, so the
  **no-ML W0/W1 CI path** (`make test`; spec/10 **T-130** imports every `app.core.*`
  module) needs it installed without torch. The scaffold originally listed numpy
  under the deferred ML file — it is moved here. Pin the resolved exact 2.x after
  first install.
- `uvicorn[standard]` bundles `websockets`, `uvloop`, `httptools`, `watchfiles`.
- **SSE needs NO extra dep:** FastAPI ≥0.135 ships `from fastapi.sse import
  EventSourceResponse, ServerSentEvent`. Do **not** add `sse-starlette`. (SSE is
  incompatible with `GZipMiddleware` — compress the *result* blob at the route
  level instead, not via GZip middleware on the SSE stream.)
- `python-multipart` is required for `UploadFile` (multipart form parsing).

`backend/requirements-dev.txt` — the scaffold already has `-r requirements.txt`,
`pytest==9.0.3`, `httpx==0.28.1`. The build **adds `ruff`** (the linter; `pyproject.toml`
already has a `[tool.ruff]` section but ruff was never in the deps):
```
-r requirements.txt
pytest==9.0.3
httpx==0.28.1          # FastAPI TestClient transport (matches scaffold pin)
ruff==0.14.*           # lint + format (dev only) — ADDED by the build
```
> This supersedes both the scaffold (which lacks `ruff`) and spec/10 §7's earlier
> "add nothing" note: §1–§4 tests need no new backend dep **except `ruff`** for
> `make lint`. Keep `httpx==0.28.1` (the on-disk pin), do not widen it.

**Create the venv with Python 3.12** (the scaffold's `.venv` is 3.10 — remove and
recreate):
```bash
cd backend
rm -rf .venv                                   # scaffold venv is Python 3.10
python3.12 -m venv .venv
./.venv/bin/pip install --upgrade pip
./.venv/bin/pip install -r requirements-dev.txt
```
**Build-time edits this implies (not "only the Makefile"):**
- `Makefile` `setup-backend`: `python3` → `python3.12 -m venv .venv` (and the
  `test-*` target edits in [10 §7](10-testing-strategy.md)).
- `backend/pyproject.toml`: leave `requires-python = ">=3.10"` (a *floor* — Python
  3.12 satisfies it; the lib runs on 3.10–3.14) and `ruff target-version = "py310"`
  (lint to the floor for forward-compat). D4 mandates the **dev/run interpreter** is
  3.12; it does **not** require raising the package floor. No pyproject change is
  forced — but if you prefer to pin the floor up, set both to `py312` consciously.

---

## 4. ML dependencies (deferred until this point; Apple-Silicon / MPS)

These are multi-GB and **never committed**. Installed only after the lightweight
path works. The default adapter combo is **VGGT + CoTracker3** (D1).

### 4.1 PyTorch (MPS) — install-then-freeze procedure
PyTorch's exact MPS-compatible build for these models can only be validated on the
target Mac, so the procedure is deterministic but the final pins are frozen at
build time:

```bash
cd backend
# Mac wheels are CPU+MPS (no CUDA index URL on macOS):
./.venv/bin/pip install torch torchvision
./.venv/bin/python - <<'PY'
import torch; print("torch", torch.__version__, "mps", torch.backends.mps.is_available())
PY
# Expect torch >= 2.12.0 and mps True. Then freeze the resolved versions:
./.venv/bin/pip freeze | grep -Ei '^(torch|torchvision|opencv-python|imageio|huggingface-hub|einops|vggt)\b' >> requirements-ml.txt
# CoTracker3 is a torch.hub runtime pull (not a pip dist) — record its commit by hand (§4.4)
```
- **Validated floor:** `torch >= 2.5.0` (first with stable MPS autocast); current
  stable is **2.12.0** (ARM64 wheels, Python 3.10–3.14). Pin whatever resolves
  (expected `2.12.0`).
- **fp16 nuance (C3):** MPS *can* do fp16/bf16 autocast (beta) — we still run
  **fp32** because the VGGT MPS path does and half-precision is incomplete for
  these models. Adapters force fp32 on MPS.

### 4.2 Shared ML utilities (pin into `requirements-ml.txt`)
```
opencv-python==4.*    # video decode → frames
imageio[ffmpeg]==2.*  # robust short-clip decode fallback
huggingface-hub==0.*  # pull weights; resolve & freeze exact
einops==0.8.*
```
(numpy is **not** here — it lives in the lightweight `requirements.txt`, §3, because
the core needs it without torch.)
Freeze exact resolved versions after install (same `pip freeze` pattern).

### 4.3 VGGT (static reconstructor — default)
- **Source (pin the git ref so DoD §9.5 "frozen exact pins" is satisfiable):**
  `pip install "git+https://github.com/facebookresearch/vggt.git@<resolved-sha>"`
  (import `from vggt.models.vggt import VGGT`). Record the resolved commit and add
  `vggt @ git+https://github.com/facebookresearch/vggt.git@<sha>` to
  `requirements-ml.txt` — an unpinned URL is **not** a frozen pin.
- **Weights (default, NC):** `facebook/VGGT-1B` — `VGGT.from_pretrained("facebook/VGGT-1B")` (pulls via huggingface-hub). License **cc-by-nc-4.0**.
- **Weights (commercial, optional):** `facebook/VGGT-1B-Commercial` — custom
  `vggt-aup-license` (commercial OK **except military**), **gated**: requires
  completing the HF access form first. Not the default (avoids gating in local dev).
- **MPS (negative knowledge, C3):** upstream is CUDA/CPU-only. The `VggtAdapter`
  applies the **community-port pattern** (`github.com/jmanhype/vggt-mps`, MIT,
  reference only): set `device="mps"`, force **fp32**, and **do not** wrap in
  `torch.cuda.amp.autocast`. Input is a **set of frames** `[S,3,H,W]` rescaled to
  **width 518 px** — the backend decodes+subsamples the video first (spec/06).
- Outputs used: world point map (+conf) → static cloud; depth + camera
  (extrinsics/intrinsics) → for lifting CoTracker3 tracks to 3D.

### 4.4 CoTracker3 (2D tracker → 3D via VGGT depth — default)
- **Source / weights:** `torch.hub.load("facebookresearch/co-tracker",
  "cotracker3_offline")` (also `cotracker3_online`); HF weights repo
  `facebook/cotracker3`. License **cc-by-nc-4.0**.
- **Pinning (it is NOT a pip dist → never in `pip freeze`):** record the resolved
  hub commit as a comment in `requirements-ml.txt`, e.g.
  `# cotracker3: torch.hub facebookresearch/co-tracker @ <sha> (runtime pull)`.
  `torch.hub.load` needs network on first run (or a warmed `~/.cache/torch/hub`);
  pre-cache it in the deploy image (spec/11).
- **MPS:** first-class — CoTracker selects `cuda > mps > cpu` automatically
  (merged upstream). Output `pred_tracks (B,T,N,2)` + `pred_visibility (B,T,N,1)`;
  the adapter lifts 2D→3D with VGGT depth+intrinsics, producing `Tracks` (spec/05).

### 4.5 SpatialTrackerV2 (optional — cloud/CUDA only)
- **Repo:** `github.com/henry123-boy/SpaTrackerV2` (slug casing: `SpaTrackerV2`).
- **Weights:** `Yuxihenry/SpatialTrackerV2_Front` / `-Online` / `-Offline` (HF).
- **License:** **CC-BY-NC-4.0 on the code itself** (GitHub shows `NOASSERTION` —
  this is a CC-detection gap, **not** permissive). Weights assume NC.
- **MPS (negative knowledge):** **CUDA-only** — upstream pins
  `torch==2.4.1+cu124`. Not Mac-installable as-is; this adapter is **cloud/optional**
  (spec/11). Do not add to the local MVP install.

### 4.6 Pi3 / π³ (optional)
- **Repo:** `github.com/yyfz/Pi3`; **weights:** `yyfz233/Pi3`.
- **License:** **code BSD-3-Clause (commercial OK)**; **weights CC-BY-NC-4.0**
  per README (HF inconsistently tags `bsd-2-clause` → treat as NC).
- **MPS (negative knowledge):** **no official MPS** (PR #153 open/unmerged);
  `demo_gradio.py` hard-fails without CUDA. Not a Mac default.

### 4.7 OpenD4RT (optional — the unofficial open D4RT)
- **Repo:** `github.com/Lijiaxin0111/Open-d4rt` (Apache-2.0); **weights:**
  `Lijiaxin0111/OpenD4RT`. GPU/PyTorch-oriented; **MPS unverified** (do not assume
  it runs on the Mac without testing). Wires into the `OpenD4RTAdapter` placeholder.
- Official Google DeepMind D4RT remains **unreleased** (decision-log §F).

---

## 5. Apple-Silicon / MPS environment

Set in the backend process environment for any MPS adapter (the adapter sets it
**before importing torch**):
```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1   # route unimplemented MPS ops to CPU (slower)
```
- Some ops may still lack MPS coverage; the fallback handles most. If a specific op
  fails even with fallback, the documented remedy is to run that adapter on the
  optional cloud-GPU deploy (spec/11), not to block the MVP.
- **Memory:** measured on the actual 36 GB Mac during the build (spec/10 gate) —
  do **not** assert per-frame GB numbers (the "7 GB/8 GB" figures are unverified).
  Repo-confirmed floor: VGGT runs on an 8 GB+ RAM Mac; 36 GB is ample for short clips.

---

## 6. Environment variables

Backend (`backend/.env.example`, `MAYAVIUS_`-prefixed). The scaffold `app/config.py`
defines only the first three; the **`(new)`** rows are `Settings` fields the build
**adds** (spec/06 §8 is the behavioral authority):
| Var | Settings field | Default | Meaning |
|-----|----------------|---------|---------|
| `MAYAVIUS_CORS_ORIGINS` | `cors_origins` | `["http://localhost:3000"]` | allowed frontend origins (existing) |
| `MAYAVIUS_MAX_CLIP_FRAMES` | `max_clip_frames` | `24` | short-clip frame cap; clamped `min(.,64)` at request build (existing) |
| `MAYAVIUS_MAX_UPLOAD_MB` | `max_upload_mb` | `64` | upload size guard (existing) |
| `MAYAVIUS_ADAPTER` | `adapter` **(new)** | `vggt+cotracker3` | active adapter id (spec/06 §4.6); `fake` = no-ML fixture mode |
| `MAYAVIUS_DEVICE` | `device` **(new)** | `mps` | `mps` \| `cpu` \| `cuda` (cloud) |
| `MAYAVIUS_TARGET_FPS` | `target_fps` **(new)** | `12.0` | subsample target + MV4D playback fps |
| `MAYAVIUS_MOTION_THRESH` | `motion_thresh` **(new)** | `0.95` | static/dynamic split percentile (spec/06 §5 step 5) |
| `MAYAVIUS_CONF_THRESH` | `conf_thresh` **(new)** | `0.5` | static-point confidence cull floor (spec/06 §5 step 6) |
| `MAYAVIUS_VGGT_WEIGHTS` | `vggt_weights` **(new)** | `facebook/VGGT-1B` | VGGT checkpoint (commercial swap = `facebook/VGGT-1B-Commercial`) |
| `MAYAVIUS_RESULT_DIR` | `result_dir` **(new)** | `outputs` (resolved absolute → `backend/outputs`) | result-blob store; served by `/result` |
| `MAYAVIUS_RUN_MPS_SMOKE` | — (test-only) | unset | opt-in flag for the MPS smoke test (spec/10 §5); not a `Settings` field |

> `MAYAVIUS_RESULT_DIR` default `outputs` is **resolved absolute against the backend
> package root at startup** (`Path(__file__).resolve().parents[1] / settings.result_dir`,
> from `app/config.py` → `backend/outputs`; the worker `mkdir(parents=True,
> exist_ok=True)`). A bare cwd-relative `./backend/outputs` would resolve to
> `backend/backend/outputs` because the run/test commands `cd backend` first — so it
> is anchored, not cwd-relative ([06 §8](06-backend-spec.md)). The on-disk
> `.gitignore` already ignores `backend/outputs/` — generated blobs are never
> committable (the committed golden test fixture lives under `backend/tests/` and is
> intentionally tracked). Add the `(new)` rows to `app/config.py` `Settings` and to
> `.env.example`.

Frontend (`frontend/.env.example`; see `src/config.ts`):
| Var | Default | Meaning |
|-----|---------|---------|
| `NEXT_PUBLIC_API_BASE_URL` | `http://localhost:8000` | backend base URL |
| `NEXT_PUBLIC_SITE_URL` | `http://localhost:3000` | deployed frontend origin — drives `metadataBase`, canonical/OG URLs, `sitemap.ts`, `robots.ts`. **A real deploy MUST set this** (else share cards/sitemap point at localhost — spec/11 §3, M18). |

---

## 7. License summary (ship-blocking facts)

| Component | License | Commercial? |
|-----------|---------|-------------|
| mayavius source code | **MIT** (D2) | ✅ |
| three.js, Spark, R3F, drei, zustand, Next, Tailwind | MIT | ✅ |
| `VGGT-1B` weights (default) | cc-by-nc-4.0 | ❌ NC |
| `VGGT-1B-Commercial` weights | `vggt-aup-license` (no military, gated) | ✅* non-OSI, gated |
| `cotracker3` weights | cc-by-nc-4.0 | ❌ NC |
| SpatialTrackerV2 code+weights | CC-BY-NC-4.0 | ❌ NC |
| Pi3 code / weights | BSD-3 / CC-BY-NC-4.0 | code ✅ / weights ❌ |
| OpenD4RT | Apache-2.0 | ✅ |

**MVP ships with NC weights, clearly labeled** (D2). The README and the `/jobs`
metadata surface `weights_license` for the active adapter. No commercial-friendly
tracker exists in the set — the track-ribbon feature is research/NC for now.

---

## 8. What must never be committed (handover; `.gitignore` already covers)
Model weights (`*.pt/*.pth/*.ckpt/*.safetensors/*.onnx`), `.venv/`,
`node_modules/`, `.next/`, env files (`.env*` except `.env.example`),
downloaded sample-video binaries beyond the curated corpus.
