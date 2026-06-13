# 11 — Deployment & Launch

How mayavius ships and how it earns stars. Two deployment targets and a launch
playbook. **The local Mac path is primary and self-sufficient; everything in
§2 (cloud GPU) is optional.** Decisions referenced: D1 (VGGT + CoTracker3),
D2 (MIT code / NC weights), D9 (HF Space, dedicated GPU), D10 (3–4 sample
clips). Facts come from [08-dependencies-and-env.md](08-dependencies-and-env.md)
and [decisions/decision-log.md](decisions/decision-log.md); the wire format is
[05-data-contract.md](05-data-contract.md); the sample corpus lives in
[10-testing-strategy.md](10-testing-strategy.md).

> **Order of priority:** ship the local path → publish the README + GIF →
> stand up the hosted demo with preloaded examples → coordinate the launch.
> A stranger must reach a *shareable, screenshot-able result in ten seconds*
> (handover §5). That is the entire star mechanic.

---

## 1. LOCAL deployment — primary, "runs on a Mac, no cloud"

This path is itself a README selling point and the only path the MVP **requires**.
It must work on a 36 GB Apple-Silicon Mac with **no cloud GPU**, MPS (fp32), the
default **VGGT + CoTracker3** combo, and NC weights.

### 1.1 The three-command story (the README's "Quick start")

| Step | Command | What it does |
|------|---------|--------------|
| 1 | `make setup` | venv (Python 3.12) + lightweight backend deps; `npm install` frontend |
| 2a | `make dev-backend` | `uvicorn app.main:app --reload --port 8000` |
| 2b | `make dev-frontend` | `next dev` → http://localhost:3000 |
| 3 | open **http://localhost:3000** | landing page; drop a clip → watch it reconstruct |

`make setup` installs the **lightweight** backend only (no torch). The heavy ML
deps are a deliberate, separate one-time step (§1.2) so cloning + first build is
fast and the 501-stubbed scaffold runs immediately.

> **Two Makefile edits required at build time** (flagged in
> [08-dependencies-and-env.md](08-dependencies-and-env.md) §1, §3): `setup-backend`
> must use `python3.12 -m venv .venv` (currently `python3`), and a `setup-ml`
> target should wrap §1.2. These are the only deployment-driven Makefile changes;
> do not rearchitect the existing targets.

### 1.2 One-time ML setup (heavy, separate, never committed)

Weights are multi-GB and **never committed** (`.gitignore` covers
`*.pt/*.pth/*.ckpt/*.safetensors/*.onnx`). Run once after the lightweight path
works, exactly per [08-dependencies-and-env.md](08-dependencies-and-env.md) §4:

```bash
cd backend
./.venv/bin/pip install torch torchvision          # Mac wheels = CPU+MPS, no CUDA index
./.venv/bin/pip install -r requirements-ml.txt      # numpy/opencv/imageio/hf-hub/einops + VGGT + CoTracker3
export PYTORCH_ENABLE_MPS_FALLBACK=1                 # set BEFORE torch import (adapter also sets it)
make dev-backend
```

First reconstruction triggers a one-time `huggingface-hub` weight pull
(`facebook/VGGT-1B`, `facebook/cotracker3`). Default weights are **NC** and
ungated — no HF login needed for local dev (the commercial gated path is §2.4).

| Concern | Local resolution |
|---------|------------------|
| Device | `MAYAVIUS_DEVICE=mps` (default); `cpu` fallback works, slower |
| Precision | **fp32** on MPS by choice (C3 — half-precision MPS is beta/incomplete for these ports), not because MPS can't do fp16 |
| Unimplemented MPS op | `PYTORCH_ENABLE_MPS_FALLBACK=1` routes it to CPU. If an op still hard-fails (e.g. Conv1d gaps), the documented remedy is "run that adapter on the optional cloud Space (§2)", **not** block the MVP |
| Clip length | `MAYAVIUS_MAX_CLIP_FRAMES=24` (≤64), subsampled — short clips only (handover §4.6) |
| Memory | measured on the actual 36 GB Mac at the [10-testing-strategy.md](10-testing-strategy.md) gate; do **not** assert per-frame GB numbers (unverified). Repo floor: VGGT runs on 8 GB+ RAM |

### 1.3 What "no cloud required to VIEW" means

The viewer is a static client (`THREE.Points` + custom shader + Line2 ribbons,
Path 1). A **shared result link** (`/view/[id]`) only needs the MV4D blob over
HTTP + the static frontend — **no GPU on the viewing device**. Only the *initial
reconstruction* needs MPS/GPU. This split is the whole pitch: heavy backend,
cheap shareable viewer (CLAUDE.md "compute is asymmetric").

---

## 2. OPTIONAL GPU deployment — Hugging Face Space (D9)

For a hosted demo that feels snappy to strangers. **Optional**: the local path
works without any of this. The same hexagonal backend, the same adapters, the
same MV4D wire format — only `MAYAVIUS_DEVICE=cuda` and (optionally) a richer
active adapter differ. **Swapping the device/host must not touch
`app/core`** (the hexagonal mandate holds in deployment too).

### 2.1 Why HF Space + dedicated GPU (not Spaces free tier, not Vercel-for-backend)

| Option | Verdict | Why |
|--------|---------|-----|
| **HF Space, dedicated GPU** | **chosen (D9)** | persistent GPU, CUDA, Docker SDK, free model-card/weight ecosystem, audience already on HF; SpatialTrackerV2's CUDA-only path becomes runnable here |
| HF Space, free CPU tier | rejected | no GPU → VGGT/CoTracker too slow; defeats "snappy for strangers" |
| Vercel / serverless for backend | rejected | no persistent GPU, cold starts, multi-GB weights blow function limits; Vercel is for the *frontend* (§3) |
| Self-managed cloud VM | alternative | more control/cost, more ops; use only if HF GPU quota/cost is blocking |

### 2.2 Container & dependency layout

A `backend/Dockerfile` (build artifact; **build from the repo root** so the
`COPY backend/ …` paths resolve) for the Space. Python 3.12 (D4) is **not** in
the jammy base repos, so add the deadsnakes PPA and run everything inside a 3.12
venv (the system `python3` is 3.10 — never install or run under it):

```dockerfile
# build context = repo root:  docker build -f backend/Dockerfile .
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y software-properties-common ffmpeg \
 && add-apt-repository -y ppa:deadsnakes/ppa && apt-get update \
 && apt-get install -y python3.12 python3.12-venv \
 && rm -rf /var/lib/apt/lists/*
# Dedicated 3.12 venv → every pip install AND the CMD use 3.12, not system 3.10:
RUN python3.12 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
WORKDIR /app
COPY backend/requirements.txt backend/requirements-ml.txt ./
# CUDA wheels (cu124), NOT the Mac CPU+MPS wheels — device-specific install:
RUN pip install --upgrade pip \
 && pip install --index-url https://download.pytorch.org/whl/cu124 torch torchvision \
 && pip install -r requirements.txt -r requirements-ml.txt
COPY backend/ ./
ENV MAYAVIUS_DEVICE=cuda
ENV MAYAVIUS_CORS_ORIGINS='["https://<frontend-host>"]'
EXPOSE 7860
CMD ["uvicorn","app.main:app","--host","0.0.0.0","--port","7860"]
```

| Item | Local (Mac) | Space (CUDA GPU) |
|------|-------------|------------------|
| torch wheels | CPU+MPS (`pip install torch`) | `cu124` index-url |
| `MAYAVIUS_DEVICE` | `mps` | `cuda` |
| Default adapter | `vggt+cotracker3` | `vggt+cotracker3` (same default) |
| Richer optional adapter | n/a (CUDA-only) | `SpatialTrackerV2Adapter` (single-model dynamic 3D tracks + geometry) |
| Port | 8000 | 7860 (HF Space convention) |
| Weight pull | first job | bake into image **or** HF cache volume |

Adapters do **not** change between hosts — `MAYAVIUS_ADAPTER` selects the combo;
the CUDA-only ones simply become *available* on the Space (negative knowledge
from §2.3 is why they aren't local).

### 2.3 Richer dynamic tracks on CUDA (and why not on Mac)

On the GPU Space, `SpatialTrackerV2Adapter` produces dynamic 3D tracks +
geometry from a single model — richer ribbons than the lift-2D-to-3D default.

> **Negative knowledge (do not relitigate):**
> - **SpatialTrackerV2 is CUDA-only** — upstream pins `torch==2.4.1+cu124`; not
>   Mac-installable. Cloud/Space adapter only.
> - **Pi3 / π³ has no official MPS path** (PR #153 unmerged) — never the Mac
>   default; on the Space it is GPU-runnable but adds no tracks (static only).
> - **OpenD4RT** (`Lijiaxin0111/Open-d4rt`, Apache-2.0) is GPU/PyTorch-oriented,
>   **MPS unverified** — Space-only until tested.
> - **MegaSaM** is rejected entirely (optimization-based SLAM, ~0.7 FPS, not a
>   single feedforward pass) — wrong for an interactive viewer regardless of host.

### 2.4 License gate on deployment (ship-blocking)

The host's commercial posture **must** match the active weights' license. The
adapter layer surfaces `AdapterInfo.weights_license`
([06-backend-spec.md](06-backend-spec.md)); deployment gates on it, and the
`/jobs` metadata + README label the active license (D2).

| Active weights | License | Allowed host posture |
|----------------|---------|----------------------|
| `facebook/VGGT-1B` (default) | cc-by-nc-4.0 | **non-commercial Space only** |
| `facebook/cotracker3` (default) | cc-by-nc-4.0 | **non-commercial Space only** |
| `SpatialTrackerV2` | code CC-BY-NC-4.0; weights unconfirmed → treated as NC (decision-log §D) | non-commercial only |
| `Pi3` weights | cc-by-nc-4.0 (code BSD-3) | non-commercial (weights) |
| `OpenD4RT` | Apache-2.0 | commercial OK |
| `facebook/VGGT-1B-Commercial` | `vggt-aup-license` (no military, **gated**) | commercial host, static-only |

**Rules:**
1. The default MVP demo Space runs **NC weights** → it is a **non-commercial**
   demo, labeled as such on the Space card + in-app.
2. **No commercial-friendly tracker exists** → the **track-ribbon feature is
   research/NC** on any host. A commercial deployment can run **static-only**
   VGGT-1B-Commercial but **cannot** ship motion ribbons until a permissive
   tracker is sourced. State this plainly; do not hide it.
3. **VGGT-1B-Commercial gating:** a commercial host must complete the HF AUP
   access form first (an HF token with access), then set the weights repo id.
   Document in the README; the *default* path avoids gating so local dev is
   frictionless.

### 2.5 Operational notes for the Space

- **SSE caveat (C7):** progress streaming uses FastAPI built-in
  `from fastapi.sse import EventSourceResponse, ServerSentEvent` (**not**
  `sse-starlette`). SSE is incompatible with `GZipMiddleware` — **compress the
  result blob at the `/jobs/{id}/result` route**, not via GZip middleware.
- **Result caching:** results are immutable → serve with brotli/gzip +
  `Cache-Control: public, max-age=31536000, immutable` so shareable links stay
  fast and cheap ([05-data-contract.md](05-data-contract.md) §4).
- **CORS:** set `MAYAVIUS_CORS_ORIGINS` to the deployed frontend origin (default
  `["http://localhost:3000"]` is local-only).
- **Concurrency:** the async job queue (`app/jobs/queue.py`) serializes GPU work;
  one in-flight reconstruction at a time on a single-GPU Space (queue the rest).
- **Cost guard:** keep `MAYAVIUS_MAX_CLIP_FRAMES` / `MAYAVIUS_MAX_UPLOAD_MB`
  tight on the public Space; reconstruction is the only expensive operation.

---

## 3. FRONTEND deployment (static/SSR) → points at the Space

The frontend is a Next.js 16 App Router app. It deploys independently of the
backend and needs **two** env vars ([08 §6](08-dependencies-and-env.md)):
`NEXT_PUBLIC_API_BASE_URL` (the backend/Space) **and** `NEXT_PUBLIC_SITE_URL` (the
deployed frontend origin — drives `metadataBase`, canonical + OG/Twitter image
URLs, `sitemap.ts`, `robots.ts`). Setting only the API base ships share cards and
sitemap pointing at `localhost:3000` — breaking the very SEO/share surface §3 is
about. The viewer is client-only and GPU-free.

| Target | When | How |
|--------|------|-----|
| **Local `next dev`** | development (primary) | `make dev-frontend`; `NEXT_PUBLIC_API_BASE_URL=http://localhost:8000`, `NEXT_PUBLIC_SITE_URL=http://localhost:3000` |
| **Vercel (SSR)** — recommended host | live demo | needs SSR for `generateMetadata` share cards on `/view/[id]`; set **both** `NEXT_PUBLIC_API_BASE_URL` (the Space) **and** `NEXT_PUBLIC_SITE_URL` (the deployed origin); landing + sitemap/robots are server-rendered for SEO |
| **Static export** — alternative | if no SSR host | `next build` with static export; **trade-off:** loses per-result dynamic `generateMetadata` (share cards degrade to a single static card). Use only if Vercel/SSR is unavailable; SSR is preferred because the rich per-result share card is the virality surface |

The SEO/share surfaces (`app/page.tsx` static landing, `app/view/[id]/page.tsx`
async-`params` + `generateMetadata`, `sitemap.ts`, `robots.ts`) are already
scaffolded; deployment just needs an SSR-capable host to realize per-result
cards. **The local path needs none of §3 — `make dev-frontend` is enough.**

---

## 4. Example-video set (the preloaded "magic before upload")

Magic must be visible **before** any upload. The hosted demo (and the README
GIF) lead with **3–4 stunning preloaded examples** (D10) the user can open in
one click → instant `/view/[id]` result, no waiting, no GPU.

- **Corpus authority:** the set is defined and validated in
  [10-testing-strategy.md](10-testing-strategy.md) (3–4 short ≤3 s CC-licensed
  clips). This file does **not** redefine it.
- **Selection criteria (for picking the demo subset):** (1) clear moving
  foreground over a stable background (shows ribbons + static cloud split — the
  signature visual); (2) ≤3 s so MV4D stays under the ~12 MB payload target;
  (3) CC-licensed, attribution recorded; (4) visually striking as a still
  (it becomes the share-card image).
- **Form on the demo:** ship each as a **pre-baked MV4D blob** served at a
  stable `/view/<slug>` permalink (no reconstruction at view time). Concretely: the
  backend lifespan **seeds** each `assets/samples/<slug>.mv4d` as a terminal `done`
  job via `JobQueue.seed_example` ([06 §6](06-backend-spec.md)), so
  `GET /jobs/<slug>/result` resolves after every boot and `/view/<slug>` uses the
  same fetch+decode path as a real result. This is what makes the demo instant and
  the link screenshot-able.
- **Never commit** the source video binaries beyond the curated corpus
  (handover; `.gitignore`). The pre-baked MV4D demo blobs are small enough to
  ship as release assets / served by the Space.

---

## 5. STAR MECHANICS — the launch playbook (handover §5/§8)

> **README top line (verbatim target):** *open-source, lightweight, interactive,
> no GPU required to view, running the actual frontier 4D research models — drop
> in your own video.*

Apps go viral when a stranger can try them in **ten seconds** and get a
shareable, screenshot-able result. The required surfaces:

| # | Surface | Spec |
|---|---------|------|
| 1 | **README opens with an animated GIF** | the GIF *is* the pitch: orbit + timeline scrub + a moving object trailing ribbons over a static cloud (bullet-time freeze-and-orbit beat). Above the fold, before any prose. Loop ≤6 s |
| 2 | **Hosted link, 3–4 preloaded examples** | magic before upload (§4); one-click `/view/<slug>` |
| 3 | **Shareable result URLs** | `/view/[id]` with per-result `generateMetadata` rich share card (the virality surface; needs SSR host §3) |
| 4 | **"Runs locally on a Mac" path** | the §1 three-command quick start, prominent for the dev/HN crowd ("no GPU required to view; MPS for your own clips") |
| 5 | **Coordinated launch** | HN + r/computervision + X, same day. Template = the **"Show HN: Spark"** thread (lead with the visual, the live link, and the "try your own" hook; be present to answer in-thread) |

**README structure (build the README to this skeleton):**
1. GIF (the pitch) + one-line tagline (the top line above).
2. Live demo link + "try a preloaded example" buttons.
3. "Runs locally on a Mac" three-command quick start (§1.1).
4. What it is: feedforward 4D — colored point cloud + 3D track ribbons, Path 1.
5. How it works: drop clip → FastAPI async job → MV4D blob → client viewer.
6. **License honesty block:** MIT code; **default weights are non-commercial**
   (VGGT-1B, cotracker3 = cc-by-nc-4.0), clearly labeled; commercial static-only
   path = VGGT-1B-Commercial (gated AUP); no commercial tracker exists yet (D2).
7. Architecture one-liner + link to [04-architecture.md](04-architecture.md)
   (hexagonal, swappable adapters; Path 2 / Spark 4DGS designed-for, not built).
8. Roadmap: Path 2 (Spark `<SplatMesh>` at the `Scene4D` seam), OpenD4RT adapter,
   commercial tracker — framed as future direction, not MVP scope.

**Launch-day checklist:**
- [ ] Demo Space is up, warm, and the 3–4 examples open instantly.
- [ ] README GIF renders on GitHub (file-size sane, autoplay-on-load).
- [ ] Share card verified — paste a `/view/[id]` link into X/Slack → OG card renders. The per-result dynamic image is `app/view/[id]/opengraph-image.tsx` via `next/og` (owned by [07 §8](07-frontend-spec.md), 1200×630, result thumbnail + title). **MVP-acceptable fallback:** if the frame-thumbnail pipeline isn't ready, ship the static `/og.png` with per-result `title`/`description` — the card still renders.
- [ ] License label visible in-app and in README.
- [ ] §6 novelty sweep re-run and findings recorded.
- [ ] HN/r-cv/X posts drafted; author available for the first few hours.

---

## 6. Pre-launch novelty sweep (do this right before launch)

This is a **fast-moving space**; the differentiation is the **combination**
(open + feedforward + point-cloud + track ribbons + client-only shareable viewer
+ GPU-free viewing), not any single component. Absence of a competitor is
unfalsifiable, so re-verify immediately before the post and **state novelty
defensively ("none surfaced as of <date>"), never absolutely** (decision-log §F).

**Sweep procedure (record dated findings in the decision log):**

| Source | Query intent |
|--------|--------------|
| GitHub | open browser-upload feedforward 4D viewer; shareable 4D result links; "video to 4D" web apps |
| arXiv / Papers with Code | new feedforward 4D-from-casual-video (e.g. MoRe-class, TracksTo4D successors) |
| Hugging Face Spaces | VGGT/Any4D/D4RT-style demo Spaces — note: server-side, transient, no permalink (the gap we fill) |
| D4RT status | official Google DeepMind D4RT still unreleased? (was unreleased 2026-06-13); OpenD4RT progress |
| X / blogs | "Show HN"-style launches of anything adjacent |

**If a direct open competitor to *this exact app* has appeared:** surface it,
sharpen the differentiation in the README and the launch post, **do not silently
proceed** (handover §2). If none surfaced, ship — and say "none surfaced as of
\<launch date\>".

---

## 7. Deployment summary

| Concern | Local (primary) | HF Space (optional) | Frontend host |
|---------|-----------------|---------------------|---------------|
| Required for MVP? | **yes** | no | no (local `next dev` suffices) |
| Device | MPS (fp32) / CPU | CUDA | n/a (client-only viewer) |
| Adapters | vggt+cotracker3 | + SpatialTrackerV2 / Pi3 / OpenD4RT | n/a |
| Weights license gate | NC, labeled | NC demo (or gated commercial static-only) | n/a |
| GPU to **view** result | **no** | no | no |
| GPU to **reconstruct** | MPS | CUDA | n/a |
| Touches `app/core`? | **no** | **no** | n/a |

The local Mac path is the product's spine; the Space makes it snappy for
strangers; the frontend host makes results shareable. The star mechanics turn
all three into GitHub stars.
