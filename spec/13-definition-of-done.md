# 13 — Definition of Done

**Objective, checkable acceptance gate for the mayavius MVP** (the *project*, not
this spec session). The MVP is **done** when every box below is green — each item
is binary and, where possible, carries a command whose exit/observed result is the
proof. Aspirational language is banned: if it isn't checkable on **the 36 GB
Apple-Silicon Mac**, it isn't in this list.

Authority cross-refs: locked decisions [03-decisions-locked.md](03-decisions-locked.md);
wire format [05-data-contract.md](05-data-contract.md); deps/env
[08-dependencies-and-env.md](08-dependencies-and-env.md). The wave acceptance
tests live in `spec/09` (waves) and the test stack in `spec/10` (tests); this file
*aggregates* their gates, it does not redefine them — if `spec/09`/`spec/10` and
this file ever disagree, those win on their own subject and this file is updated.

How to read a row: **[ ]** = not done, **[x]** = verified done. A reviewer signs
off only when the listed command/observation passes on a clean checkout of the
target Mac (`MAYAVIUS_DEVICE=mps`, default adapter `vggt+cotracker3`).

---

## 0. One-command summary gate

Done means **all** of the following return success on a fresh clone + `make setup`
+ ML install (§08.4):

```bash
make test            # backend pytest (no mps/gpu) + frontend vitest + tsc (per the W0.T4 Makefile; §1, §2, §7)
make lint            # frontend eslint clean — separate target, NOT part of `make test` (§7)
( cd frontend && npx tsc --noEmit )   # 0 type errors (§7)
( cd backend && ./.venv/bin/pytest -q )  # 0 failures incl. hexagonal + conformance (§1)
```

If any of those is non-zero, the MVP is **not** done regardless of the boxes below.

---

## 1. Backend tests green (pytest 9)

| # | Gate | Verification | Source |
|---|------|--------------|--------|
| 1.1 | [ ] Full backend suite passes | `cd backend && ./.venv/bin/pytest -q` → exit 0, **0 failures/errors** | spec/10 |
| 1.2 | [ ] **Hexagonal import test** passes — `app.core.*` imports **no** module in the banned set `{fastapi, starlette, torch, uvicorn, numpy.*cuda, app.adapters.*}` (NumPy non-cuda is allowed) — the exact set in [10 §1.2 T-130](10-testing-strategy.md) | dedicated test walks `app.core`'s import graph and asserts none appear; `cd backend && ./.venv/bin/pytest -q -k hexagonal` → exit 0 | CLAUDE.md "pure core"; [03 hard constraints](03-decisions-locked.md) (handover §3 hexagonal mandate) |
| 1.3 | [ ] Core depends only on `ReconstructionPort` — `app/core/services/reconstruction_service.py` references `app.core.ports.reconstruction_port.ReconstructionPort`, never a concrete adapter class | `grep -RnE "VggtAdapter|CoTracker3Adapter|SpatialTrackerV2Adapter|Pi3Adapter|OpenD4RTAdapter|import torch|import fastapi" backend/app/core/` → **no matches** | CLAUDE.md hexagonal |
| 1.4 | [ ] No stub remains reachable on the default happy path — `submit_job`/`get_job`/`get_result` (`app/api/routes/jobs.py`) no longer `raise HTTPException(501)`; encoder no longer `raise NotImplementedError` | `grep -RnE "501_NOT_IMPLEMENTED|NotImplementedError" backend/app/api backend/app/wire backend/app/core` → no matches on the default combo path (optional adapters §3.x may still 501 by design) | spec/06 |
| 1.5 | [ ] Encoder consumes `Scene4D` (not the placeholder `ReconstructionResult`) and returns an `MV4D` v1 buffer | unit test builds a `Scene4D`, calls `encode_reconstruction(scene) -> bytes`, asserts magic `MV4D`, `version==1`, `posBits==16` | [05 §5.1](05-data-contract.md) |

---

## 2. MV4D round-trip & version parity (the wire-format seam)

| # | Gate | Verification | Source |
|---|------|--------------|--------|
| 2.1 | [ ] **MV4D_VERSION parity** — backend and frontend both export the constant and it equals `1` | `grep -RnE "MV4D_VERSION\s*=\s*1" backend/app/wire/encoder.py frontend/src/lib/wire/decoder.ts` → matches in **both** files | [05 §7](05-data-contract.md) |
| 2.2 | [ ] **Cross-format conformance test passes** — a fixture `MV4D` blob (checked into the test corpus) decodes in `decoder.ts` to the *exact* counts/AABB/positions the encoder produced from the same `Scene4D` | Vitest reads the shared fixture; asserts `frameCount`, `aabbMin/Max`, `static.count`, `dynamic.frames[*].count`, `tracks.count`, and a sampled dequantized position match within ±1 quantum | [05 §7](05-data-contract.md); spec/10 |
| 2.3 | [ ] Byte-for-byte stability — re-encoding the canonical fixture `Scene4D` yields a buffer identical to the committed golden `.mv4d` | `cd backend && ./.venv/bin/pytest -q -k golden` → exit 0 | [05](05-data-contract.md) "byte-for-byte compatible" |
| 2.4 | [ ] Decoder error contract honored — bad magic, `version!=1`, `posBits!=16`, OOB/misaligned section each throw a typed `Mv4dDecodeError`; never a partial scene | Vitest cases for each malformed input assert a throw | [05 §8](05-data-contract.md) |
| 2.5 | [ ] Caps enforced — culls `T>64`, `N_s>150k` (lowest `static_conf`), dynamic `>20k/frame` (**deterministic fixed-seed uniform subsample**, [06 §5 step 7](06-backend-spec.md)), `M>4096`; logs actual counts; target payload ≤ 12 MB (hard ceiling 24 MB) | unit test feeds over-cap arrays, asserts cull/subsample + a logged size line | [05 §4](05-data-contract.md) |
| 2.6 | [ ] Coordinate convention — encoded positions are right-handed +X right/+Y up/−Z forward; the adapter (not the frontend) does the transform | conformance fixture asserts a known world point lands where Three.js expects it | [05 §2](05-data-contract.md) |

---

## 3. End-to-end on the 36 GB Apple-Silicon Mac (the real product)

The headline gate. **No cloud GPU.** `MAYAVIUS_DEVICE=mps`, default adapter
`vggt+cotracker3`, weights `facebook/VGGT-1B` + `facebook/cotracker3` (NC, §08.7).

| # | Gate | Verification | Source |
|---|------|--------------|--------|
| 3.1 | [ ] Both servers start clean | `make dev-backend` (uvicorn :8000) and `make dev-frontend` (:3000) run without error; `curl -fsS localhost:8000/health` → 200 | [08 §6](08-dependencies-and-env.md) |
| 3.2 | [ ] **MPS is actually used** | backend log on job start prints `device=mps`; `python -c "import torch;print(torch.backends.mps.is_available())"` → `True` | [08 §4.1](08-dependencies-and-env.md), [03 hard constraints](03-decisions-locked.md) |
| 3.3 | [ ] **Upload → job id** — `POST /jobs` (multipart `UploadFile`) on a short clip returns **202** + a `job_id` | `curl -fsS -X POST -F clip=@assets/samples/<clip>.mp4 localhost:8000/jobs` → 202 + JSON with non-empty **`job_id`** (spec/06 §7) | [handover §4.4](03-decisions-locked.md); spec/06 |
| 3.4 | [ ] **Reconstruct on MPS** — `GET /jobs/{id}` is pollable, reports `status` + `progress ∈ [0,1]`, and reaches `done`; the SSE stream is the dedicated route **`GET /jobs/{id}/stream`** (`response_class=EventSourceResponse`, `fastapi.sse`, not `sse-starlette` — spec/06 §7) | poll loop ends in `status:done`; `curl -N localhost:8000/jobs/{id}/stream` streams `ServerSentEvent`s ending in a terminal `done` event | [03 C7](03-decisions-locked.md); [08 §3](08-dependencies-and-env.md) |
| 3.5 | [ ] **Result is a binary MV4D blob** — `GET /jobs/{id}/result` returns `application/octet-stream` beginning with `MV4D`, with an immutable cache header (compression is the serving layer's job; the app adds **no** `GZipMiddleware`, spec/06 §7) | `curl -fsS localhost:8000/jobs/{id}/result -o out.mv4d`; `head -c4 out.mv4d` == `MV4D`; size ≤ 12 MB target | [05 §4](05-data-contract.md); spec/06 |
| 3.6 | [ ] **Interactive playback** in the browser — open `/view/{id}`: the static cloud renders, then dynamic frames + track ribbons; **orbit**, **timeline scrub**, **play/pause**, **loop**, and **bullet-time** (freeze current frame + orbit) all work | manual + Playwright e2e (spec/10): drive each control, assert canvas updates / store state | handover §1; [03 render Path 1](03-decisions-locked.md) |
| 3.7 | [ ] **No-GPU-to-view** — the same `/view/{id}` loads and is fully interactive on a machine with no discrete GPU / no MPS / backend offline (decode from a saved blob) | load `out.mv4d` via the viewer with backend stopped; all controls work | handover §2; README line (§6) |
| 3.8 | [ ] Active **weight license surfaced** — `/jobs` metadata returns the running adapter's `weights_license` (e.g. `cc-by-nc-4.0`) | `GET /jobs/{id}` (or result metadata) contains `weights_license` | [03 D2](03-decisions-locked.md); [08 §7](08-dependencies-and-env.md) |

---

## 4. Default adapter combo produces a real Scene4D

| # | Gate | Verification | Source |
|---|------|--------------|--------|
| 4.1 | [ ] `VggtAdapter` yields a non-empty **static cloud** (colored points + depth + camera) on MPS, fp32, via the community-port pattern (`device="mps"`, **no** cuda autocast, frames rescaled to width 518) | adapter test on a sample clip → `Scene4D.static_positions.shape[0] > 0`, `static_colors` present, `cameras` non-None | [08 §4.3](08-dependencies-and-env.md); [03 D1](03-decisions-locked.md) |
| 4.2 | [ ] `CoTracker3Adapter` yields **≥ 1 dynamic 3D track ribbon** — 2D tracks lifted to 3D via VGGT depth+intrinsics, first-class MPS | adapter test → `Scene4D.tracks.positions.shape[0] >= 1`, `visibility` present | [08 §4.4](08-dependencies-and-env.md); [03 D1](03-decisions-locked.md) |
| 4.3 | [ ] The combined `ReconstructionService` returns a `Scene4D` with **static cloud + ≥1 dynamic track ribbon** that survives encode → decode → render | full-path test: service → `encode_reconstruction` → `decodeReconstruction` → Scene4D mounts in `Scene.tsx` with a visible ribbon | [05 §5](05-data-contract.md) |
| 4.4 | [ ] **Negative-knowledge gates respected** — optional adapters do **not** run on the Mac default and are documented dead ends, not silent failures | `Pi3Adapter`/`SpatialTrackerV2Adapter`/`OpenD4RTAdapter` raise a clear "not a Mac/MPS default — see spec/08 §4.5–4.7" error when selected with `MAYAVIUS_DEVICE=mps`; π³ = no MPS, SpatialTrackerV2 = CUDA-only documented in code | [08 §4.5–4.7](08-dependencies-and-env.md); [decision-log §E](decisions/decision-log.md) |

---

## 5. Sample corpus renders with expected qualitative results

3–4 short (≤ 3 s) CC-licensed clips bundled as preloaded examples (D10). Each has a
written **expected qualitative result** in `spec/10`/`spec/11`; "done" = each clip
matches its description, not a pixel diff.

| # | Gate | Verification |
|---|------|--------------|
| 5.1 | [ ] 3–4 sample clips present + CC-licensed + recorded in the corpus manifest; **no video binaries committed beyond the curated corpus** | manifest lists each clip + its license + its expected result |
| 5.2 | [ ] Each clip reconstructs end-to-end (§3) and renders its **expected** result (e.g. "static room + one moving hand traces a ribbon", "background stable while subject walks"). The moving subject animates as a **dense** colored cluster (the moving subset of VGGT's per-frame world points, [06 §5 step 5](06-backend-spec.md)); if the build took the logged **sparse fallback**, the qualitative description still holds (ribbon-led motion over a stable cloud) and the choice is recorded. | run each through `/view/{id}`; observer confirms the description; recorded in spec/10 acceptance |
| 5.3 | [ ] At least one sample is the **README GIF** source (§6.1) | the committed GIF is produced from a corpus clip |

---

## 6. README & virality surface

| # | Gate | Verification |
|---|------|--------------|
| 6.1 | [ ] README **opens** with an animated GIF of the viewer in action | first screenful of `README.md` references an in-repo/asset GIF; renders on GitHub |
| 6.2 | [ ] README states **"runs locally on a Mac"** and a **no-GPU-to-view** line near the top | `grep -niE "runs locally on a Mac|No GPU required to view" README.md` → matches in the opening section |
| 6.3 | [ ] README/`/jobs` clearly label the default **NC weight license** (honest redistribution posture, D2) | `grep -niE "cc-by-nc|non-commercial" README.md` → present |
| 6.4 | [ ] **Shareable `/view/[id]` links** load with correct share-card metadata — per-result `generateMetadata` (async `params`) emits OG/Twitter title+image+description | `curl -fsS localhost:3000/view/<id>` HTML contains `og:title`/`og:image`/`twitter:card`; Playwright asserts non-default per-result values | [03 D5 / handover §4.3](03-decisions-locked.md); CLAUDE.md |
| 6.5 | [ ] `sitemap.ts` + `robots.ts` resolve and landing `/` is indexable (Server Component, no `'use client'`) | `curl -fsS localhost:3000/sitemap.xml` + `/robots.txt` → 200; `grep -L "use client" frontend/src/app/page.tsx` confirms server component |

---

