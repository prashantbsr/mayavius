# 09 — Execution Plan (waves → tasks)

The build order. Five waves; each wave is a **human-gated boundary** (do not start
wave N+1 until every task in wave N passes its acceptance tests). Within a wave,
tasks may run in parallel (Claude-allowed transitions). Every task lists: **files**
it creates/edits, **acceptance criteria**, and the **exact verify command** — whose
green `T-xxx` ids ([10-testing-strategy.md](10-testing-strategy.md)) are the
definition of "task done". The mirror handover for the build session is
[../EXECUTE.md](../EXECUTE.md).

**Principle:** Waves 0–2 ship the entire app **with no ML** (a `FakeAdapter` + a
fixture-mode backend), so the wire seam, API, and viewer are fully proven before a
single multi-GB weight is downloaded. ML lands in Wave 3, behind the on-device MPS
gate. This makes the hard constraint (runs on a 36 GB Mac) the *last* risk retired,
on top of an already-tested app.

Stub convention (handover): every not-yet-built handler **returns 501 / raises
`NotImplementedError`** with a message pointing at the relevant spec section — never
a silent no-op. `T-301` asserts the 501 state pre-implementation and flips to the
2xx lifecycle tests at the Wave-1 cutover.

---

## Wave overview

| Wave | Theme | ML? | Exit gate (all green) |
|------|-------|-----|------------------------|
| **W0** | Contracts: domain, port, MV4D encoder+decoder (both langs), test harness | no | `T-100–103`, `T-105–107`, `T-120–122`, `T-130`, `T-150–155`, `T-160`, `T-180`, `T-200–203` (`T-104` caps → **W1.T1**, where `enforce_caps` lives) |
| **W1** | Backend: service post-process + async job queue + FastAPI surface (FixtureAdapter) | no | `T-300`, `T-302–308`, `T-310` (`T-301` flips/retires at the W1 cutover) |
| **W2** | Frontend: viewer (point cloud, ribbons, HUD, playback, bullet-time) + landing/share (fixture-mode) | no | `T-400–407`, `npm run build/lint`, `tsc` (`T-601` corpus listing is a **W4** gate — W2 wires the gallery to one seeded fixture example) |
| **W3** | ML adapters on MPS: VGGT + CoTracker3 combo (the on-device gate) | **yes** | `T-500`, `T-510`, `T-511`, gated `T-310` |
| **W4** | Sample corpus, launch assets (README/GIF/LICENSE), aesthetics polish, optional deploy | yes | `T-600`, `T-601`, [13-definition-of-done.md](13-definition-of-done.md) fully checked |

---

## Wave 0 — Contracts & test harness (no ML)

The wire seam is the spine; build it first and prove it round-trips in both
languages before anything depends on it.

| Task | Files (create/edit) | Acceptance | Verify |
|------|---------------------|-----------|--------|
| **W0.T1 Domain + port + errors** | `backend/app/core/domain/models.py` (replace placeholder → `Scene4D`/`Tracks`/`CameraTrack`/`ReconstructionRequest`), `core/domain/errors.py` (new, error hierarchy), `core/ports/reconstruction_port.py` (final port + `AdapterInfo` + `ProgressSink`), `backend/tests/fakes/fake_adapter.py` (new) | Port matches [06 §2](06-backend-spec.md); `FakeAdapter` is a concrete `ReconstructionPort` returning a deterministic `Scene4D`; **core imports no fastapi/torch/adapter** | `T-120 T-121 T-122 T-130` |
| **W0.T2 MV4D encoder + Python ref decoder** | `backend/app/wire/encoder.py` (implement `encode_reconstruction` + `MV4D_VERSION=1`), `backend/app/wire/decoder.py` (new Python reference decoder, mirrors [05 §3](05-data-contract.md)), `backend/tests/wire/*` | Encoder emits MV4D v1 per [05](05-data-contract.md); Python encode→decode round-trips within quant tolerance; the encoder assumes an already-capped `Scene4D` (does NOT cull — caps are W1.T1's `enforce_caps`, [06 §5](06-backend-spec.md)); (T-200 byte-stability is gated in W0.T4, which commits the fixture) | `T-100 T-101 T-102 T-103 T-105 T-106 T-107` |
| **W0.T3 Frontend decoder + types + client retype** | `frontend/src/lib/wire/decoder.ts` (implement → `Mv4dScene`, zero-copy, `MV4D_VERSION=1`, `Mv4dDecodeError`), `frontend/src/types/index.ts` (delete placeholder `ReconstructionResult` → `Mv4dScene`; rename `JobStatus` `succeeded`→`done`), **`frontend/src/lib/api/client.ts`** (retype `fetchResult` → `Promise<Mv4dScene>`, drop the `ReconstructionResult` import — required for whole-frontend `tsc` to pass, per [05 §5.2](05-data-contract.md)). Also export the pure-TS **`dequantize(q, min, max)`** helper from `lib/wire/decoder.ts` (the off-GPU mirror of the vertex-shader dequant, [05 §2](05-data-contract.md)) for T-160 | Decoder returns zero-copy views; bad magic/version/bounds throw typed error; `dequantize()` matches the encoder inverse; `tsc --noEmit` clean | `T-150 T-151 T-152 T-153 T-154 T-155 T-160 T-180` |
| **W0.T4 Test harness + golden fixture + Makefile** | `backend/tests/fixtures/golden_scene.mv4d` (committed binary asset, <4 KB), `backend/pyproject.toml` (markers `mps`/`gpu` under `[tool.pytest.ini_options]` + `--strict-markers`, [10 §7](10-testing-strategy.md)), `frontend/vitest.config.ts` (+`@vitejs/plugin-react`,`jsdom`), `frontend/playwright.config.ts`, `frontend/package.json` (devDeps + `test`/`test:e2e` scripts), `frontend/src/lib/wire/__fixtures__/tiny.mv4d`, **`Makefile`** (per [10 §7](10-testing-strategy.md): `test-backend` → `pytest -m "not mps and not gpu"`, `test-frontend` → `vitest + tsc`, **new** `test-e2e`/`test-mps`; `setup-backend` → `python3.12`) | Golden fixture decodes identically in Python (T-200) **and** TS (T-202); reverse tiny vector (T-203) decodes in Python; `MV4D_VERSION` parity; `make test`/`test-e2e`/`test-mps` targets exist | `T-200 T-201 T-202 T-203` |

**W0 gate:** the MV4D format round-trips Python↔TS (`T-200/T-202/T-203`), version
parity holds (`T-201`), and the hexagonal import guard is green (`T-130`).
Command: `make test` (backend `pytest -m "not mps and not gpu"` + frontend `vitest` + `tsc`).

---

## Wave 1 — Backend service + async API (no ML, FakeAdapter)

| Task | Files | Acceptance | Verify |
|------|-------|-----------|--------|
| **W1.T1 Service post-processing** | `backend/app/core/services/reconstruction_service.py` (validate/caps, `smooth_and_cull`, `enforce_caps` — **pure numpy**, [06 §5](06-backend-spec.md)). The static/dynamic **split is NOT here** — it's adapter-side in `pipeline/assemble.py` (W3.T1, [06 §4.6/§5](06-backend-spec.md)) | `run()` delegates to the port (which returns an **already-split** `Scene4D`) then applies smooth/cull/caps; empty → `EmptyReconstructionError`; no torch in core (T-130 stays green) | `T-120 T-104 T-130` |
| **W1.T2 Async job queue + worker** | `backend/app/jobs/queue.py` (`JobQueue.submit/status/result/events`, `Job`, in-process worker via `run_in_executor`, SSE event push — [06 §6](06-backend-spec.md)) | Submitting runs the service off-thread; status transitions `queued→running→done`; progress monotonic in `[0,1]`; failure → `failed` + `{code,message}` | (covered by W1.T3 lifecycle tests) |
| **W1.T3 FastAPI surface + errors + config** | `backend/app/api/routes/jobs.py` (implement the 4 stubs), `api/sse.py`, `api/errors.py` (`http_status_for`), `api/deps.py` (`get_queue`, lifespan adapter wiring via registry), `app/main.py` (lifespan, health fields), `app/config.py` (+`adapter`,`device`,`target_fps`,`motion_thresh`,`conf_thresh`,`vggt_weights`,`result_dir` — resolve `result_dir` absolute at startup, [06 §8](06-backend-spec.md)), `backend/app/adapters/registry.py` (id→factory; default `vggt+cotracker3`), **`backend/tests/test_health.py`** (rewrite exact-dict equality → membership check, [06 §7](06-backend-spec.md)); lifespan **seeds** present `assets/samples/*.mv4d` via `JobQueue.seed_example` (slug = filename stem) | Endpoints exactly per [06 §7](06-backend-spec.md); 415/413/404/409 mapping; SSE via `fastapi.sse` (not sse-starlette), stream not behind GZip; `/result` immutable cache header; runs with the `fake` fixture adapter (`MAYAVIUS_ADAPTER=fake` → `FixtureAdapter`, [06 §4.6](06-backend-spec.md)) | `T-300 T-302 T-303 T-304 T-305 T-306 T-307 T-308 T-310` |

**W1 gate:** full job lifecycle works against the `fake` `FixtureAdapter` — `POST /jobs`
→ poll/SSE → `/result` returns a valid MV4D blob that the Python reference decoder
parses (`T-304` chains `T-100`). `T-301` (501 stub) is now superseded by `T-302…T-304`.
Command: `make test-backend`.

---

## Wave 2 — Frontend viewer + landing/share (no ML, fixture-mode backend)

Backend runs in **fixture mode** (`MAYAVIUS_ADAPTER=fake` serving the golden/example
MV4D) so the entire viewer is provable with no GPU. Tasks W2.T2–T4 can run in
parallel after W2.T1 (store/client) lands.

| Task | Files | Acceptance | Verify |
|------|-------|-----------|--------|
| **W2.T1 Store + API client** | `frontend/src/lib/state/viewerStore.ts` (extend: `scene/loadState/progress/error/cameraMode/frameCount` + actions, [07 §4](07-frontend-spec.md)), `lib/api/client.ts` (`submitClip/getJobStatus/streamJob/fetchResult`, SSE+poll fallback), `src/config.ts` (viewer tunables) | Store keeps scaffold fields; actions per [07 §4.2](07-frontend-spec.md); client hits [06](06-backend-spec.md) endpoints; decode errors → `loadState='error'` | `T-170 T-171` |
| **W2.T2 PointCloud + buildScene** | `frontend/src/components/viewer/PointCloud.tsx` (static+dynamic, custom dequant `ShaderMaterial`), `lib/viewer/buildScene.ts`, `components/viewer/Scene.tsx` (replace icosahedron; **keep Path-2 seam comment**) | u16 positions stay `Uint16` to the GPU (no CPU `Float32` expand); shader dequant matches [05 §2](05-data-contract.md); static drawn every frame, dynamic swaps per `t` | `T-160`; e2e `T-401` |
| **W2.T3 TrackRibbons** | `frontend/src/components/viewer/TrackRibbons.tsx` (`Line2`/`LineSegments2`, visibility gaps, grow-with-`t`, per-track color) | ribbons render with gaps where `isVisible(m,t)==0`; grow during playback; full ribbon when frozen | e2e `T-405` (visual) |
| **W2.T4 HUD + playback + camera** | `frontend/src/components/viewer/ui/{Timeline,PlaybackControls,BulletTimeButton,ProgressOverlay}.tsx`, `components/viewer/ViewerOverlay.tsx`, `PlaybackDriver` (`useFrame`), `ViewerCanvas.tsx` wiring (camera modes) | HUD is plain DOM, talks only to the store (no THREE import); play advances time on the loop; loop wraps; bullet-time freezes + free-orbits | `T-403 T-404 T-405` |
| **W2.T5 Landing + share route + SEO** | `frontend/src/app/page.tsx` (Hero+UploadDropzone+ExampleGallery), `components/{Hero,UploadDropzone,ExampleGallery}.tsx`, `app/view/[id]/page.tsx` (enrich `generateMetadata`, await `params`), `app/view/[id]/opengraph-image.tsx`, `app/sitemap.ts` (examples), **`assets/samples/example.mv4d`** (small committed MV4D — the W2 fixture example), **`frontend/e2e/fixtures/tiny.mp4`** (few-KB CC0 clip for the T-402 upload flow; bytes ignored in fixture mode, license recorded per [10 §6](10-testing-strategy.md)) | landing is a Server Component (indexable); upload → `/view/{jobId}`; result page emits OG/twitter cards; `ssr:false` boundary intact; `ExampleGallery` links to **`/view/example`** (the pinned slug seeded by the lifespan from `assets/samples/example.mv4d`, [06 §6](06-backend-spec.md)) — full C-1..C-4 corpus wiring is W4.T1 | `T-400 T-406 T-407` |

**W2 gate:** Playwright `T-400…T-407` green in fixture mode (upload → progressive
reveal → scrub → play/loop → bullet-time orbit → copy & reload share link);
`npm run build`, `npm run lint`, `npx tsc --noEmit` clean.
Command: `make test-frontend && make test-e2e`.

---

## Wave 3 — ML adapters on MPS (the on-device gate) — **needs the 36 GB Mac**

Install ML deps (`spec/08 §4` install-then-freeze) only now. Build the default combo
and prove it on MPS. The optional CUDA/no-MPS adapters are written as **honest
stubs** that raise `UnsupportedDeviceError` on MPS (not silent failures).

| Task | Files | Acceptance | Verify |
|------|-------|-----------|--------|
| **W3.T1 Pipeline utils** | `backend/app/pipeline/{decode,lift,assemble,quantize}.py` (`decode_and_subsample`; the §5 step 4 2D→3D lift formula; `assemble_scene4d(geo,tr,request)→RAW Scene4D`; AABB/quantize — kept out of adapters so adapters stay thin) + the `GeometryResult`/`TrackResult` structs ([06 §4.5a](06-backend-spec.md)) | decode→subsample→width-518 frames; lift unprojects (u_px,v_px)+depth→world via the §5 step 4 formula + §4.1a axis-flip; `assemble_scene4d` does **raw assembly only** (no split/cull); helpers pure (numpy/opencv), no FastAPI | unit (decode/lift on a tiny fixture clip) |
| **W3.T2 VggtAdapter** | `backend/app/adapters/vggt_adapter.py` (MPS, fp32, no cuda.autocast, `PYTORCH_ENABLE_MPS_FALLBACK=1` before torch import; weights `facebook/VGGT-1B`) | returns `Scene4D` static cloud + depth + camera in mayavius world space; `info.weights_license="cc-by-nc-4.0"` | part of `T-510` |
| **W3.T3 CoTracker3Adapter** | `backend/app/adapters/cotracker3_adapter.py` (`torch.hub` cotracker3_offline, MPS; lift via VGGT depth) | returns `Tracks` (positions/visibility/colors); 2D→3D lift correct | part of `T-510` |
| **W3.T4 Combo + registry wiring** | `backend/app/adapters/combo.py` (`VggtCoTracker3Adapter`), `adapters/registry.py` (default `vggt+cotracker3`), `requirements-ml.txt` (frozen pins) | combo runs VGGT once, feeds depth to the lift; `MAYAVIUS_ADAPTER=vggt+cotracker3` resolves at startup | `T-310` (gated, real adapter) |
| **W3.T5 Optional adapters (honest stubs)** | `backend/app/adapters/{spatialtracker_adapter,pi3_adapter,open_d4rt_adapter}.py` | each sets `info.mps_capable=False` and raises `UnsupportedDeviceError` on `device∈{mps,cpu}` with a message naming the constraint (CUDA-only / no-MPS / unverified); marked `@pytest.mark.gpu`, skipped on Mac | `T-310` skip-with-reason |

**W3 gate (on the 36 GB Mac, opt-in):** `T-500` (MPS available) + `T-510`
(VGGT+CoTracker3 on one ≤3 s sample → valid `Scene4D` within caps, ≥1 track; **time
+ peak memory recorded, not asserted**) + `T-511` (any CPU-fallback op logged).
Command: `make test-mps` (`MAYAVIUS_RUN_MPS_SMOKE=1 pytest -m mps -s`).

---

## Wave 4 — Corpus, launch assets, polish, optional deploy

| Task | Files | Acceptance | Verify |
|------|-------|-----------|--------|
| **W4.T1 Sample corpus + example results** | `assets/samples/{C-1…C-4}.{mp4,json}` (CC-licensed, ≤3 s, sidecar with `source_url`/`license`), pre-baked `assets/samples/<slug>.mv4d` blobs seeded at startup via `JobQueue.seed_example` ([06 §6](06-backend-spec.md)); `.gitignore` corpus stanza ([10 §6](10-testing-strategy.md)) | 3–4 clips present, each licensed + sized per [10 §6](10-testing-strategy.md); `ExampleGallery` lists C-1..C-4, each opening a seeded `/view/<slug>` | `T-600 T-601` |
| **W4.T2 Launch assets** | `README.md` (opens with animated GIF; "runs locally on a Mac"; "no GPU to view"; honest MIT-code/NC-weights), `LICENSE` (MIT), `frontend/public/og.png` (1200×630 static branded fallback), optional `backend/Dockerfile` (per [11 §2.2](11-deployment-and-launch.md), built from repo root; `deploy/` holds only the HF Space README/config) | README top line + GIF above the fold; LICENSE present; no weights/.venv/node_modules committed | manual ([13](13-definition-of-done.md)) |
| **W4.T3 Aesthetics polish** | shader uniforms / point-size / color / ribbon tuning in `PointCloud.tsx`/`TrackRibbons.tsx` on **real** reconstructions (risk #4) | a real sample renders screenshot-ably (stable static bg + readable moving cluster + ribbons) | manual visual on `C-1` |
| **W4.T4 (optional) GPU deploy** | `deploy/` HF Space (Docker, `MAYAVIUS_DEVICE=cuda`, may enable `spatialtracker_v2`), frontend env to the Space | hosted demo reachable; license-gated; **local path still works without it** | manual ([11](11-deployment-and-launch.md)) |

**W4 gate / project done:** every box in [13-definition-of-done.md](13-definition-of-done.md)
checked, incl. a full **local** E2E on a real clip on the 36 GB Mac.

---

## Build harness (how to run a wave — mirrored in EXECUTE.md)

- **Fan out by task** within a wave (parallel subagents, in git worktrees where
  two tasks touch disjoint files; serialize tasks that edit the same file —
  e.g. W2.T2 and W2.T4 both touch `ViewerCanvas.tsx`/`Scene.tsx`, so order them).
- **Adversarial code review per task:** before a task is marked done, a reviewer
  agent checks the diff against that task's **acceptance criteria + `T-xxx`** and
  the hexagonal import rule. A task is done only when its tests are green **and**
  the review passes.
- **Wave-gated transitions:** Claude may move between tasks *within* a wave
  autonomously; **crossing a wave boundary is human-gated** (the human confirms the
  exit gate is green before W(N+1) starts). W3 specifically requires the human to
  run `make test-mps` on the physical 36 GB Mac.
- **On failure:** retry the task up to 2× (fix → re-test → re-review); if still
  failing, **stop and escalate to the human** with the failing `T-xxx` + diff — do
  not weaken a test or the spec to pass.

## Dependency graph (what unblocks what)

```
W0.T1 ─┬─> W1.T1 ─> W1.T3 ─> [W1 gate] ─> W2.* ─> [W2 gate] ─> W3.* ─> [W3 gate] ─> W4.*
W0.T2 ─┤            W1.T2 ─┘
W0.T3 ─┤   (W0.T4 needs T1–T3 outputs for fixtures)
W0.T4 ─┘
```
Wire seam (W0.T2/T3) blocks everything that serializes/deserializes; the port
(W0.T1) blocks the service/queue; the API (W1) blocks the viewer's real data path
(W2 uses fixture mode until then); the viewer + service (W0–W2) are all provable
**before** ML (W3), which is the only wave that needs the Mac's GPU/MPS + weights.
