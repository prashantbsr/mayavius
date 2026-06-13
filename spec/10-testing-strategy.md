# 10 — Testing Strategy

How mayavius proves it works **on the 36 GB Apple-Silicon Mac** without a cloud
GPU. Tests are the executable form of the locked decisions: the
[MV4D v1](05-data-contract.md) wire format must round-trip identically in two
languages, the [hexagonal boundary](03-decisions-locked.md) must be impossible to
violate silently, and the viewer must actually scrub/play/orbit a real
reconstruction.

Every test below carries a **named id** (`T-xxx`). [09-execution-plan.md](09-execution-plan.md)
cites these ids as per-task acceptance gates; "done" for a task = its listed
`T-xxx` are green. Ids are stable — renaming one is a spec change.

**Tool stack (D8, see [08-dependencies-and-env.md](08-dependencies-and-env.md) §2/§3):**

| Layer | Tool | Pin / source | Scope |
|-------|------|--------------|-------|
| Backend unit + integration | `pytest` | `9.0.3` (`requirements-dev.txt`) | encoder, port/service, API, adapters |
| Backend HTTP | FastAPI `TestClient` via `httpx` | `httpx==0.28.*` | job lifecycle, status codes |
| Frontend unit | `vitest` + `@vitejs/plugin-react` + `jsdom` | added in this file | decoder, store, shader helpers |
| Frontend e2e | `@playwright/test` | added in this file | full viewer flow in a real browser |
| Frontend types | `tsc --noEmit` | `typescript 5.9.3` | type contract incl. `Mv4dScene` |
| MPS smoke | `pytest` (gated marker) | `requirements-ml.txt` (real torch) | VGGT+CoTracker3 on MPS, on-Mac only |

**Test philosophy (the quality bar applied):**
- **Verify, don't assert blind.** The MPS smoke test (T-510) *measures and records*
  wall-time + peak memory; it never pre-asserts a GB number (decision-log §E,H —
  the "7 GB/8 GB" figures are unverified blogspam).
- **Make wrong architecture fail CI.** The hexagonal import test (T-130) fails the
  build if `app.core` ever imports FastAPI/torch/an adapter.
- **One format, two languages, proven equal.** The cross-implementation
  round-trip (T-200/T-201) is the single test that guards the
  `encoder.py`↔`decoder.ts` seam.
- **Local-first.** All of §1–§4 + §6 run on the Mac with **no weights and no
  torch**. Only §5 (MPS smoke) needs the multi-GB ML deps; it is opt-in and
  skipped by default in CI.

---

## 1. Unit tests

### 1.1 Backend — wire format (`backend/tests/wire/`)

The encoder consumes the `Scene4D` domain model and emits an MV4D v1 buffer
([05 §5.1](05-data-contract.md)). These tests own the encoder + quantization math.
A small **Python reference decoder** lives at `backend/app/wire/decoder.py` (new,
mirrors `encoder.py`) so the backend can round-trip without the browser; it
re-implements [05 §3](05-data-contract.md) byte-for-byte and is itself covered.

| id | test | asserts |
|----|------|---------|
| **T-100** | `test_encode_decode_roundtrip` | Build a representative `Scene4D` (static + dynamic + tracks + cameras), `encode_reconstruction()` → `decode()` in Python → every section recovered: counts exact; colors `u8` exact; visibility bitmask exact; camera poses/intrinsics exact within `f32`; positions within **quantization tolerance** (see T-101). |
| **T-101** | `test_quantization_tolerance` | For random points in a known AABB, max dequant error per axis `≤ (aabbMax−aabbMin)/65535` (½-ULP of the 16-bit grid). Degenerate axis (`aabbMax==aabbMin`) → all `q=0`, dequant == `aabbMin` ([05 §2](05-data-contract.md)). |
| **T-102** | `test_header_and_directory` | Magic `"MV4D"`, `version==1`, `posBits==16`, `flags` bits match present sections, `sectionCount` correct; every section `byteOffset` is **8-byte aligned**; directory offsets/lengths stay inside the buffer ([05 §3.1–3.3](05-data-contract.md)). |
| **T-103** | `test_aabb_spans_all_sections` | AABB computed over static ∪ dynamic ∪ tracks; no point quantizes out of `[0,65535]` (clamp never engaged for in-AABB input) ([05 §5.1](05-data-contract.md)). |
| **T-104** | `test_caps_enforced` (**W1.T1 service** test — `enforce_caps` lives in core/services, not the encoder) | An over-cap `Scene4D` (`T>64`, `N_s>150k`, dynamic `>20k`/frame, `M>4096`) passed to `enforce_caps` is culled/subsampled to within caps (static by lowest `static_conf`; **dynamic by deterministic fixed-seed uniform random subsample** — confidence-free, [06 §5 step 7](06-backend-spec.md); tracks by mean-visibility; frames uniform); the encoder (given the already-capped scene) logs the final counts + actual payload size. There is **no separate >24 MB exception** — escalating cull keeps it under the ceiling ([05 §4](05-data-contract.md), [06 §5 step 7](06-backend-spec.md)); the only encode-path raise is `EmptyReconstructionError` when culling removes everything (covered by T-120 / [06 §5](06-backend-spec.md)). |
| **T-105** | `test_optional_sections_omitted` | A static-only scene (no tracks/cameras/dynamic) emits only `HAS_STATIC`; absent sections produce no directory entry; `HAS_STATIC_CONF` / `HAS_TRACK_COLOR` toggle the optional sub-arrays. |
| **T-106** | `test_empty_dynamic_frame` | A frame with `pointCount==0` is valid and round-trips (its `frameDir` entry has `pointCount=0`) ([05 §3.5](05-data-contract.md)). |
| **T-107** | `test_version_constant` | `from app.wire.encoder import MV4D_VERSION; assert MV4D_VERSION == 1` (constant exists for parity check T-201). |

### 1.2 Backend — port / service / hexagon (`backend/tests/core/`)

A `FakeAdapter(ReconstructionPort)` (in `backend/tests/fakes/fake_adapter.py`)
returns a deterministic small `Scene4D`. **No torch, no weights** — this is how the
core is tested in CI.

| id | test | asserts |
|----|------|---------|
| **T-120** | `test_service_delegates_to_port` | `ReconstructionService(FakeAdapter()).run(req)` returns the adapter's `Scene4D` with the documented **core** post-processing applied — `smooth_and_cull` + `enforce_caps` ([06 §5](06-backend-spec.md)). The static/dynamic **split is NOT a core step** (it's adapter-side, [06 §4.6](06-backend-spec.md)): `FakeAdapter` returns an **already-split** deterministic `Scene4D` (static + dynamic + tracks), so the service runs only smooth/cull/caps on it. |
| **T-121** | `test_fake_adapter_satisfies_port` | `FakeAdapter` is a concrete `ReconstructionPort`; `reconstruct()` returns a valid `Scene4D` that encodes cleanly (chains into T-100). Reusable as the **adapter-contract** suite (T-310). |
| **T-122** | `test_adapter_license_tag_surfaced` | Each adapter exposes `weights_license` (D2); `FakeAdapter` returns a known tag; surfaced into job metadata (spec/06). |
| **T-130** | **`test_core_imports_no_framework`** — **the hexagonal import test** | Import every module under `app.core.*` in a subprocess, then assert `sys.modules` contains **no** key in the banned set **`{fastapi, starlette, torch, uvicorn, numpy.*cuda, app.adapters.*}`**. Fails CI if the core's dependency edge is ever violated. **NumPy (non-cuda) is allowed** (domain model uses it); FastAPI/starlette/torch/uvicorn/adapters are not. See note below. (This exact set is mirrored in DoD §1.2.) |

> **T-130 mechanics (make wrong architecture hard to write):** run a clean
> `python -c "import importlib, pkgutil, app.core, sys; [importlib.import_module(m.name) for m in pkgutil.walk_packages(app.core.__path__, 'app.core.')]; banned={k for k in sys.modules if k.split('.')[0] in {'fastapi','starlette','torch','uvicorn'} or k.startswith('app.adapters') or ('cuda' in k and k.split('.')[0]=='numpy')}; assert not banned, banned"`
> in a subprocess so import side effects don't leak from other tests. This is the
> single guardrail behind the [hexagonal mandate](03-decisions-locked.md).

### 1.3 Frontend — decoder / store / shader (`frontend/src/**/*.test.ts`)

Vitest with `environment: 'jsdom'`. No WebGL context needed (shader helpers are
pure math; the GLSL itself is covered by e2e T-4xx visual presence).

| id | test | asserts |
|----|------|---------|
| **T-150** | `decoder.golden` | `decodeReconstruction(goldenBuffer)` → `Mv4dScene` with correct `version/frameCount/fps/aabbMin/aabbMax`, `static.positionsQ` is a `Uint16Array` **view** (zero-copy: `.buffer === goldenBuffer`), counts match. Golden buffer = the committed fixture (T-200). |
| **T-151** | `decoder.dynamic_slicing` | `scene.dynamic.frames[t]` returns the correct sub-view per `frameDir` (`startPoint`/`pointCount`), incl. an empty frame ([05 §3.5](05-data-contract.md)). |
| **T-152** | `decoder.tracks_visibility` | `scene.tracks.isVisible(m,t)` matches the packed LSB-first bitmask ([05 §3.6](05-data-contract.md)); track-color present only when `HAS_TRACK_COLOR`. |
| **T-153** | `decoder.errors` — bad magic | A buffer with wrong magic throws typed `Mv4dDecodeError` (not a generic `Error`), never returns a partial scene ([05 §8](05-data-contract.md)). |
| **T-154** | `decoder.errors` — bad version | `version=2` throws `Mv4dDecodeError` (unsupported major); `posBits≠16` throws; section bounds overflow throws; misaligned offset throws. |
| **T-155** | `decoder.version_constant` | `MV4D_VERSION === 1` exported from `decoder.ts` (parity with backend, T-201). |
| **T-160** | `shader.dequant` | The pure-TS **`dequantize(q, min, max)`** helper — exported from `lib/wire/decoder.ts` ([09 W0.T3](09-execution-plan.md)) — matches `p = aabbMin + q/65535*(aabbMax−aabbMin)`, the Python encoder's inverse for sample qs (the GPU vertex-shader math, verified off-GPU). |
| **T-170** | `viewerStore.actions` | `play/pause` set `isPlaying`; `setTime` clamps/sets `time∈[0,1]`; `toggleLoop` flips `loop`; `setFrozen(true)` sets `frozen` (bullet-time). Defaults: `time=0, isPlaying=false, loop=true, frozen=false` (matches scaffold). |
| **T-171** | `viewerStore.transient` | High-frequency `setTime` calls do not throw and leave a single final value (scrubber path; store is the R3F-loop write target outside React renders). |

### 1.4 Frontend — type contract

| id | test | asserts |
|----|------|---------|
| **T-180** | `tsc --noEmit` | Whole frontend typechecks; `Mv4dScene` (replacing the placeholder `ReconstructionResult`) is consistent across `decoder.ts`, `lib/api/client.ts`, `types/index.ts` ([05 §5.2](05-data-contract.md)). Run as `make typecheck`. |

---

## 2. The cross-implementation round-trip test (the seam guard)

**This is the most load-bearing test in the repo.** `encoder.py` and `decoder.ts`
are two implementations of one format ([05](05-data-contract.md)); nothing else
forces them to agree. We commit a **conformance vector** and decode it from both
sides.

**The golden fixture** — `backend/tests/fixtures/golden_scene.mv4d` (the worked
micro-example of [05 §6](05-data-contract.md) expanded to exercise all four
sections: small `T`, a handful of static points, ≥1 dynamic frame incl. an empty
one, ≥2 tracks with mixed visibility, cameras). It is a **binary asset** (small,
< 4 KB) committed to the repo — it is *not* a model weight, so the no-commit rule
([08 §8](08-dependencies-and-env.md)) does not apply.

| id | test | direction | asserts |
|----|------|-----------|---------|
| **T-200** | `test_golden_fixture_is_canonical` (pytest) | Python | The committed `golden_scene.mv4d` is **regenerated** from a hard-coded `Scene4D` literal and compared **byte-for-byte** to the on-disk file. If the encoder changes output, this fails until the fixture (and `decoder.ts`) are updated in the same commit — enforces [05 §7](05-data-contract.md). |
| **T-201** | `version.parity` (pytest reads both) | both | `MV4D_VERSION` in `encoder.py` (Python) **equals** `MV4D_VERSION` in `decoder.ts` (parsed from source / a tiny shared JSON `wire/version.json`, see note). Both equal `1`. |
| **T-202** | `decoder.golden_conformance` (vitest) | TS | Vitest loads the **same** `golden_scene.mv4d` (imported as an `ArrayBuffer`), decodes it, and asserts the recovered values equal a hard-coded expectation table identical to T-100's Python expectation. Python-encoded → TS-decoded proven correct. |
| **T-203** | `encoder.reverse_conformance` (pytest) | TS→Py | A **tiny TS-authored vector** (`frontend/src/lib/wire/__fixtures__/tiny.mv4d`, written by a one-off `vitest` "fixture" test using a future `encode()` helper *or* hand-laid bytes) is decoded by the Python reference decoder and matches. Guards the *reverse* direction so the seam is symmetric, not one-way. |

> **Parity mechanics (T-201):** to avoid brittle source-scraping, both sides import
> the version from one place — `MV4D_VERSION = 1` is duplicated as a literal in
> `encoder.py` and `decoder.ts` (per [05 §7](05-data-contract.md), both already
> "export `MV4D_VERSION = 1`"). T-201 reads the TS constant via a regex over
> `decoder.ts` and the Python constant via import, asserting equality. If a future
> change bumps one and not the other, T-201 fails.

> **Why both directions:** the MVP only ships a backend encoder + frontend decoder,
> so T-200/T-202 are the primary guard. T-203 is **cheap insurance** that the
> format spec (not one impl's quirks) is the authority — required because Path-2 /
> future tools may encode on the client.

---

## 3. Integration tests (backend)

FastAPI `TestClient` (sync, via `httpx`) drives the real ASGI app with a
`FakeAdapter` wired in (dependency override) — **no torch**. Covers the async job
model: `POST /jobs` (202) → `GET /jobs/{id}` (status + `0..1` progress) →
`GET /jobs/{id}/result` (binary MV4D), plus `/health`.

| id | test | asserts |
|----|------|---------|
| **T-300** | `test_health` | `GET /health` → `200`. **"Extends" = rewrite** the scaffold's exact-dict-equality assertion (`== {"status":"ok"}`) to a **membership** check (`status=="ok"` **and** `{"adapter","device","weights_license"} ⊆ keys`), since the build adds those fields ([06 §7](06-backend-spec.md)). `tests/test_health.py` is owned by **W1.T3** — this is a logged rewrite, not a forbidden weakening. |
| **T-301** | `test_stub_returns_501_now` | **Pre-implementation gate.** While handlers are stubs, `POST /jobs`, `GET /jobs/{id}`, `GET /jobs/{id}/result` return `501` with a message pointing at spec/06 (matches scaffold). This test **flips** to T-302/303/304 once the route is implemented — they are mutually exclusive by build phase (spec/09 marks the cutover). |
| **T-302** | `test_job_submit_returns_id` | `POST /jobs` multipart `clip=<small mp4>` → `202` + JSON `{ "job_id": ... }`. |
| **T-303** | `test_job_lifecycle_poll` | Poll `GET /jobs/{id}`: the observed statuses ⊆ `{queued,running,done}` (the `FixtureAdapter` emits `progress(0.25)`/`progress(0.75)` so a non-terminal `running` is reliably observable, [06 §4.6](06-backend-spec.md)), `progress` monotonically non-decreasing in `[0,1]` (0.0 at running-start → 1.0 at done), terminal `done` exposes `weights_license` + `adapter_id` metadata (D2). Assert the observed-status **set** + monotonicity, not a specific live `running` snapshot (no wall-clock race). |
| **T-304** | `test_job_result_is_mv4d` | After `done`, `GET /jobs/{id}/result` → `200`, `Content-Type: application/octet-stream`, body begins with magic `"MV4D"` and decodes via the Python reference decoder (chains T-100). Long-lived/immutable cache header present (shareable) ([05 §4](05-data-contract.md)). |
| **T-305** | `test_unknown_job_404` | `GET /jobs/{bogus}` → `404`. |
| **T-306** | `test_clip_frame_cap` | Uploading a clip exceeding `MAYAVIUS_MAX_CLIP_FRAMES` is accepted but **subsampled** to the cap (verified via result `frameCount ≤ 64`); over-long clips never produce `T>64` ([05 §4](05-data-contract.md), [08 §6](08-dependencies-and-env.md)). |
| **T-307** | `test_upload_rejections` | Upload above `MAYAVIUS_MAX_UPLOAD_MB` → **`413`** (Payload Too Large). Non-video / wrong content-type → **`415`** (Unsupported Media Type, per [06 §2.2](06-backend-spec.md) `UnsupportedMediaError`). Missing `clip` field → **`422`** (FastAPI validation). |
| **T-308** | `test_sse_progress_stream` | `GET /jobs/{id}/stream` (dedicated SSE route, `response_class=EventSourceResponse` yielding `ServerSentEvent`; `fastapi.sse`, **not** `sse-starlette`, C7 — [06 §7](06-backend-spec.md)) emits ordered progress events (`data`=poll JSON, `event`=status) ending in a terminal `done`/`failed` event; route is **not** behind `GZipMiddleware` ([08 §3](08-dependencies-and-env.md)). |
| **T-310** | **adapter-contract suite** (parametrized) | One suite, run against **`FakeAdapter` always** and the **real `VggtAdapter`+`CoTracker3Adapter` only when gated** (marker `mps`, §5). Asserts: `reconstruct(req)` returns a `Scene4D` honoring caps; coordinate system is mayavius world space (+X right, +Y up, −Z forward) per [05 §2](05-data-contract.md); `weights_license` populated. This is how "swapping models must not touch the core" is proven testable. |

> **Negative-knowledge gates (do not write green tests for dead ends):**
> `SpatialTrackerV2Adapter`, `Pi3Adapter`, `OpenD4RTAdapter` are **CUDA/GPU-only or
> no-MPS** (decision-log §D/E) — their contract tests are marked `@pytest.mark.gpu`
> and **skipped on the Mac** with a skip reason naming the constraint
> (SpatialTrackerV2 = CUDA `cu124` pin; Pi3 = no official MPS, PR #153 unmerged;
> OpenD4RT = MPS unverified). They are *documented as skipped*, not silently absent.

---

