# 03 — Locked Decisions

These are **fixed**. Do not relitigate them during the build. Each carries its
rationale; the evidence behind the re-verified ones is in
[decisions/decision-log.md](decisions/decision-log.md). If the build session
believes a locked decision is wrong, it must **stop and flag**, not improvise
(handover §0).

---

## Part 1 — Handover §4 locked decisions (verbatim, with rationale)

1. **Render = animated colored point cloud + 3D track ribbons (Path 1). MVP.**
   Per-timestep 3D points + per-track trajectories, rendered as `THREE.Points`
   with a custom shader plus line ribbons (`Line2`/`LineSegments2`) for tracks.
   GPU-cheap, mobile-capable, and exactly the D4RT-page aesthetic. *Why:* the
   "wow" is immediacy + interactivity, not photorealism.

2. **4D Gaussian Splatting (Path 2 / Spark) is OUT of the MVP.** Turning a
   feedforward reconstruction into renderable 4D gaussians needs per-scene
   optimization (GPU-heavy, not feedforward). Design the frontend scene layer so
   a Spark (`@sparkjsdev/spark`, MIT) `<SplatMesh>` path can drop in later at the
   `Scene.tsx` seam **without rearchitecting**. Do not build it.

3. **Frontend = React + Three.js via react-three-fiber. Stack = Next.js.** Chosen
   to satisfy the hard SEO requirement (static landing + rich result share
   cards). The UX *is* the product; this is where the polish budget goes.

4. **Backend = FastAPI, async job model.** Upload → job id → poll/stream
   progress → fetch binary result. Stream frames so the cloud appears
   progressively, not behind a spinner.

5. **Wire format = compact binary, not JSON.** Quantized positions (16-bit) +
   per-point color + per-track indices as typed-array blobs. JSON for point
   payloads is **forbidden** — it is the difference between a ~2 s and a ~40 s
   load and it gates shareable result links. Exact layout: [05-data-contract.md](05-data-contract.md).

6. **MVP targets SHORT clips.** Long-video support is the #1 scope risk; **cap
   clip length and temporally subsample** rather than chase arbitrary length.

7. **Backend models are adapters behind `ReconstructionPort`; the MVP ships ≥1
   working adapter combo.** Swapping models must not touch the core. See Part 2/D1.

## Hard constraints (handover §3, carried verbatim)

- Must **build, run, and be testable on a 36 GB Apple-Silicon Mac.** Frontend is
  fully local. Inference runs on **MPS** for short clips. Anything requiring a
  cloud GPU is an **optional** phase; the local path must work without it.
- **MPS reality:** these models run **fp32** on MPS (their working ports do; and
  half-precision MPS autocast is beta/incomplete — see decision-log §E). **VGGT**
  is the MPS-capable default; **π³ has no official MPS path** — do not make it the
  Mac default. Weights are multi-GB (one-time download; never commit).
- Compute is **asymmetric**: the viewer is cheap, inference is heavy and isolated
  behind the async-job + adapter boundary. Keep it that way.

---

## Part 2 — Decisions locked this session (2026-06-13)

### D1 — MVP default model combo: **VGGT + CoTracker3**
- **Static layer:** `VggtAdapter` → colored point cloud, depth, camera poses (MPS,
  fp32, via the community-port pattern). Default weights `facebook/VGGT-1B` (NC).
- **Dynamic layer:** `CoTracker3Adapter` → 2D point tracks, **lifted to 3D using
  VGGT depth + camera**, producing the colored 3D track ribbons. CoTracker3 has
  first-class in-tree MPS support.
- **`SpatialTrackerV2Adapter`** = additional **cloud/optional** adapter (single
  model → dynamic 3D tracks + geometry; CUDA-only, not a Mac default).
- **`Pi3Adapter`**, **`OpenD4RTAdapter`** = additional optional adapters (Pi3 no
  MPS; OpenD4RT = the now-existing unofficial open D4RT reimpl, GPU-oriented).
- *Why:* this is the lowest-friction route to the D4RT look that **runs on the
  Mac**, and it satisfies the "≥1 working adapter combo" mandate.

### D2 — Licensing: **MIT code, non-commercial research weights**
- mayavius's **own source code** is licensed **MIT**.
- Default **model weights are non-commercial** (`VGGT-1B`, `cotracker3` are
  CC-BY-NC-4.0). The repo, README, and the `/jobs` API must **clearly label** the
  active model's weight license.
- **Commercial path (documented, not default):** `facebook/VGGT-1B-Commercial`
  (custom `vggt-aup-license`, commercial OK except military, gated) for static
  geometry — but **no commercial tracker exists**, so the track-ribbon feature is
  commercially blocked until a permissively-licensed tracker is sourced.
- The adapter layer **license-tags** every model (`AdapterInfo.weights_license`)
  so deployment can gate on it (spec/06).
- *Why:* an open research project (MIT code + NC weights, honestly labeled)
  maximizes stars without misrepresenting redistribution rights.

### D3 — Project name: **`mayavius`**
- Display name and repo name = `mayavius`. `video-to-4d` is a descriptive alias
  only. Verified available across npm/GitHub/`.com`/`.dev` (decision-log §G).

### D4 — Backend runtime: **Python 3.12**
- **Supersedes** CLAUDE.md's `python 3.10`. Python 3.10 EOLs Oct 2026; all
  backend pins + torch ARM64/MPS wheels support 3.12. The venv is created with
  `python3.12`; `backend/requirements.txt` package pins are unchanged (all support
  3.12). This is the one place the spec overrides a CLAUDE.md stack pin — flagged
  in the decision log, not silent.

### D5–D10 (owned, see decision-log §A for the table)
- **D5** react-three-fiber: yes. **D6** state: Zustand. **D7** wire format: `MV4D`
  v1 (spec/05). **D8** tests: pytest 9 + Vitest + Playwright + `tsc`. **D9** deploy:
  HF Space (dedicated GPU). **D10** sample corpus: 3–4 short CC clips.
- **TypeScript stays on 5.x** (repo pins `^5`, installs 5.9.3). TS6 exists but is
  a deliberate post-MVP bump.

---

## Part 3 — Corrections to the handover (must be reflected everywhere)

These are factual corrections found during Phase 0 verification. The spec uses the
corrected versions; the handover text is **superseded** on these points.

| # | Handover said | Reality (verified) | Where it matters |
|---|---------------|--------------------|------------------|
| C1 | "VGGT … no point tracks" | VGGT has a **2D track head** (static scenes); it lacks native **dynamic 3D** tracks | spec/02, spec/06 — phrase as "static reconstruction; 2D static-scene tracks; no native dynamic 3D tracks" |
| C2 | "π³ … non-commercial / research-only" (whole) | π³ **code is BSD-3 (commercial OK)**; only **weights** are non-commercial | spec/02, spec/08 |
| C3 | "MPS = fp32 only (no fp16 autocast)" | PyTorch **does** support fp16/bf16 autocast on MPS (beta); we use fp32 **by choice** because the model ports do | spec/02, spec/06, spec/08 |
| C4 | "Pin three 0.184 / scaffold June 2026" + "python 3.10" | three 0.184.0 ✅; **Python → 3.12** (D4); **TypeScript 6 now exists**, staying on 5.x | spec/08 |
| C5 | D4RT "unreleased; future OpenD4RT" | Official still unreleased ✅, **but** an unofficial open reimpl (`Lijiaxin0111/Open-d4rt`, Apache-2.0, weights on HF) now exists | spec/02, spec/06 (`OpenD4RTAdapter`) |
| C6 | "SpatialTrackerV2 ~936★, has tracks" ✅ | code is **CC-BY-NC-4.0** (not permissive); CUDA-only; repo slug `SpaTrackerV2` | spec/06, spec/08 |
| C7 | (job streaming) | FastAPI ≥0.135 ships **built-in SSE** (`fastapi.sse`); `sse-starlette` not needed | spec/06 |
