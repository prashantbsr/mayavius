# 01 — Overview & Goals

**mayavius is a browser-native viewer for feedforward 4D scene reconstruction:
drop a short casual video into a webpage and, seconds later, orbit, scrub, and
bullet-time-freeze an interactive 3D point cloud of the scene over a stable
static background — playable on any machine, shareable as a URL.**

This file is the *why* and the *bar*. The *what/how* lives in
[04-architecture.md](04-architecture.md) onward; the fixed choices are in
[03-decisions-locked.md](03-decisions-locked.md); the binary payload is
[05-data-contract.md](05-data-contract.md). If anything here conflicts with a
locked decision, the locked decision wins — stop and flag.

---

## 1. The "wow" (be precise about it)

The product feeling, in one sentence: **"That's *my* video — and I'm flying
through it in 3D, in my browser, in seconds, and I can send the link to a
friend."** Three load-bearing words:

| Word | What it means concretely | Why it sells |
|------|--------------------------|--------------|
| **Own-clip** | The visitor uploads *their own* short casual video, not a curated demo asset. | Personal stake → the screenshot they take is of *their* scene. |
| **Shareable** | Every result is a permalink (`/view/[id]`) with a rich share card (`generateMetadata`) → the cloud renders for a stranger with **no GPU and no install**. | The virality surface; the frontend is fully client-side (see [07-frontend-spec.md](07-frontend-spec.md)). |
| **Screenshot-able** | The default camera + Path-1 aesthetic (colored `THREE.Points` + glowing track ribbons over a stable background) looks good *at rest*, in a still frame. | Stars come from the README GIF and the screenshots people post, not from a live session. |

The interactivity verbs are fixed (handover §2, [03](03-decisions-locked.md)):
**orbit · timeline scrub · play/pause/loop · bullet-time freeze-and-orbit.**
Moving objects animate as point clouds trailing **motion ribbons**; the static
background stays put. **The wow is immediacy + interactivity, not
photorealism** — this single sentence vetoes every "make it prettier at the cost
of speed/simplicity" decision downstream.

---

## 2. Goals

| # | Goal | Owner spec |
|---|------|-----------|
| G1 | Upload a short clip → get an interactive, orbitable, scrubbable 4D point-cloud playback. | [06](06-backend-spec.md), [07](07-frontend-spec.md) |
| G2 | Render = **Path 1** (colored `THREE.Points` + `Line2`/`LineSegments2` track ribbons) — the D4RT aesthetic, GPU-cheap, mobile-capable. | [07](07-frontend-spec.md) |
| G3 | Bullet-time: freeze the timeline at frame `t` and orbit the frozen instant. | [07](07-frontend-spec.md) (Zustand `frozen`) |
| G4 | Fast, shareable results via the **MV4D v1** compact binary (no JSON for point payloads). | [05](05-data-contract.md) |
| G5 | Runs **end-to-end on a 36GB Apple-Silicon Mac with zero cloud**: VGGT + CoTracker3 on MPS (fp32), async job pipeline, local viewer. | [06](06-backend-spec.md), [08](08-dependencies-and-env.md) |
| G6 | Clean **hexagonal** backend — a pure core behind `ReconstructionPort`; models are swappable adapters (`VggtAdapter`, `CoTracker3Adapter`, …). | [04](04-architecture.md), [06](06-backend-spec.md) |
| G7 | **Stars-optimized**: hosted demo, 3–4 preloaded examples, shareable URLs, a README GIF, MIT code with honest weight-license labeling. | [10](10-testing-strategy.md), [11](11-deployment-and-launch.md) |

---

## 3. Non-goals (explicitly OUT of MVP — do not build)

| Not building | Why | Where it *would* go |
|--------------|-----|---------------------|
| **Photorealism.** | The wow is immediacy, not fidelity. Point clouds + ribbons, full stop. | — |
| **4D Gaussian Splatting / Path 2** (`@sparkjsdev/spark` `<SplatMesh>`). | Feedforward → renderable 4DGS needs per-scene optimization (GPU-heavy, not feedforward). | Design the `Scene4D` seam in `Scene.tsx` for drop-in; **do not build it** (D-lock #2). |
| **Long video.** | The #1 scope risk. MVP caps clip length and **temporally subsamples** (`T ≤ 64`; default subsample target 32–48). | [05 §4 caps](05-data-contract.md), `MAYAVIUS_MAX_CLIP_FRAMES` |
| **Commercial use of the default tracker.** | No commercial-friendly point tracker exists; default weights (VGGT-1B, cotracker3) are **CC-BY-NC-4.0**. Commercial static-only path = VGGT-1B-Commercial (gated AUP), documented not default. | [08 §7](08-dependencies-and-env.md) (D2) |
| **CUDA-only / no-MPS models in the local path** (SpatialTrackerV2 CUDA-only, Pi3 no official MPS, OpenD4RT MPS-unverified, MegaSaM optimization-based). | The local Mac path must work without a cloud GPU. | optional adapters / [11 deploy](11-deployment-and-launch.md) |
| **Auth, accounts, persistence beyond the result blob, video moderation.** | Out of a stars-MVP scope; results are immutable cacheable blobs. | future |

---

## 4. Success metrics

### 4.1 Star-oriented (the product *is* its distribution)

| Metric | Target | Verified by |
|--------|--------|-------------|
| Hosted live demo reachable, no install | up at a public URL (HF Space, dedicated GPU — D9) | [11](11-deployment-and-launch.md) |
| Preloaded examples on the landing page | **3–4** short CC-licensed clips, one-click to a result | [10](10-testing-strategy.md) corpus, [11](11-deployment-and-launch.md) |
| Shareable result URLs with rich cards | `/view/[id]` resolves for a logged-out stranger; share card renders (OG/Twitter) | [07](07-frontend-spec.md) `generateMetadata` |
| README GIF | a looping orbit/bullet-time GIF, above the fold | [10](10-testing-strategy.md) |
| Honest licensing in the open repo | MIT code badge; active model `weights_license` surfaced in README + `/jobs` metadata | [08 §7](08-dependencies-and-env.md) (D2) |

### 4.2 Technical (the bar that makes the stars-play credible)

| Metric | Target | Measured |
|--------|--------|----------|
| **Result load** (blob fetch + decode → first painted cloud) | **≤ ~2 s** for an in-caps payload (≤12MB target), vs ~40s if JSON. | MV4D zero-copy decode; brotli/gzip + immutable cache ([05](05-data-contract.md)) |
| **Zero-cloud E2E** | upload → reconstruct → interactive playback, entirely on the 36GB Mac (MPS, fp32). | [10](10-testing-strategy.md) acceptance gate |
| **Time-to-first-cloud** | static `THREE.Points` painted as soon as STATIC_POINTS arrives — *before* dynamic/tracks finish (progressive, not behind a spinner). | section-directory + static-first decode ([05 §1](05-data-contract.md)); SSE progress ([06](06-backend-spec.md)) |
| **Payload size** | uncompressed ≤ **12MB** target (24MB hard ceiling); encoder logs actual. | [05 §4 caps](05-data-contract.md) |
| **Interactive frame rate** | smooth orbit/scrub on the target Mac and a typical laptop GPU (16-bit positions dequantized in-shader; no per-point CPU copy). | Path-1 shader ([07](07-frontend-spec.md)) |

> **Note — peak-memory numbers are deliberately NOT asserted here.** VGGT-on-MPS
> per-frame GB is *measured* on the actual 36GB Mac during the build (the
> "7GB/8GB" figures circulating online are unverified). Repo-confirmed floor:
> VGGT runs on an 8GB+ Mac; 36GB is ample for short clips. See
> [decision-log §E/§H](decisions/decision-log.md).

---

## 5. Target users

| User | Arrives via | Wants | We give them |
|------|-------------|-------|--------------|
| **The "drop in your own video" stranger** | a shared link / the README GIF / HN | the wow with **zero setup** | the hosted demo: upload → 4D in seconds → orbit/scrub/bullet-time → their own shareable `/view/[id]`. No GPU, no install. |
| **The dev who runs it locally** | the GitHub repo (stars) | clone → `make setup` → `make dev-backend` + `make dev-frontend` → it works on *their* Mac, and the model layer is obviously swappable | a clean hexagonal backend (`ReconstructionPort` + adapters), the MV4D contract, and an MPS path that runs with **no cloud GPU**. |

Both users are first-class. The stranger drives **stars-through-shares**; the dev
drives **stars-through-cloning** and is the audience for the architecture's
legibility (the swappable-adapter story is the extensibility pitch).

---

## 6. Citation note (handover §8 — documented future direction, NOT MVP scope)

Be explicit: **mayavius is primarily a *stars* play and is weak on *citations*
standalone.** It is a viewer + pipeline + wire format, not a novel model — there
is little for an academic paper to cite in the MVP itself.

The path to citations is **to ship mayavius as the frontend of a *citable
backend***: a future open, D4RT-style feedforward 4D decoder ("**OpenD4RT**").
The architecture already reserves the seam — `OpenD4RTAdapter` is a placeholder
adapter behind `ReconstructionPort`, and an unofficial open reimplementation now
exists to wrap (`Lijiaxin0111/Open-d4rt`, Apache-2.0; official Google DeepMind
D4RT remains unreleased — [decision-log §F](decisions/decision-log.md)). Standing
up a trained, benchmarkable OpenD4RT decoder *behind* mayavius is what would make
the combined system citable.

**This is a documented future direction, not MVP scope.** The MVP ships the
viewer + the default VGGT + CoTracker3 adapter combo (D1) and optimizes for
stars. Do not build OpenD4RT in the MVP — only keep its adapter seam intact.
