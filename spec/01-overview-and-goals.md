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

