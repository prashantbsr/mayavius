# 00 — Spec Index

The complete build spec for **mayavius** — a browser-native viewer for feedforward
4D scene reconstruction (drop a short video → orbit/scrub/bullet-time an interactive
colored point cloud + motion ribbons in the browser; shareable as a URL). This spec
is the **single source of truth** for the build. The build session
([../EXECUTE.md](../EXECUTE.md)) reads it and fills in the existing scaffolding; it
does **not** rearchitect. If code and spec disagree, **stop and flag** — do not
improvise (handover §0).

## How to read it (recommended order)

1. **[01-overview-and-goals.md](01-overview-and-goals.md)** — the *why* and the bar: the "wow", goals, non-goals, success metrics, target users, the citation note.
2. **[02-research-dossier.md](02-research-dossier.md)** — the re-verified landscape: D4RT, why the niche is open, competitors/analogues, model candidates, render substrate.
3. **[03-decisions-locked.md](03-decisions-locked.md)** — every fixed decision (handover §4 + this session's D1–D10) and the **corrections C1–C7** to the handover. Read before building.
4. **[04-architecture.md](04-architecture.md)** — components, the hexagonal dependency graph, the end-to-end sequence, the real repo tree, the Path-1/Path-2 seam, state flow.
5. **[05-data-contract.md](05-data-contract.md)** — **the MV4D v1 binary wire format.** The authority for the backend↔frontend payload; never redefined elsewhere.
6. **[06-backend-spec.md](06-backend-spec.md)** — `ReconstructionPort`, adapters, the reconstruction pipeline, the async job model, the exact FastAPI surface, error contract.
7. **[07-frontend-spec.md](07-frontend-spec.md)** — component tree, the Three.js scene graph, decoder→GPU, Zustand store, playback + bullet-time, UX flows, SEO/share cards.
8. **[08-dependencies-and-env.md](08-dependencies-and-env.md)** — verified pins, model repo IDs, licenses, the Apple-Silicon/MPS install-then-freeze procedure, env vars.
9. **[09-execution-plan.md](09-execution-plan.md)** — waves → tasks; each task has files, acceptance criteria, and a verify command (citing `T-xxx`).
10. **[10-testing-strategy.md](10-testing-strategy.md)** — unit/integration/e2e + the MV4D cross-impl round-trip, the hexagonal guard, the MPS smoke gate, the sample corpus. Owns the `T-xxx` ids.
11. **[11-deployment-and-launch.md](11-deployment-and-launch.md)** — the local-first path, the optional GPU deploy, and the GitHub-stars launch plan.
12. **[12-risk-register.md](12-risk-register.md)** — risks, likelihood/impact, mitigations, kill-switches.
13. **[13-definition-of-done.md](13-definition-of-done.md)** — the objective, checkable DoD for the project.
- **[decisions/decision-log.md](decisions/decision-log.md)** — append-only log of every decision + every re-verified fact with source + date (2026-06-13 sweep), plus the §I spec-build process record. The audit trail.
- **[decisions/blind-executor-dry-run.md](decisions/blind-executor-dry-run.md)** — the Phase-5 stop-gate record: 14 blind-executor rounds, per-wave `wouldNeedToAsk=0` build-readiness, and the convergence finding.

## Authority chain (who owns what — on conflict, the owner wins)

| Topic | Authority | Everyone else |
|-------|-----------|---------------|
| MV4D byte layout, quantization, `Scene4D`/`Mv4dScene` shape | **[05](05-data-contract.md)** | reference only; never redefine bytes |
| Versions, repo IDs, licenses, MPS install | **[08](08-dependencies-and-env.md)** + [decision-log](decisions/decision-log.md) | never invent a version/ID |
| Locked decisions D1–D10 + corrections C1–C7 | **[03](03-decisions-locked.md)** | do not relitigate |
| Test ids `T-xxx` (acceptance gates) | **[10](10-testing-strategy.md)** | [09](09-execution-plan.md) cites them |
| Repo folder structure | the **real on-disk tree** + **[04 §5](04-architecture.md)** | fill stubs, don't move files |

## The locked decisions in one line each

- **D1** default model combo = **VGGT** (static cloud + depth + camera, MPS fp32) **+ CoTracker3** (2D tracks → lifted to 3D, MPS); SpatialTrackerV2 = cloud/optional.
- **D2** mayavius code **MIT**; default model weights **non-commercial** (clearly labeled); commercial static-only path = VGGT-1B-Commercial.
- **D3** name = **mayavius**. **D4** backend = **Python 3.12** (supersedes CLAUDE.md's 3.10).
- **Render** = Path 1 (`THREE.Points` + custom shader + `Line2` ribbons); **Path 2 (Spark 4DGS) OUT of MVP**, seam-only.
- **Wire** = MV4D v1 compact binary (16-bit quantized positions, dequant in shader); JSON for point payloads forbidden.
- **Backend** = FastAPI hexagonal (pure core behind `ReconstructionPort`; adapters/api/jobs outside); async job model.

## Status

This `spec/` set + `EXECUTE.md` is the deliverable of the spec-build session. The
repo is **scaffolding** (endpoints 501, adapters `NotImplementedError`, viewer
renders a placeholder mesh). The build session executes [09](09-execution-plan.md)
wave by wave until [13](13-definition-of-done.md) is fully checked.
