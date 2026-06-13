/**
 * Runtime configuration for the mayavius frontend.
 * Values come from NEXT_PUBLIC_* env vars (see .env.example) with safe
 * local-dev defaults so the scaffold runs with zero setup.
 */

/** Public site origin — used for canonical URLs and Open Graph metadata. */
export const SITE_URL = process.env.NEXT_PUBLIC_SITE_URL ?? "http://localhost:3000";

/** Base URL of the FastAPI reconstruction backend. */
export const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

/** Product display name. */
export const SITE_NAME = "mayavius";

/** One-line positioning (placeholder copy — finalized at launch, spec/11). */
export const SITE_TAGLINE =
  "Drop in a video and orbit a live 4D reconstruction of the scene in your browser.";

// ── Viewer tunables (spec/07 §4.3) ────────────────────────────────────────────
// Env-overridable knobs for the upload→poll/SSE→reveal loop and playback. Kept
// here (not hard-coded in client.ts) so a deploy can tune them without a rebuild
// of the call sites. All read from NEXT_PUBLIC_* with the spec defaults.

/** Helper: parse a NEXT_PUBLIC_* numeric env var, falling back to `fallback`
 * when unset or unparseable (so a typo can never silently zero a cap). */
function envNumber(value: string | undefined, fallback: number): number {
  if (value == null || value === "") return fallback;
  const n = Number(value);
  return Number.isFinite(n) ? n : fallback;
}

/** Poll interval (ms) for the `GET /jobs/{id}` fallback loop (spec/07 §6 step 2). */
export const POLL_INTERVAL_MS = envNumber(
  process.env.NEXT_PUBLIC_POLL_INTERVAL_MS,
  800,
);

/** Prefer SSE (`GET /jobs/{id}/stream`); fall back to polling when false or on
 * stream error/stall (spec/07 §4.3, §6 step 2). Env "false"/"0" disables. */
export const USE_SSE: boolean = (() => {
  const v = process.env.NEXT_PUBLIC_USE_SSE;
  if (v == null || v === "") return true;
  return !(v === "false" || v === "0");
})();

/** Watchdog (ms): if no SSE event arrives within this window, fall back to
 * polling — guards a silently-stalled stream, not just an errored one (§6 step 2). */
export const SSE_WATCHDOG_MS = envNumber(
  process.env.NEXT_PUBLIC_SSE_WATCHDOG_MS,
  5000,
);

/** Playback fps used only when `scene.fps <= 0` (spec/07 §4.3, §5 loop). */
export const DEFAULT_FPS_FALLBACK = envNumber(
  process.env.NEXT_PUBLIC_DEFAULT_FPS_FALLBACK,
  24,
);

/** Client-side upload size cap (MB); mirrors backend `MAYAVIUS_MAX_UPLOAD_MB`
 * (spec/08 §6). UploadDropzone rejects larger files before submit (§6 step 1). */
export const MAX_UPLOAD_MB = envNumber(
  process.env.NEXT_PUBLIC_MAX_UPLOAD_MB,
  64,
);

/** Pinned example slugs — MUST mirror the backend's seeded sample slugs (spec/06
 * §6). W2 fixture mode ships exactly one (`example`); W4 appends the C-1..C-4
 * corpus. Imported (server-only) by `generateMetadata` + `sitemap.ts` for the
 * example/user index split (spec/07 §8); the viewer NEVER branches on it. */
export const EXAMPLE_SLUGS = [
  "example",
  "walking-person",
  "street-vehicle",
  "pet-motion",
  "static-scene",
] as const;
