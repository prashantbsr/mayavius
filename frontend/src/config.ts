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
