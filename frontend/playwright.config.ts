import { defineConfig, devices } from "@playwright/test";

// Playwright e2e config (spec/10 §4) — the W2 gate. Drives the full viewer flow
// T-400..T-407 in a real browser against the fixture-mode backend.
//
// Two auto-started servers (spec/10 §4 webServer block, unchanged): the FastAPI
// backend in FIXTURE mode (MAYAVIUS_ADAPTER=fake → the deterministic
// FixtureAdapter, so no torch / no GPU) on :8000, and `next dev` on :3000.
// Playwright's cwd is `frontend/`, so the backend command `cd ../backend` reaches
// the venv. Default project = Chromium (runs the full flow with headless software
// WebGL — see launchOptions below); the WebKit project runs the T-400 smoke ONLY
// (landing render) to catch Safari/WebGL2 differences without asserting WebGL.
export default defineConfig({
  testDir: "./e2e",
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 2 : 0,
  use: {
    baseURL: "http://localhost:3000",
    trace: "on-first-retry",
  },
  projects: [
    {
      // Default project — runs the full viewer flow (T-400..T-407). R3F needs a
      // real WebGL2 context in headless Chromium; on this machine the GPU is
      // blocklisted, so we force ANGLE's SwiftShader (software GL) and unblock it.
      // These args were verified to yield a working WebGL2 context here (the e2e
      // assertion __mayaviusDebug.staticPointCount > 0 only passes once the R3F
      // <Canvas> actually rendered — see T-401).
      name: "chromium",
