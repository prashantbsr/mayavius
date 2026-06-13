import { test, expect } from "@playwright/test";
import path from "node:path";
import { debugNumber, progressSeen } from "./helpers";

// Playwright runs specs from the project root (frontend/); the fixture is the
// committed CC0 clip under e2e/fixtures (spec/10 §6 / W2.T5).
const TINY_MP4 = path.join(process.cwd(), "e2e", "fixtures", "tiny.mp4");

// T-402 — upload.flow (spec/10 §4).
//
// Upload the bundled e2e fixture clip frontend/e2e/fixtures/tiny.mp4 (bytes are
// ignored in fixture mode, but POST /jobs validates content-type + size first so
// a REAL video file must exist). The result loads → the cloud appears, AND the
// store observed at least one intermediate progress value 0 < p < 1.
//
// No wall-clock race (mirror of T-303's framing): we assert the terminal reveal
// (staticPointCount > 0) AND assert against the viewerStore.progress field
// captured OVER THE RUN via the always-on store surface — NOT a guaranteed
// mid-flight DOM frame. The FixtureAdapter deterministically emits
// progress(0.25,"decode") then progress(0.75,"assemble") (spec/06 §4.6), both
// buffered on the per-job SSE queue, so a subscribing client always sees them.

test.describe("T-402 upload.flow", () => {
  test("upload → result loads → cloud appears with observed intermediate progress", async ({
    page,
  }) => {
    await page.goto("/");

    // Drive the hidden <input type=file> directly (the dropzone's picker).
    await page.getByTestId("upload-input").setInputFiles(TINY_MP4);

    // UploadDropzone validates type+size, POSTs /jobs, then router.push to
    // /view/<jobId>. Wait for that navigation (a 32-hex job id, not "example").
    await page.waitForURL(/\/view\/[0-9a-f]{8,}$/, { timeout: 30_000 });

    // The viewer mounts the always-on progress recorder (installed BEFORE the
    // loader effect — testObservability.ts) so every progress value the loader
    // writes is captured: the FixtureAdapter's 0.25 / 0.75 progression, then 1.0
    // on done. Wait for the recorder to exist before asserting on it.
    await page.waitForFunction(
      () => Array.isArray(window.__mayaviusProgressLog),
      undefined,
      { timeout: 30_000 },
    );

    // Terminal reveal: a real static cloud rendered.
    await page.waitForSelector("canvas", { timeout: 30_000 });
    await expect
      .poll(() => debugNumber(page, "staticPointCount", -1), {
        timeout: 30_000,
        intervals: [100, 200, 300],
      })
      .toBeGreaterThan(0);

    // Observed-set assertion (race-free, mirror of T-303's framing): over the run
    // the store's progress held at least one intermediate value 0<p<1 AND reached
    // terminal progress 1. We assert against the captured history, never a single
    // mid-flight DOM frame. (We do NOT assert global monotonicity here: the SSE
    // late-subscriber path legitimately replays the buffered running events, which
    // can revisit a lower value — that is T-303's poll-ordering concern, not this
