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
