import { test, expect } from "@playwright/test";
import { debugNumber } from "./helpers";

// T-407 — ssr_boundary (spec/10 §4).
//
// The /view/[id] route renders server-side without throwing (no `window` access
// at SSR); the <canvas> only appears AFTER hydration — confirming the
// ViewerClient `dynamic(() => import('./ViewerCanvas'), { ssr:false })` boundary
// (ssr:false is forbidden in Server Components in Next 16, so it lives in the
// Client Component ViewerClient — spec/07 §1).
//
// Proof: fetch the RAW server HTML (no browser JS) and assert it (a) is a clean
// 200 — the server render did not throw — and (b) contains NO <canvas> but DOES
// contain the dynamic-import loading fallback ("Loading viewer…"). Then load the
// page in a real browser and assert the <canvas> appears + the scene renders,
// i.e. WebGL mounted only on the client after hydration.

test.describe("T-407 ssr_boundary", () => {
  test("the view route SSRs cleanly with no canvas; the canvas mounts after hydration", async ({
    page,
    request,
    baseURL,
  }) => {
    // ── Raw server HTML (no JS executes) ────────────────────────────────────────
    const res = await request.get(`${baseURL}/view/example`);
    expect(res.status()).toBe(200); // server render did not throw
    const html = await res.text();

    // No WebGL canvas in the server-rendered markup — it is client-only.
    expect(html).not.toContain("<canvas");
