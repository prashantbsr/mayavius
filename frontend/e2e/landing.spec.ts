import { test, expect } from "@playwright/test";

// T-400 — landing.loads (spec/10 §4).
//
// GET / (the static, indexable landing — spec/07 §1/§8) renders: the hero copy
// + an example gallery are present. The WebGL `<canvas>` is NOT required here
// (the viewer is client-only and lazy — it lives on /view/[id]). This is the one
// flow the WebKit project also runs (a smoke for Safari/WebGL2 differences), so
// it asserts NO WebGL — only the static landing DOM.

test.describe("T-400 landing.loads", () => {
  test("renders the hero and the example gallery (no canvas required)", async ({
    page,
  }) => {
    await page.goto("/");

    // Hero: the H1 carries the product name (Hero.tsx → SITE_NAME "mayavius").
    const hero = page.getByRole("heading", { level: 1, name: /mayavius/i });
    await expect(hero).toBeVisible();

    // Example gallery is present with at least the seeded `example` card
    // (ExampleGallery.tsx, data-testid) → links to /view/example.
    const gallery = page.getByTestId("example-gallery");
    await expect(gallery).toBeVisible();
    const exampleCard = page.getByTestId("example-card-example");
    await expect(exampleCard).toBeVisible();
    await expect(exampleCard).toHaveAttribute("href", "/view/example");

    // The upload affordance (the only producer of submitClip) is on the landing.
    await expect(page.getByTestId("upload-dropzone")).toBeVisible();
