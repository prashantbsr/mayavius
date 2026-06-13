import { test, expect } from "@playwright/test";
import { debugNumber } from "./helpers";

// T-406 — share.link (spec/10 §4).
//
// Copy/reload /view/example in a FRESH context → the same result loads; assert
// the share-card og:* meta tags are in the document head (generateMetadata, the
// virality surface, spec/07 §8). `params` is awaited (Next 16 async params — the
// page would throw at SSR otherwise, so a clean server render proves it). The
// example route is the shareable id space (a seeded `done` job == its slug,
// spec/06 §6), so a pasted link resolves identically on a cold load.

test.describe("T-406 share.link", () => {
  test("a shared /view/example link renders rich og cards and the same result", async ({
    browser,
  }) => {
    // FRESH context = a pasted link opened by someone who never visited before.
    const context = await browser.newContext();
    const page = await context.newPage();
    try {
      const response = await page.goto("/view/example", {
        waitUntil: "domcontentloaded",
      });
      // The route renders server-side without throwing (200, not a 500 — proves
      // `params` was awaited; an un-awaited Promise would crash generateMetadata).
      expect(response?.status()).toBe(200);

      // og:* share-card tags present in the head (the virality surface, spec/07 §8).
      const ogTitle = page.locator('head meta[property="og:title"]');
      const ogDesc = page.locator('head meta[property="og:description"]');
