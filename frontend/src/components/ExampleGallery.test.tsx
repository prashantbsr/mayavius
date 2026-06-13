import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { EXAMPLE_SLUGS } from "@/config";
import { ExampleGallery } from "./ExampleGallery";

// T-601 — `corpus.examples_listed` (spec/10 §6). The landing example gallery MUST
// list the C-1..C-4 corpus clips so each is wired to a result route (/view/<slug>);
// the e2e click-through (T-401 chain) is covered by the e2e gate, so a jsdom/vitest
// unit test that checks the EXAMPLE_SLUGS wiring + the rendered cards is sufficient
// here (the task brief).
//
// `ExampleGallery` is a Server Component built from plain `next/link` cards (no
// client hooks), so it renders cleanly to static markup — we assert each corpus
// slug appears as a `/view/<slug>` link with its fixed role label (spec/10 §6 table).

// The four corpus slugs (C-1..C-4) and their fixed role labels (spec/10 §6 table).
const CORPUS: ReadonlyArray<readonly [string, string]> = [
  ["walking-person", "Walking person"],
  ["street-vehicle", "Street vehicle"],
  ["pet-motion", "Pet in motion"],
  ["static-scene", "Static scene (control)"],
];

describe("T-601 corpus.examples_listed", () => {
  it("EXAMPLE_SLUGS includes the W2 example + the four W4 corpus slugs", () => {
    // Mirrors the backend's pinned seed slugs (spec/06 §6) — the wiring the
    // gallery, sitemap, and generateMetadata all read.
    expect(EXAMPLE_SLUGS).toContain("example");
    for (const [slug] of CORPUS) {
      expect(EXAMPLE_SLUGS).toContain(slug);
