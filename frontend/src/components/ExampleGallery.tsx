import Link from "next/link";
import { EXAMPLE_SLUGS } from "@/config";

// ExampleGallery — a Server Component shell (spec/07 §1, §6 step 6). Each card is
// a plain <Link> to /view/<slug>, so no client interactivity is needed and the
// gallery stays server-rendered (indexable). An example opens the viewer with NO
// upload and NO reconstruction: the backend lifespan pre-seeds each committed
// assets/samples/<slug>.mv4d as a terminal `done` job under id == slug (spec/06
// §6), so /view/<slug> uses the IDENTICAL fetch+decode path as a real result.
//
// W2 fixture mode ships exactly one slug — `example` (spec/07 §4.3). W4.T1 appends
// the C-1..C-4 corpus (`walking-person`/`street-vehicle`/`pet-motion`/`static-scene`)
// with their fixed role labels (spec/10 §6 table). Each card still uses the static
// /og.png thumbnail (no per-clip thumbnail pipeline until later — spec/07 §6 step 6).

/** Fixed role labels for the C-1..C-4 corpus slugs (spec/10 §6 table). The slug
 * IS the name; these are the human role labels shown on each gallery card. */
const CORPUS_LABELS: Record<string, string> = {
  "walking-person": "Walking person",
  "street-vehicle": "Street vehicle",
  "pet-motion": "Pet in motion",
  "static-scene": "Static scene (control)",
};

/** Human label for a slug. W2: the single `example` slug → "Example scene".
 * W4: the corpus slugs map to their fixed C-1..C-4 role names (spec/10 §6). */
function slugLabel(slug: string): string {
  if (slug === "example") return "Example scene";
  if (slug in CORPUS_LABELS) return CORPUS_LABELS[slug];
  // Generic title-case fallback for any future slug added before its wiring.
  return slug
    .split("-")
    .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
    .join(" ");
}

export function ExampleGallery() {
  return (
    <section
      aria-label="Example scenes"
      data-testid="example-gallery"
      className="flex w-full max-w-3xl flex-col items-center gap-4"
    >
      <h2 className="text-sm font-medium uppercase tracking-wide opacity-60">
        Or open an example — no upload, no GPU
      </h2>
      <ul className="grid w-full grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
        {EXAMPLE_SLUGS.map((slug) => (
          <li key={slug}>
            <Link
              href={`/view/${slug}`}
              data-testid={`example-card-${slug}`}
              className="group flex flex-col overflow-hidden rounded-xl border border-foreground/15 transition-colors hover:border-foreground/50"
            >
              {/* W2 thumbnail = the static /og.png (spec/07 §6 step 6). Plain
                  <img>: /og.png is a 1200x630 asset (W2.T5) served from public/. */}
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src="/og.png"
                alt={`${slugLabel(slug)} preview`}
