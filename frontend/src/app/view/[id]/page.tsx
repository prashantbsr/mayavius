import type { Metadata } from "next";
import { ViewerClient } from "@/components/viewer/ViewerClient";
import { EXAMPLE_SLUGS, SITE_NAME } from "@/config";

// Shareable result route. This is the SEO/virality surface: a pasted result
// link renders a rich preview card (title + Open Graph image) so it is
// screenshot-able and shareable (handover §5 "star mechanics", spec/07 §8).
type Props = { params: Promise<{ id: string }> };

/** Example/user classification with no backend round-trip (spec/07 §8): an id is
 * an example iff it is in `EXAMPLE_SLUGS` (server-only list, mirrors the backend
 * pinned seed slugs — spec/06 §6). `EXAMPLE_SLUGS` is `readonly`, so widen for
 * `.includes`. */
function isExample(id: string): boolean {
  return (EXAMPLE_SLUGS as readonly string[]).includes(id);
}

export async function generateMetadata({ params }: Props): Promise<Metadata> {
  const { id } = await params; // params is async in Next 16 — must be awaited
  const example = isExample(id);

  const title = example
    ? `Example scene · ${SITE_NAME}`
    : `Reconstruction ${id}`;
  const description = example
    ? `Orbit a pre-seeded 4D reconstruction example on ${SITE_NAME} — no upload, no GPU.`
    : `An interactive 4D reconstruction on ${SITE_NAME}.`;
  const canonical = `/view/${id}`;

  return {
    title,
    description,
    alternates: { canonical },
    // Example results (D10) are indexable so they earn organic traffic and seed
    // the share loop (spec/07 §8). User-generated results stay out of the index
    // (avoid indexing junk) but still emit rich OG/twitter cards below.
    robots: example
      ? { index: true, follow: true }
      : { index: false, follow: false },
    // opengraph-image.tsx overrides og:image for this route; these tags carry
    // the title/description/url that make the pasted link a rich card.
    openGraph: {
      type: "website",
      siteName: SITE_NAME,
      url: canonical,
      title,
      description,
    },
    twitter: {
      card: "summary_large_image",
      title,
      description,
    },
  };
}

export default async function ViewPage({ params }: Props) {
  const { id } = await params;
  return (
    <main className="flex flex-1">
      <ViewerClient resultId={id} />
    </main>
  );
}
