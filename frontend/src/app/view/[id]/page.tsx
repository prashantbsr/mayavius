import type { Metadata } from "next";
import { ViewerClient } from "@/components/viewer/ViewerClient";
import { SITE_NAME } from "@/config";

// Shareable result route. This is the SEO/virality surface: a pasted result
// link renders a rich preview card (title + Open Graph image) so it is
// screenshot-able and shareable (handover §5 "star mechanics").
type Props = { params: Promise<{ id: string }> };

export async function generateMetadata({ params }: Props): Promise<Metadata> {
  const { id } = await params; // params is async in Next 16 — must be awaited
  const title = `Reconstruction ${id}`;
  const description = `An interactive 4D reconstruction on ${SITE_NAME}.`;
  return {
    title,
    description,
    alternates: { canonical: `/view/${id}` },
    // User-generated result pages stay out of the search index.
    robots: { index: false, follow: false },
    openGraph: {
      type: "website",
      title,
      description,
      // TODO(spec/11): per-result dynamic OG image via next/og for share cards.
      images: [{ url: "/og.png", width: 1200, height: 630, alt: title }],
    },
    twitter: { card: "summary_large_image", title, description, images: ["/og.png"] },
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
