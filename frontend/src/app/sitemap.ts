import type { MetadataRoute } from "next";
import { EXAMPLE_SLUGS, SITE_URL } from "@/config";

// Indexed routes only (spec/07 §8): the landing + the pre-seeded example results
// (D10 — they are `index:true` in generateMetadata, so listing them lets them
// earn organic traffic and seed the share loop). Per-result /view/[id] URLs for
// USER jobs are user-generated and intentionally excluded — they are noindex'd at
// the route level and never enumerated here.
export default function sitemap(): MetadataRoute.Sitemap {
  const now = new Date();
  return [
    {
      url: SITE_URL,
      lastModified: now,
      changeFrequency: "weekly",
      priority: 1,
    },
    // Example results (mirrors the backend's pinned seed slugs, spec/06 §6).
    ...EXAMPLE_SLUGS.map((slug) => ({
      url: `${SITE_URL}/view/${slug}`,
      lastModified: now,
      changeFrequency: "monthly" as const,
      priority: 0.8,
    })),
  ];
}
