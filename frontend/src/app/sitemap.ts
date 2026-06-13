import type { MetadataRoute } from "next";
import { SITE_URL } from "@/config";

// Static, indexable routes only. Per-result /view/[id] URLs are user-generated
// and intentionally excluded here (they are noindex'd at the route level).
export default function sitemap(): MetadataRoute.Sitemap {
  return [
    {
      url: SITE_URL,
      lastModified: new Date(),
      changeFrequency: "weekly",
      priority: 1,
    },
  ];
}
