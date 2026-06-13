import type { MetadataRoute } from "next";
import { SITE_URL } from "@/config";

// Generated /robots.txt — see app/sitemap.ts for the indexed routes.
export default function robots(): MetadataRoute.Robots {
  return {
    rules: { userAgent: "*", allow: "/" },
    sitemap: `${SITE_URL}/sitemap.xml`,
    host: SITE_URL,
  };
}
