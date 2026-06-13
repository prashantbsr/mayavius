import { Hero } from "@/components/Hero";
import { UploadDropzone } from "@/components/UploadDropzone";
import { ExampleGallery } from "@/components/ExampleGallery";

// Landing page — a Server Component, so it is fully static and SEO-indexable
// (spec/07 §1, §8 — Server Components by default; only UploadDropzone and the
// gallery cards are 'use client' islands). Composes the hero copy, the upload
// dropzone (the single producer of a new submitClip), and the example gallery
// (pre-seeded, no-reconstruction results).
export default function Home() {
  return (
    <main className="flex flex-1 flex-col items-center gap-12 p-8 py-16">
      <Hero />
      <UploadDropzone />
      <ExampleGallery />
    </main>
  );
}
