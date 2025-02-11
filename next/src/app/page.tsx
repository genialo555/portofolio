"use client";

import dynamic from "next/dynamic";

// Loading component
const LoadingHero = () => (
  <div className="w-full h-screen flex items-center justify-center">
    <div className="space-y-8 text-center animate-pulse">
      <div className="space-y-4">
        <div className="h-12 w-64 bg-gray-200 rounded mx-auto" />
        <div className="h-6 w-96 bg-gray-200 rounded mx-auto" />
      </div>
      <div className="flex items-center justify-center gap-16 pt-4">
        <div className="h-6 w-32 bg-gray-200 rounded" />
        <div className="h-6 w-32 bg-gray-200 rounded" />
      </div>
    </div>
  </div>
);

// Client-side only Hero component
const Hero = dynamic(
  () => import("@/components/ui/Hero").then((mod) => mod.Hero),
  {
    ssr: false,
    loading: () => <LoadingHero />,
  }
);

export default function Home() {
  return <Hero />;
}
