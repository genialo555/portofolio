"use client"

import Link from "next/link"
import { SparklesCore } from "../projects/components/sparkles"
import { LinkedInPosts } from "./components/linkedin-posts"

export default function BlogPage() {
  return (
    <div className="relative min-h-screen bg-background">
      {/* Sparkles Background */}
      <div className="absolute inset-0 h-screen w-full">
        <SparklesCore
          id="tsparticlesfullpage"
          background="transparent"
          minSize={0.6}
          maxSize={1.4}
          particleDensity={100}
          className="w-full h-full"
          particleColor="#475569"
        />
      </div>

      {/* Back Arrow */}
      <Link 
        href="/"
        className="fixed top-8 left-8 z-50 p-3 rounded-full bg-white/90 shadow-lg hover:bg-white hover:scale-110 transition-all duration-200 group"
      >
        <svg 
          xmlns="http://www.w3.org/2000/svg" 
          width="24" 
          height="24" 
          viewBox="0 0 24 24" 
          fill="none" 
          stroke="currentColor" 
          strokeWidth="2" 
          strokeLinecap="round" 
          strokeLinejoin="round" 
          className="transform group-hover:-translate-x-1 transition-transform duration-200"
        >
          <path d="M19 12H5M12 19l-7-7 7-7"/>
        </svg>
      </Link>

      {/* Content */}
      <div className="relative z-10 container mx-auto px-4 py-16">
        <div className="max-w-4xl mx-auto space-y-8">
          <div className="text-center space-y-4">
            <h1 className="text-4xl font-bold tracking-tight">Blog</h1>
            <p className="text-xl text-muted-foreground">
              Mes derniers articles et réflexions sur le développement, l'IA et l'innovation
            </p>
          </div>

          <LinkedInPosts />
        </div>
      </div>
    </div>
  )
} 