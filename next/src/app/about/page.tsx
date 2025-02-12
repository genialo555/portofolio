"use client"

import Link from "next/link"
import { PhotoCarousel } from "./components/photo-carousel"
import { SparklesCore } from "../projects/components/sparkles"

export default function AboutPage() {
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
      <div className="relative z-10 flex min-h-screen">
        {/* Carousel à gauche avec padding */}
        <div className="w-[400px] shrink-0 pl-32">
          <PhotoCarousel />
        </div>

        {/* Contenu à droite */}
        <div className="flex-1 flex items-center justify-center p-16">
          <div className="max-w-2xl space-y-8 bg-white/80 backdrop-blur-sm rounded-2xl p-8">
            <h1 className="text-4xl font-bold tracking-tight">À propos</h1>
            <div className="prose prose-gray dark:prose-invert">
              <p className="text-xl text-muted-foreground">
                Développeur passionné avec une expertise en développement web full-stack et une affinité particulière pour les interfaces utilisateur innovantes.
              </p>
              <p>
                Fort de plusieurs années d'expérience dans le développement d'applications web modernes, je combine créativité et expertise technique pour créer des solutions numériques performantes et esthétiques.
              </p>
              <p>
                Ma stack technique principale inclut :
              </p>
              <ul>
                <li>React & Next.js pour le développement frontend</li>
                <li>Node.js & NestJS pour le backend</li>
                <li>TypeScript pour un code robuste et maintenable</li>
                <li>Tailwind CSS pour un design moderne et responsive</li>
              </ul>
              <p>
                Je suis constamment à la recherche de nouveaux défis et d'opportunités d'apprentissage dans le domaine du développement web.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
} 