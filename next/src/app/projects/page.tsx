"use client";

import dynamic from "next/dynamic";
import Link from "next/link";
import { Suspense } from "react";

const LayoutGrid = dynamic(
  () => import("./components/project-grid").then((mod) => mod.LayoutGrid),
  { ssr: false }
);

const SparklesCore = dynamic(
  () => import("./components/sparkles").then((mod) => mod.SparklesCore),
  { ssr: false }
);

// Loading component for the grid
const GridSkeleton = () => (
  <div className="grid grid-cols-1 gap-8 animate-pulse">
    {[...Array(6)].map((_, i) => (
      <div
        key={i}
        className="relative bg-gray-200 rounded-xl h-[400px]"
      />
    ))}
  </div>
);

export default function ProjectsPage() {
  const cards = [
    {
      id: "ecotrack",
      title: "EcoTrack",
      description: "Une application de suivi d'empreinte carbone avec planification d'itinéraires écologiques et recommandations personnalisées.",
      thumbnail: "https://images.unsplash.com/photo-1501854140801-50d01698950b?q=80&w=1920&auto=format&fit=crop",
      type: "project" as const,
      tags: ["Next.js", "ML", "API Maps", "TypeScript"],
      demoUrl: "/projects/ecotrack",
      features: [
        "Planification d'itinéraires écologiques",
        "Modal interactif de trajet",
        "Calcul d'empreinte carbone",
        "Recommandations IA"
      ],
      className: "md:col-span-2 h-[400px]"
    },
    {
      id: "route-planner",
      title: "Planificateur d'Itinéraire",
      description: "Un widget interactif pour planifier des itinéraires écologiques avec calcul d'empreinte carbone en temps réel.",
      thumbnail: "https://images.unsplash.com/photo-1520531158340-44015069e78e?q=80&w=1920&auto=format&fit=crop",
      type: "widget" as const,
      tags: ["React", "Maps API", "Real-time", "UX/UI"],
      demoUrl: "/projects/route-planner",
      features: [
        "Multi-mode transport",
        "Calcul CO2 en temps réel",
        "Qualité de l'air",
        "Notifications"
      ],
      className: "md:col-span-1 h-[400px]"
    },
    {
      id: "chatbot",
      title: "AI Chatbot",
      description: "Un système de débat multi-agents utilisant Gemini pour générer des discussions argumentées avec synthèse automatique.",
      thumbnail: "https://images.unsplash.com/photo-1470071459604-3b5ec3a7fe05?q=80&w=1920&auto=format&fit=crop",
      type: "project" as const,
      tags: ["Next.js", "Gemini API", "TypeScript", "Framer Motion"],
      demoUrl: "/chat",
      features: [
        "Débat multi-agents",
        "Arguments pour et contre",
        "Synthèse automatique",
        "Interface responsive"
      ],
      className: "md:col-span-1 h-[400px]"
    },
    {
      id: "golddy",
      title: "Golddy",
      description: "Plateforme d'analyse avancée pour influenceurs Instagram, offrant des insights détaillés et des recommandations d'optimisation.",
      thumbnail: "https://images.unsplash.com/photo-1449495169669-7b118f960251?q=80&w=1920&auto=format&fit=crop",
      type: "widget" as const,
      tags: ["Vue.js", "ML", "API Instagram", "Analytics"],
      demoUrl: "/projects/golddy",
      features: [
        "Analyse de performance des posts",
        "Prédiction d'engagement",
        "Recommandations de contenu",
        "Analyse de l'audience"
      ],
      className: "md:col-span-2 h-[400px]"
    },
    {
      id: "dashboard",
      title: "Analytics Dashboard",
      description: "Tableau de bord analytique en temps réel avec visualisations de données interactives.",
      thumbnail: "https://images.unsplash.com/photo-1426604966848-d7adac402bff?q=80&w=1920&auto=format&fit=crop",
      type: "widget" as const,
      tags: ["React", "D3.js", "WebSocket", "TypeScript"],
      demoUrl: "/projects/golddy#dashboard",
      features: [
        "Visualisations temps réel",
        "Filtres dynamiques",
        "Export de données",
        "Personnalisation"
      ],
      className: "md:col-span-2 h-[400px]"
    },
    {
      id: "portfolio",
      title: "Portfolio Interactif",
      description: "Mon portfolio personnel avec des animations avancées, intégration ML et démonstrations de composants.",
      thumbnail: "https://images.unsplash.com/photo-1506744038136-46273834b3fb?q=80&w=1920&auto=format&fit=crop",
      type: "project" as const,
      tags: ["Next.js", "Framer Motion", "Tailwind", "TypeScript"],
      demoUrl: "/",
      features: [
        "Animations fluides",
        "Design responsive",
        "Mode sombre",
        "Composants réutilisables"
      ],
      className: "md:col-span-1 h-[400px]"
    }
  ];

  return (
    <div className="relative min-h-screen bg-white overflow-x-hidden">
      {/* Back Arrow */}
      <Link 
        href="/"
        className="fixed top-6 left-6 lg:top-8 lg:left-8 z-50 p-2 lg:p-3 rounded-full bg-white/90 shadow-lg hover:bg-white hover:scale-110 transition-all duration-200 group"
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
          className="transform group-hover:-translate-x-1 transition-transform duration-200 w-5 h-5 lg:w-6 lg:h-6"
        >
          <path d="M19 12H5M12 19l-7-7 7-7"/>
        </svg>
      </Link>

      {/* Sparkles effect */}
      <div className="absolute inset-0 h-screen w-full">
        <Suspense fallback={null}>
          <SparklesCore
            id="tsparticlesfullpage"
            background="transparent"
            minSize={0.6}
            maxSize={1.4}
            particleDensity={100}
            className="w-full h-full"
            particleColor="#475569"
          />
        </Suspense>
      </div>

      {/* Content */}
      <div className="relative z-10 min-h-screen flex flex-col md:flex-row">
        {/* Titre et description */}
        <div className="md:sticky md:top-0 md:left-0 w-full md:w-[350px] lg:w-[400px] p-6 sm:p-8 md:p-10 lg:p-12 md:h-screen flex flex-col md:justify-center bg-white/80 backdrop-blur-sm shrink-0">
          <div className="space-y-4 md:space-y-5 lg:space-y-6">
            <h1 className="text-3xl sm:text-4xl md:text-5xl lg:text-6xl font-bold tracking-tight text-gray-900">
              Mes Projets
            </h1>
            <p className="text-base sm:text-lg md:text-xl lg:text-2xl text-gray-600 leading-relaxed">
              Une sélection de mes projets récents, mettant en avant mes compétences en développement full-stack, ML et IA.
            </p>
          </div>
        </div>

        {/* Grid des projets */}
        <div className="flex-1 p-4 sm:p-6 md:p-8 lg:p-10">
          <div className="max-w-[1000px] mx-auto">
            <Suspense fallback={<GridSkeleton />}>
              <LayoutGrid cards={cards} />
            </Suspense>
          </div>
        </div>
      </div>
    </div>
  );
}