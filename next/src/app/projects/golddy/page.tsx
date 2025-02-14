"use client"

import { motion } from "framer-motion"
import Link from "next/link"
import { Button } from "@/components/ui/button"
import Image from "next/image"
import { Particles } from "@/components/ui/particles"

export default function GoddyPage() {
  return (
    <div className="relative min-h-screen bg-background">
      {/* Particles Background */}
      <div className="absolute inset-0 z-0">
        <Particles
          className="absolute inset-0"
          quantity={100}
          staticity={30}
          color="#6b7280"
          ease={100}
          refresh={false}
          vx={-0.2}
          vy={0.2}
        />
      </div>

      <div className="container py-16 space-y-8 relative z-10">
        {/* Bouton de retour */}
        <Link 
          href="/projects"
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

        {/* En-tête */}
        <div className="space-y-4">
          <h1 className="text-4xl font-bold tracking-tight">
            Golddy
          </h1>
          <p className="text-xl text-muted-foreground max-w-2xl">
            Plateforme d&apos;analyse avancée pour influenceurs Instagram, combinant ML et analytics pour optimiser la performance des contenus.
          </p>
        </div>

        {/* Aperçu */}
        <motion.div 
          className="rounded-lg border bg-card p-8"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
        >
          <div className="relative aspect-video w-full overflow-hidden rounded-md">
            <Image
              src="/Screenshot from 2025-02-14 14-21-02.png"
              alt="Golddy Home Page"
              fill
              className="object-cover"
              quality={100}
              priority
            />
          </div>
        </motion.div>

        {/* Documentation */}
        <div className="space-y-12">
          {/* Fonctionnalités principales */}
          <div className="rounded-lg border bg-card/50 backdrop-blur-sm p-8">
            <h2 className="text-2xl font-semibold mb-6">Fonctionnalités principales</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {[
                {
                  icon: "📊",
                  feature: "Analyse détaillée des performances des posts Instagram",
                },
                {
                  icon: "🤖",
                  feature: "Prédiction d'engagement basée sur le ML",
                },
                {
                  icon: "💡",
                  feature: "Recommandations personnalisées de contenu",
                },
                {
                  icon: "👥",
                  feature: "Analyse démographique et comportementale de l'audience",
                },
                {
                  icon: "📈",
                  feature: "Suivi des tendances et des hashtags",
                },
                {
                  icon: "📅",
                  feature: "Planification optimisée des publications",
                },
                {
                  icon: "📑",
                  feature: "Rapports automatisés et insights",
                },
              ].map((item, index) => (
                <div
                  key={index}
                  className="flex items-start gap-4 p-4 rounded-lg bg-white/5 hover:bg-white/10 transition-colors"
                >
                  <span className="text-2xl">{item.icon}</span>
                  <span className="text-sm text-black">{item.feature}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Technologies utilisées */}
          <div className="rounded-lg border bg-card/50 backdrop-blur-sm p-8">
            <h2 className="text-2xl font-semibold mb-6">Stack Technique</h2>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              {[
                {
                  category: "Frontend",
                  tech: "Vue.js 3",
                  details: "Composition API",
                  icon: "🎨"
                },
                {
                  category: "ML",
                  tech: "TensorFlow.js",
                  details: "Prédictions en temps réel",
                  icon: "🧠"
                },
                {
                  category: "API",
                  tech: "Instagram Graph",
                  details: "Intégration native",
                  icon: "📱"
                },
                {
                  category: "Backend",
                  tech: "Python FastAPI",
                  details: "Haute performance",
                  icon: "⚡"
                },
                {
                  category: "Database",
                  tech: "PostgreSQL",
                  details: "Stockage robuste",
                  icon: "💾"
                },
                {
                  category: "Cache",
                  tech: "Redis",
                  details: "Performance optimale",
                  icon: "🚀"
                },
              ].map((item, index) => (
                <div
                  key={index}
                  className="p-4 rounded-lg bg-white/5 hover:bg-white/10 transition-colors"
                >
                  <div className="flex items-center gap-2 mb-2">
                    <span className="text-2xl">{item.icon}</span>
                    <span className="text-sm font-medium text-black">{item.category}</span>
                  </div>
                  <div className="text-black font-medium">{item.tech}</div>
                  <div className="text-xs text-black">{item.details}</div>
                </div>
              ))}
            </div>
          </div>

          {/* Architecture */}
          <div className="rounded-lg border bg-card/50 backdrop-blur-sm p-8">
            <h2 className="text-2xl font-semibold mb-6">Architecture</h2>
            <div className="bg-white/5 rounded-lg p-6">
              <div className="flex items-start gap-4">
                <div className="p-3 rounded-full bg-primary/20 text-primary">
                  <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
                  </svg>
                </div>
                <div className="space-y-2">
                  <p className="text-black">
                    Architecture moderne basée sur des microservices, permettant une scalabilité optimale
                    et une maintenance facilitée. Le frontend Vue.js communique avec un backend Python qui orchestre différents
                    services : analyse de données, ML, cache, et stockage.
                  </p>
                  <div className="flex flex-wrap gap-2 mt-4">
                    <span className="px-2 py-1 text-xs rounded-full bg-primary/20 text-primary">Microservices</span>
                    <span className="px-2 py-1 text-xs rounded-full bg-primary/20 text-primary">Scalable</span>
                    <span className="px-2 py-1 text-xs rounded-full bg-primary/20 text-primary">Maintenable</span>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Modèles ML */}
          <div className="rounded-lg border bg-card/50 backdrop-blur-sm p-8">
            <h2 className="text-2xl font-semibold mb-6">Modèles ML</h2>
            <div className="bg-white/5 rounded-lg p-6">
              <div className="flex items-start gap-4">
                <div className="p-3 rounded-full bg-primary/20 text-primary">
                  <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
                  </svg>
                </div>
                <div className="space-y-4">
                  <p className="text-black">
                    Les modèles de machine learning sont entraînés sur des millions de posts Instagram pour prédire
                    l'engagement et générer des recommandations pertinentes.
                  </p>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    {[
                      { factor: "Timing", icon: "⏰" },
                      { factor: "Contenu visuel", icon: "🖼️" },
                      { factor: "Texte", icon: "📝" },
                      { factor: "Hashtags", icon: "#️⃣" },
                    ].map((item, index) => (
                      <div key={index} className="p-3 rounded-lg bg-white/5 text-center">
                        <div className="text-2xl mb-2">{item.icon}</div>
                        <div className="text-xs text-black">{item.factor}</div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Section Dashboard Analytics */}
          <div className="mt-16">
            <h2 className="text-2xl font-semibold mb-6">Dashboard Analytics</h2>
            <p className="text-lg text-muted-foreground mb-8">
              Un tableau de bord interactif offrant des visualisations en temps réel des performances et des insights.
            </p>
            
            <div className="flex flex-col items-center gap-4">
              <div className="bg-white/5 backdrop-blur-sm rounded-lg p-8 w-full">
                <p className="text-center text-black mb-4">
                  Accédez au tableau de bord complet pour visualiser toutes les analyses et statistiques en temps réel.
                </p>
                <div className="flex justify-center">
                  <Link href="/projects/golddy/dashboard">
                    <Button className="bg-primary text-primary-foreground hover:bg-primary/90">
                      Voir le Dashboard
                    </Button>
                  </Link>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
} 