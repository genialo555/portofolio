"use client"

import Link from "next/link"
import { Button } from "@/components/ui/button"
import { motion } from "framer-motion"
import { SparklesCore } from "@/app/projects/components/sparkles"

export default function ChatbotPage() {
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

      <div className="relative z-10 container mx-auto px-4 py-16">
        <div className="max-w-4xl mx-auto space-y-8">
          {/* Titre et Description */}
          <div className="text-center space-y-4">
            <h1 className="text-4xl font-bold tracking-tight">AI Chatbot</h1>
            <p className="text-xl text-muted-foreground">
              Un système de débat multi-agents utilisant Gemini pour générer des discussions argumentées avec synthèse automatique.
            </p>
          </div>

          {/* Aperçu */}
          <motion.div 
            className="rounded-lg border bg-card p-8"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
          >
            <div className="space-y-8">
              {/* Fonctionnalités */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="space-y-4">
                  <h2 className="text-2xl font-semibold">Fonctionnalités</h2>
                  <ul className="space-y-2">
                    <li className="flex items-center gap-2">
                      <span className="text-primary">→</span>
                      <span>Débat multi-agents avec positions pour et contre</span>
                    </li>
                    <li className="flex items-center gap-2">
                      <span className="text-primary">→</span>
                      <span>Génération de synthèse automatique</span>
                    </li>
                    <li className="flex items-center gap-2">
                      <span className="text-primary">→</span>
                      <span>Interface adaptative et responsive</span>
                    </li>
                    <li className="flex items-center gap-2">
                      <span className="text-primary">→</span>
                      <span>Support multimédia (images, vidéos)</span>
                    </li>
                  </ul>
                </div>

                <div className="space-y-4">
                  <h2 className="text-2xl font-semibold">Technologies</h2>
                  <ul className="space-y-2">
                    <li className="flex items-center gap-2">
                      <span className="text-primary">→</span>
                      <span>Next.js avec TypeScript</span>
                    </li>
                    <li className="flex items-center gap-2">
                      <span className="text-primary">→</span>
                      <span>API Gemini pour le traitement du langage</span>
                    </li>
                    <li className="flex items-center gap-2">
                      <span className="text-primary">→</span>
                      <span>Framer Motion pour les animations</span>
                    </li>
                    <li className="flex items-center gap-2">
                      <span className="text-primary">→</span>
                      <span>Tailwind CSS pour le style</span>
                    </li>
                  </ul>
                </div>
              </div>

              {/* Appel à l'action */}
              <div className="flex flex-col items-center gap-4 pt-8">
                <p className="text-center text-muted-foreground">
                  Prêt à lancer un débat ? Choisissez votre sujet et laissez les agents IA débattre !
                </p>
                <Link href="/chat">
                  <Button size="lg" className="mt-4">
                    Lancer un débat
                  </Button>
                </Link>
              </div>
            </div>
          </motion.div>
        </div>
      </div>
    </div>
  )
} 