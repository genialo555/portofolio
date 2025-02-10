"use client"

import { useState, useEffect } from "react"
import { RouteModal } from "./components/route-modal"
import { LoadingState } from "./components/loading-state"
import { AnimatePresence, motion } from "framer-motion"
import { ShimmerButton } from "@/components/ui/shimmer-button"
import { AirQualityWidget } from "./components/air-quality-widget"
import { NotificationsWidget } from "./components/notifications-widget"

export default function RouteModalDemo() {
  const [isLoading, setIsLoading] = useState(false)
  const [isOpen, setIsOpen] = useState(false)
  const [isBlurring, setIsBlurring] = useState(false)

  useEffect(() => {
    let timer: NodeJS.Timeout;
    if (isLoading) {
      timer = setTimeout(() => {
        setIsBlurring(true);
        setTimeout(() => {
          setIsLoading(false);
          setTimeout(() => {
            setIsBlurring(false);
            setIsOpen(true);
          }, 800);
        }, 1600);
      }, 15000);
    }
    return () => {
      if (timer) clearTimeout(timer);
    };
  }, [isLoading]);

  const handleClick = () => {
    setIsLoading(true)
  }

  return (
    <div className="container py-16 space-y-8">
      <AnimatePresence mode="wait">
        {isLoading && (
          <LoadingState onFinish={() => null} />
        )}
      </AnimatePresence>

      {/* Blur Transition */}
      <AnimatePresence>
        {isBlurring && (
          <motion.div
            initial={{ opacity: 0, filter: "blur(0px)" }}
            animate={{ 
              opacity: 1, 
              filter: "blur(8px)",
              transition: {
                opacity: { duration: 0.8, ease: [0.4, 0, 0.2, 1] },
                filter: { duration: 1.2, ease: [0.4, 0, 0.2, 1] }
              }
            }}
            exit={{ 
              opacity: 0,
              filter: "blur(0px)",
              transition: {
                opacity: { duration: 0.8, ease: [0.4, 0, 0.2, 1] },
                filter: { duration: 0.6, ease: [0.4, 0, 0.2, 1] }
              }
            }}
            className="fixed inset-0 z-50 bg-background/80 backdrop-blur-md"
          />
        )}
      </AnimatePresence>

      {/* Widgets */}
      <AirQualityWidget isOpen={isOpen} />
      <NotificationsWidget isOpen={isOpen} />

      {/* En-tête */}
      <div className="space-y-4">
        <h1 className="text-4xl font-bold tracking-tight">
          Modal d&apos;Itinéraire
        </h1>
        <p className="text-xl text-muted-foreground max-w-2xl">
          Un composant interactif pour la planification d&apos;itinéraires écologiques, utilisé dans le projet EcoTrack.
        </p>
      </div>

      {/* Zone de démonstration */}
      <div className="flex flex-col items-center justify-center min-h-[400px] rounded-lg border bg-card p-8">
        <ShimmerButton
          onClick={handleClick}
          background="hsl(var(--primary))"
          shimmerColor="rgba(255, 255, 255, 0.2)"
          className="font-medium text-base"
        >
          Planifier un itinéraire
        </ShimmerButton>
      </div>

      {/* Modal */}
      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ opacity: 0, scale: 0.98 }}
            animate={{ 
              opacity: 1,
              scale: 1,
              transition: {
                duration: 0.8,
                ease: [0.4, 0, 0.2, 1]
              }
            }}
            exit={{ opacity: 0, scale: 0.98 }}
          >
            <RouteModal 
              isOpen={isOpen} 
              onClose={() => setIsOpen(false)} 
            />
          </motion.div>
        )}
      </AnimatePresence>

      {/* Documentation */}
      <div className="prose prose-gray dark:prose-invert max-w-none">
        <h2>Fonctionnalités</h2>
        <ul>
          <li>Recherche d&apos;adresses avec autocomplétion</li>
          <li>Calcul d&apos;itinéraires multimodaux</li>
          <li>Estimation de l&apos;empreinte carbone</li>
          <li>Visualisation sur carte interactive</li>
          <li>Suggestions d&apos;alternatives écologiques</li>
          <li>Suivi de la qualité de l&apos;air en temps réel</li>
          <li>Notifications de trafic et alertes</li>
        </ul>

        <h2>Utilisation</h2>
        <p>
          Ce composant est conçu pour être facilement intégré dans n&apos;importe quelle application React/Next.js.
          Il utilise l&apos;API Google Maps pour la géolocalisation et le calcul d&apos;itinéraires, ainsi qu&apos;un
          modèle ML personnalisé pour les recommandations écologiques.
        </p>

        <h2>Technologies utilisées</h2>
        <ul>
          <li>Next.js 13+ avec App Router</li>
          <li>API Google Maps & Places</li>
          <li>TensorFlow.js pour les prédictions</li>
          <li>Tailwind CSS pour le style</li>
          <li>Framer Motion pour les animations</li>
        </ul>
      </div>
    </div>
  )
} 