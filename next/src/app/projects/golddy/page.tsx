"use client"

import { motion } from "framer-motion"

export default function GoddyPage() {
  return (
    <div className="container py-16 space-y-8">
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
        <div className="aspect-video bg-muted rounded-md">
          {/* Placeholder pour la démo/capture d'écran */}
        </div>
      </motion.div>

      {/* Documentation */}
      <div className="prose prose-gray dark:prose-invert max-w-none">
        <h2>Fonctionnalités principales</h2>
        <ul>
          <li>Analyse détaillée des performances des posts Instagram</li>
          <li>Prédiction d&apos;engagement basée sur le ML</li>
          <li>Recommandations personnalisées de contenu</li>
          <li>Analyse démographique et comportementale de l&apos;audience</li>
          <li>Suivi des tendances et des hashtags</li>
          <li>Planification optimisée des publications</li>
          <li>Rapports automatisés et insights</li>
        </ul>

        <h2>Technologies utilisées</h2>
        <ul>
          <li>Vue.js 3 avec Composition API</li>
          <li>TensorFlow.js pour les prédictions</li>
          <li>API Instagram Graph</li>
          <li>Python FastAPI pour le backend</li>
          <li>PostgreSQL pour le stockage des données</li>
          <li>Redis pour le cache</li>
        </ul>

        <h2>Architecture</h2>
        <p>
          Golddy utilise une architecture moderne basée sur des microservices, permettant une scalabilité optimale
          et une maintenance facilitée. Le frontend Vue.js communique avec un backend Python qui orchestre différents
          services : analyse de données, ML, cache, et stockage.
        </p>

        <h2>Modèles ML</h2>
        <p>
          Les modèles de machine learning sont entraînés sur des millions de posts Instagram pour prédire
          l&apos;engagement et générer des recommandations pertinentes. Ils prennent en compte de nombreux facteurs :
          timing, contenu visuel, texte, hashtags, et historique des performances.
        </p>
      </div>
    </div>
  )
} 