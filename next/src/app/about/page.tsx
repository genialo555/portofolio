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
        className="fixed top-6 left-6 md:top-8 md:left-8 z-50 p-2 md:p-3 rounded-full bg-white/90 shadow-lg hover:bg-white hover:scale-110 transition-all duration-200 group"
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
          className="transform group-hover:-translate-x-1 transition-transform duration-200 w-5 h-5 md:w-6 md:h-6"
        >
          <path d="M19 12H5M12 19l-7-7 7-7"/>
        </svg>
      </Link>

      {/* Content */}
      <div className="relative z-10 flex flex-col md:flex-row min-h-screen">
        {/* Carousel à gauche avec padding */}
        <div className="w-full md:w-[400px] h-[300px] md:h-auto md:shrink-0 md:pl-32 sticky top-0">
          <PhotoCarousel />
        </div>

        {/* Contenu à droite */}
        <div className="flex-1 flex items-start md:items-center justify-center p-6 md:p-16">
          <div className="max-w-2xl space-y-6 bg-white/80 backdrop-blur-sm rounded-2xl p-6 md:p-8">
            <div className="space-y-4 md:space-y-6">
              <p className="text-lg md:text-xl text-muted-foreground leading-relaxed">
                Après dix ans en tant que chef cuisinier, j'ai troqué les fourneaux contre le code pour me consacrer pleinement au développement web, à l'automatisation et à l'intelligence artificielle.
              </p>
              <p className="text-base text-muted-foreground leading-relaxed">
                Passionné par l'innovation et la tech, je conçois des solutions performantes qui optimisent les processus et apportent un impact réel.
              </p>
            </div>

            <div className="space-y-4 md:space-y-6">
              <p className="text-base text-muted-foreground leading-relaxed">
                <span className="font-medium">Golddy</span> illustre parfaitement mon expertise en scraping et automatisation. Ce système avancé permet d'extraire et d'analyser des données à grande échelle sur Instagram, avec une approche optimisée pour la gestion de gros volumes d'informations. J'ai travaillé sur la récupération et l'exploitation de données stratégiques en temps réel, en structurant un pipeline de scraping intelligent et scalable.
              </p>
              <p className="text-base text-muted-foreground leading-relaxed">
                Je développe également <span className="font-medium">LIA</span>, un assistant IA intégré à EcoTrack, capable d'interagir avec les utilisateurs pour les aider à analyser leur empreinte carbone et optimiser leurs décisions. Basé sur l'API Gemini, LIA agit comme un middle-end intelligent, connectant le frontend et le backend tout en apportant des fonctionnalités avancées comme l'analyse de données, les recommandations automatisées et la gestion sécurisée des interactions utilisateur.
              </p>
            </div>

            <div className="space-y-4 md:space-y-6">
              <p className="text-base text-muted-foreground leading-relaxed">
                Sur le plan technique, je maîtrise des technologies comme React, Next.js et NestJS, avec une architecture robuste et sécurisée reposant sur PostgreSQL, MongoDB et Docker. Je mets en place des pipelines CI/CD, garantissant un déploiement fluide sur Google Cloud, Vercel et Heroku. Mon approche allie sécurité, scalabilité et optimisation, avec une attention particulière à la gestion des rôles et à la protection des données.
              </p>
              <p className="text-base text-muted-foreground leading-relaxed">
                Toujours avide d'explorer de nouvelles perspectives, je me spécialise également en machine learning appliqué avec une maîtrise de TensorFlow, PyTorch et Pandas. J'explore des réseaux neuronaux avancés (LSTM, Transformers) et des techniques d'optimisation pour intégrer l'IA dans des solutions concrètes.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
} 