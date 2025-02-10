import Link from "next/link"
import { ProjectCard } from "./components/project-card"

export default function ProjectsPage() {
  return (
    <div className="container py-16 space-y-16">
      {/* En-tête */}
      <div className="space-y-4">
        <h1 className="text-4xl font-bold tracking-tight">
          Mes Projets
        </h1>
        <p className="text-xl text-muted-foreground max-w-2xl">
          Une sélection de mes projets récents, mettant en avant mes compétences en développement full-stack, ML et IA.
        </p>
      </div>

      {/* Liste des projets */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
        {/* EcoTrack */}
        <ProjectCard
          title="EcoTrack"
          description="Une application de suivi d'empreinte carbone avec planification d'itinéraires écologiques et recommandations personnalisées."
          tags={["Next.js", "ML", "API Maps", "TypeScript"]}
          image="/projects/ecotrack.png"
          demoUrl="/projects/ecotrack"
          features={[
            "Planification d'itinéraires écologiques",
            "Modal interactif de trajet",
            "Calcul d'empreinte carbone",
            "Recommandations IA"
          ]}
        />

        {/* Portfolio */}
        <ProjectCard
          title="Portfolio Interactif"
          description="Mon portfolio personnel avec des animations avancées, intégration ML et démonstrations de composants."
          tags={["Next.js", "Three.js", "ML", "WebGL"]}
          image="/projects/portfolio.png"
          demoUrl="/projects/portfolio"
          features={[
            "Animations WebGL",
            "Chatbot ML multi-agents",
            "Intégrations API IA",
            "Composants interactifs"
          ]}
        />

        {/* Golddy */}
        <ProjectCard
          title="Golddy"
          description="Plateforme d'analyse avancée pour influenceurs Instagram, offrant des insights détaillés et des recommandations d'optimisation."
          tags={["Vue.js", "ML", "API Instagram", "Analytics"]}
          image="/projects/golddy.png"
          demoUrl="/projects/golddy"
          features={[
            "Analyse de performance des posts",
            "Prédiction d'engagement",
            "Recommandations de contenu",
            "Analyse de l'audience"
          ]}
        />
      </div>

      {/* Section Démos Techniques */}
      <div className="space-y-8">
        <h2 className="text-3xl font-bold tracking-tight">
          Démos Techniques
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {/* Ces sections seront implémentées plus tard */}
          <div className="p-6 rounded-lg border bg-card text-card-foreground hover:border-primary transition-colors">
            <h3 className="text-xl font-semibold mb-2">Modal d&apos;Itinéraire</h3>
            <p className="text-muted-foreground mb-4">
              Démonstration du composant de planification d&apos;itinéraire utilisé dans EcoTrack.
            </p>
            <Link 
              href="/projects/demos/route-modal"
              className="text-primary hover:underline"
            >
              Voir la démo →
            </Link>
          </div>

          <div className="p-6 rounded-lg border bg-card text-card-foreground hover:border-primary transition-colors">
            <h3 className="text-xl font-semibold mb-2">Chatbot ML</h3>
            <p className="text-muted-foreground mb-4">
              Exemple d&apos;interaction avec différents agents ML spécialisés.
            </p>
            <Link 
              href="/projects/demos/chatbot"
              className="text-primary hover:underline"
            >
              Voir la démo →
            </Link>
          </div>

          <div className="p-6 rounded-lg border bg-card text-card-foreground hover:border-primary transition-colors">
            <h3 className="text-xl font-semibold mb-2">API IA</h3>
            <p className="text-muted-foreground mb-4">
              Démonstration des wrappers d&apos;API IA et de leurs utilisations.
            </p>
            <Link 
              href="/projects/demos/ai-api"
              className="text-primary hover:underline"
            >
              Voir la démo →
            </Link>
          </div>
        </div>
      </div>
    </div>
  )
} 