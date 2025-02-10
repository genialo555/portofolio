import Image from "next/image"
import Link from "next/link"
import { cn } from "@/lib/utils"
import { ShimmerButton } from "@/components/ui/shimmer-button"

interface ProjectCardProps {
  title: string
  description: string
  tags: string[]
  image: string
  demoUrl: string
  features: string[]
}

export function ProjectCard({
  title,
  description,
  tags,
  image,
  demoUrl,
  features,
}: ProjectCardProps) {
  return (
    <div className="group relative overflow-hidden rounded-lg border bg-card text-card-foreground">
      {/* Image du projet avec effet de hover */}
      <div className="relative h-48 overflow-hidden">
        <Image
          src={image}
          alt={title}
          width={600}
          height={300}
          className="object-cover transition-transform duration-300 group-hover:scale-105"
        />
        <div className="absolute inset-0 bg-gradient-to-t from-background/80 to-transparent" />
      </div>

      {/* Contenu */}
      <div className="p-6 space-y-4">
        {/* Tags */}
        <div className="flex flex-wrap gap-2">
          {tags.map((tag) => (
            <span
              key={tag}
              className="px-2 py-1 text-xs rounded-full bg-primary/10 text-primary"
            >
              {tag}
            </span>
          ))}
        </div>

        {/* Titre et description */}
        <div className="space-y-2">
          <h3 className="text-2xl font-bold tracking-tight">{title}</h3>
          <p className="text-muted-foreground">{description}</p>
        </div>

        {/* Liste des fonctionnalités */}
        <ul className="space-y-2">
          {features.map((feature) => (
            <li key={feature} className="flex items-center text-sm text-muted-foreground">
              <svg
                className="mr-2 h-4 w-4 text-primary"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"
                />
              </svg>
              {feature}
            </li>
          ))}
        </ul>

        {/* Lien vers la démo */}
        <Link href={demoUrl} className="block">
          <ShimmerButton
            className="w-full text-sm font-normal"
            background="hsl(var(--card))"
            shimmerColor="rgba(0, 0, 0, 0.1)"
            shimmerDuration="3s"
            shimmerSize="0.1em"
          >
            <span className="text-primary">Voir le projet</span>
            <svg
              className="ml-2 h-4 w-4 text-primary"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={1.5}
                d="M14 5l7 7m0 0l-7 7m7-7H3"
              />
            </svg>
          </ShimmerButton>
        </Link>
      </div>
    </div>
  )
} 