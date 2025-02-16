"use client";

import { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { cn } from "@/lib/utils";
import Image from "next/image";
import { Card } from "@/types";
import { WavesBackground } from "@/components/ui/waves-background";
import { LoadingState } from "@/app/projects/demos/route-modal/components/loading-state";
import { useRouter } from "next/navigation";
import Link from "next/link";

// Separate the image component for better performance
const ImageComponent = ({ card }: { card: Card }) => {
  return (
    <div className="absolute inset-0 h-full w-full">
      <Image
        src={card.thumbnail}
        alt={card.title}
        fill
        className="object-cover rounded-xl"
        sizes="(max-width: 768px) 100vw, (max-width: 1200px) 50vw, 33vw"
        loading="lazy"
        quality={75}
      />
      <div className="absolute inset-0 bg-black/20 group-hover:bg-black/40 transition-colors duration-200 rounded-xl" />
      <div className="absolute inset-0 p-6 text-white flex flex-col justify-end">
        <h3 className="text-2xl font-bold mb-2">{card.title}</h3>
        <p className="text-sm opacity-90">{card.description}</p>
      </div>
    </div>
  );
};

// Separate the selected card component
const SelectedCard = ({ project, onClose }: { project: Card; onClose: () => void }) => {
  const [isLoading, setIsLoading] = useState(false);
  const [showModal, setShowModal] = useState(true);
  const router = useRouter();

  const handleLoadingComplete = () => {
    setIsLoading(false);
    router.push('/projects/route-planner');
  };

  const handleLiveDemo = (e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (project.title === "Planificateur d'itinéraire") {
      router.push('/projects/demos/route-modal');
    } else if (project.demoUrl) {
      window.open(project.demoUrl, '_blank');
    }
  };

  if (isLoading) {
    return (
      <AnimatePresence>
        <LoadingState onFinish={handleLoadingComplete} />
      </AnimatePresence>
    );
  }

  if (!showModal) return null;

  return (
    <motion.div
      layoutId={`card-${project.id}`}
      className="fixed inset-0 z-[100] flex items-center justify-center p-4"
      onClick={onClose}
    >
      <motion.div
        className="absolute inset-0 bg-black/60 backdrop-blur-sm"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
      />
      <motion.div
        className="relative w-full max-w-2xl rounded-2xl bg-white/80 backdrop-blur-sm p-6"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="relative aspect-video w-full overflow-hidden rounded-lg">
          <Image
            src={project.thumbnail}
            alt={project.title}
            fill
            className="object-cover"
          />
        </div>
        <div className="mt-4">
          <h2 className="text-2xl font-semibold text-gray-900">{project.title}</h2>
          <p className="mt-2 text-gray-600">{project.description}</p>
          
          <div className="mt-6 flex flex-wrap gap-4">
            {project.demoUrl && (
              <div className="flex gap-2">
                {project.title === "Golddy" ? (
                  <>
                    <Link
                      href={project.demoUrl}
                      className="inline-flex items-center gap-2 rounded-lg bg-black px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-gray-900"
                    >
                      Voir la page
                      <span className="text-xs">↗</span>
                    </Link>
                    <Link
                      href={`${project.demoUrl}#dashboard`}
                      className="inline-flex items-center gap-2 rounded-lg border border-gray-300 px-4 py-2 text-sm font-medium text-gray-900 transition-colors hover:bg-gray-50"
                    >
                      Voir le dashboard
                      <span className="text-xs">↗</span>
                    </Link>
                  </>
                ) : project.title === "AI Chatbot" ? (
                  <>
                    <Link
                      href={project.demoUrl}
                      className="inline-flex items-center gap-2 rounded-lg bg-black px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-gray-900"
                    >
                      Voir la page
                      <span className="text-xs">↗</span>
                    </Link>
                    <Link
                      href="/chat"
                      className="inline-flex items-center gap-2 rounded-lg border border-gray-300 px-4 py-2 text-sm font-medium text-gray-900 transition-colors hover:bg-gray-50"
                    >
                      Lancer un débat
                      <span className="text-xs">↗</span>
                    </Link>
                  </>
                ) : project.title === "Planificateur d'itinéraire" ? (
                  <button
                    onClick={handleLiveDemo}
                    className="inline-flex items-center gap-2 rounded-lg bg-black px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-gray-900"
                  >
                    Live Demo
                    <span className="text-xs">↗</span>
                  </button>
                ) : (
                  <Link
                    href={project.demoUrl}
                    className="inline-flex items-center gap-2 rounded-lg bg-black px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-gray-900"
                  >
                    Live Demo
                    <span className="text-xs">↗</span>
                  </Link>
                )}
              </div>
            )}
            {project.githubUrl && (
              <a
                href={project.githubUrl}
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-2 rounded-lg border border-gray-300 px-4 py-2 text-sm font-medium text-gray-900 transition-colors hover:bg-gray-50"
              >
                View Code
                <span className="text-xs">↗</span>
              </a>
            )}
          </div>
        </div>
        <button
          onClick={onClose}
          className="absolute top-4 right-4 rounded-full p-2 text-gray-500 transition-colors hover:bg-gray-100"
        >
          <svg
            className="h-6 w-6"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M6 18L18 6M6 6l12 12"
            />
          </svg>
        </button>
      </motion.div>
    </motion.div>
  );
};

export function LayoutGrid({ cards }: { cards: Card[] }) {
  const [selected, setSelected] = useState<Card | null>(null);
  const [lastSelected, setLastSelected] = useState<Card | null>(null);

  const handleClick = (card: Card) => {
    setLastSelected(selected);
    setSelected(card);
  };

  const handleOutsideClick = () => {
    setLastSelected(selected);
    setSelected(null);
  };

  return (
    <div className="w-full">
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
        <AnimatePresence>
          {cards.map((card, i) => (
            <motion.div
              key={card.id}
              layoutId={`card-${card.id}`}
              className={cn(
                "relative",
                "md:col-span-1",
                card.className?.includes("md:col-span-2") && "md:col-span-2",
              )}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: 20 }}
              transition={{
                duration: 0.3,
                delay: i * 0.1,
                ease: "easeOut",
              }}
            >
              <motion.div
                onClick={() => handleClick(card)}
                className={cn(
                  "relative overflow-hidden w-full h-[300px] md:h-[400px] rounded-xl shadow-lg cursor-pointer",
                  "bg-white hover:shadow-xl transition-shadow duration-200"
                )}
                layoutId={`card-content-${card.id}`}
                transition={{ duration: 0.3 }}
              >
                <ImageComponent card={card} />
              </motion.div>
            </motion.div>
          ))}
        </AnimatePresence>
        <AnimatePresence>
          {selected && (
            <SelectedCard project={selected} onClose={handleOutsideClick} />
          )}
        </AnimatePresence>
      </div>
    </div>
  );
}
