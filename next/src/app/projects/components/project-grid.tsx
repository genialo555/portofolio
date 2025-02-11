"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { cn } from "@/lib/utils";
import Image from "next/image";
import { Card } from "@/types";

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
const SelectedCard = ({ selected }: { selected: Card }) => {
  return (
    <div className="absolute inset-0 p-8 flex flex-col items-center">
      <div className="relative w-full h-1/2 mb-4">
        <Image
          src={selected.thumbnail}
          alt={selected.title}
          fill
          className="object-cover rounded-xl"
          sizes="(max-width: 768px) 100vw, (max-width: 1200px) 50vw, 33vw"
          quality={75}
        />
      </div>
      <h2 className="text-3xl font-bold mb-4">{selected.title}</h2>
      <p className="text-gray-600 mb-4 text-center">{selected.description}</p>
      <div className="flex flex-wrap gap-2 justify-center mb-4">
        {selected.tags.map((tag) => (
          <span
            key={tag}
            className="px-3 py-1 bg-gray-100 rounded-full text-sm text-gray-600"
          >
            {tag}
          </span>
        ))}
      </div>
      <div className="space-y-2">
        {selected.features.map((feature) => (
          <div key={feature} className="flex items-center gap-2">
            <svg
              className="w-5 h-5 text-green-500"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M5 13l4 4L19 7"
              />
            </svg>
            <span>{feature}</span>
          </div>
        ))}
      </div>
    </div>
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
    <div className="max-w-7xl mx-auto">
      <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
        <AnimatePresence>
          {cards.map((card, i) => (
            <motion.div
              key={card.id}
              layoutId={`card-${card.id}`}
              className={cn(card.className, "relative")}
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
                  "relative overflow-hidden w-full h-full rounded-xl shadow-lg cursor-pointer",
                  selected?.id === card.id
                    ? "fixed inset-0 h-2/3 w-full md:w-2/3 m-auto z-50 flex justify-center items-center flex-wrap flex-col"
                    : lastSelected?.id === card.id
                    ? "z-40 bg-white"
                    : "bg-white hover:shadow-xl transition-shadow duration-200"
                )}
                layoutId={`card-content-${card.id}`}
                transition={{ duration: 0.3 }}
              >
                {selected?.id === card.id ? (
                  <SelectedCard selected={selected} />
                ) : (
                  <ImageComponent card={card} />
                )}
              </motion.div>
            </motion.div>
          ))}
        </AnimatePresence>
        <AnimatePresence>
          {selected?.id && (
            <motion.div
              onClick={handleOutsideClick}
              className="fixed inset-0 bg-black/50 z-40"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.2 }}
            />
          )}
        </AnimatePresence>
      </div>
    </div>
  );
}
