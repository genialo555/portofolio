"use client";

import { motion } from "framer-motion";
import { useEffect } from "react";
import dynamic from "next/dynamic";

const FlickeringGrid = dynamic(
  () => import("@/components/ui/flickering-grid").then((mod) => mod.FlickeringGrid),
  { ssr: false }
);

interface LoadingStateProps {
    onFinish: () => void;
    isMobile?: boolean;
}

export function LoadingState({ onFinish, isMobile = false }: LoadingStateProps) {
    useEffect(() => {
        const timer = setTimeout(onFinish, 6000);
        return () => clearTimeout(timer);
    }, [onFinish]);

    return (
        <motion.div 
            className="fixed inset-0 z-[2000] flex items-center justify-center overflow-hidden"
            initial={{ opacity: 0 }}
            animate={{ 
                opacity: 1,
                transition: {
                    duration: 0.5,
                    ease: [0.4, 0, 0.2, 1]
                }
            }}
            exit={{ 
                opacity: 0,
                transition: { 
                    duration: 0.5, 
                    ease: [0.4, 0, 0.2, 1] 
                }
            }}
        >
            {/* Background avec la grille */}
            <motion.div 
                className="absolute inset-0 bg-background/95"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ duration: 0.5 }}
            >
                <FlickeringGrid 
                    color="rgb(0, 0, 0)"
                    maxOpacity={0.2}
                    flickerChance={0.2}
                    squareSize={isMobile ? 4 : 6}
                    gridGap={isMobile ? 6 : 8}
                />
            </motion.div>

            {/* Contenu */}
            <motion.div 
                className="relative z-[2001] text-center space-y-4 px-4"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.3, duration: 0.8 }}
            >
                <motion.h2
                    className="text-xl md:text-2xl font-semibold text-foreground"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ delay: 0.5, duration: 0.8 }}
                >
                    {isMobile ? "Préparation de votre itinéraire mobile" : "Préparation de votre itinéraire"}
                </motion.h2>
                <motion.p
                    className="text-sm md:text-base text-muted-foreground"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ delay: 0.7, duration: 0.8 }}
                >
                    Veuillez patienter pendant que nous préparons votre trajet{isMobile ? " sur mobile" : ""}...
                </motion.p>
            </motion.div>
        </motion.div>
    );
}