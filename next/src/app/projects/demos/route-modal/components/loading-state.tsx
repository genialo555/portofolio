"use client";

import { motion } from "framer-motion";

function FloatingPaths({ position }: { position: number }) {
    const paths = Array.from({ length: 36 }, (_, i) => ({
        id: i,
        d: `M-${380 - i * 5 * position} -${189 + i * 6}C-${
            380 - i * 5 * position
        } -${189 + i * 6} -${312 - i * 5 * position} ${216 - i * 6} ${
            152 - i * 5 * position
        } ${343 - i * 6}C${616 - i * 5 * position} ${470 - i * 6} ${
            684 - i * 5 * position
        } ${875 - i * 6} ${684 - i * 5 * position} ${875 - i * 6}`,
        color: `rgba(15,23,42,${0.1 + i * 0.03})`,
        width: 0.5 + i * 0.03,
    }));

    return (
        <div className="absolute inset-0 pointer-events-none">
            <svg
                className="w-full h-full text-slate-950 dark:text-white"
                viewBox="0 0 696 316"
                fill="none"
            >
                <title>Background Paths</title>
                {paths.map((path) => (
                    <motion.path
                        key={path.id}
                        d={path.d}
                        stroke="currentColor"
                        strokeWidth={path.width}
                        strokeOpacity={0.1 + path.id * 0.02}
                        initial={{ pathLength: 0.3, opacity: 0.4 }}
                        animate={{
                            pathLength: 1,
                            opacity: [0.2, 0.4, 0.2],
                            pathOffset: [0, 1, 0],
                        }}
                        transition={{
                            duration: 8.5,
                            repeat: Number.POSITIVE_INFINITY,
                            ease: [0.4, 0, 0.2, 1],
                            opacity: {
                                duration: 4,
                                repeat: Number.POSITIVE_INFINITY,
                                ease: "easeInOut"
                            }
                        }}
                    />
                ))}
            </svg>
        </div>
    );
}

interface LoadingStateProps {
    onFinish: () => void;
}

export function LoadingState({ onFinish }: LoadingStateProps) {
    return (
        <motion.div 
            className="fixed inset-0 z-50 bg-background flex items-center justify-center"
            initial={{ opacity: 0 }}
            animate={{ 
                opacity: 1,
                transition: {
                    duration: 1,
                    ease: [0.4, 0, 0.2, 1]
                }
            }}
            exit={{ 
                opacity: 0,
                transition: { 
                    duration: 1.6, 
                    ease: [0.4, 0, 0.2, 1] 
                }
            }}
        >
            <div className="relative min-h-screen w-full flex items-center justify-center overflow-hidden">
                <div className="absolute inset-0">
                    <FloatingPaths position={1} />
                    <FloatingPaths position={-1} />
                </div>

                <div className="relative z-10 text-center">
                    <motion.div
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ 
                            delay: 0.3, 
                            duration: 1.2, 
                            ease: [0.4, 0, 0.2, 1] 
                        }}
                        className="text-2xl font-semibold mb-4"
                    >
                        Préparation de votre itinéraire
                    </motion.div>
                    <motion.div
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ 
                            delay: 0.6, 
                            duration: 1.2, 
                            ease: [0.4, 0, 0.2, 1] 
                        }}
                        className="text-muted-foreground"
                    >
                        Chargement des données en cours...
                    </motion.div>
                </div>
            </div>
        </motion.div>
    );
} 