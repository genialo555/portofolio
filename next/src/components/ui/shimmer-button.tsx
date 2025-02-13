"use client";

import React, { useState } from "react";
import { cn } from "@/lib/utils";
import { Vortex } from "./loading-vortex";
import { motion, AnimatePresence } from "framer-motion";

export interface ShimmerButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  shimmerColor?: string;
  shimmerSize?: string;
  borderRadius?: string;
  shimmerDuration?: string;
  background?: string;
  className?: string;
  children?: React.ReactNode;
  isLoading?: boolean;
}

const ShimmerButton = React.forwardRef<HTMLButtonElement, ShimmerButtonProps>(
  (
    {
      shimmerColor = "rgba(255, 255, 255, 0.4)",
      shimmerSize = "0.1em",
      shimmerDuration = "1.5s",
      borderRadius = "0.5rem",
      background = "transparent",
      className,
      children,
      onClick,
      disabled,
      isLoading,
      ...props
    },
    ref,
  ) => {
    const [showVortex, setShowVortex] = useState(false);

    const handleClick = async (e: React.MouseEvent<HTMLButtonElement>) => {
      if (!disabled && !isLoading && onClick) {
        setShowVortex(true);
        await onClick(e);
        setShowVortex(false);
      }
    };

    return (
      <>
        <button
          ref={ref}
          className={cn(
            "group relative flex items-center justify-center overflow-hidden whitespace-nowrap",
            "px-6 py-3 transition-all duration-300",
            "disabled:pointer-events-none disabled:opacity-50",
            "bg-gradient-to-r from-blue-600 via-indigo-600 to-violet-600",
            "hover:from-blue-500 hover:via-indigo-500 hover:to-violet-500",
            "border border-blue-500/50",
            "text-white font-medium",
            "shadow-[0_0_20px_rgba(79,70,229,0.3)] hover:shadow-[0_0_25px_rgba(79,70,229,0.5)]",
            "active:scale-[0.98] hover:scale-[1.02]",
            "rounded-full",
            className,
          )}
          style={{
            borderRadius: borderRadius === "0.5rem" ? "9999px" : borderRadius,
          }}
          onClick={handleClick}
          disabled={disabled || isLoading}
          {...props}
        >
          <AnimatePresence>
            {isLoading && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="absolute inset-0 flex items-center justify-center bg-black/10 backdrop-blur-sm"
              >
                <div className="h-4 w-4 animate-spin rounded-full border-2 border-white border-t-transparent" />
              </motion.div>
            )}
          </AnimatePresence>

          {/* Effet de brillance */}
          <div className="absolute inset-0 overflow-hidden rounded-[inherit]">
            <div
              className="absolute inset-0 h-[100cqh] animate-shimmer"
              style={{
                background: `linear-gradient(90deg, transparent 0%, ${shimmerColor} 50%, transparent 100%)`,
                transform: "translateX(-100%)",
                animation: `shimmer ${shimmerDuration} infinite linear`,
                backgroundSize: "200% 100%",
              }}
            />
          </div>

          {/* Contenu */}
          <span className={cn("relative z-10", isLoading && "invisible")}>
            {children}
          </span>

          {/* Effet de bordure brillante */}
          <div
            className="absolute inset-0 rounded-[inherit] opacity-20 transition-all duration-500 group-hover:opacity-40"
            style={{
              background: `linear-gradient(to bottom right, rgba(255,255,255,0.2), rgba(255,255,255,0.1), rgba(255,255,255,0))`,
            }}
          />

          {/* Effet de hover */}
          <div
            className="absolute inset-0 rounded-[inherit] opacity-0 transition-opacity duration-500 group-hover:opacity-100"
            style={{
              background: "linear-gradient(rgba(255,255,255,0.1), transparent)",
            }}
          />
        </button>

        <AnimatePresence>
          {showVortex && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="fixed inset-0 z-50"
              onClick={() => setShowVortex(false)}
            >
              <div className="absolute inset-0 bg-background/80 backdrop-blur-sm" />
              <div className="absolute inset-0 flex items-center justify-center">
                <div className="relative h-32 w-32">
                  <Vortex
                    particleCount={200}
                    baseHue={220}
                    backgroundColor="transparent"
                    baseSpeed={0.5}
                    rangeSpeed={1}
                    baseRadius={0.5}
                    rangeRadius={1}
                  />
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </>
    );
  },
);

ShimmerButton.displayName = "ShimmerButton";

export { ShimmerButton }; 