"use client"

import { useState, useCallback, useEffect, Dispatch, SetStateAction } from "react"
import { cn } from "@/lib/utils"
import Link from "next/link"

interface MenuProps {
  onOpenChange: Dispatch<SetStateAction<boolean>>;
  isOpen: boolean;
}

export const Menu = ({ onOpenChange, isOpen }: MenuProps) => {
  const [isAnimating, setIsAnimating] = useState(false)
  const [isPreloaded, setIsPreloaded] = useState(false)

  // Précharger le menu
  useEffect(() => {
    setIsPreloaded(true)
  }, [])

  // Utilisation de useCallback pour mémoriser la fonction
  const toggleMenu = useCallback((open: boolean) => {
    if (isAnimating) return
    setIsAnimating(true)
    onOpenChange(open)
  }, [onOpenChange, isAnimating])

  // Gérer la fin de l'animation
  useEffect(() => {
    if (isAnimating) {
      const timer = setTimeout(() => {
        setIsAnimating(false)
      }, 200)
      return () => clearTimeout(timer)
    }
  }, [isAnimating])

  // Mémoriser les liens pour éviter les re-renders inutiles
  const menuLinks = [
    { href: "/", label: "ACCUEIL", number: "01" },
    { href: "/projects", label: "PROJETS", number: "02" },
    { href: "/about", label: "À PROPOS", number: "03" },
    { href: "/contact", label: "CONTACT", number: "04" }
  ]

  const socialLinks = [
    { href: "https://instagram.com", label: "Instagram" },
    { href: "https://behance.com", label: "Behance" }
  ]

  if (!isPreloaded) return null

  return (
    <>
      {/* Domino Button - Optimisé avec transform3d pour de meilleures performances */}
      <button
        onClick={() => toggleMenu(!isOpen)}
        className={cn(
          "fixed top-8 right-8 z-[9999] group",
          "transform-gpu"
        )}
        style={{ perspective: "1000px" }}
        aria-label="Toggle menu"
        disabled={isAnimating}
      >
        <div 
          className="relative w-[65px] h-[35px] rounded-md overflow-hidden transform-gpu"
          style={{ transformStyle: "preserve-3d" }}
        >
          {/* Face avant du domino */}
          <div 
            className={cn(
              "absolute inset-0 border border-white/10 rounded-md",
              "transition-transform duration-200 ease-out will-change-transform",
              "bg-transparent transform-gpu",
              isOpen ? "[transform:rotateY(90deg)]" : "[transform:rotateY(0deg)]"
            )}
            style={{ 
              backfaceVisibility: "hidden", 
              transformOrigin: "right",
              WebkitBackfaceVisibility: "hidden"
            }}
          >
            <div className="absolute inset-0 flex items-center justify-center">
              <div className="grid grid-cols-3 gap-x-2.5 gap-y-1.5">
                {[...Array(9)].map((_, i) => (
                  <div 
                    key={i}
                    className={cn(
                      "w-1.5 h-1.5 rounded-full bg-transparent ring-1 ring-black",
                      i % 2 === 1 && "invisible"
                    )}
                  />
                ))}
              </div>
            </div>
          </div>

          {/* Face arrière du domino */}
          <div 
            className={cn(
              "absolute inset-0 border border-white/10 rounded-md",
              "transition-transform duration-200 ease-out will-change-transform",
              "bg-transparent transform-gpu",
              isOpen ? "[transform:rotateY(0deg)]" : "[transform:rotateY(-90deg)]"
            )}
            style={{ 
              backfaceVisibility: "hidden", 
              transformOrigin: "left",
              WebkitBackfaceVisibility: "hidden"
            }}
          >
            <div className="absolute inset-0 flex items-center justify-center">
              <div className="grid grid-cols-3 gap-x-2.5 gap-y-1.5">
                {[...Array(9)].map((_, i) => (
                  <div 
                    key={i}
                    className={cn(
                      "w-1.5 h-1.5 rounded-full bg-transparent ring-1 ring-black",
                      i % 2 === 1 && "invisible"
                    )}
                  />
                ))}
              </div>
            </div>
          </div>

          {/* Effet de hover optimisé */}
          <div 
            className={cn(
              "absolute inset-0 rounded-md opacity-0 transition-opacity duration-200",
              "bg-gradient-to-r from-primary/20 to-primary/10",
              "group-hover:opacity-100"
            )} 
          />
        </div>
      </button>

      {/* Menu Panel - Optimisé avec will-change et hardware acceleration */}
      <div
        className={cn(
          "fixed inset-0 bg-white/10 backdrop-blur-[2px]",
          "transition-all duration-200 ease-out will-change-transform",
          "flex items-center justify-center z-[9990] transform-gpu",
          isOpen ? "opacity-100 visible" : "opacity-0 invisible"
        )}
        style={{ 
          transform: "translateZ(0)",
          WebkitBackfaceVisibility: "hidden"
        }}
        onClick={() => toggleMenu(false)}
      >
        <nav 
          className="relative w-full max-w-4xl px-8 py-12 z-[9991] transform-gpu"
          onClick={e => e.stopPropagation()}
        >
          <div className="flex flex-col gap-6 relative z-[9992]">
            {menuLinks.map(({ href, label, number }) => (
              <Link 
                key={href}
                href={href}
                className="group flex items-baseline gap-4 text-black hover:text-primary transition-all duration-200 hover:scale-105 transform-gpu"
                onClick={() => toggleMenu(false)}
              >
                <span className="text-sm font-mono opacity-50 group-hover:opacity-100">{number}</span>
                <span className="text-5xl font-light tracking-wide">{label}</span>
              </Link>
            ))}
          </div>

          <div className="mt-12 flex justify-center gap-8 relative z-[9992]">
            {socialLinks.map(({ href, label }) => (
              <a 
                key={href}
                href={href} 
                target="_blank" 
                rel="noopener noreferrer"
                className="text-black/80 hover:text-primary transition-all duration-200 hover:scale-105 transform-gpu"
                onClick={(e) => e.stopPropagation()}
              >
                ↗ {label}
              </a>
            ))}
          </div>
        </nav>
      </div>
    </>
  )
}