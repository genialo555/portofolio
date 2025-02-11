"use client"

import { useState } from "react"
import { cn } from "@/lib/utils"
import Link from "next/link"

interface MenuProps {
  onOpenChange?: (isOpen: boolean) => void
}

export function Menu({ onOpenChange }: MenuProps) {
  const [isOpen, setIsOpen] = useState(false)

  const toggleMenu = (open: boolean) => {
    setIsOpen(open)
    onOpenChange?.(open)
  }

  return (
    <>
      {/* Menu Button */}
      <button
        onClick={() => toggleMenu(!isOpen)}
        className="fixed top-8 right-8 z-[9999] w-12 h-12 flex items-center justify-center group"
        aria-label="Toggle menu"
      >
        <div className="relative w-8 h-8">
          <span
            className={cn(
              "absolute h-0.5 bg-foreground transition-all duration-300",
              "left-0 right-0 group-hover:bg-primary",
              isOpen ? "top-1/2 -translate-y-1/2 rotate-45 w-8" : "top-2 w-6"
            )}
          />
          <span
            className={cn(
              "absolute top-1/2 -translate-y-1/2 h-0.5 transition-all duration-300",
              "right-0 group-hover:bg-primary",
              isOpen ? "w-0 opacity-0 bg-transparent" : "w-4 opacity-100 bg-foreground"
            )}
          />
          <span
            className={cn(
              "absolute h-0.5 bg-foreground transition-all duration-300",
              "right-0 group-hover:bg-primary",
              isOpen ? "top-1/2 -translate-y-1/2 -rotate-45 w-8 left-0" : "bottom-2 w-8"
            )}
          />
        </div>
      </button>

      {/* Menu Panel */}
      <div
        className={cn(
          "fixed inset-0 bg-white/10 backdrop-blur-[2px] transition-all duration-500",
          "flex items-center justify-center z-[9990]",
          isOpen ? "opacity-100 pointer-events-auto" : "opacity-0 pointer-events-none"
        )}
      >
        <nav className="relative w-full max-w-4xl px-8 py-12 z-[9991]">
          <div className="flex flex-col gap-6 relative z-[9992]">
            <Link 
              href="/"
              className="group flex items-baseline gap-4 text-black hover:text-primary transition-all duration-200 hover:scale-105"
              onClick={() => toggleMenu(false)}
            >
              <span className="text-sm font-mono opacity-50 group-hover:opacity-100">01</span>
              <span className="text-5xl font-light tracking-wide">ACCUEIL</span>
            </Link>
            
            <Link 
              href="/projects"
              className="group flex items-baseline gap-4 text-black hover:text-primary transition-all duration-200 hover:scale-105"
              onClick={() => toggleMenu(false)}
            >
              <span className="text-sm font-mono opacity-50 group-hover:opacity-100">02</span>
              <span className="text-5xl font-light tracking-wide">PROJETS</span>
            </Link>
            
            <Link 
              href="/about"
              className="group flex items-baseline gap-4 text-black hover:text-primary transition-all duration-200 hover:scale-105"
              onClick={() => toggleMenu(false)}
            >
              <span className="text-sm font-mono opacity-50 group-hover:opacity-100">03</span>
              <span className="text-5xl font-light tracking-wide">À PROPOS</span>
            </Link>
            
            <Link 
              href="/contact"
              className="group flex items-baseline gap-4 text-black hover:text-primary transition-all duration-200 hover:scale-105"
              onClick={() => toggleMenu(false)}
            >
              <span className="text-sm font-mono opacity-50 group-hover:opacity-100">04</span>
              <span className="text-5xl font-light tracking-wide">CONTACT</span>
            </Link>
          </div>

          <div className="mt-12 flex justify-center gap-8 relative z-[9992]">
            <a 
              href="https://instagram.com" 
              target="_blank" 
              rel="noopener noreferrer"
              className="text-black/80 hover:text-primary transition-all duration-200 hover:scale-105"
              onClick={(e) => e.stopPropagation()}
            >
              ↗ Instagram
            </a>
            <a 
              href="https://behance.com" 
              target="_blank" 
              rel="noopener noreferrer"
              className="text-black/80 hover:text-primary transition-all duration-200 hover:scale-105"
              onClick={(e) => e.stopPropagation()}
            >
              ↗ Behance
            </a>
          </div>
        </nav>
      </div>
    </>
  )
}