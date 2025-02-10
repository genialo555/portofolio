"use client"

import { useState } from "react"
import { cn } from "@/lib/utils"
import Link from "next/link"
import { Menu } from "./Menu"
import { SplashCursor } from "./SplashCursor"
import { WavesBackground } from "./waves-background"

export function Hero() {
  const [isMenuOpen, setIsMenuOpen] = useState(false)

  return (
    <div className="relative w-full h-screen flex items-center justify-center overflow-hidden">
      {/* Menu */}
      <Menu onOpenChange={setIsMenuOpen} />
      
      {/* Container for both animations */}
      <div className="absolute inset-0 w-full h-full">
        {/* Left Side - Waves Animation */}
        <WavesBackground isMenuOpen={isMenuOpen} />

        {/* Right Side - Splash Cursor Animation */}
        <div className={cn(
          "absolute top-0 right-0 h-full overflow-hidden transition-all duration-500 ease-in-out",
          isMenuOpen ? "w-1/2 opacity-100" : "w-0 opacity-0"
        )}>
          <SplashCursor />
        </div>
      </div>
      
      {/* Main Content */}
      <div className={cn(
        "relative z-[2000] text-center max-w-3xl px-4 space-y-8 transition-all duration-500",
        isMenuOpen ? "opacity-0" : "opacity-100"
      )}>
        <div className="space-y-4">
          <h1 className="text-4xl font-bold tracking-tight">
            Hey, je suis <span className="text-primary">Jeremie Nunez</span>
          </h1>
          <p className="text-xl text-muted-foreground">
            Je suis développeur Full Stack, passionné par le développement web & mobile
          </p>
        </div>

        <div className="flex items-center justify-center gap-16 pt-4">
          <Link 
            href="/projects" 
            className="text-lg text-muted-foreground hover:text-primary transition-colors duration-300"
            onClick={(e) => {
              if (isMenuOpen) {
                setIsMenuOpen(false);
              }
            }}
          >
            → voir mes projets
          </Link>
          <Link 
            href="/about" 
            className="text-lg text-muted-foreground hover:text-primary transition-colors duration-300"
            onClick={(e) => {
              if (isMenuOpen) {
                setIsMenuOpen(false);
              }
            }}
          >
            → en savoir plus
          </Link>
        </div>
      </div>
    </div>
  )
} 