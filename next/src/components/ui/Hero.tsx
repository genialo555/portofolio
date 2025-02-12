"use client";

import { useState } from "react";
import { cn } from "@/lib/utils";
import Link from "next/link";
import dynamic from "next/dynamic";
import { Dispatch, SetStateAction } from "react";

interface MenuProps {
  onOpenChange: Dispatch<SetStateAction<boolean>>;
  isOpen: boolean;
}

const Menu = dynamic(() => import("./Menu").then((mod) => mod.Menu), {
  ssr: false,
  loading: () => (
    <div className="w-[65px] h-[25px] bg-transparent border border-black rounded-md" />
  ),
});

const WavesBackground = dynamic(
  () => import("./waves-background").then((mod) => mod.WavesBackground),
  {
    ssr: false,
  }
);

const SplashCursor = dynamic(
  () => import("./SplashCursor").then((mod) => mod.SplashCursor),
  {
    ssr: false,
  }
);

// Separate the main content for better performance
const MainContent = ({ isMenuOpen }: { isMenuOpen: boolean }) => {
  return (
    <div
      className={cn(
        "relative z-[2000] text-center max-w-3xl px-4 space-y-8 transition-all duration-500",
        isMenuOpen ? "opacity-0" : "opacity-100"
      )}
    >
      <div className="space-y-4">
        <h1 className="text-4xl md:text-6xl font-bold tracking-tight">
          HEY, JE SUIS <span className="text-primary">JEREMIE</span>
        </h1>
        <h2 className="text-2xl md:text-4xl font-light tracking-wide text-muted-foreground">
          MAIS VOUS POUVEZ M'APPELER <span className="font-medium">JEREM</span>
        </h2>
        <p className="text-lg md:text-xl text-muted-foreground mt-4">
          Je suis développeur Full Stack, passionné par le<br />
          développement web & machine learning
        </p>
      </div>

      <div className="flex items-center justify-center gap-16 pt-4">
        <Link
          href="/projects"
          className="text-lg text-muted-foreground hover:text-primary transition-colors duration-300"
        >
          → voir mes projets
        </Link>
        <Link
          href="/about"
          className="text-lg text-muted-foreground hover:text-primary transition-colors duration-300"
        >
          → en savoir plus
        </Link>
      </div>
    </div>
  );
};

export function Hero() {
  const [isMenuOpen, setIsMenuOpen] = useState(false);

  return (
    <div className="relative w-full h-screen flex items-center justify-center overflow-hidden">
      {/* Menu avec son bouton */}
      <Menu onOpenChange={setIsMenuOpen} isOpen={isMenuOpen} />

      {/* Container for both animations */}
      <div className="absolute inset-0 w-full h-full">
        {/* Left Side - Waves Animation */}
        <WavesBackground isMenuOpen={isMenuOpen} />

        {/* Right Side - Splash Cursor Animation */}
        {isMenuOpen && (
          <div
            className={cn(
              "absolute top-0 right-0 h-full overflow-hidden transition-all duration-500 ease-in-out",
              isMenuOpen ? "w-1/2 opacity-100" : "w-0 opacity-0"
            )}
          >
            <SplashCursor />
          </div>
        )}
      </div>

      {/* Main Content */}
      <MainContent isMenuOpen={isMenuOpen} />
    </div>
  );
}