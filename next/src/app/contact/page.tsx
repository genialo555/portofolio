"use client"

import Link from "next/link"
import { ContactForm } from "./components/contact-form"
import { IconCloud } from "./components/icon-cloud"
import { SplashCursor } from "@/components/ui/SplashCursor"

const iconSlugs = [
  "typescript",
  "javascript",
  "react",
  "nextdotjs",
  "nodedotjs",
  "vuedotjs",
  "tailwindcss",
  "prisma",
  "postgresql",
  "mongodb",
  "docker",
  "git",
  "github",
  "vercel",
  "figma",
]

export default function ContactPage() {
  return (
    <div className="relative min-h-screen bg-background">
      {/* Back Arrow */}
      <Link 
        href="/"
        className="fixed top-8 left-8 z-[9999] p-3 rounded-full bg-white/90 shadow-lg hover:bg-white hover:scale-110 transition-all duration-200 group"
      >
        <svg 
          xmlns="http://www.w3.org/2000/svg" 
          width="24" 
          height="24" 
          viewBox="0 0 24 24" 
          fill="none" 
          stroke="currentColor" 
          strokeWidth="2" 
          strokeLinecap="round" 
          strokeLinejoin="round" 
          className="transform group-hover:-translate-x-1 transition-transform duration-200"
        >
          <path d="M19 12H5M12 19l-7-7 7-7"/>
        </svg>
      </Link>

      {/* Splash Background */}
      <div className="absolute inset-0 z-0">
        <SplashCursor
          SIM_RESOLUTION={128}
          DYE_RESOLUTION={1024}
          DENSITY_DISSIPATION={3}
          VELOCITY_DISSIPATION={1.5}
          PRESSURE={0.6}
          PRESSURE_ITERATIONS={20}
          CURL={20}
          SPLAT_RADIUS={0.3}
          SPLAT_FORCE={6000}
          SHADING={true}
          COLOR_UPDATE_SPEED={10}
          BACK_COLOR={{ r: 0, g: 0, b: 0 }}
          TRANSPARENT={true}
        />
      </div>

      <div className="container relative z-10 h-screen grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Contact Form */}
        <div className="flex items-center justify-center">
          <ContactForm />
        </div>

        {/* Icon Cloud */}
        <div className="hidden lg:flex items-center justify-center">
          <IconCloud iconSlugs={iconSlugs} />
        </div>
      </div>
    </div>
  )
} 