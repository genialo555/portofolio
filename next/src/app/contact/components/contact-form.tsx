"use client"

import { motion } from "framer-motion"
import { Button } from "@/components/ui/button"

const badges = [
  { label: "Design Web", delay: 0 },
  { label: "Full Stack", delay: 0.1 },
  { label: "Machine Learning", delay: 0.2 },
  { label: "Automatisation", delay: 0.3 }
]

export function ContactForm() {
  return (
    <div className="w-full max-w-3xl mx-auto">
      <div className="flex flex-col items-center text-center space-y-4">
        <div className="flex gap-2 items-center flex-wrap justify-center">
          {badges.map((badge, i) => (
            <motion.span
              key={badge.label}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: badge.delay, duration: 0.5 }}
              className="px-3 py-1 text-sm rounded-full bg-white/10 backdrop-blur-sm hover:bg-white/20 transition-colors cursor-default"
            >
              {badge.label}
            </motion.span>
          ))}
        </div>

        <motion.h1 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.4, duration: 0.5 }}
          className="text-4xl font-semibold"
        >
          Une idée de projet ?
        </motion.h1>
        
        <motion.p 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.5, duration: 0.5 }}
          className="text-lg text-muted-foreground"
        >
          N'hésitez pas à me contacter !
        </motion.p>

        <motion.a
          href="https://www.linkedin.com/in/jérémie-nunez"
          target="_blank"
          rel="noopener noreferrer"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.55, duration: 0.5 }}
          className="flex items-center gap-2 text-muted-foreground hover:text-primary transition-colors"
          whileHover={{ scale: 1.05 }}
        >
          <svg
            xmlns="http://www.w3.org/2000/svg"
            width="24"
            height="24"
            viewBox="0 0 24 24"
            fill="currentColor"
            className="w-6 h-6"
          >
            <path d="M20.5 2h-17A1.5 1.5 0 002 3.5v17A1.5 1.5 0 003.5 22h17a1.5 1.5 0 001.5-1.5v-17A1.5 1.5 0 0020.5 2zM8 19H5v-9h3zM6.5 8.25A1.75 1.75 0 118.3 6.5a1.78 1.78 0 01-1.8 1.75zM19 19h-3v-4.74c0-1.42-.6-1.93-1.38-1.93A1.74 1.74 0 0013 14.19a.66.66 0 000 .14V19h-3v-9h2.9v1.3a3.11 3.11 0 012.7-1.4c1.55 0 3.36.86 3.36 3.66z"></path>
          </svg>
          <span className="text-sm font-medium">LinkedIn</span>
        </motion.a>

        <motion.div 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.6, duration: 0.5 }}
          className="flex gap-4 mt-4"
        >
          <Button 
            size="lg" 
            className="bg-black text-white hover:bg-black/90 hover:scale-105 transition-all duration-300"
            onClick={() => window.open('https://calendly.com', '_blank')}
          >
            Planifier un appel
          </Button>
          
          <Button 
            size="lg" 
            variant="outline" 
            className="border-2 hover:scale-105 transition-all duration-300"
            onClick={() => window.location.href = 'mailto:jeremienunezpro@gmail.com'}
          >
            <motion.svg 
              xmlns="http://www.w3.org/2000/svg" 
              width="20" 
              height="20" 
              viewBox="0 0 24 24" 
              fill="none" 
              stroke="currentColor" 
              strokeWidth="2" 
              strokeLinecap="round" 
              strokeLinejoin="round"
              whileHover={{ scale: 1.1 }}
              whileTap={{ scale: 0.9 }}
            >
              <rect width="20" height="16" x="2" y="4" rx="2"/>
              <path d="m22 7-8.97 5.7a1.94 1.94 0 0 1-2.06 0L2 7"/>
            </motion.svg>
          </Button>

          <Button 
            size="lg" 
            variant="outline" 
            className="border-2 hover:scale-105 transition-all duration-300"
            onClick={() => window.open('https://wa.me/33670303478', '_blank')}
          >
            <motion.svg 
              xmlns="http://www.w3.org/2000/svg" 
              width="20" 
              height="20" 
              viewBox="0 0 24 24" 
              fill="none" 
              stroke="currentColor" 
              strokeWidth="2" 
              strokeLinecap="round" 
              strokeLinejoin="round"
              whileHover={{ scale: 1.1 }}
              whileTap={{ scale: 0.9 }}
            >
              <path d="M22 16.92v3a2 2 0 0 1-2.18 2 19.79 19.79 0 0 1-8.63-3.07 19.5 19.5 0 0 1-6-6 19.79 19.79 0 0 1-3.07-8.67A2 2 0 0 1 4.11 2h3a2 2 0 0 1 2 1.72 12.84 12.84 0 0 0 .7 2.81 2 2 0 0 1-.45 2.11L8.09 9.91a16 16 0 0 0 6 6l1.27-1.27a2 2 0 0 1 2.11-.45 12.84 12.84 0 0 0 2.81.7A2 2 0 0 1 22 16.92z"/>
            </motion.svg>
          </Button>
        </motion.div>
      </div>
    </div>
  )
} 