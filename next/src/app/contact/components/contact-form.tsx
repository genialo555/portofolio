"use client"

import { motion } from "framer-motion"
import { Button } from "@/components/ui/button"

const badges = [
  { label: "Branding", delay: 0 },
  { label: "Graphic Design", delay: 0.1 },
  { label: "Web Application", delay: 0.2 },
  { label: "UI-UX", delay: 0.3 }
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
          Any questions about Design?
        </motion.h1>
        
        <motion.p 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.5, duration: 0.5 }}
          className="text-lg text-muted-foreground"
        >
          Feel free to reach out to me!
        </motion.p>

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
            Book a call
          </Button>
          
          <Button 
            size="lg" 
            variant="outline" 
            className="border-2 hover:scale-105 transition-all duration-300"
            onClick={() => window.location.href = 'mailto:contact@example.com'}
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
            onClick={() => window.open('https://wa.me/1234567890', '_blank')}
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