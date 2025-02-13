"use client"

import { useState, useEffect } from "react"
import { RouteModal } from "./components/route-modal"
import { RouteModalMobile } from "./components/route-modal.mobile"
import { LoadingState } from "./components/loading-state"
import { AnimatePresence } from "framer-motion"
import { useRouter } from "next/navigation"

export default function RouteModalDemo() {
  const [isLoading, setIsLoading] = useState(true)
  const [showModal, setShowModal] = useState(false)
  const [isMobile, setIsMobile] = useState(false)
  const router = useRouter()

  useEffect(() => {
    // Démarrer avec le loading state
    setIsLoading(true)
    setShowModal(false)

    // Détecter si on est sur mobile
    const checkMobile = () => {
      setIsMobile(window.innerWidth < 768)
    }
    
    checkMobile()
    window.addEventListener('resize', checkMobile)
    
    return () => window.removeEventListener('resize', checkMobile)
  }, [])

  const handleLoadingComplete = () => {
    setIsLoading(false)
    setShowModal(true)
  }

  const handleClose = () => {
    router.push("/projects")
  }

  return (
    <div className="min-h-screen w-full bg-background">
      <AnimatePresence>
        {isLoading && (
          <LoadingState 
            onFinish={handleLoadingComplete} 
            isMobile={isMobile}
          />
        )}
      </AnimatePresence>

      <AnimatePresence>
        {showModal && !isLoading && (
          isMobile ? (
            <RouteModalMobile 
              isOpen={true} 
              onClose={handleClose} 
            />
          ) : (
            <RouteModal 
              isOpen={true} 
              onClose={handleClose} 
            />
          )
        )}
      </AnimatePresence>
    </div>
  )
}