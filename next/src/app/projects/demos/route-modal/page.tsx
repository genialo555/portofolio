"use client"

import { useState, useEffect } from "react"
import { RouteModal } from "./components/route-modal"
import { LoadingState } from "./components/loading-state"
import { AnimatePresence } from "framer-motion"
import { useRouter } from "next/navigation"

export default function RouteModalDemo() {
  const [isLoading, setIsLoading] = useState(true)
  const [showModal, setShowModal] = useState(false)
  const router = useRouter()

  useEffect(() => {
    // Démarrer avec le loading state
    setIsLoading(true)
    setShowModal(false)
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
          <LoadingState onFinish={handleLoadingComplete} />
        )}
      </AnimatePresence>

      <AnimatePresence>
        {showModal && !isLoading && (
          <RouteModal 
            isOpen={true} 
            onClose={handleClose} 
          />
        )}
      </AnimatePresence>
    </div>
  )
}