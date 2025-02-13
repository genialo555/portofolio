"use client"

import { useRouter } from "next/navigation"
import { useEffect } from "react"

export default function ChatbotPage() {
  const router = useRouter()

  useEffect(() => {
    // Rediriger vers la nouvelle page de chat
    router.replace("/chat")
  }, [router])

  return null
} 