"use client"

import { useRouter } from "next/navigation"
import { useEffect } from "react"

export default function RoutePlannerPage() {
  const router = useRouter()

  useEffect(() => {
    // Rediriger vers la page du modal avec le loading state
    router.replace("/projects/demos/route-modal")
  }, [router])

  return null
}
