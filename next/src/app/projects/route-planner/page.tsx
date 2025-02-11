"use client"

import { RouteModal } from "../demos/route-modal/components/route-modal"

export default function RoutePlannerPage() {
  return (
    <div className="min-h-screen bg-background">
      <RouteModal isOpen={true} onClose={() => window.history.back()} />
    </div>
  )
}
