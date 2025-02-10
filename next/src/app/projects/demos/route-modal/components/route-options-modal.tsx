"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"

interface RouteOptionsModalProps {
  isOpen: boolean
  onClose: () => void
  origin: string
  destination: string
}

interface CarRoute {
  duration: string
  distance: string
  type: "fastest" | "economic"
  mainRoads: string[]
  traffic: "low" | "medium" | "high"
}

interface TransitRoute {
  duration: string
  type: "train" | "bus"
  changes: number
  price: string
  departure: string
  arrival: string
}

interface PlaneRoute {
  duration: string
  type: "direct" | "connection"
  price: string
  departure: string
  arrival: string
  airline: string
}

type RouteType = {
  car: CarRoute[]
  transit: TransitRoute[]
  plane: PlaneRoute[]
}

// Données mockées pour les itinéraires
const MOCK_ROUTES: RouteType = {
  car: [
    {
      duration: "2h15",
      distance: "180km",
      type: "fastest",
      mainRoads: ["A1", "A86"],
      traffic: "medium",
    },
    {
      duration: "2h45",
      distance: "165km",
      type: "economic",
      mainRoads: ["N104", "D445"],
      traffic: "low",
    }
  ],
  transit: [
    {
      duration: "2h30",
      type: "train",
      changes: 1,
      price: "45€",
      departure: "10:15",
      arrival: "12:45"
    },
    {
      duration: "3h00",
      type: "bus",
      changes: 2,
      price: "22€",
      departure: "10:00",
      arrival: "13:00"
    }
  ],
  plane: [
    {
      duration: "1h15",
      type: "direct",
      price: "120€",
      departure: "10:30",
      arrival: "11:45",
      airline: "Air France"
    }
  ]
}

export function RouteOptionsModal({ isOpen, onClose, origin, destination }: RouteOptionsModalProps) {
  const [selectedMode, setSelectedMode] = useState<"car" | "transit" | "plane">("car")

  if (!isOpen) return null

  const renderRouteDetails = (route: CarRoute | TransitRoute | PlaneRoute) => {
    if ('mainRoads' in route) {
      // This is a CarRoute
      return (
        <>
          <div>Principaux axes : {route.mainRoads.join(", ")}</div>
          <div>Trafic : {route.traffic === "low" ? "Fluide" : "Moyen"}</div>
        </>
      )
    } else if ('changes' in route) {
      // This is a TransitRoute
      return (
        <>
          <div>Type : {route.type === "train" ? "Train" : "Bus"}</div>
          <div>Départ : {route.departure} - Arrivée : {route.arrival}</div>
          <div>Changements : {route.changes}</div>
        </>
      )
    } else {
      // This is a PlaneRoute
      return (
        <>
          <div>Compagnie : {route.airline}</div>
          <div>Départ : {route.departure} - Arrivée : {route.arrival}</div>
          <div>Vol {route.type}</div>
        </>
      )
    }
  }

  return (
    <div className="p-6">
      <div className="mb-4">
        <h2 className="text-lg font-semibold mb-2">Options d&apos;itinéraires</h2>
      </div>

      <div className="grid gap-6">
        {/* Sélecteur de mode */}
        <div className="flex gap-2">
          <Button
            variant={selectedMode === "car" ? "default" : "outline"}
            onClick={() => setSelectedMode("car")}
            className="flex-1"
          >
            🚗 Voiture
          </Button>
          <Button
            variant={selectedMode === "transit" ? "default" : "outline"}
            onClick={() => setSelectedMode("transit")}
            className="flex-1"
          >
            🚆 Train/Bus
          </Button>
          <Button
            variant={selectedMode === "plane" ? "default" : "outline"}
            onClick={() => setSelectedMode("plane")}
            className="flex-1"
          >
            ✈️ Avion
          </Button>
        </div>

        {/* Liste des itinéraires */}
        <div className="space-y-4">
          {MOCK_ROUTES[selectedMode].map((route, index) => (
            <div
              key={index}
              className="rounded-lg border bg-card p-4 hover:border-primary transition-colors cursor-pointer"
            >
              <div className="flex items-start justify-between">
                <div className="space-y-1">
                  <div className="font-medium">{route.duration}</div>
                  {'distance' in route && (
                    <div className="text-sm text-muted-foreground">
                      {route.distance}
                    </div>
                  )}
                </div>
                <Badge variant="secondary">
                  {'price' in route ? route.price : (route.type === "fastest" ? "Rapide" : "Économique")}
                </Badge>
              </div>

              <div className="mt-4 text-sm text-muted-foreground">
                {renderRouteDetails(route)}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
} 