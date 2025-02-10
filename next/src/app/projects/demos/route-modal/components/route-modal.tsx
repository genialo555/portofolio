"use client"

import { useState } from "react"
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter,
} from "@/components/ui/dialog"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { RouteOptionsModal } from "./route-options-modal"
import { RouteMapModal } from "./route-map-modal"
import { RouteCarbonModal } from "./route-carbon-modal"
import { AirQualityWidget } from "./air-quality-widget"
import { NotificationsWidget } from "./notifications-widget"
import { cn } from "@/lib/utils"

interface RouteModalProps {
  isOpen: boolean
  onClose: () => void
}

type TransportMode = "drive" | "transit" | "bike" | "walk"

export function RouteModal({ isOpen, onClose }: RouteModalProps) {
  const [origin, setOrigin] = useState("")
  const [destination, setDestination] = useState("")
  const [mode, setMode] = useState<TransportMode>("drive")

  if (!isOpen) return null

  return (
    <div className="fixed inset-0 z-50 bg-background/80 backdrop-blur-sm animate-in fade-in-0 duration-300">
      {/* Widgets flottants */}
      <div className="relative z-[60]">
        <AirQualityWidget isOpen={true} />
        <NotificationsWidget isOpen={true} />
      </div>

      <div className="container max-w-7xl h-full flex items-center justify-center">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-12 p-4">
          {/* Left Column - Input Form */}
          <div className="bg-card rounded-lg shadow-lg animate-in slide-in-from-left-8 duration-500 fill-mode-forwards">
            <div className="p-6 space-y-6">
              <div>
                <h2 className="text-lg font-semibold mb-2">Planifier un itinéraire</h2>
                <p className="text-sm text-muted-foreground">
                  Entrez votre point de départ et votre destination pour obtenir un itinéraire écologique.
                </p>
              </div>

              {/* Transport Mode Selection */}
              <div className="grid grid-cols-4 gap-2">
                <Button
                  variant={mode === "drive" ? "default" : "outline"}
                  size="sm"
                  onClick={() => setMode("drive")}
                  className="flex flex-col items-center py-4 h-auto"
                >
                  <span className="text-xl mb-1">🚗</span>
                  <span className="text-xs">Voiture</span>
                </Button>
                <Button
                  variant={mode === "transit" ? "default" : "outline"}
                  size="sm"
                  onClick={() => setMode("transit")}
                  className="flex flex-col items-center py-4 h-auto"
                >
                  <span className="text-xl mb-1">🚆</span>
                  <span className="text-xs">Transport</span>
                </Button>
                <Button
                  variant={mode === "bike" ? "default" : "outline"}
                  size="sm"
                  onClick={() => setMode("bike")}
                  className="flex flex-col items-center py-4 h-auto"
                >
                  <span className="text-xl mb-1">🚲</span>
                  <span className="text-xs">Vélo</span>
                </Button>
                <Button
                  variant={mode === "walk" ? "default" : "outline"}
                  size="sm"
                  onClick={() => setMode("walk")}
                  className="flex flex-col items-center py-4 h-auto"
                >
                  <span className="text-xl mb-1">🚶</span>
                  <span className="text-xs">À pied</span>
                </Button>
              </div>

              <div className="grid gap-4">
                <div className="grid gap-2">
                  <Label htmlFor="origin">Point de départ</Label>
                  <Input
                    id="origin"
                    placeholder="Entrez une adresse..."
                    value={origin}
                    onChange={(e) => setOrigin(e.target.value)}
                  />
                </div>

                <div className="grid gap-2">
                  <Label htmlFor="destination">Destination</Label>
                  <Input
                    id="destination"
                    placeholder="Entrez une adresse..."
                    value={destination}
                    onChange={(e) => setDestination(e.target.value)}
                  />
                </div>

                <Button 
                  className="w-full"
                  disabled={!origin || !destination}
                >
                  Rechercher
                </Button>
              </div>
            </div>
          </div>

          {/* Map Card */}
          <div className={cn(
            "bg-card rounded-lg shadow-lg",
            "animate-in slide-in-from-bottom-8 duration-500 fill-mode-forwards",
            "delay-300"
          )}>
            <RouteMapModal
              isOpen={true}
              onClose={() => {}}
              origin={origin}
              destination={destination}
              mode={mode}
            />
          </div>

          {/* Carbon Card */}
          <div className={cn(
            "bg-card rounded-lg shadow-lg",
            "animate-in slide-in-from-right-8 duration-500 fill-mode-forwards",
            "delay-500"
          )}>
            <RouteCarbonModal
              isOpen={true}
              onClose={() => {}}
              origin={origin}
              destination={destination}
              mode={mode === "transit" ? "transit" : mode === "drive" ? "car" : undefined}
            />
          </div>
        </div>

        {/* Close button */}
        <div className={cn(
          "flex justify-center mt-4",
          "animate-in fade-in-0 duration-300",
          "delay-700"
        )}>
          <Button 
            variant="outline" 
            onClick={onClose}
            className="bg-background"
          >
            Fermer
          </Button>
        </div>
      </div>
    </div>
  )
} 