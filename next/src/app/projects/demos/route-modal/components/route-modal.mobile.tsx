import { useState } from "react"
import { motion } from "framer-motion"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { NotificationsWidgetMobile } from "./notifications-widget.mobile"
import { AirQualityWidgetMobile } from "./air-quality-widget.mobile"
import { RouteMapModal } from "./route-map-modal"
import { RouteCarbonModal } from "./route-carbon-modal"
import { cn } from "@/lib/utils"

type TransportMode = "drive" | "transit" | "bike" | "walk"

interface RouteModalMobileProps {
  isOpen: boolean
  onClose: () => void
}

export function RouteModalMobile({ isOpen, onClose }: RouteModalMobileProps) {
  const [origin, setOrigin] = useState("")
  const [destination, setDestination] = useState("")
  const [mode, setMode] = useState<TransportMode>("drive")

  if (!isOpen) return null

  const handleClose = (e?: React.MouseEvent) => {
    e?.preventDefault()
    e?.stopPropagation()
    onClose()
  }

  return (
    <motion.div 
      className="fixed inset-0 z-[1500] bg-background/80 backdrop-blur-sm overflow-y-auto"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      transition={{ duration: 0.3 }}
    >
      {/* Bouton de fermeture */}
      <motion.button
        onClick={handleClose}
        className="fixed top-4 right-4 z-[1520] rounded-full p-2 bg-background shadow-lg hover:bg-background/80 text-foreground/60 hover:text-foreground transition-all duration-200"
        whileHover={{ scale: 1.1 }}
        whileTap={{ scale: 0.95 }}
      >
        <svg
          className="h-5 w-5"
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M6 18L18 6M6 6l12 12"
          />
        </svg>
      </motion.button>

      {/* Contenu empilé */}
      <div className="container px-4 py-20 space-y-4">
        {/* Widgets en haut */}
        <NotificationsWidgetMobile isOpen={true} />
        <AirQualityWidgetMobile isOpen={true} />

        {/* Formulaire */}
        <div className="bg-card rounded-lg shadow-lg p-4 space-y-4">
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
              className="flex flex-col items-center py-3 h-auto"
            >
              <span className="text-lg mb-1">🚗</span>
              <span className="text-xs">Voiture</span>
            </Button>
            <Button
              variant={mode === "transit" ? "default" : "outline"}
              size="sm"
              onClick={() => setMode("transit")}
              className="flex flex-col items-center py-3 h-auto"
            >
              <span className="text-lg mb-1">🚆</span>
              <span className="text-xs">Transport</span>
            </Button>
            <Button
              variant={mode === "bike" ? "default" : "outline"}
              size="sm"
              onClick={() => setMode("bike")}
              className="flex flex-col items-center py-3 h-auto"
            >
              <span className="text-lg mb-1">🚲</span>
              <span className="text-xs">Vélo</span>
            </Button>
            <Button
              variant={mode === "walk" ? "default" : "outline"}
              size="sm"
              onClick={() => setMode("walk")}
              className="flex flex-col items-center py-3 h-auto"
            >
              <span className="text-lg mb-1">🚶</span>
              <span className="text-xs">À pied</span>
            </Button>
          </div>

          <div className="space-y-3">
            <div>
              <Label htmlFor="origin">Point de départ</Label>
              <Input
                id="origin"
                placeholder="Entrez une adresse..."
                value={origin}
                onChange={(e) => setOrigin(e.target.value)}
                className="mt-1"
              />
            </div>

            <div>
              <Label htmlFor="destination">Destination</Label>
              <Input
                id="destination"
                placeholder="Entrez une adresse..."
                value={destination}
                onChange={(e) => setDestination(e.target.value)}
                className="mt-1"
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

        {/* Carte */}
        <div className="bg-card rounded-lg shadow-lg overflow-hidden">
          <RouteMapModal
            isOpen={true}
            onClose={() => {}}
            origin={origin}
            destination={destination}
            mode={mode}
          />
        </div>

        {/* Empreinte carbone */}
        <div className="bg-card rounded-lg shadow-lg overflow-hidden">
          <RouteCarbonModal
            isOpen={true}
            onClose={() => {}}
            origin={origin}
            destination={destination}
            mode={mode === "transit" ? "transit" : mode === "drive" ? "car" : undefined}
          />
        </div>
      </div>
    </motion.div>
  )
} 