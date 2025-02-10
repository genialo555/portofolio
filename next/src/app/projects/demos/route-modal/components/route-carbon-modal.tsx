"use client"

import { Badge } from "@/components/ui/badge"

interface RouteCarbonModalProps {
  isOpen: boolean
  onClose: () => void
  origin: string
  destination: string
  distance?: number
  mode?: "car" | "transit" | "plane"
}

// Constantes pour le calcul CO2 (g/km)
const CO2_FACTORS = {
  car: 220,    // Moyenne pour une voiture essence
  transit: 35, // Train moyenne distance
  plane: 285   // Vol court-courrier
}

export function RouteCarbonModal({ 
  isOpen, 
  onClose, 
  origin, 
  destination,
  distance = 180, // Distance par défaut en km
  mode = "car"    // Mode par défaut
}: RouteCarbonModalProps) {
  if (!isOpen) return null

  // Calcul des émissions
  const emissions = {
    selected: Math.round(distance * CO2_FACTORS[mode] / 1000), // en kg
    alternatives: {
      car: Math.round(distance * CO2_FACTORS.car / 1000),
      transit: Math.round(distance * CO2_FACTORS.transit / 1000),
      plane: Math.round(distance * CO2_FACTORS.plane / 1000)
    }
  }

  // Équivalences concrètes
  const equivalents = {
    trees: Math.round(emissions.selected / 25), // Un arbre absorbe environ 25kg de CO2 par an
    meals: Math.round(emissions.selected / 7),  // Un repas carné moyen = 7kg de CO2
    phone: Math.round(emissions.selected * 2)   // 1kg de CO2 = 2 charges de smartphone
  }

  return (
    <div className="p-6">
      <div className="mb-4">
        <h2 className="text-lg font-semibold mb-2">Impact environnemental</h2>
      </div>

      <div className="grid gap-6">
        {/* Émissions du trajet */}
        <div className="space-y-4">
          <div className="flex items-baseline justify-between">
            <h3 className="text-lg font-medium">Émissions CO₂</h3>
            <div className="flex items-center gap-2">
              <span className="text-2xl font-bold">{emissions.selected}</span>
              <span className="text-muted-foreground">kg</span>
            </div>
          </div>

          {/* Comparaison des modes */}
          <div className="grid gap-2">
            <div className="flex items-center justify-between rounded-lg border p-3">
              <div className="flex items-center gap-2">
                🚗 Voiture
                {mode === "car" && <Badge variant="secondary">Sélectionné</Badge>}
              </div>
              <span>{emissions.alternatives.car} kg</span>
            </div>
            <div className="flex items-center justify-between rounded-lg border p-3">
              <div className="flex items-center gap-2">
                🚆 Train
                {mode === "transit" && <Badge variant="secondary">Sélectionné</Badge>}
              </div>
              <span>{emissions.alternatives.transit} kg</span>
            </div>
            <div className="flex items-center justify-between rounded-lg border p-3">
              <div className="flex items-center gap-2">
                ✈️ Avion
                {mode === "plane" && <Badge variant="secondary">Sélectionné</Badge>}
              </div>
              <span>{emissions.alternatives.plane} kg</span>
            </div>
          </div>
        </div>

        {/* Équivalences */}
        <div className="space-y-4">
          <h3 className="font-medium">En perspective</h3>
          <div className="grid gap-3">
            <div className="flex items-center justify-between rounded-lg border p-3">
              <span>🌳 Absorption par des arbres</span>
              <span>{equivalents.trees} arbres/an</span>
            </div>
            <div className="flex items-center justify-between rounded-lg border p-3">
              <span>🍖 Équivalent en repas</span>
              <span>{equivalents.meals} repas</span>
            </div>
            <div className="flex items-center justify-between rounded-lg border p-3">
              <span>📱 Charges de smartphone</span>
              <span>{equivalents.phone} charges</span>
            </div>
          </div>
        </div>

        {/* Conseil */}
        <div className="rounded-lg bg-muted p-4 text-sm">
          <p>
            💡 <strong>Conseil :</strong> Privilégiez le train quand c&apos;est possible.
            Il émet en moyenne 6 fois moins de CO₂ que la voiture et 8 fois moins que l&apos;avion.
          </p>
        </div>
      </div>
    </div>
  )
} 