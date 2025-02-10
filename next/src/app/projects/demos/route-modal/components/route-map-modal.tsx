"use client"

interface RouteMapModalProps {
  isOpen: boolean
  onClose: () => void
  origin: string
  destination: string
  mode?: "drive" | "transit" | "bike" | "walk"
}

// Données mockées pour les étapes
const MOCK_STEPS = [
  {
    instruction: "Prenez l'autoroute A1 en direction de Paris",
    distance: "120 km",
    duration: "1h10",
    details: "Trafic fluide"
  },
  {
    instruction: "Continuez sur le périphérique extérieur",
    distance: "12 km",
    duration: "15 min",
    details: "Trafic modéré"
  },
  {
    instruction: "Sortie 4 : Direction La Défense",
    distance: "5 km",
    duration: "8 min",
    details: "Trafic dense"
  },
  {
    instruction: "Arrivée à destination",
    distance: "0.5 km",
    duration: "2 min",
    details: "Zone urbaine"
  }
]

const getModeIcon = (mode: RouteMapModalProps["mode"]) => {
  switch (mode) {
    case "drive":
      return "🚗"
    case "transit":
      return "🚆"
    case "bike":
      return "🚲"
    case "walk":
      return "🚶"
    default:
      return "🚗"
  }
}

export function RouteMapModal({ isOpen, onClose, origin, destination, mode = "drive" }: RouteMapModalProps) {
  if (!isOpen) return null

  return (
    <div className="p-6">
      <div className="mb-4">
        <h2 className="text-lg font-semibold mb-2">
          <span className="mr-2">{getModeIcon(mode)}</span>
          Carte et étapes
        </h2>
      </div>

      <div className="grid gap-6">
        {/* Carte placeholder */}
        <div className="relative aspect-[16/9] w-full overflow-hidden rounded-xl border bg-muted">
          <div className="absolute inset-0 flex items-center justify-center text-muted-foreground">
            Carte Google Maps
          </div>
        </div>

        {/* Liste des étapes */}
        <div className="space-y-4">
          <h3 className="font-medium">Étapes de l&apos;itinéraire</h3>
          <div className="space-y-2 max-h-[300px] overflow-y-auto pr-2">
            {MOCK_STEPS.map((step, index) => (
              <div
                key={index}
                className="flex items-start gap-4 rounded-lg border bg-card p-4"
              >
                <div className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-primary text-primary-foreground">
                  {index + 1}
                </div>
                <div className="grid gap-1">
                  <div className="font-medium">{step.instruction}</div>
                  <div className="text-sm text-muted-foreground">
                    {step.distance} • {step.duration}
                  </div>
                  <div className="text-sm text-muted-foreground">
                    {step.details}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
} 