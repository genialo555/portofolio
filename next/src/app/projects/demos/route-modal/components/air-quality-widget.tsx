"use client"

import { cn } from "@/lib/utils"

interface AirQualityWidgetProps {
  isOpen: boolean
}

export function AirQualityWidget({ isOpen }: AirQualityWidgetProps) {
  if (!isOpen) return null

  return (
    <div className={cn(
      "absolute top-8 left-8",
      "w-64 bg-card rounded-lg shadow-lg",
      "animate-in slide-in-from-left-8 duration-500",
      "border bg-background/95"
    )}>
      <div className="p-4">
        <div className="flex items-center justify-between mb-3">
          <h3 className="font-semibold">Qualité de l'air</h3>
          <div className="h-2 w-2 rounded-full bg-green-500"></div>
        </div>

        <div className="space-y-3">
          <div className="flex justify-between items-center">
            <span className="text-sm text-muted-foreground">Indice global</span>
            <span className="font-medium">Bon (75/100)</span>
          </div>

          <div className="space-y-2">
            <div className="flex justify-between text-sm">
              <span>PM2.5</span>
              <span className="text-green-500">12 µg/m³</span>
            </div>
            <div className="flex justify-between text-sm">
              <span>NO2</span>
              <span className="text-yellow-500">28 µg/m³</span>
            </div>
            <div className="flex justify-between text-sm">
              <span>O3</span>
              <span className="text-green-500">48 µg/m³</span>
            </div>
          </div>

          <div className="text-xs text-muted-foreground mt-2">
            Dernière mise à jour: il y a 5 min
          </div>
        </div>
      </div>
    </div>
  )
} 