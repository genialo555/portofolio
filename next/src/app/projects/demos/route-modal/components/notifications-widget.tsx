"use client"

import { cn } from "@/lib/utils"

interface NotificationsWidgetProps {
  isOpen: boolean
}

const MOCK_NOTIFICATIONS = [
  {
    id: 1,
    title: "Trafic dense",
    message: "Ralentissement sur A86 direction Paris",
    time: "Il y a 2 min",
    type: "warning"
  },
  {
    id: 2,
    title: "Transport en commun",
    message: "RER B : trafic perturbé",
    time: "Il y a 15 min",
    type: "alert"
  },
  {
    id: 3,
    title: "Météo",
    message: "Risque de pluie sur votre trajet",
    time: "Il y a 30 min",
    type: "info"
  }
]

export function NotificationsWidget({ isOpen }: NotificationsWidgetProps) {
  if (!isOpen) return null

  return (
    <div className={cn(
      "absolute top-8 right-8",
      "w-80 bg-card rounded-lg shadow-lg",
      "animate-in slide-in-from-right-8 duration-500",
      "border bg-background/95"
    )}>
      <div className="p-4">
        <div className="flex items-center justify-between mb-4">
          <h3 className="font-semibold">Notifications</h3>
          <span className="text-xs text-muted-foreground">3 nouvelles</span>
        </div>

        <div className="space-y-3">
          {MOCK_NOTIFICATIONS.map((notif) => (
            <div
              key={notif.id}
              className={cn(
                "p-3 rounded-md border",
                "hover:bg-accent/50 transition-colors",
                "cursor-pointer"
              )}
            >
              <div className="flex justify-between items-start mb-1">
                <span className="font-medium">{notif.title}</span>
                <span className="text-xs text-muted-foreground">{notif.time}</span>
              </div>
              <p className="text-sm text-muted-foreground">{notif.message}</p>
            </div>
          ))}
        </div>

        <button className="w-full mt-3 text-sm text-primary hover:underline">
          Voir toutes les notifications
        </button>
      </div>
    </div>
  )
} 