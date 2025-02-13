"use client"

import { AIModel } from "../page"
import { AI_MODELS } from "../services/ai-models"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"

interface ModelSelectorProps {
  selectedModel: AIModel
  onModelChange: (model: AIModel) => void
}

export function ModelSelector({ selectedModel, onModelChange }: ModelSelectorProps) {
  return (
    <div className="space-y-2">
      <label className="text-sm font-medium text-foreground/60">
        Modèle d'IA
      </label>
      <Select
        value={selectedModel}
        onValueChange={(value: AIModel) => onModelChange(value)}
      >
        <SelectTrigger>
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          {AI_MODELS.map((model) => (
            <SelectItem
              key={model.id}
              value={model.id}
              className="space-y-1.5"
            >
              <div className="font-medium">{model.name}</div>
              <div className="text-xs text-muted-foreground">
                {model.description}
              </div>
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
    </div>
  )
} 