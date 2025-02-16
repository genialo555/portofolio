"use client"

import { useState, useEffect } from "react"
import { AIModel, CrossModelConfig, DebateRole } from "../types"
import { AI_MODELS, DEFAULT_MODEL_CONFIG } from "../services/ai-models"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { motion } from "framer-motion"
import { cn } from "@/lib/utils"

interface ModelSelectorProps {
  selectedModels?: CrossModelConfig;
  onModelChange: (role: DebateRole, model: AIModel) => void;
  className?: string;
}

const roleLabels: Record<DebateRole, string> = {
  pour: "Arguments Pour",
  contre: "Arguments Contre",
  synthese: "Synthèse"
};

const roleColors: Record<DebateRole, string> = {
  pour: "bg-blue-500",
  contre: "bg-red-500",
  synthese: "bg-green-500"
};

const roles: DebateRole[] = ["pour", "contre", "synthese"];

export function ModelSelector({ 
  selectedModels = DEFAULT_MODEL_CONFIG, 
  onModelChange, 
  className 
}: ModelSelectorProps) {
  const [localModels, setLocalModels] = useState<CrossModelConfig>(selectedModels);

  useEffect(() => {
    setLocalModels(selectedModels);
  }, [selectedModels]);

  const handleModelChange = (role: DebateRole, modelId: AIModel) => {
    const newModels = {
      ...localModels,
      [role]: modelId,
    };
    setLocalModels(newModels);
    onModelChange(role, modelId);
  };

  return (
    <div className={cn("space-y-4", className)}>
      <label className="text-sm font-medium text-foreground/60">
        Configuration des modèles
      </label>
      
      <div className="grid gap-4">
        {roles.map((role) => {
          const selectedModel = AI_MODELS.find(model => model.id === localModels[role]);
          
          return (
            <div key={role} className="space-y-2">
              <label className="text-sm font-medium flex items-center gap-2">
                <div className={cn("w-2 h-2 rounded-full", roleColors[role])} />
                {roleLabels[role]}
              </label>
              
              <Select
                value={localModels[role]}
                onValueChange={(value: AIModel) => handleModelChange(role, value)}
              >
                <SelectTrigger className="w-full bg-background/50 backdrop-blur-sm">
                  <SelectValue>
                    <div className="flex items-center gap-2">
                      <div className={cn("w-2 h-2 rounded-full", roleColors[role], "animate-pulse")} />
                      <span>{selectedModel?.name}</span>
                    </div>
                  </SelectValue>
                </SelectTrigger>
                <SelectContent>
                  {AI_MODELS
                    .filter(model => !model.recommendedRoles || model.recommendedRoles.includes(role))
                    .map((model) => (
                      <SelectItem
                        key={model.id}
                        value={model.id}
                        className="space-y-1.5 py-3 cursor-pointer hover:bg-accent"
                      >
                        <div className="flex items-center gap-2">
                          <div className={cn("w-2 h-2 rounded-full", roleColors[role])} />
                          <div>
                            <div className="font-medium">{model.name}</div>
                            <div className="text-xs text-muted-foreground">
                              {model.description}
                            </div>
                          </div>
                        </div>
                      </SelectItem>
                    ))}
                </SelectContent>
              </Select>
            </div>
          );
        })}
      </div>
    </div>
  );
} 