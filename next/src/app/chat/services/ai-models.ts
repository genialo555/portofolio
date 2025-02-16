import { AIModel, ModelConfig, DebateRole, CrossModelConfig } from "../types"
import { generateQwenResponse } from "./qwen-service"
import { generateDeepSeekResponse } from "./deepseek-service"

const HUGGING_FACE_API_URL = "https://api-inference.huggingface.co/models"

export interface Message {
  id: string;
  content: string;
  role: "pour" | "contre" | "synthese";
  timestamp: Date;
  media?: {
    type: "image" | "video";
    url: string;
    alt?: string;
  };
}

export interface Conversation {
  id: string;
  title: string;
  messages: Message[];
  lastUpdated: Date;
  model: AIModel;
}

export const AI_MODELS: ModelConfig[] = [
  {
    id: "gemini",
    name: "Gemini Pro",
    description: "Modèle le plus avancé de Google, excellent pour tous les rôles",
    modelId: "google/gemini-pro",
    maxLength: 4096,
    temperature: 0.7,
    recommendedRoles: ["pour", "contre", "synthese"]
  },
  {
    id: "qwen-max",
    name: "Qwen 2.5 Max",
    description: "Version la plus puissante de Qwen, excellente pour la synthèse",
    modelId: "qwen-max",
    maxLength: 8192,
    temperature: 0.7,
    recommendedRoles: ["synthese", "pour"]
  },
  {
    id: "qwen-plus",
    name: "Qwen 2.5 Plus",
    description: "Version équilibrée de Qwen, bon rapport performance/coût",
    modelId: "qwen-plus",
    maxLength: 4096,
    temperature: 0.7,
    recommendedRoles: ["pour", "contre"]
  },
  {
    id: "qwen-turbo",
    name: "Qwen 2.5 Turbo",
    description: "Version rapide de Qwen, idéale pour les réponses courtes",
    modelId: "qwen-turbo",
    maxLength: 2048,
    temperature: 0.7,
    recommendedRoles: ["pour", "contre"]
  },
  {
    id: "deepseek",
    name: "DeepSeek Chat",
    description: "Modèle très performant pour l'argumentation avec un large contexte",
    modelId: "deepseek-chat",
    maxLength: 8192,
    temperature: 0.7,
    recommendedRoles: ["pour", "contre"]
  },
  {
    id: "deepseek-reasoner",
    name: "DeepSeek Reasoner",
    description: "Version spécialisée dans le raisonnement et l'analyse",
    modelId: "deepseek-reasoner",
    maxLength: 8192,
    temperature: 0.7,
    recommendedRoles: ["synthese", "contre"]
  }
]

export const DEFAULT_MODEL_CONFIG: CrossModelConfig = {
  pour: "gemini",
  contre: "deepseek",
  synthese: "qwen-max"
}

export async function generateResponse(
  message: string,
  modelConfig: ModelConfig,
  apiKey: string
) {
  if (modelConfig.id.startsWith('qwen-')) {
    return generateQwenResponse(
      message,
      modelConfig.id,
      modelConfig.maxLength,
      modelConfig.temperature
    );
  }
  
  if (modelConfig.id.startsWith('deepseek')) {
    return generateDeepSeekResponse(
      message,
      modelConfig.id,
      modelConfig.maxLength,
      modelConfig.temperature
    );
  }

  try {
    const response = await fetch(
      `${HUGGING_FACE_API_URL}/${modelConfig.modelId}`,
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${apiKey}`,
        },
        body: JSON.stringify({
          inputs: message,
          parameters: {
            max_new_tokens: modelConfig.maxLength,
            temperature: modelConfig.temperature,
            top_p: 0.9,
            do_sample: true,
          },
        }),
      }
    )

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`)
    }

    const result = await response.json()
    return result[0].generated_text
  } catch (error) {
    console.error("Error calling API:", error)
    throw error
  }
}

export function getModelConfig(modelId: AIModel): ModelConfig {
  const config = AI_MODELS.find(model => model.id === modelId)
  if (!config) {
    throw new Error(`Model config not found for ${modelId}`)
  }
  return config
} 