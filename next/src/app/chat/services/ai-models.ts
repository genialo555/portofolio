import { AIModel } from "../page"

const HUGGING_FACE_API_URL = "https://api-inference.huggingface.co/models"

export interface ModelConfig {
  id: AIModel
  name: string
  description: string
  modelId: string
  maxLength: number
  temperature: number
}

export const AI_MODELS: ModelConfig[] = [
  {
    id: "qwen",
    name: "Qwen 1.5",
    description: "Modèle multilingue performant de Alibaba",
    modelId: "Qwen/Qwen1.5-7B-Chat",
    maxLength: 2048,
    temperature: 0.7
  },
  {
    id: "mistral",
    name: "Mistral 7B",
    description: "Modèle français open source très performant",
    modelId: "mistralai/Mistral-7B-Instruct-v0.2",
    maxLength: 2048,
    temperature: 0.7
  },
  {
    id: "yi",
    name: "Yi 34B",
    description: "Modèle multilingue de 01.AI",
    modelId: "01-ai/Yi-34B-Chat",
    maxLength: 4096,
    temperature: 0.7
  },
  {
    id: "openchat",
    name: "OpenChat 3.5",
    description: "Alternative open source à GPT-3.5",
    modelId: "openchat/openchat-3.5",
    maxLength: 2048,
    temperature: 0.7
  },
  {
    id: "solar",
    name: "Solar 10.7B",
    description: "Modèle performant de Upstage",
    modelId: "upstage/SOLAR-10.7B-Instruct-v1.0",
    maxLength: 2048,
    temperature: 0.7
  }
]

export async function generateResponse(
  message: string,
  modelConfig: ModelConfig,
  apiKey: string
) {
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
    console.error("Error calling Hugging Face API:", error)
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