import OpenAI from "openai";
import { AIModel } from "../types";

// Log des variables d'environnement au chargement du module
console.log("Chargement des variables d'environnement DeepSeek:", {
  KEY_1: process.env.NEXT_PUBLIC_DEEPSEEK_API_KEY_1 ? "Défini" : "Non défini",
  KEY_2: process.env.NEXT_PUBLIC_DEEPSEEK_API_KEY_2 ? "Défini" : "Non défini"
});

const DEEPSEEK_API_KEYS = {
  key1: process.env.NEXT_PUBLIC_DEEPSEEK_API_KEY_1 || "",
  key2: process.env.NEXT_PUBLIC_DEEPSEEK_API_KEY_2 || ""
};

const DEEPSEEK_MODELS = {
  "deepseek": {
    modelName: "deepseek-chat",
    maxTokens: 8192,
    contextWindow: 64000
  },
  "deepseek-reasoner": {
    modelName: "deepseek-reasoner",
    maxTokens: 8192,
    contextWindow: 64000
  }
};

const createDeepSeekClient = (apiKey: string) => {
  // Vérifier que la clé API est au format UUID attendu
  if (!apiKey.match(/^517-[a-f0-9]{16}-[1-8]$/)) {
    throw new Error("Invalid DeepSeek API key format");
  }

  return new OpenAI({
    apiKey,
    baseURL: "https://api.deepseek.com/v1",
    defaultHeaders: {
      'Content-Type': 'application/json',
    },
    defaultQuery: {
      'api-version': '2024-02',
    },
  });
};

export async function generateDeepSeekResponse(
  message: string,
  modelId: AIModel,
  maxTokens?: number,
  temperature: number = 0.7
) {
  // Vérifier que c'est un modèle DeepSeek
  if (!modelId.startsWith('deepseek')) {
    throw new Error(`Invalid DeepSeek model: ${modelId}`);
  }

  // Récupérer la configuration du modèle
  const modelConfig = DEEPSEEK_MODELS[modelId as keyof typeof DEEPSEEK_MODELS];
  if (!modelConfig) {
    throw new Error(`Model config not found for ${modelId}`);
  }

  // Alterner entre les deux clés API pour la répartition de charge
  const apiKey = Math.random() < 0.5 ? DEEPSEEK_API_KEYS.key1 : DEEPSEEK_API_KEYS.key2;
  
  const client = createDeepSeekClient(apiKey);

  try {
    const completion = await client.chat.completions.create({
      model: modelConfig.modelName,
      messages: [
        {
          role: "user",
          content: message
        }
      ],
      max_tokens: maxTokens || modelConfig.maxTokens,
      temperature: temperature,
      stream: false
    });

    return completion.choices[0].message.content;
  } catch (error) {
    console.error("Error calling DeepSeek API:", error);
    throw error;
  }
} 