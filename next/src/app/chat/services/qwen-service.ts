import OpenAI from "openai";
import { AIModel } from "../types";

// Log des variables d'environnement au chargement du module
console.log("Chargement des variables d'environnement Qwen:", {
  KEY_1: process.env.NEXT_PUBLIC_QWEN_API_KEY_1 ? "Défini" : "Non défini",
  KEY_2: process.env.NEXT_PUBLIC_QWEN_API_KEY_2 ? "Défini" : "Non défini"
});

const QWEN_API_KEYS = {
  "qwen-max": {
    key1: process.env.NEXT_PUBLIC_QWEN_API_KEY_1 || "",
    key2: process.env.NEXT_PUBLIC_QWEN_API_KEY_2 || ""
  },
  "qwen-plus": {
    key1: process.env.NEXT_PUBLIC_QWEN_API_KEY_1 || "",
    key2: process.env.NEXT_PUBLIC_QWEN_API_KEY_2 || ""
  },
  "qwen-turbo": {
    key1: process.env.NEXT_PUBLIC_QWEN_API_KEY_1 || "",
    key2: process.env.NEXT_PUBLIC_QWEN_API_KEY_2 || ""
  }
};

const QWEN_MODELS = {
  "qwen-max": {
    modelName: "qwen-max",
    maxTokens: 8192
  },
  "qwen-plus": {
    modelName: "qwen-plus",
    maxTokens: 4096
  },
  "qwen-turbo": {
    modelName: "qwen-turbo",
    maxTokens: 2048
  }
};

const createQwenClient = (apiKey: string) => {
  // Vérifier que la clé API est au format UUID attendu
  if (!apiKey.match(/^517-[a-f0-9]{16}-[1-8]$/)) {
    throw new Error("Invalid Qwen API key format");
  }

  return new OpenAI({
    apiKey,
    baseURL: "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
  });
};

export async function generateQwenResponse(
  message: string,
  modelId: AIModel,
  maxTokens?: number,
  temperature: number = 0.7
) {
  // Vérifier que c'est un modèle Qwen
  if (!modelId.startsWith('qwen-')) {
    throw new Error(`Invalid Qwen model: ${modelId}`);
  }

  // Récupérer la configuration du modèle
  const modelConfig = QWEN_MODELS[modelId as keyof typeof QWEN_MODELS];
  if (!modelConfig) {
    throw new Error(`Model config not found for ${modelId}`);
  }

  // Alterner entre les deux clés API pour la répartition de charge
  const keys = QWEN_API_KEYS[modelId as keyof typeof QWEN_API_KEYS];
  const apiKey = Math.random() < 0.5 ? keys.key1 : keys.key2;
  
  const client = createQwenClient(apiKey);

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
      temperature: temperature
    });

    return completion.choices[0].message.content;
  } catch (error) {
    console.error("Error calling Qwen API:", error);
    throw error;
  }
} 