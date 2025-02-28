import { AIModel } from "../types";

// Configuration des modèles Qwen
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

// Utiliser une API côté serveur au lieu d'appeler directement les services
async function callServerApi(data: any): Promise<any> {
  try {
    const response = await fetch(`/api/qwen`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(data),
    });
    
    if (!response.ok) {
      throw new Error(`Erreur API: ${response.status}`);
    }
    
    return await response.json();
  } catch (error) {
    const errorMsg = error instanceof Error ? error.message : "Erreur inconnue";
    console.error(`Erreur API Qwen:`, errorMsg);
    throw new Error(`Échec de la génération de contenu`);
  }
}

export async function generateQwenResponse(
  message: string,
  modelId: AIModel,
  maxTokens?: number,
  temperature: number = 0.7
) {
  // Vérifier que c'est un modèle Qwen
  if (!modelId.startsWith('qwen-')) {
    throw new Error(`Modèle Qwen invalide : ${modelId}`);
  }

  // Récupérer la configuration du modèle
  const modelConfig = QWEN_MODELS[modelId as keyof typeof QWEN_MODELS];
  if (!modelConfig) {
    throw new Error(`Configuration du modèle non trouvée pour ${modelId}`);
  }

  try {
    // Appeler l'API côté serveur
    const result = await callServerApi({
      message,
      modelId,
      maxTokens: maxTokens || modelConfig.maxTokens,
      temperature
    });

    return result.text;
  } catch (error) {
    console.error("Erreur lors de l'appel à l'API Qwen:", error);
    throw error;
  }
} 