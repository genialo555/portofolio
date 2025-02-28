import { AIModel } from "../types";

// Configuration des modèles DeepSeek
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

// Utiliser une API côté serveur au lieu d'appeler directement les services
async function callServerApi(data: any): Promise<any> {
  try {
    const response = await fetch(`/api/deepseek`, {
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
    console.error(`Erreur API DeepSeek:`, errorMsg);
    throw new Error(`Échec de la génération de contenu`);
  }
}

export async function generateDeepSeekResponse(
  message: string,
  modelId: AIModel,
  maxTokens?: number,
  temperature: number = 0.7
) {
  // Vérifier que c'est un modèle DeepSeek
  if (!modelId.startsWith('deepseek')) {
    throw new Error(`Modèle DeepSeek invalide : ${modelId}`);
  }

  // Récupérer la configuration du modèle
  const modelConfig = DEEPSEEK_MODELS[modelId as keyof typeof DEEPSEEK_MODELS];
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
    console.error("Erreur lors de l'appel à l'API DeepSeek:", error);
    throw error;
  }
} 