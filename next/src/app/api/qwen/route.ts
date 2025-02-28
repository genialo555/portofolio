import OpenAI from "openai";
import { NextRequest, NextResponse } from "next/server";

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

// Fonction pour récupérer les clés API
function getApiKeys(): { key1: string, key2: string } {
  const key1 = process.env.QWEN_API_KEY_1;
  const key2 = process.env.QWEN_API_KEY_2;
  
  if (!key1 && !key2) {
    throw new Error("Aucune clé API Qwen configurée");
  }
  
  return {
    key1: key1 || "",
    key2: key2 || ""
  };
}

// Fonction pour valider le format de la clé API
function validateAPIKey(apiKey: string): boolean {
  if (!apiKey) return false;
  // Accepter n'importe quel format de clé non vide
  return apiKey.length > 0;
}

// Fonction pour créer un client Qwen
function createQwenClient(apiKey: string) {
  if (!validateAPIKey(apiKey)) {
    throw new Error("Format de clé API Qwen invalide");
  }

  return new OpenAI({
    apiKey,
    baseURL: "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
  });
}

// Fonction principale pour gérer les requêtes POST
export async function POST(request: NextRequest): Promise<NextResponse> {
  try {
    // Récupérer les données de la requête
    const { message, modelId, maxTokens, temperature } = await request.json();
    
    // Valider les paramètres requis
    if (!message || !modelId) {
      return NextResponse.json(
        { error: "Paramètres manquants: message et modelId sont requis" },
        { status: 400 }
      );
    }
    
    // Vérifier que c'est un modèle Qwen
    if (!modelId.startsWith('qwen-')) {
      return NextResponse.json(
        { error: `Modèle Qwen invalide : ${modelId}` },
        { status: 400 }
      );
    }
    
    // Récupérer la configuration du modèle
    const modelConfig = QWEN_MODELS[modelId as keyof typeof QWEN_MODELS];
    if (!modelConfig) {
      return NextResponse.json(
        { error: `Configuration du modèle non trouvée pour ${modelId}` },
        { status: 400 }
      );
    }
    
    // Récupérer les clés API
    const keys = getApiKeys();
    const apiKey = Math.random() < 0.5 ? keys.key1 : keys.key2;
    
    // Créer le client Qwen
    const client = createQwenClient(apiKey);
    
    // Appeler l'API Qwen
    const completion = await client.chat.completions.create({
      model: modelConfig.modelName,
      messages: [
        {
          role: "user",
          content: message
        }
      ],
      max_tokens: maxTokens || modelConfig.maxTokens,
      temperature: temperature || 0.7
    });
    
    // Extraire le contenu de la réponse
    const content = completion.choices[0].message.content;
    if (!content) {
      throw new Error("Réponse vide du modèle Qwen");
    }
    
    // Retourner la réponse
    return NextResponse.json({ text: content });
  } catch (error) {
    console.error("Erreur API Qwen:", error);
    return NextResponse.json(
      { error: "Échec de la génération de contenu" },
      { status: 500 }
    );
  }
} 