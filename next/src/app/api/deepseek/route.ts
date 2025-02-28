import OpenAI from "openai";
import { NextRequest, NextResponse } from "next/server";

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

// Fonction pour récupérer les clés API
function getApiKeys(): { key1: string, key2: string } {
  const key1 = process.env.DEEPSEEK_API_KEY_1;
  const key2 = process.env.DEEPSEEK_API_KEY_2;
  
  if (!key1 && !key2) {
    throw new Error("Aucune clé API DeepSeek configurée");
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

// Fonction pour créer un client DeepSeek
function createDeepSeekClient(apiKey: string) {
  if (!validateAPIKey(apiKey)) {
    throw new Error("Format de clé API DeepSeek invalide");
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
    
    // Vérifier que c'est un modèle DeepSeek
    if (!modelId.startsWith('deepseek')) {
      return NextResponse.json(
        { error: `Modèle DeepSeek invalide : ${modelId}` },
        { status: 400 }
      );
    }
    
    // Récupérer la configuration du modèle
    const modelConfig = DEEPSEEK_MODELS[modelId as keyof typeof DEEPSEEK_MODELS];
    if (!modelConfig) {
      return NextResponse.json(
        { error: `Configuration du modèle non trouvée pour ${modelId}` },
        { status: 400 }
      );
    }
    
    // Récupérer les clés API
    const keys = getApiKeys();
    const apiKey = Math.random() < 0.5 ? keys.key1 : keys.key2;
    
    // Créer le client DeepSeek
    const client = createDeepSeekClient(apiKey);
    
    // Appeler l'API DeepSeek
    const completion = await client.chat.completions.create({
      model: modelConfig.modelName,
      messages: [
        {
          role: "user",
          content: message
        }
      ],
      max_tokens: maxTokens || modelConfig.maxTokens,
      temperature: temperature || 0.7,
      stream: false
    });
    
    // Extraire le contenu de la réponse
    const content = completion.choices[0].message.content;
    if (!content) {
      throw new Error("Réponse vide du modèle DeepSeek");
    }
    
    // Retourner la réponse
    return NextResponse.json({ text: content });
  } catch (error) {
    console.error("Erreur API DeepSeek:", error);
    return NextResponse.json(
      { error: "Échec de la génération de contenu" },
      { status: 500 }
    );
  }
} 