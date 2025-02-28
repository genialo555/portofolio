import { AIModel, ModelConfig, DebateRole, CrossModelConfig } from "../types";
import { generateQwenResponse } from "./qwen-service";
import { generateDeepSeekResponse } from "./deepseek-service";

// **Constantes**
const HUGGING_FACE_API_URL = "https://api-inference.huggingface.co/models";
const MAX_INPUT_LENGTH = 500; // Limite de caractères pour les entrées utilisateur
const MIN_RESPONSE_LENGTH = 50; // Longueur minimale pour une réponse valide
const MAX_RESPONSE_TIME_MS = 10000; // Temps de réponse maximal avant avertissement (10s)

// Désactiver les logs sensibles en production
const isDev = process.env.NODE_ENV === "development";

// **Types**
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

// **Liste des modèles AI disponibles avec leurs configurations**
export const AI_MODELS: ModelConfig[] = [
  {
    id: "gemini",
    name: "Gemini Pro",
    description: "Modèle le plus avancé de Google, excellent pour tous les rôles",
    modelId: "google/gemini-pro",
    maxLength: 4096,
    temperature: 0.7,
    recommendedRoles: ["pour", "contre", "synthese"],
  },
  {
    id: "qwen-max",
    name: "Qwen 2.5 Max",
    description: "Version la plus puissante de Qwen, excellente pour la synthèse",
    modelId: "qwen-max",
    maxLength: 8192,
    temperature: 0.7,
    recommendedRoles: ["synthese", "pour"],
  },
  {
    id: "qwen-plus",
    name: "Qwen 2.5 Plus",
    description: "Version équilibrée de Qwen, bon rapport performance/coût",
    modelId: "qwen-plus",
    maxLength: 4096,
    temperature: 0.7,
    recommendedRoles: ["pour", "contre"],
  },
  {
    id: "qwen-turbo",
    name: "Qwen 2.5 Turbo",
    description: "Version rapide de Qwen, idéale pour les réponses courtes",
    modelId: "qwen-turbo",
    maxLength: 2048,
    temperature: 0.7,
    recommendedRoles: ["pour", "contre"],
  },
  {
    id: "deepseek",
    name: "DeepSeek Chat",
    description: "Modèle très performant pour l'argumentation avec un large contexte",
    modelId: "deepseek-chat",
    maxLength: 8192,
    temperature: 0.7,
    recommendedRoles: ["pour", "contre"],
  },
  {
    id: "deepseek-reasoner",
    name: "DeepSeek Reasoner",
    description: "Version spécialisée dans le raisonnement et l'analyse",
    modelId: "deepseek-reasoner",
    maxLength: 8192,
    temperature: 0.7,
    recommendedRoles: ["synthese", "contre"],
  },
];

// **Configuration par défaut pour les rôles dans un débat**
export const DEFAULT_MODEL_CONFIG: CrossModelConfig = {
  pour: "gemini",
  contre: "deepseek",
  synthese: "qwen-max",
};

// **Validation des entrées utilisateur**
function validateInput(input: string, field: string): string {
  const trimmed = input.trim();
  if (!trimmed) throw new Error(`Le ${field} ne peut pas être vide`);
  if (trimmed.length > MAX_INPUT_LENGTH) throw new Error(`Le ${field} est trop long (max ${MAX_INPUT_LENGTH} caractères)`);
  return trimmed.replace(/[<>&;]/g, ""); // Supprime les caractères dangereux
}

// **Fonction utilitaire pour charger les variables d'environnement en toute sécurité**
function loadEnvVariable(key: string): string {
  const value = process.env[key];
  if (value === undefined || value === null) {
    throw new Error(`La variable d'environnement ${key} est manquante. Vérifiez votre configuration.`);
  }
  return value;
}

// **Appel API sécurisé avec détection d'anomalies**
async function callHuggingFaceApi(apiKey: string, modelId: string, message: string, config: ModelConfig): Promise<string> {
  const startTime = performance.now();
  try {
    const response = await fetch(`${HUGGING_FACE_API_URL}/${modelId}`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${apiKey}`,
      },
      body: JSON.stringify({
        inputs: message,
        parameters: {
          max_new_tokens: config.maxLength,
          temperature: config.temperature,
          top_p: 0.9,
          do_sample: true,
        },
      }),
    });

    if (!response.ok) {
      throw new Error(`Erreur HTTP ! statut : ${response.status}`);
    }

    const result = await response.json();
    const text = result[0].generated_text.trim();
    const duration = performance.now() - startTime;

    // Détection d'anomalies
    if (duration > MAX_RESPONSE_TIME_MS) console.warn(`Temps de réponse excessif pour ${modelId} : ${duration}ms`);
    if (text.length < MIN_RESPONSE_LENGTH) throw new Error("Réponse trop courte");
    if (/erreur|impossible/i.test(text)) throw new Error("Réponse invalide");

    return text;
  } catch (error) {
    const duration = performance.now() - startTime;
    const errorMsg = error instanceof Error ? error.message : "Erreur inconnue";
    if (isDev) console.error(`Erreur API pour ${modelId} après ${duration}ms :`, errorMsg);
    throw new Error(`Échec de la génération de contenu`);
  }
}

// **Fonction principale pour générer une réponse en fonction du modèle**
export async function generateResponse(message: string, modelConfig: ModelConfig, apiKey: string): Promise<string> {
  const validMessage = validateInput(message, "message");

  // Vérification de sécurité pour s'assurer que modelConfig.id est une chaîne
  if (!modelConfig.id) {
    throw new Error("ID du modèle non défini");
  }

  // Vérification que l'ID est bien un AIModel valide
  const isValidAIModel = (id: string): id is AIModel => {
    return ["gemini", "qwen-max", "qwen-plus", "qwen-turbo", "deepseek", "deepseek-reasoner"].includes(id as AIModel);
  };

  if (!isValidAIModel(modelConfig.id)) {
    throw new Error(`ID de modèle invalide: ${modelConfig.id}`);
  }

  const modelId = modelConfig.id;
  const maxLength = modelConfig.maxLength ?? 4096;
  const temperature = modelConfig.temperature ?? 0.7;

  if (["qwen-max", "qwen-plus", "qwen-turbo"].includes(modelId)) {
    return generateQwenResponse(validMessage, modelId, maxLength, temperature);
  }

  if (["deepseek", "deepseek-reasoner"].includes(modelId)) {
    return generateDeepSeekResponse(validMessage, modelId, maxLength, temperature);
  }

  // Pour les modèles Hugging Face
  return callHuggingFaceApi(apiKey, modelConfig.modelId, validMessage, modelConfig);
}

// **Fonction utilitaire pour récupérer la configuration d'un modèle**
export function getModelConfig(modelId: AIModel): ModelConfig {
  const config = AI_MODELS.find((model) => model.id === modelId);
  if (!config) {
    throw new Error(`Configuration du modèle non trouvée pour ${modelId}`);
  }
  return config;
}