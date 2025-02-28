import { GoogleGenerativeAI } from "@google/generative-ai";

// Désactiver les logs sensibles en production
const isDev = process.env.NODE_ENV === "development";

// Types
export type AgentRole = "pour" | "contre" | "synthese";

// Configuration des agents
interface AgentConfig {
  name: string;
  role: AgentRole;
  personality: string;
}

// Validation des variables d'environnement
function loadEnvVariable(key: string, name: string): string {
  const value = process.env[key];
  if (!value) {
    console.warn(`Variable d'environnement ${name} manquante. Utilisation d'une valeur par défaut.`);
    return 'not-configured';
  }
  return value;
}

// Configuration des agents avec sécurisation
function getAgentConfigs(): AgentConfig[] {
  return [
    {
      name: "Agent Pour 1",
      role: "pour",
      personality: "Vous êtes un défenseur optimiste qui mise sur des arguments constructifs et structurés."
    },
    {
      name: "Agent Pour 2",
      role: "pour",
      personality: "Vous êtes un soutien pragmatique apportant exemples concrets et données chiffrées."
    },
    {
      name: "Agent Contre 1",
      role: "contre",
      personality: "Vous êtes un critique rationnel qui privilégie la logique et les failles potentielles."
    },
    {
      name: "Agent Contre 2",
      role: "contre",
      personality: "Vous êtes un analyste pointu qui met en lumière les risques et limites."
    }
  ];
}

// Prompts sécurisés
const PROMPTS = {
  agent: (agent: AgentConfig, topic: string, context: string = "") => `
    ${agent.personality}
    Sujet : ${topic}
    ${context ? `Contexte : ${context}\n` : ""}
    En tant que ${agent.role === "pour" ? "partisan" : "opposant"}, développez votre position :
    1. Exposez clairement votre thèse
    2. Fournissez 2-3 arguments clés
    3. Illustrez avec des exemples ou faits pertinents
    Soyez concis, persuasif et évitez les redondances.
  `,
  synthesis: (topic: string, pourArgs: string[], contreArgs: string[]) => `
    Rôle : Médiateur impartial
    Sujet : ${topic}
    Arguments Pour :
    ${pourArgs.map((arg, i) => `${i + 1}. ${arg}`).join("\n")}
    Arguments Contre :
    ${contreArgs.map((arg, i) => `${i + 1}. ${arg}`).join("\n")}
    Produisez une synthèse objective :
    1. Résumez les positions principales
    2. Identifiez les convergences possibles
    3. Concluez avec une perspective équilibrée
    Restez neutre et factuel.
  `
};

// Validation des entrées
function validateInput(input: string, field: string): string {
  const trimmed = input.trim();
  if (!trimmed) throw new Error(`Le ${field} ne peut pas être vide`);
  if (trimmed.length > 500) throw new Error(`Le ${field} est trop long (max 500 caractères)`);
  return trimmed.replace(/[<>&;]/g, ""); // Supprime les caractères dangereux
}

// Utiliser une API côté serveur au lieu d'appeler directement les services
async function callServerApi(endpoint: string, data: any): Promise<any> {
  try {
    const response = await fetch(`/api/${endpoint}`, {
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
    if (isDev) console.error(`Erreur API:`, errorMsg);
    throw new Error(`Échec de la génération de contenu`);
  }
}

// Réponse d'un agent
export async function getAgentResponse(
  agent: AgentConfig,
  topic: string,
  context: string = ""
): Promise<string> {
  const validTopic = validateInput(topic, "sujet");
  const validContext = context ? validateInput(context, "contexte") : "";
  
  try {
    const result = await callServerApi('gemini', {
      role: agent.role,
      topic: validTopic,
      context: validContext,
      personality: agent.personality
    });
    
    return result.text;
  } catch (error) {
    if (isDev) console.error("Erreur lors de l'appel à l'agent:", error);
    return "Désolé, je n'ai pas pu générer une réponse. Veuillez réessayer.";
  }
}

// Synthèse
export async function getSynthesis(
  topic: string,
  pourArguments: string[],
  contreArguments: string[]
): Promise<string> {
  const validTopic = validateInput(topic, "sujet");
  
  try {
    const result = await callServerApi('synthesis', {
      topic: validTopic,
      pourArguments,
      contreArguments
    });
    
    return result.text;
  } catch (error) {
    if (isDev) console.error("Erreur lors de la synthèse:", error);
    return "Désolé, je n'ai pas pu générer une synthèse. Veuillez réessayer.";
  }
}

// Débat complet avec gestion sécurisée
export async function runDebate(
  topic: string,
  context: string = ""
): Promise<{
  pour: string[];
  contre: string[];
  synthese: string;
}> {
  try {
    const validTopic = validateInput(topic, "sujet");
    const validContext = context ? validateInput(context, "contexte") : "";

    if (isDev) console.log(`Débat lancé : "${validTopic}"`);

    // Utiliser l'API côté serveur
    const result = await callServerApi('debate', {
      topic: validTopic,
      context: validContext
    });

    if (isDev) console.log("Débat terminé avec succès");
    return result;
  } catch (error) {
    const errorMsg = error instanceof Error ? error.message : "Erreur inconnue";
    if (isDev) console.error("Échec du débat:", errorMsg);
    
    // Retourner des données de secours en cas d'erreur
    return {
      pour: ["Désolé, je n'ai pas pu générer d'arguments pour."],
      contre: ["Désolé, je n'ai pas pu générer d'arguments contre."],
      synthese: "Désolé, je n'ai pas pu générer une synthèse."
    };
  }
}