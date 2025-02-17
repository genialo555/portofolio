import { GoogleGenerativeAI } from "@google/generative-ai";

export type AgentRole = "pour" | "contre" | "synthese";

// Log détaillé des variables d'environnement au chargement du module
console.log("=== Débogage des variables d'environnement ===");
console.log("Variables d'environnement brutes:", {
  POUR_1: process.env.NEXT_PUBLIC_GEMINI_API_KEY_POUR_1,
  POUR_2: process.env.NEXT_PUBLIC_GEMINI_API_KEY_POUR_2,
  CONTRE_1: process.env.NEXT_PUBLIC_GEMINI_API_KEY_CONTRE_1,
  CONTRE_2: process.env.NEXT_PUBLIC_GEMINI_API_KEY_CONTRE_2
});

// Vérification de l'environnement
console.log("Environnement:", {
  NODE_ENV: process.env.NODE_ENV,
  IS_BROWSER: typeof window !== 'undefined'
});

interface AgentConfig {
  name: string;
  role: AgentRole;
  apiKey: string;
  personality: string;
}

// Configuration des agents avec vérification
const AGENT_CONFIGS: AgentConfig[] = [
  {
    name: "Agent Pour 1",
    role: "pour",
    apiKey: process.env.NEXT_PUBLIC_GEMINI_API_KEY_POUR_1 || "",
    personality: "Vous êtes un agent qui défend le point de vue positif avec des arguments constructifs et bien structurés."
  },
  {
    name: "Agent Pour 2",
    role: "pour",
    apiKey: process.env.NEXT_PUBLIC_GEMINI_API_KEY_POUR_2 || "",
    personality: "Vous êtes un agent qui renforce les arguments positifs avec des exemples concrets et des données."
  },
  {
    name: "Agent Contre 1",
    role: "contre",
    apiKey: process.env.NEXT_PUBLIC_GEMINI_API_KEY_CONTRE_1 || "",
    personality: "Vous êtes un agent qui présente des contre-arguments logiques et réfléchis."
  },
  {
    name: "Agent Contre 2",
    role: "contre",
    apiKey: process.env.NEXT_PUBLIC_GEMINI_API_KEY_CONTRE_2 || "",
    personality: "Vous êtes un agent qui soulève des points critiques importants avec une approche analytique."
  }
];

// Log de vérification des configurations
console.log("=== Vérification des configurations des agents ===");
AGENT_CONFIGS.forEach(agent => {
  console.log(`${agent.name}:`, {
    hasApiKey: !!agent.apiKey,
    keyLength: agent.apiKey?.length || 0,
    keyStart: agent.apiKey ? agent.apiKey.substring(0, 10) + "..." : "non définie"
  });
});

const GEMINI_API_KEYS = {
  pour: {
    key1: process.env.NEXT_PUBLIC_GEMINI_API_KEY_POUR_1 || "",
    key2: process.env.NEXT_PUBLIC_GEMINI_API_KEY_POUR_2 || ""
  },
  contre: {
    key1: process.env.NEXT_PUBLIC_GEMINI_API_KEY_CONTRE_1 || "",
    key2: process.env.NEXT_PUBLIC_GEMINI_API_KEY_CONTRE_2 || ""
  }
};

// Fonction pour générer le prompt de l'agent
function generateAgentPrompt(
  agent: AgentConfig,
  topic: string,
  context: string = ""
): string {
  const basePrompt = `${agent.personality}

Sujet de discussion : ${topic}

${context ? `Contexte de la discussion :\n${context}\n` : ""}

En tant qu'agent ${agent.role === "pour" ? "favorable" : "critique"}, donnez votre point de vue sur ce sujet.
Structurez votre réponse avec :
1. Votre position principale
2. 2-3 arguments clés
3. Des exemples ou preuves pour appuyer vos arguments

Répondez de manière concise mais convaincante.`;

  return basePrompt;
}

// Fonction pour générer le prompt de synthèse
function generateSynthesisPrompt(
  topic: string,
  pourArguments: string[],
  contreArguments: string[]
): string {
  return `En tant que médiateur objectif, faites une synthèse équilibrée du débat suivant :

Sujet : ${topic}

Arguments POUR :
${pourArguments.map((arg, i) => `${i + 1}. ${arg}`).join("\n")}

Arguments CONTRE :
${contreArguments.map((arg, i) => `${i + 1}. ${arg}`).join("\n")}

Veuillez fournir :
1. Un résumé des principaux points de vue des deux côtés
2. Les points de convergence éventuels
3. Une conclusion nuancée qui tient compte des différentes perspectives

Restez neutre et objectif dans votre synthèse.`;
}

// Fonction pour obtenir une réponse d'un agent
export async function getAgentResponse(
  agent: AgentConfig,
  topic: string,
  context: string = ""
): Promise<string> {
  try {
    if (!agent.apiKey) {
      console.error(`Clé API manquante pour l'agent ${agent.name}`);
      throw new Error("API key not configured");
    }

    console.log(`Tentative d'appel API pour ${agent.name} avec la clé: ${agent.apiKey.substring(0, 10)}...`);

    const response = await fetch('https://generativelanguage.googleapis.com/v1/models/gemini-pro:generateContent', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'x-goog-api-key': agent.apiKey,
      },
      body: JSON.stringify({
        contents: [{
          parts: [{
            text: generateAgentPrompt(agent, topic, context)
          }]
        }]
      })
    });

    if (!response.ok) {
      throw new Error(`API request failed: ${response.statusText}`);
    }

    const result = await response.json();
    
    if (!result.candidates?.[0]?.content?.parts?.[0]?.text) {
      throw new Error("La réponse a été bloquée par les filtres de sécurité. Veuillez reformuler votre demande.");
    }

    return result.candidates[0].content.parts[0].text;
  } catch (error: any) {
    console.error(`Erreur lors de l'appel à l'API pour l'agent ${agent.name}:`, error);
    throw error;
  }
}

// Fonction pour obtenir une synthèse
export async function getSynthesis(
  topic: string,
  pourArguments: string[],
  contreArguments: string[]
): Promise<string> {
  try {
    const apiKey = AGENT_CONFIGS[0].apiKey;
    if (!apiKey) {
      throw new Error("API key not configured for synthesis");
    }

    const prompt = generateSynthesisPrompt(topic, pourArguments, contreArguments);

    const response = await fetch('https://generativelanguage.googleapis.com/v1/models/gemini-pro:generateContent', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'x-goog-api-key': apiKey,
      },
      body: JSON.stringify({
        contents: [{
          parts: [{
            text: prompt
          }]
        }]
      })
    });

    if (!response.ok) {
      throw new Error(`API request failed: ${response.statusText}`);
    }

    const result = await response.json();
    
    if (!result.candidates?.[0]?.content?.parts?.[0]?.text) {
      throw new Error("La synthèse a été bloquée par les filtres de sécurité. Veuillez reformuler la demande.");
    }

    return result.candidates[0].content.parts[0].text;
  } catch (error: any) {
    console.error("Erreur lors de la génération de la synthèse:", error);
    throw new Error(`Erreur lors de la génération de la synthèse: ${error.message}`);
  }
}

// Fonction pour lancer un débat complet
export async function runDebate(topic: string): Promise<{
  pour: string[];
  contre: string[];
  synthese: string;
}> {
  try {
    console.log("Démarrage du débat sur le sujet :", topic);
    
    // Vérification des clés API
    const missingKeys = AGENT_CONFIGS.filter(agent => !agent.apiKey);
    if (missingKeys.length > 0) {
      throw new Error(`Clés API manquantes pour les agents: ${missingKeys.map(a => a.name).join(', ')}`);
    }

    // Obtenir les réponses des agents "pour"
    console.log("Obtention des arguments POUR...");
    const pourResponses = await Promise.all(
      AGENT_CONFIGS.filter(agent => agent.role === "pour")
        .map(agent => getAgentResponse(agent, topic))
    );

    // Obtenir les réponses des agents "contre"
    console.log("Obtention des arguments CONTRE...");
    const contreResponses = await Promise.all(
      AGENT_CONFIGS.filter(agent => agent.role === "contre")
        .map(agent => getAgentResponse(agent, topic))
    );

    // Générer la synthèse
    console.log("Génération de la synthèse...");
    const synthese = await getSynthesis(topic, pourResponses, contreResponses);

    return {
      pour: pourResponses,
      contre: contreResponses,
      synthese
    };
  } catch (error) {
    console.error("Erreur lors du débat complet:", error);
    throw error;
  }
} 