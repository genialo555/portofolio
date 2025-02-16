import { GoogleGenerativeAI } from "@google/generative-ai";

export type AgentRole = "pour" | "contre" | "synthese";

// Log des variables d'environnement au chargement du module
console.log("Chargement des variables d'environnement:", {
  POUR_1: process.env.NEXT_PUBLIC_GEMINI_API_KEY_POUR_1 ? "Défini" : "Non défini",
  POUR_2: process.env.NEXT_PUBLIC_GEMINI_API_KEY_POUR_2 ? "Défini" : "Non défini",
  CONTRE_1: process.env.NEXT_PUBLIC_GEMINI_API_KEY_CONTRE_1 ? "Défini" : "Non défini",
  CONTRE_2: process.env.NEXT_PUBLIC_GEMINI_API_KEY_CONTRE_2 ? "Défini" : "Non défini",
});

interface AgentConfig {
  name: string;
  role: AgentRole;
  apiKey: string;
  personality: string;
}

// Configuration des agents avec vérification des clés
export const AGENT_CONFIGS: AgentConfig[] = [
  {
    name: "Agent Pour 1",
    role: "pour",
    apiKey: (() => {
      const key = process.env.NEXT_PUBLIC_GEMINI_API_KEY_POUR_1 || "";
      console.log("Initialisation Agent Pour 1 - Clé API:", key ? "Présente" : "Manquante");
      return key;
    })(),
    personality: "Je suis un agent qui défend activement le point de vue favorable, en m'appuyant sur des arguments logiques et des exemples concrets."
  },
  {
    name: "Agent Pour 2",
    role: "pour",
    apiKey: (() => {
      const key = process.env.NEXT_PUBLIC_GEMINI_API_KEY_POUR_2 || "";
      console.log("Initialisation Agent Pour 2 - Clé API:", key ? "Présente" : "Manquante");
      return key;
    })(),
    personality: "Je suis un agent qui soutient la position favorable en explorant les avantages et les opportunités potentielles."
  },
  {
    name: "Agent Contre 1",
    role: "contre",
    apiKey: (() => {
      const key = process.env.NEXT_PUBLIC_GEMINI_API_KEY_CONTRE_1 || "";
      console.log("Initialisation Agent Contre 1 - Clé API:", key ? "Présente" : "Manquante");
      return key;
    })(),
    personality: "Je suis un agent qui présente des contre-arguments réfléchis et soulève des points de vigilance importants."
  },
  {
    name: "Agent Contre 2",
    role: "contre",
    apiKey: (() => {
      const key = process.env.NEXT_PUBLIC_GEMINI_API_KEY_CONTRE_2 || "";
      console.log("Initialisation Agent Contre 2 - Clé API:", key ? "Présente" : "Manquante");
      return key;
    })(),
    personality: "Je suis un agent qui examine de manière critique les potentiels inconvénients et risques à considérer."
  }
];

const GEMINI_API_KEYS = {
  pour: {
    key1: process.env.GEMINI_API_KEY_POUR_1 || "",
    key2: process.env.GEMINI_API_KEY_POUR_2 || ""
  },
  contre: {
    key1: process.env.GEMINI_API_KEY_CONTRE_1 || "",
    key2: process.env.GEMINI_API_KEY_CONTRE_2 || ""
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
    // Vérifier si la clé API est présente et bien formatée
    if (!agent.apiKey) {
      console.error(`Clé API manquante pour l'agent ${agent.name}`);
      return "Erreur de configuration : clé API manquante.";
    }

    if (!agent.apiKey.startsWith('AI')) {
      console.error(`Format de clé API invalide pour l'agent ${agent.name}. La clé doit commencer par 'AI'`);
      return "Erreur de configuration : format de clé API invalide.";
    }

    console.log(`Configuration de l'agent ${agent.name}:`, {
      role: agent.role,
      apiKeyPrefix: agent.apiKey.substring(0, 10),
      apiKeyLength: agent.apiKey.length
    });

    const genAI = new GoogleGenerativeAI(agent.apiKey);
    console.log(`Instance GoogleGenerativeAI créée pour ${agent.name}`);
    
    const model = genAI.getGenerativeModel({ model: "gemini-pro" });
    console.log(`Modèle gemini-pro obtenu pour ${agent.name}`);

    const prompt = generateAgentPrompt(agent, topic, context);
    console.log(`Prompt généré pour ${agent.name}:`, prompt);

    console.log(`Début de l'appel API pour ${agent.name}...`);
    const result = await model.generateContent(prompt);
    console.log(`Réponse reçue de l'API pour ${agent.name}:`, result);
    
    // Vérifier si la réponse a été bloquée
    if (!result.response.text()) {
      console.error(`Réponse bloquée pour l'agent ${agent.name}`);
      throw new Error("La réponse a été bloquée par les filtres de sécurité. Veuillez reformuler votre demande.");
    }

    return result.response.text();
  } catch (error: any) {
    console.error(`Erreur détaillée pour l'agent ${agent.name}:`, {
      error: error.toString(),
      stack: error.stack,
      message: error.message,
      name: error.name
    });
    
    // Personnaliser le message d'erreur
    if (error?.toString().includes("SAFETY")) {
      return "Désolé, je ne peux pas répondre à cette demande car elle a été bloquée par les filtres de sécurité. Veuillez reformuler votre question de manière plus appropriée.";
    } else if (error?.toString().includes("API Key")) {
      return `Erreur d'authentification avec l'API pour l'agent ${agent.name}. Détails : ${error.message}`;
    } else {
      return `Une erreur est survenue lors de la génération de la réponse pour l'agent ${agent.name}. Erreur : ${error.message || error.toString()}`;
    }
  }
}

// Fonction pour obtenir une synthèse
export async function getSynthesis(
  topic: string,
  pourArguments: string[],
  contreArguments: string[]
): Promise<string> {
  try {
    // Utiliser une des clés API pour la synthèse
    const apiKey = AGENT_CONFIGS[0].apiKey;
    if (!apiKey) {
      console.error("Clé API manquante pour la synthèse");
      return "Erreur de configuration : clé API manquante pour la synthèse.";
    }

    console.log("Tentative de synthèse avec la clé :", apiKey.substring(0, 10));

    const genAI = new GoogleGenerativeAI(apiKey);
    const model = genAI.getGenerativeModel({ model: "gemini-pro" });

    const prompt = generateSynthesisPrompt(topic, pourArguments, contreArguments);
    console.log("Prompt de synthèse :", prompt);

    const result = await model.generateContent(prompt);
    console.log("Réponse de synthèse reçue :", result);
    
    // Vérifier si la réponse a été bloquée
    if (!result.response.text()) {
      console.error("Synthèse bloquée par les filtres de sécurité");
      throw new Error("La synthèse a été bloquée par les filtres de sécurité.");
    }

    return result.response.text();
  } catch (error: any) {
    console.error("Erreur détaillée pour la synthèse:", error);
    
    // Personnaliser le message d'erreur
    if (error?.toString().includes("SAFETY")) {
      return "Désolé, la synthèse a été bloquée par les filtres de sécurité. Veuillez reformuler le sujet de manière plus appropriée.";
    } else if (error?.toString().includes("API Key")) {
      return "Erreur d'authentification avec l'API pour la synthèse. Veuillez vérifier la configuration.";
    } else {
      return `Une erreur est survenue lors de la génération de la synthèse. Erreur : ${error.message || error.toString()}`;
    }
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
    console.log("Variables d'environnement disponibles :", {
      POUR_1: process.env.NEXT_PUBLIC_GEMINI_API_KEY_POUR_1?.substring(0, 10),
      POUR_2: process.env.NEXT_PUBLIC_GEMINI_API_KEY_POUR_2?.substring(0, 10),
      CONTRE_1: process.env.NEXT_PUBLIC_GEMINI_API_KEY_CONTRE_1?.substring(0, 10),
      CONTRE_2: process.env.NEXT_PUBLIC_GEMINI_API_KEY_CONTRE_2?.substring(0, 10),
    });

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
    return {
      pour: ["Erreur lors de la génération des arguments pour."],
      contre: ["Erreur lors de la génération des arguments contre."],
      synthese: "Une erreur est survenue lors de la génération du débat. Veuillez réessayer."
    };
  }
} 