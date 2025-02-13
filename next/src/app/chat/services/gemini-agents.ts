import { GoogleGenerativeAI } from "@google/generative-ai";

export type AgentRole = "pour" | "contre" | "synthese";

interface AgentConfig {
  name: string;
  role: AgentRole;
  apiKey: string;
  personality: string;
}

// Configuration des agents
export const AGENT_CONFIGS: AgentConfig[] = [
  {
    name: "Agent Pour 1",
    role: "pour",
    apiKey: process.env.GEMINI_API_KEY_POUR_1 || "",
    personality: "Je suis un agent qui défend activement le point de vue favorable, en m'appuyant sur des arguments logiques et des exemples concrets."
  },
  {
    name: "Agent Pour 2",
    role: "pour",
    apiKey: process.env.GEMINI_API_KEY_POUR_2 || "",
    personality: "Je suis un agent qui soutient la position favorable en explorant les avantages et les opportunités potentielles."
  },
  {
    name: "Agent Contre 1",
    role: "contre",
    apiKey: process.env.GEMINI_API_KEY_CONTRE_1 || "",
    personality: "Je suis un agent qui présente des contre-arguments réfléchis et soulève des points de vigilance importants."
  },
  {
    name: "Agent Contre 2",
    role: "contre",
    apiKey: process.env.GEMINI_API_KEY_CONTRE_2 || "",
    personality: "Je suis un agent qui examine de manière critique les potentiels inconvénients et risques à considérer."
  }
];

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
    const genAI = new GoogleGenerativeAI(agent.apiKey);
    const model = genAI.getGenerativeModel({ model: "gemini-pro" });

    const prompt = generateAgentPrompt(agent, topic, context);
    const result = await model.generateContent(prompt);
    const response = result.response;
    return response.text();
  } catch (error) {
    console.error(`Error with agent ${agent.name}:`, error);
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
    // Utiliser une des clés API pour la synthèse
    const genAI = new GoogleGenerativeAI(AGENT_CONFIGS[0].apiKey);
    const model = genAI.getGenerativeModel({ model: "gemini-pro" });

    const prompt = generateSynthesisPrompt(topic, pourArguments, contreArguments);
    const result = await model.generateContent(prompt);
    const response = result.response;
    return response.text();
  } catch (error) {
    console.error("Error generating synthesis:", error);
    throw error;
  }
}

// Fonction pour lancer un débat complet
export async function runDebate(topic: string): Promise<{
  pour: string[];
  contre: string[];
  synthese: string;
}> {
  try {
    // Obtenir les réponses des agents "pour"
    const pourResponses = await Promise.all(
      AGENT_CONFIGS.filter(agent => agent.role === "pour")
        .map(agent => getAgentResponse(agent, topic))
    );

    // Obtenir les réponses des agents "contre"
    const contreResponses = await Promise.all(
      AGENT_CONFIGS.filter(agent => agent.role === "contre")
        .map(agent => getAgentResponse(agent, topic))
    );

    // Générer la synthèse
    const synthese = await getSynthesis(topic, pourResponses, contreResponses);

    return {
      pour: pourResponses,
      contre: contreResponses,
      synthese
    };
  } catch (error) {
    console.error("Error running debate:", error);
    throw error;
  }
} 