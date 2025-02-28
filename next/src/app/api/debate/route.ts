import { GoogleGenerativeAI } from "@google/generative-ai";
import { NextRequest, NextResponse } from "next/server";

// Interface pour la configuration des agents
interface AgentConfig {
  name: string;
  role: "pour" | "contre";
  apiKey: string;
  personality: string;
}

// Fonction pour récupérer les configurations des agents
function getAgentConfigs(): AgentConfig[] {
  // Récupérer les clés API
  const pourApiKey = process.env.GEMINI_API_KEY_POUR || process.env.GEMINI_API_KEY;
  const contreApiKey = process.env.GEMINI_API_KEY_CONTRE || process.env.GEMINI_API_KEY;
  
  if (!pourApiKey || !contreApiKey) {
    throw new Error("Clés API Gemini non configurées");
  }
  
  return [
    {
      name: "Agent Pour 1",
      role: "pour",
      apiKey: pourApiKey,
      personality: "Vous êtes un défenseur optimiste qui mise sur des arguments constructifs et structurés."
    },
    {
      name: "Agent Pour 2",
      role: "pour",
      apiKey: pourApiKey,
      personality: "Vous êtes un soutien pragmatique apportant exemples concrets et données chiffrées."
    },
    {
      name: "Agent Contre 1",
      role: "contre",
      apiKey: contreApiKey,
      personality: "Vous êtes un critique rationnel qui privilégie la logique et les failles potentielles."
    },
    {
      name: "Agent Contre 2",
      role: "contre",
      apiKey: contreApiKey,
      personality: "Vous êtes un analyste pointu qui met en lumière les risques et limites."
    }
  ];
}

// Fonction pour générer le prompt d'un agent
function generateAgentPrompt(agent: AgentConfig, topic: string, context: string = ""): string {
  return `
    ${agent.personality}
    Sujet : ${topic}
    ${context ? `Contexte : ${context}\n` : ""}
    En tant que ${agent.role === "pour" ? "partisan" : "opposant"}, développez votre position :
    1. Exposez clairement votre thèse
    2. Fournissez 2-3 arguments clés
    3. Illustrez avec des exemples ou faits pertinents
    Soyez concis, persuasif et évitez les redondances.
  `;
}

// Fonction pour générer le prompt de synthèse
function generateSynthesisPrompt(topic: string, pourArguments: string[], contreArguments: string[]): string {
  return `
    Rôle : Médiateur impartial
    Sujet : ${topic}
    Arguments Pour :
    ${pourArguments.map((arg, i) => `${i + 1}. ${arg}`).join("\n")}
    Arguments Contre :
    ${contreArguments.map((arg, i) => `${i + 1}. ${arg}`).join("\n")}
    Produisez une synthèse objective :
    1. Résumez les positions principales
    2. Identifiez les convergences possibles
    3. Concluez avec une perspective équilibrée
    Restez neutre et factuel.
  `;
}

// Fonction pour appeler l'API Gemini
async function callGeminiApi(apiKey: string, prompt: string, temperature: number = 0.7, maxTokens: number = 1024): Promise<string> {
  const genAI = new GoogleGenerativeAI(apiKey);
  const model = genAI.getGenerativeModel({
    model: "gemini-pro",
    generationConfig: {
      temperature: temperature,
      maxOutputTokens: maxTokens,
    },
  });
  
  const result = await model.generateContent(prompt);
  const text = result.response.text();
  
  // Vérifier que la réponse n'est pas trop courte
  if (text.length < 50) {
    throw new Error("Réponse trop courte");
  }
  
  return text;
}

// Fonction principale pour gérer les requêtes POST
export async function POST(request: NextRequest): Promise<NextResponse> {
  try {
    // Récupérer les données de la requête
    const { topic, context } = await request.json();
    
    // Valider les paramètres requis
    if (!topic) {
      return NextResponse.json(
        { error: "Paramètre manquant: sujet est requis" },
        { status: 400 }
      );
    }
    
    // Récupérer les configurations des agents
    const agents = getAgentConfigs();
    
    // Générer les réponses des agents
    const pourAgents = agents.filter(agent => agent.role === "pour");
    const contreAgents = agents.filter(agent => agent.role === "contre");
    
    // Générer les arguments pour
    const pourPromises = pourAgents.map(agent => 
      callGeminiApi(agent.apiKey, generateAgentPrompt(agent, topic, context))
    );
    const pourResponses = await Promise.all(pourPromises);
    
    // Générer les arguments contre
    const contrePromises = contreAgents.map(agent => 
      callGeminiApi(agent.apiKey, generateAgentPrompt(agent, topic, context))
    );
    const contreResponses = await Promise.all(contrePromises);
    
    // Récupérer la clé API pour la synthèse
    const syntheseApiKey = process.env.GEMINI_API_KEY_SYNTHESE || process.env.GEMINI_API_KEY;
    if (!syntheseApiKey) {
      throw new Error("Clé API Gemini pour la synthèse non configurée");
    }
    
    // Générer la synthèse
    const synthesePrompt = generateSynthesisPrompt(topic, pourResponses, contreResponses);
    const synthese = await callGeminiApi(syntheseApiKey, synthesePrompt, 0.5, 1500);
    
    // Retourner les résultats
    return NextResponse.json({
      pour: pourResponses,
      contre: contreResponses,
      synthese: synthese
    });
  } catch (error) {
    console.error("Erreur API Débat:", error);
    return NextResponse.json(
      { error: "Échec de la génération du débat" },
      { status: 500 }
    );
  }
} 