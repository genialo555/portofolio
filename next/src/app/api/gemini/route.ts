import { GoogleGenerativeAI } from "@google/generative-ai";
import { NextRequest, NextResponse } from "next/server";

// Fonction pour récupérer la clé API en fonction du rôle
function getApiKey(role: string): string {
  // Utiliser la clé API appropriée en fonction du rôle
  const apiKey = role === "pour" 
    ? process.env.GEMINI_API_KEY_POUR || process.env.GEMINI_API_KEY 
    : process.env.GEMINI_API_KEY_CONTRE || process.env.GEMINI_API_KEY;
  
  if (!apiKey) {
    throw new Error("Clé API Gemini non configurée");
  }
  
  return apiKey;
}

// Fonction pour générer le prompt en fonction des paramètres
function generatePrompt(role: string, topic: string, context: string, personality: string): string {
  return `
    ${personality}
    Sujet : ${topic}
    ${context ? `Contexte : ${context}\n` : ""}
    En tant que ${role === "pour" ? "partisan" : "opposant"}, développez votre position :
    1. Exposez clairement votre thèse
    2. Fournissez 2-3 arguments clés
    3. Illustrez avec des exemples ou faits pertinents
    Soyez concis, persuasif et évitez les redondances.
  `;
}

// Fonction principale pour gérer les requêtes POST
export async function POST(request: NextRequest): Promise<NextResponse> {
  try {
    // Récupérer les données de la requête
    const { role, topic, context, personality } = await request.json();
    
    // Valider les paramètres requis
    if (!role || !topic) {
      return NextResponse.json(
        { error: "Paramètres manquants: rôle et sujet sont requis" },
        { status: 400 }
      );
    }
    
    // Récupérer la clé API
    const apiKey = getApiKey(role);
    
    // Générer le prompt
    const prompt = generatePrompt(role, topic, context || "", personality || "");
    
    // Initialiser l'API Gemini
    const genAI = new GoogleGenerativeAI(apiKey);
    const model = genAI.getGenerativeModel({
      model: "gemini-pro",
      generationConfig: {
        temperature: 0.7,
        maxOutputTokens: 1024,
      },
    });
    
    // Appeler l'API Gemini
    const result = await model.generateContent(prompt);
    const text = result.response.text();
    
    // Vérifier que la réponse n'est pas trop courte
    if (text.length < 50) {
      throw new Error("Réponse trop courte");
    }
    
    // Retourner la réponse
    return NextResponse.json({ text });
  } catch (error) {
    console.error("Erreur API Gemini:", error);
    return NextResponse.json(
      { error: "Échec de la génération de contenu" },
      { status: 500 }
    );
  }
} 