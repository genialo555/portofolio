import { GoogleGenerativeAI } from "@google/generative-ai";
import { NextRequest, NextResponse } from "next/server";

// Fonction pour générer le prompt de synthèse
function generatePrompt(topic: string, pourArguments: string[], contreArguments: string[]): string {
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

// Fonction principale pour gérer les requêtes POST
export async function POST(request: NextRequest): Promise<NextResponse> {
  try {
    // Récupérer les données de la requête
    const { topic, pourArguments, contreArguments } = await request.json();
    
    // Valider les paramètres requis
    if (!topic || !pourArguments || !contreArguments) {
      return NextResponse.json(
        { error: "Paramètres manquants: sujet, arguments pour et contre sont requis" },
        { status: 400 }
      );
    }
    
    // Récupérer la clé API
    const apiKey = process.env.GEMINI_API_KEY_SYNTHESE || process.env.GEMINI_API_KEY;
    if (!apiKey) {
      return NextResponse.json(
        { error: "Clé API Gemini non configurée" },
        { status: 500 }
      );
    }
    
    // Générer le prompt
    const prompt = generatePrompt(topic, pourArguments, contreArguments);
    
    // Initialiser l'API Gemini
    const genAI = new GoogleGenerativeAI(apiKey);
    const model = genAI.getGenerativeModel({
      model: "gemini-pro",
      generationConfig: {
        temperature: 0.5, // Température plus basse pour une synthèse plus factuelle
        maxOutputTokens: 1500,
      },
    });
    
    // Appeler l'API Gemini
    const result = await model.generateContent(prompt);
    const text = result.response.text();
    
    // Vérifier que la réponse n'est pas trop courte
    if (text.length < 100) {
      throw new Error("Réponse trop courte");
    }
    
    // Retourner la réponse
    return NextResponse.json({ text });
  } catch (error) {
    console.error("Erreur API Synthèse:", error);
    return NextResponse.json(
      { error: "Échec de la génération de la synthèse" },
      { status: 500 }
    );
  }
} 