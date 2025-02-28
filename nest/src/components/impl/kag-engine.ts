import { v4 as uuidv4 } from 'uuid';
import { ComponentType, ComponentRegistration } from '../registry';
import { ComponentPriority, ComponentStatus, CoordinationContext } from '../../handlers/coordination-handler';
import { Logger } from '../../utils/logger';
import { AgentType, AgentResponse } from '../../types/agent.types';

/**
 * Configuration du moteur KAG
 */
export interface KagEngineConfig {
  maxGenerationDepth: number;         // Profondeur maximum de génération récursive
  modelProvider: string;              // Fournisseur du modèle de génération
  temperature: number;                // Température pour la génération (créativité)
  maxTokensPerGeneration: number;     // Nombre maximum de tokens par génération
  knowledgeSources: string[];         // Sources de connaissances à consulter
  enhancementTechniques: string[];    // Techniques d'amélioration à appliquer
  reasoningSteps: boolean;            // Inclure les étapes de raisonnement
  factCheckingEnabled: boolean;       // Activer la vérification des faits
  confidenceThreshold: number;        // Seuil de confiance minimal
}

/**
 * Structure d'une connaissance générée
 */
export interface GeneratedKnowledge {
  id: string;                         // Identifiant unique
  content: string;                    // Contenu de la connaissance 
  sourceType: 'inference' | 'synthesis' | 'deduction' | 'extrapolation'; // Type de source
  confidenceScore: number;            // Score de confiance (0-1)
  reasoningPath?: string[];           // Chemin de raisonnement (si activé)
  relatedConcepts: string[];          // Concepts associés
  generationTimestamp: number;        // Timestamp de génération
  metadata?: Record<string, any>;     // Métadonnées additionnelles
}

/**
 * Résultat global du moteur KAG
 */
export interface KagEngineResult {
  originalQuery: string;              // Requête originale
  generatedKnowledge: GeneratedKnowledge[]; // Connaissances générées
  synthesizedResponse: string;        // Réponse synthétisée
  reasoningGraph?: {                 // Graphe de raisonnement (si disponible)
    nodes: Array<{id: string, type: string, content: string}>;
    edges: Array<{source: string, target: string, relationship: string}>;
  };
  confidenceScores: {                 // Scores de confiance détaillés
    overall: number;                  // Score global
    factual: number;                  // Précision factuelle
    relevance: number;                // Pertinence par rapport à la requête
    coherence: number;                // Cohérence du raisonnement
  };
  performance: {                      // Métriques de performance
    totalGenerationTime: number;      // Temps total de génération (ms)
    inferenceSteps: number;           // Nombre d'étapes d'inférence
    tokensGenerated: number;          // Nombre total de tokens générés
  };
}

/**
 * Moteur de Knowledge Augmented Generation
 * Ce composant génère de nouvelles connaissances à partir de
 * raisonnements sur les informations existantes et la requête
 */
export class KagEngine {
  private logger: Logger;
  private config: KagEngineConfig;
  
  // Référentiel de modèles de raisonnement
  private reasoningTemplates: Record<string, string[]> = {
    deduction: [
      "Si {premise1} et {premise2}, alors {conclusion}",
      "Étant donné que {premise}, on peut déduire que {conclusion}",
      "Puisque {observation} est vrai dans ce contexte, {conclusion} doit également être vrai"
    ],
    induction: [
      "D'après plusieurs exemples comme {example1} et {example2}, on peut généraliser que {conclusion}",
      "Le pattern observé dans {observations} suggère que {conclusion}",
      "Sur la base des cas {case1} et {case2}, la règle générale semble être {conclusion}"
    ],
    abduction: [
      "L'observation {observation} pourrait être expliquée par {hypothesis}",
      "La meilleure explication pour {phenomenon} serait {explanation}",
      "Face à {observation}, l'hypothèse la plus plausible est {hypothesis}"
    ],
    analogy: [
      "De la même façon que {knownDomain}, dans {targetDomain} on peut considérer que {conclusion}",
      "Si l'on compare {situation1} avec {situation2}, on peut inférer que {conclusion}",
      "Par analogie avec {reference}, on peut appliquer {principle} à {target}"
    ]
  };
  
  /**
   * Crée une instance du moteur KAG
   * @param logger Instance du logger
   * @param config Configuration du moteur (optionnelle)
   */
  constructor(logger: Logger, config?: Partial<KagEngineConfig>) {
    this.logger = logger;
    
    // Configuration par défaut
    this.config = {
      maxGenerationDepth: 3,
      modelProvider: 'defaultProvider',
      temperature: 0.7,
      maxTokensPerGeneration: 1000,
      knowledgeSources: ['contextual', 'domain-expertise', 'logical-inference'],
      enhancementTechniques: ['conceptual-expansion', 'counterfactual-reasoning', 'constraint-satisfaction'],
      reasoningSteps: true,
      factCheckingEnabled: true,
      confidenceThreshold: 0.6,
      ...config
    };
  }
  
  /**
   * Crée l'enregistrement du composant pour le registre
   * @returns Enregistrement du composant
   */
  public createRegistration(): ComponentRegistration {
    return {
      id: `kag-engine-${uuidv4().substring(0, 8)}`,
      type: ComponentType.KAG_ENGINE,
      name: "Moteur KAG Avancé",
      description: "Génère de nouvelles connaissances par raisonnement et inférence sur les informations existantes",
      version: "1.0.0",
      priority: ComponentPriority.HIGH,
      executeFunction: this.execute.bind(this),
      isEnabled: true
    };
  }

  /**
   * Exécute le moteur KAG sur un contexte donné
   * @param context Contexte de coordination
   * @returns Résultat du moteur KAG
   */
  private async execute(context: CoordinationContext): Promise<KagEngineResult> {
    const startTime = Date.now();
    const query = context.query;
    
    this.logger.debug(`[${context.traceId}] Démarrage du moteur KAG pour: "${query.substring(0, 50)}..."`);
    
    try {
      // 1. Récupérer les informations contextuelles 
      const contextualData = this.extractContextualData(context);
      
      // 2. Identifier les concepts clés et relations pertinentes
      const concepts = this.identifyKeyConcepts(query, contextualData);
      
      // 3. Générer des connaissances par inférence
      const generatedKnowledge = await this.generateKnowledge(query, concepts, contextualData);
      
      // 4. Évaluer la qualité des connaissances générées
      const evaluatedKnowledge = this.evaluateKnowledge(generatedKnowledge);
      
      // 5. Filtrer les connaissances selon le seuil de confiance
      const filteredKnowledge = evaluatedKnowledge.filter(
        k => k.confidenceScore >= this.config.confidenceThreshold
      );
      
      // 6. Synthétiser une réponse à partir des connaissances générées
      const synthesizedResponse = this.synthesizeResponse(filteredKnowledge, query);
      
      // 7. Construire le graphe de raisonnement (si activé)
      const reasoningGraph = this.config.reasoningSteps ? 
        this.buildReasoningGraph(filteredKnowledge) : undefined;
      
      // 8. Calculer les scores de confiance globaux
      const confidenceScores = this.calculateConfidenceScores(filteredKnowledge, query);
      
      // 9. Collecter les métriques de performance
      const endTime = Date.now();
      const performance = {
        totalGenerationTime: endTime - startTime,
        inferenceSteps: filteredKnowledge.length,
        tokensGenerated: this.estimateTokenCount(filteredKnowledge)
      };
      
      // 10. Construire le résultat
      const result: KagEngineResult = {
        originalQuery: query,
        generatedKnowledge: filteredKnowledge,
        synthesizedResponse,
        reasoningGraph,
        confidenceScores,
        performance
      };
      
      this.logger.debug(`[${context.traceId}] Moteur KAG terminé en ${performance.totalGenerationTime}ms, ${filteredKnowledge.length} connaissances générées`);
      
      return result;
      
    } catch (error) {
      this.logger.error(`[${context.traceId}] Erreur dans le moteur KAG: ${error.message}`);
      throw error;
    }
  }

  /**
   * Extrait les données contextuelles pertinentes du contexte de coordination
   * @param context Contexte de coordination
   * @returns Données contextuelles pertinentes
   */
  private extractContextualData(context: CoordinationContext): any {
    const contextualData: any = {};
    
    // Rechercher des résultats d'analyse de requête
    if ((context as any).componentResults) {
      const queryAnalysisResult = Object.values((context as any).componentResults)
        .find((result: any) => result && (result as any).queryType !== undefined);
      
      if (queryAnalysisResult) {
        contextualData.queryAnalysis = queryAnalysisResult;
      }
      
      // Rechercher des résultats RAG
      const ragResult = Object.values((context as any).componentResults)
        .find((result: any) => result && (result as any).contextualKnowledge !== undefined);
        
      if (ragResult) {
        contextualData.retrievedKnowledge = ragResult;
      }
    }
    
    return contextualData;
  }

  /**
   * Identifie les concepts clés et leurs relations
   * @param query Requête utilisateur
   * @param contextualData Données contextuelles
   * @returns Liste des concepts identifiés
   */
  private identifyKeyConcepts(query: string, contextualData: any): string[] {
    const concepts: Set<string> = new Set();
    
    // Extraire des mots-clés de la requête (simplifié)
    const queryWords = query.toLowerCase()
      .replace(/[.,?!;:'"()]/g, '')
      .split(' ')
      .filter(word => word.length > 3);
      
    queryWords.forEach(word => concepts.add(word));
    
    // Ajouter des concepts depuis l'analyse de requête
    if (contextualData.queryAnalysis) {
      const { keywords, entities, domains } = contextualData.queryAnalysis;
      
      if (keywords) keywords.forEach((kw: string) => concepts.add(kw));
      if (entities) entities.forEach((entity: string) => concepts.add(entity));
      if (domains) domains.forEach((domain: string) => concepts.add(domain));
    }
    
    // Ajouter des concepts depuis les connaissances récupérées
    if (contextualData.retrievedKnowledge) {
      const { retrievedDocuments } = contextualData.retrievedKnowledge;
      
      if (retrievedDocuments) {
        retrievedDocuments.forEach((doc: any) => {
          // Extraire des mots-clés du titre
          const titleWords = doc.documentTitle.toLowerCase()
            .replace(/[.,?!;:'"()]/g, '')
            .split(' ')
            .filter((word: string) => word.length > 3);
            
          titleWords.forEach((word: string) => concepts.add(word));
        });
      }
    }
    
    return Array.from(concepts);
  }

  /**
   * Génère des connaissances nouvelles par raisonnement
   * @param query Requête utilisateur
   * @param concepts Concepts identifiés
   * @param contextualData Données contextuelles
   * @returns Liste des connaissances générées
   */
  private async generateKnowledge(
    query: string, 
    concepts: string[], 
    contextualData: any
  ): Promise<GeneratedKnowledge[]> {
    const generatedKnowledge: GeneratedKnowledge[] = [];
    
    // Simuler différents types de génération de connaissances
    
    // 1. Inférence à partir des connaissances récupérées
    if (contextualData.retrievedKnowledge) {
      const inference = this.generateInference(
        contextualData.retrievedKnowledge, 
        query
      );
      
      if (inference) {
        generatedKnowledge.push(inference);
      }
    }
    
    // 2. Synthèse de plusieurs sources
    if (contextualData.retrievedKnowledge && 
        contextualData.retrievedKnowledge.retrievedDocuments && 
        contextualData.retrievedKnowledge.retrievedDocuments.length >= 2) {
      
      const synthesis = this.generateSynthesis(
        contextualData.retrievedKnowledge.retrievedDocuments,
        concepts
      );
      
      if (synthesis) {
        generatedKnowledge.push(synthesis);
      }
    }
    
    // 3. Génération par déduction logique
    const deduction = this.generateDeduction(concepts, query);
    if (deduction) {
      generatedKnowledge.push(deduction);
    }
    
    // 4. Génération par extrapolation
    const extrapolation = this.generateExtrapolation(concepts, contextualData);
    if (extrapolation) {
      generatedKnowledge.push(extrapolation);
    }
    
    return generatedKnowledge;
  }
  
  /**
   * Génère une connaissance par inférence depuis les documents récupérés
   * @param retrievedKnowledge Connaissances récupérées
   * @param query Requête utilisateur
   * @returns Connaissance générée ou null
   */
  private generateInference(retrievedKnowledge: any, query: string): GeneratedKnowledge | null {
    if (!retrievedKnowledge.contextualKnowledge) {
      return null;
    }
    
    // Simuler une inférence à partir du contexte
    const context = retrievedKnowledge.contextualKnowledge;
    const inferenceContent = `En analysant les informations disponibles, on peut inférer que ${
      this.simulateContentGeneration(context, query, 'inference')
    }`;
    
    return {
      id: uuidv4(),
      content: inferenceContent,
      sourceType: 'inference',
      confidenceScore: 0.85,
      reasoningPath: this.config.reasoningSteps ? [
        "Analyse des informations contextuelles",
        "Identification des relations causales",
        "Application du raisonnement inductif"
      ] : undefined,
      relatedConcepts: this.selectRandomElements(query.split(' ').filter(w => w.length > 3), 3),
      generationTimestamp: Date.now()
    };
  }
  
  /**
   * Génère une connaissance par synthèse de plusieurs documents
   * @param documents Documents récupérés
   * @param concepts Concepts identifiés
   * @returns Connaissance générée ou null
   */
  private generateSynthesis(documents: any[], concepts: string[]): GeneratedKnowledge | null {
    if (!documents || documents.length < 2) {
      return null;
    }
    
    // Extraire des snippets de texte
    const snippets = documents.map(doc => doc.snippetText).filter(Boolean);
    
    if (snippets.length < 2) {
      return null;
    }
    
    const synthesisContent = `En combinant plusieurs sources, on peut synthétiser que ${
      this.simulateContentGeneration(snippets.join(" "), concepts.join(" "), 'synthesis')
    }`;
    
    return {
      id: uuidv4(),
      content: synthesisContent,
      sourceType: 'synthesis',
      confidenceScore: 0.78,
      reasoningPath: this.config.reasoningSteps ? [
        "Identification des points communs entre sources",
        "Résolution des contradictions apparentes",
        "Construction d'une vue unifiée"
      ] : undefined,
      relatedConcepts: this.selectRandomElements(concepts, 3),
      generationTimestamp: Date.now()
    };
  }
  
  /**
   * Génère une connaissance par déduction logique
   * @param concepts Concepts identifiés
   * @param query Requête utilisateur
   * @returns Connaissance générée ou null
   */
  private generateDeduction(concepts: string[], query: string): GeneratedKnowledge | null {
    if (concepts.length < 2) {
      return null;
    }
    
    // Sélectionner un modèle de déduction aléatoire
    const deductionTemplates = this.reasoningTemplates.deduction;
    const template = deductionTemplates[Math.floor(Math.random() * deductionTemplates.length)];
    
    // Remplir le template avec des concepts
    let content = template;
    if (content.includes("{premise1}") && content.includes("{premise2}")) {
      const [c1, c2] = this.selectRandomElements(concepts, 2);
      content = content
        .replace("{premise1}", this.capitalizeFirstLetter(c1))
        .replace("{premise2}", c2)
        .replace("{conclusion}", this.simulateContentGeneration(query, [c1, c2].join(" "), 'deduction'));
    } else if (content.includes("{premise}")) {
      const premise = this.selectRandomElements(concepts, 1)[0];
      content = content
        .replace("{premise}", this.capitalizeFirstLetter(premise))
        .replace("{conclusion}", this.simulateContentGeneration(query, premise, 'deduction'));
    } else if (content.includes("{observation}")) {
      const observation = this.selectRandomElements(concepts, 1)[0];
      content = content
        .replace("{observation}", this.capitalizeFirstLetter(observation))
        .replace("{conclusion}", this.simulateContentGeneration(query, observation, 'deduction'));
    }
    
    return {
      id: uuidv4(),
      content,
      sourceType: 'deduction',
      confidenceScore: 0.82,
      reasoningPath: this.config.reasoningSteps ? [
        "Identification des prémisses valides",
        "Application des règles d'inférence",
        "Dérivation de conclusions logiques"
      ] : undefined,
      relatedConcepts: this.selectRandomElements(concepts, 3),
      generationTimestamp: Date.now()
    };
  }
  
  /**
   * Génère une connaissance par extrapolation
   * @param concepts Concepts identifiés
   * @param contextualData Données contextuelles
   * @returns Connaissance générée ou null
   */
  private generateExtrapolation(concepts: string[], contextualData: any): GeneratedKnowledge | null {
    if (concepts.length < 3) {
      return null;
    }
    
    const extrapolationContent = `En extrapolant à partir des tendances observées, on peut anticiper que ${
      this.simulateContentGeneration(concepts.join(" "), JSON.stringify(contextualData).substring(0, 100), 'extrapolation')
    }`;
    
    return {
      id: uuidv4(),
      content: extrapolationContent,
      sourceType: 'extrapolation',
      confidenceScore: 0.68, // Confiance plus faible car extrapolation
      reasoningPath: this.config.reasoningSteps ? [
        "Identification des tendances actuelles",
        "Projection des évolutions possibles",
        "Estimation des scénarios probables"
      ] : undefined,
      relatedConcepts: this.selectRandomElements(concepts, 3),
      generationTimestamp: Date.now(),
      metadata: {
        uncertaintyLevel: 'medium',
        timeHorizon: 'near-term'
      }
    };
  }
  
  /**
   * Simule la génération de contenu pour l'exemple
   * Dans une implémentation réelle, ceci serait remplacé par un appel à un LLM
   * @param input Texte d'entrée
   * @param context Contexte supplémentaire
   * @param generationType Type de génération
   * @returns Texte généré
   */
  private simulateContentGeneration(input: string, context: string, generationType: string): string {
    // Ceci est une simulation simplifiée pour l'exemple
    const generationTemplates: Record<string, string[]> = {
      'inference': [
        "les stratégies marketing les plus efficaces combinent contenu de valeur et personnalisation avancée",
        "l'analyse des données client révèle des patterns comportementaux exploitables pour l'optimisation des conversions",
        "la combinaison d'intelligence artificielle et d'expertise humaine produit les meilleurs résultats commerciaux"
      ],
      'synthesis': [
        "les différentes sources convergent vers l'importance de l'expérience client omnicanale comme facteur clé de succès",
        "l'intégration des technologies émergentes doit s'accompagner d'une refonte des processus métier pour maximiser l'impact",
        "les entreprises adoptant une approche data-driven tout en préservant la dimension humaine surpassent leurs concurrents"
      ],
      'deduction': [
        "cette approche devrait logiquement conduire à une amélioration mesurable des indicateurs de performance",
        "ce principe, appliqué systématiquement, permettrait d'optimiser significativement le retour sur investissement marketing",
        "cette méthode constitue une solution viable aux défis identifiés dans le contexte actuel"
      ],
      'extrapolation': [
        "les prochaines évolutions du marché favoriseront les acteurs ayant investi dans des capacités d'analyse prédictive",
        "la convergence des technologies d'IA et d'automatisation transformera radicalement les stratégies d'acquisition client",
        "l'évolution des comportements consommateurs nécessitera une personnalisation encore plus fine des parcours d'achat"
      ]
    };
    
    // Sélectionner un template aléatoire pour le type de génération
    const templates = generationTemplates[generationType] || generationTemplates['inference'];
    return templates[Math.floor(Math.random() * templates.length)];
  }

  /**
   * Évalue la qualité des connaissances générées
   * @param knowledge Liste des connaissances à évaluer
   * @returns Connaissances avec scores de confiance ajustés
   */
  private evaluateKnowledge(knowledge: GeneratedKnowledge[]): GeneratedKnowledge[] {
    // Dans une implémentation réelle, une évaluation plus sophistiquée serait effectuée
    return knowledge.map(k => {
      // Ajustement du score de confiance en fonction du type de source
      let adjustedScore = k.confidenceScore;
      
      // Les extrapolations sont généralement moins fiables
      if (k.sourceType === 'extrapolation') {
        adjustedScore *= 0.9;
      }
      
      // Vérification des faits si activée
      if (this.config.factCheckingEnabled) {
        // Simuler un processus de vérification des faits
        // Un ajustement aléatoire pour l'exemple, remplacer par une vraie vérification
        const factCheckingAdjustment = Math.random() * 0.2 - 0.1; // -0.1 à +0.1
        adjustedScore = Math.max(0, Math.min(1, adjustedScore + factCheckingAdjustment));
      }
      
      return {
        ...k,
        confidenceScore: adjustedScore
      };
    });
  }

  /**
   * Synthétise une réponse à partir des connaissances générées
   * @param knowledge Connaissances générées
   * @param query Requête originale
   * @returns Réponse synthétisée
   */
  private synthesizeResponse(knowledge: GeneratedKnowledge[], query: string): string {
    if (knowledge.length === 0) {
      return "Aucune connaissance pertinente n'a pu être générée pour cette requête.";
    }
    
    // Trier les connaissances par score de confiance
    const sortedKnowledge = [...knowledge].sort((a, b) => b.confidenceScore - a.confidenceScore);
    
    // Construire une introduction contextuelle
    let response = `En réponse à votre question sur ${query.substring(0, 30)}..., voici une analyse basée sur les connaissances générées:\n\n`;
    
    // Intégrer les connaissances les plus pertinentes
    for (const k of sortedKnowledge) {
      response += `${k.content}\n\n`;
    }
    
    // Ajouter une conclusion
    response += "Cette analyse combine des inférences logiques et des synthèses de sources pertinentes, ";
    response += `avec un niveau de confiance global de ${Math.round(this.calculateAverageConfidence(knowledge) * 100)}%.`;
    
    return response;
  }

  /**
   * Construit un graphe de raisonnement à partir des connaissances
   * @param knowledge Connaissances générées
   * @returns Graphe de raisonnement
   */
  private buildReasoningGraph(knowledge: GeneratedKnowledge[]): KagEngineResult['reasoningGraph'] {
    const nodes: Array<{id: string, type: string, content: string}> = [];
    const edges: Array<{source: string, target: string, relationship: string}> = [];
    
    // Créer un nœud pour chaque connaissance
    knowledge.forEach(k => {
      nodes.push({
        id: k.id,
        type: k.sourceType,
        content: k.content.substring(0, 50) + "..."
      });
      
      // Créer des nœuds pour les concepts liés
      k.relatedConcepts.forEach((concept, index) => {
        const conceptId = `${k.id}-concept-${index}`;
        nodes.push({
          id: conceptId,
          type: 'concept',
          content: concept
        });
        
        // Lier la connaissance au concept
        edges.push({
          source: k.id,
          target: conceptId,
          relationship: 'relates_to'
        });
      });
      
      // Ajouter des étapes de raisonnement si disponibles
      if (k.reasoningPath) {
        k.reasoningPath.forEach((step, index) => {
          const stepId = `${k.id}-step-${index}`;
          nodes.push({
            id: stepId,
            type: 'reasoning_step',
            content: step
          });
          
          // Lier à l'étape précédente ou à la connaissance
          if (index === 0) {
            edges.push({
              source: k.id,
              target: stepId,
              relationship: 'first_step'
            });
          } else {
            edges.push({
              source: `${k.id}-step-${index-1}`,
              target: stepId,
              relationship: 'next_step'
            });
          }
        });
      }
    });
    
    // Créer des liens entre connaissances partageant des concepts
    for (let i = 0; i < knowledge.length; i++) {
      for (let j = i + 1; j < knowledge.length; j++) {
        const k1 = knowledge[i];
        const k2 = knowledge[j];
        
        // Vérifier les concepts communs
        const sharedConcepts = k1.relatedConcepts.filter(c => k2.relatedConcepts.includes(c));
        
        if (sharedConcepts.length > 0) {
          edges.push({
            source: k1.id,
            target: k2.id,
            relationship: `shares_concepts:${sharedConcepts.join(',')}`
          });
        }
      }
    }
    
    return { nodes, edges };
  }

  /**
   * Calcule les scores de confiance détaillés
   * @param knowledge Connaissances générées
   * @param query Requête originale
   * @returns Scores de confiance
   */
  private calculateConfidenceScores(
    knowledge: GeneratedKnowledge[], 
    query: string
  ): KagEngineResult['confidenceScores'] {
    if (knowledge.length === 0) {
      return {
        overall: 0,
        factual: 0,
        relevance: 0,
        coherence: 0
      };
    }
    
    // Calculer le score global (moyenne pondérée)
    const overallScore = this.calculateAverageConfidence(knowledge);
    
    // Calculer le score de précision factuelle
    // Dans une implémentation réelle, ceci impliquerait une vérification plus avancée
    const factualScore = knowledge
      .filter(k => k.sourceType === 'inference' || k.sourceType === 'synthesis')
      .reduce((sum, k) => sum + k.confidenceScore, 0) 
      / knowledge.filter(k => k.sourceType === 'inference' || k.sourceType === 'synthesis').length || 0;
    
    // Calculer le score de pertinence
    // Simuler pour l'exemple
    const relevanceScore = 0.85;
    
    // Calculer le score de cohérence
    // Simuler pour l'exemple
    const coherenceScore = knowledge.length > 1 ? 0.82 : 0.95;
    
    return {
      overall: overallScore,
      factual: factualScore,
      relevance: relevanceScore,
      coherence: coherenceScore
    };
  }

  /**
   * Calcule la moyenne des scores de confiance
   * @param knowledge Connaissances générées
   * @returns Score de confiance moyen
   */
  private calculateAverageConfidence(knowledge: GeneratedKnowledge[]): number {
    if (knowledge.length === 0) {
      return 0;
    }
    return knowledge.reduce((sum, k) => sum + k.confidenceScore, 0) / knowledge.length;
  }

  /**
   * Estime le nombre de tokens générés
   * @param knowledge Connaissances générées
   * @returns Estimation du nombre de tokens
   */
  private estimateTokenCount(knowledge: GeneratedKnowledge[]): number {
    let totalTokens = 0;
    
    knowledge.forEach(k => {
      // Estimation simplifiée: environ 1 token pour 4 caractères
      totalTokens += Math.ceil(k.content.length / 4);
      
      // Ajouter les tokens pour le chemin de raisonnement
      if (k.reasoningPath) {
        totalTokens += Math.ceil(k.reasoningPath.join(' ').length / 4);
      }
    });
    
    return totalTokens;
  }

  /**
   * Sélectionne aléatoirement n éléments d'un tableau
   * @param array Tableau source
   * @param n Nombre d'éléments à sélectionner
   * @returns Éléments sélectionnés
   */
  private selectRandomElements<T>(array: T[], n: number): T[] {
    if (n >= array.length) {
      return [...array];
    }
    
    const result: T[] = [];
    const copy = [...array];
    
    for (let i = 0; i < n; i++) {
      const randomIndex = Math.floor(Math.random() * copy.length);
      result.push(copy[randomIndex]);
      copy.splice(randomIndex, 1);
    }
    
    return result;
  }

  /**
   * Met en majuscule la première lettre d'une chaîne
   * @param str Chaîne à modifier
   * @returns Chaîne avec première lettre en majuscule
   */
  private capitalizeFirstLetter(str: string): string {
    if (!str || str.length === 0) return str;
    return str.charAt(0).toUpperCase() + str.slice(1);
  }
} 