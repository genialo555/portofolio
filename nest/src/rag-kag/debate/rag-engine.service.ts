import { Injectable, Inject } from '@nestjs/common';
import { LOGGER_TOKEN, ILogger } from '../utils/logger-tokens';
import { PromptsService, PromptTemplateType } from '../prompts/prompts.service';
import { ApiProviderFactory } from '../apis/api-provider-factory.service';
import { UserQuery, RagAnalysis, ConfidenceLevel, ApiType } from '../types';
import { KnowledgeGraphService, GraphSearchOptions, GraphSearchResult, KnowledgeSource } from '../core/knowledge-graph.service';
import { EventBusService, RagKagEventType } from '../core/event-bus.service';

/**
 * Service pour la génération augmentée par récupération (RAG)
 * Combiné avec le KnowledgeGraphService pour une récupération plus précise
 */
@Injectable()
export class RagEngineService {
  constructor(
    @Inject(LOGGER_TOKEN) private readonly logger: ILogger,
    private readonly promptsService: PromptsService,
    private readonly apiProviderFactory: ApiProviderFactory,
    private readonly knowledgeGraph: KnowledgeGraphService,
    private readonly eventBus: EventBusService
  ) {
    this.logger.info('Service RAG Engine initialisé');
  }
  
  /**
   * Génère une analyse RAG pour une requête
   * @param query Requête utilisateur
   * @returns Analyse RAG
   */
  async generateAnalysis(query: UserQuery): Promise<RagAnalysis> {
    const startTime = Date.now();
    const text = typeof query === 'string' ? query : query.text;
    
    this.logger.debug(`Génération d'une analyse RAG pour: "${text.substring(0, 50)}${text.length > 50 ? '...' : ''}"`, {
      queryLength: text.length
    });
    
    // Émettre un événement de début d'analyse RAG
    this.eventBus.emit({
      type: RagKagEventType.RAG_RETRIEVAL_STARTED,
      source: 'RagEngineService',
      payload: { query: text }
    });

    try {
      // Rechercher dans le graphe de connaissances
      const graphResults = await this.searchKnowledgeGraph(text);
      
      // Construire un contexte à partir des résultats du graphe
      const graphContext = this.buildContextFromGraphResults(graphResults, text);
      
      // Générer l'analyse avec l'API
      const apiResponse = await this.apiProviderFactory.generateResponse(ApiType.GOOGLE_AI, `
Vous êtes un expert en analyse de sources d'information. Analysez le contenu suivant et fournissez une réponse structurée.

REQUÊTE: ${text}

CONTEXTE RÉCUPÉRÉ:
${graphContext}

Veuillez fournir:
1. Une analyse factuelle basée sur le contexte fourni
2. Une évaluation de la pertinence et de la qualité des sources
3. Les concepts clés identifiés dans l'information récupérée
4. Les lacunes d'information éventuelles

Répondez de manière factuelle, concise et structurée.
      `);
      
      // Analyser la réponse pour extraire les métriques
      const confidence = this.evaluateConfidence(apiResponse.response);
      
      // Construire l'analyse RAG
      const analysis: RagAnalysis = {
        content: apiResponse.response,
        sources: graphResults.nodes.map(node => node.label),
        processingTime: Date.now() - startTime,
        confidence: this.mapConfidenceScore(confidence),
        confidenceScore: confidence,
        sourceType: 'RAG',
        factualityScore: this.evaluateFactuality(apiResponse.response, graphResults),
        relevanceScore: this.evaluateRelevance(apiResponse.response, text),
        retrievedDocuments: graphResults.nodes.map(node => ({
          id: node.id,
          content: node.content,
          type: node.type,
          confidence: node.confidence
        })),
        sourcesUsed: ['knowledge_graph', 'google_ai'],
        timestamp: new Date()
      };
      
      // Stocker les résultats dans le graphe de connaissances
      this.storeAnalysisInGraph(text, analysis);
      
      // Émettre un événement de fin d'analyse RAG
      this.eventBus.emit({
        type: RagKagEventType.RAG_RETRIEVAL_COMPLETED,
        source: 'RagEngineService',
        payload: {
          query: text,
          processingTime: analysis.processingTime,
          confidence: analysis.confidenceScore,
          nodesRetrieved: graphResults.nodes.length
        }
      });
      
      return analysis;
    } catch (error) {
      this.logger.error(`Erreur lors de la génération de l'analyse RAG: ${error.message}`, {
        error: error.stack,
        query: text
      });
      
      // Émettre un événement d'erreur
      this.eventBus.emit({
        type: RagKagEventType.QUERY_ERROR,
        source: 'RagEngineService',
        payload: {
          query: text,
          error: error.message
        }
      });
      
      return {
        content: "Désolé, une erreur s'est produite lors de l'analyse des informations récupérées.",
        sources: [],
        processingTime: Date.now() - startTime,
        confidence: 'LOW',
        confidenceScore: 0.2,
        sourceType: 'RAG',
        error: error.message
      };
    }
  }
  
  /**
   * Recherche dans le graphe de connaissances
   * @param query Texte de la requête
   * @returns Résultats de la recherche
   */
  private async searchKnowledgeGraph(query: string): Promise<GraphSearchResult> {
    const options: GraphSearchOptions = {
      maxDepth: 2,
      minConfidence: 0.3,
      sortByRelevance: true,
      maxResults: 15,
      includeRelations: true
    };
    
    return this.knowledgeGraph.search(query, options);
  }
  
  /**
   * Construit un contexte à partir des résultats du graphe
   * @param results Résultats de la recherche dans le graphe
   * @param query Requête originale
   * @returns Contexte formaté
   */
  private buildContextFromGraphResults(results: GraphSearchResult, query: string): string {
    if (results.nodes.length === 0) {
      return "Aucune information pertinente n'a été trouvée dans le graphe de connaissances.";
    }
    
    // Trier les nœuds par score de pertinence
    const sortedNodes = [...results.nodes].sort((a, b) => {
      const scoreA = results.relevanceScores?.get(a.id) || 0;
      const scoreB = results.relevanceScores?.get(b.id) || 0;
      return scoreB - scoreA;
    });
    
    // Extraire le texte contextuel des nœuds les plus pertinents
    let context = `Informations récupérées pour la requête "${query}":\n\n`;
    
    for (let i = 0; i < sortedNodes.length; i++) {
      const node = sortedNodes[i];
      const relevanceScore = results.relevanceScores?.get(node.id) || 0;
      
      if (relevanceScore < 0.2) {
        continue; // Ignorer les nœuds peu pertinents
      }
      
      context += `[Source ${i + 1}] ${node.label} (confiance: ${(node.confidence * 100).toFixed(0)}%, pertinence: ${(relevanceScore * 100).toFixed(0)}%)\n`;
      context += `Type: ${node.type}\n`;
      context += `Contenu: ${node.content}\n\n`;
    }
    
    return context;
  }
  
  /**
   * Évalue la confiance dans l'analyse générée
   * @param content Contenu de l'analyse
   * @returns Score de confiance (0-1)
   */
  private evaluateConfidence(content: string): number {
    // Dans une implémentation réelle, on utiliserait un modèle plus sophistiqué
    // Indicateurs positifs
    const positiveIndicators = [
      "clairement indiqué",
      "selon les sources",
      "comme démontré par",
      "comme illustré par",
      "d'après l'analyse",
      "les données montrent"
    ];
    
    // Indicateurs de faible confiance
    const uncertaintyIndicators = [
      "il est possible que",
      "peut-être",
      "il semble que",
      "il n'est pas clair",
      "information limitée",
      "manque de données",
      "incertain",
      "difficile à déterminer"
    ];
    
    // Calculer le score
    let positiveCount = 0;
    let uncertaintyCount = 0;
    
    for (const indicator of positiveIndicators) {
      if (content.toLowerCase().includes(indicator)) {
        positiveCount++;
      }
    }
    
    for (const indicator of uncertaintyIndicators) {
      if (content.toLowerCase().includes(indicator)) {
        uncertaintyCount++;
      }
    }
    
    // Normaliser les scores
    const positiveScore = Math.min(positiveCount / positiveIndicators.length, 1);
    const uncertaintyScore = Math.min(uncertaintyCount / uncertaintyIndicators.length, 1);
    
    // Calculer le score final (0-1)
    return 0.5 + (positiveScore * 0.5) - (uncertaintyScore * 0.4);
  }
  
  /**
   * Évalue la factualité de l'analyse générée
   * @param content Contenu de l'analyse
   * @param graphResults Résultats du graphe de connaissances
   * @returns Score de factualité (0-1)
   */
  private evaluateFactuality(content: string, graphResults: GraphSearchResult): number {
    if (graphResults.nodes.length === 0) {
      return 0.3; // Faible score si aucune source
    }
    
    // Calculer la confiance moyenne des nœuds
    const avgNodeConfidence = graphResults.nodes.reduce((sum, node) => sum + node.confidence, 0) / graphResults.nodes.length;
    
    // Vérifier que le contenu fait référence aux sources
    const mentionsSourcesScore = content.includes("source") || content.includes("référence") ? 0.3 : 0;
    
    // Score de base basé sur le nombre de sources
    const sourceCountScore = Math.min(graphResults.nodes.length / 10, 0.3);
    
    // Score final
    return avgNodeConfidence * 0.4 + sourceCountScore + mentionsSourcesScore;
  }
  
  /**
   * Évalue la relevance de l'analyse générée par rapport à la requête
   * @param content Contenu de l'analyse
   * @param query Requête originale
   * @returns Score de relevance (0-1)
   */
  private evaluateRelevance(content: string, query: string): number {
    const queryKeywords = new Set(
      query.toLowerCase()
        .replace(/[.,!?;:()"']/g, ' ')
        .split(/\s+/)
        .filter(word => word.length > 3)
    );
    
    // Compter les mots-clés de la requête présents dans le contenu
    let keywordMatches = 0;
    
    for (const keyword of queryKeywords) {
      if (content.toLowerCase().includes(keyword)) {
        keywordMatches++;
      }
    }
    
    // Calculer le score de correspondance des mots-clés
    const keywordScore = queryKeywords.size > 0 ? keywordMatches / queryKeywords.size : 0;
    
    // Facteur de longueur - une réponse trop courte est probablement moins pertinente
    const lengthFactor = Math.min(content.length / 500, 1) * 0.3;
    
    // Vérifier si la réponse contient des concepts clés liés à la requête
    const directAnswerIndicators = [
      "en réponse à votre question",
      "pour répondre à",
      "concernant votre demande",
      "en ce qui concerne"
    ];
    
    let directAnswerScore = 0;
    for (const indicator of directAnswerIndicators) {
      if (content.toLowerCase().includes(indicator)) {
        directAnswerScore = 0.2;
        break;
      }
    }
    
    // Score final
    return keywordScore * 0.5 + lengthFactor + directAnswerScore;
  }
  
  /**
   * Convertit un score de confiance numérique en niveau de confiance
   * @param score Score numérique (0-1)
   * @returns Niveau de confiance
   */
  private mapConfidenceScore(score: number): ConfidenceLevel {
    if (score >= 0.7) {
      return 'HIGH';
    } else if (score >= 0.4) {
      return 'MEDIUM';
    } else {
      return 'LOW';
    }
  }
  
  /**
   * Stocke l'analyse RAG dans le graphe de connaissances
   * @param query Requête analysée
   * @param analysis Analyse générée
   */
  private storeAnalysisInGraph(query: string, analysis: RagAnalysis): void {
    try {
      // Créer un nœud pour l'analyse
      const analysisNodeId = this.knowledgeGraph.addNode({
        label: `RAG Analysis: ${query.substring(0, 30)}${query.length > 30 ? '...' : ''}`,
        type: 'RAG_ANALYSIS',
        content: analysis.content,
        confidence: analysis.confidenceScore || 0.5,
        source: KnowledgeSource.RAG
      });
      
      // Rechercher le nœud de requête existant
      const graphResults = this.knowledgeGraph.search(query, {
        nodeTypes: ['QUERY'],
        maxResults: 1,
        maxDepth: 0,
        minConfidence: 0
      });
      
      // Lier l'analyse à la requête si elle existe
      if (graphResults.nodes.length > 0) {
        const queryNodeId = graphResults.nodes[0].id;
        
        this.knowledgeGraph.addFact(
          queryNodeId,
          'HAS_RAG_ANALYSIS',
          analysisNodeId,
          analysis.confidenceScore || 0.5,
          {
            bidirectional: true,
            weight: 0.9
          }
        );
      }
      
      this.logger.debug(`Analyse RAG stockée dans le graphe de connaissances`, {
        analysisId: analysisNodeId,
        confidence: analysis.confidenceScore
      });
    } catch (error) {
      this.logger.error(`Erreur lors du stockage de l'analyse RAG dans le graphe: ${error.message}`, {
        error: error.stack
      });
    }
  }
} 