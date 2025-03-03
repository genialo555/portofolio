import { Injectable, Inject } from '@nestjs/common';
import { LOGGER_TOKEN, ILogger } from '../utils/logger-tokens';
import { PromptsService, PromptTemplateType } from '../prompts/prompts.service';
import { ApiProviderFactory } from '../apis/api-provider-factory.service';
import { UserQuery, KagAnalysis, ConfidenceLevel, ApiType } from '../types';
import { EventBusService, RagKagEventType } from '../core/event-bus.service';
import { KnowledgeGraphService, KnowledgeSource } from '../core/knowledge-graph.service';

/**
 * Service pour la génération augmentée par connaissance (KAG)
 * Utilise l'EventBusService et le KnowledgeGraphService pour des analyses plus riches
 */
@Injectable()
export class KagEngineService {
  constructor(
    @Inject(LOGGER_TOKEN) private readonly logger: ILogger,
    private readonly promptsService: PromptsService,
    private readonly apiProviderFactory: ApiProviderFactory,
    private readonly eventBus: EventBusService,
    private readonly knowledgeGraph: KnowledgeGraphService
  ) {
    this.logger.info('Service KAG Engine initialisé');
  }
  
  /**
   * Analyse une requête en utilisant les connaissances internes du modèle
   * @param query Requête utilisateur
   * @returns Analyse KAG
   */
  async analyzeQuery(query: UserQuery): Promise<KagAnalysis> {
    this.logger.debug('Analyse KAG de la requête', { queryId: query.sessionId });
    
    try {
      // 1. Obtenir le template de prompt KAG
      const kagPrompt = this.promptsService.getPromptTemplate(PromptTemplateType.KAG_ANALYSIS);
      
      // 2. Préparer le prompt complet avec la requête
      const fullPrompt = this.prepareKagPrompt(kagPrompt, query);
      
      // 3. Appeler le modèle via l'API
      const kagResponse = await this.apiProviderFactory.generateResponse(
        'qwen', // Utiliser Qwen pour l'analyse KAG
        fullPrompt,
        {
          temperature: 0.2, // Température basse pour une analyse factuelle
          top_p: 0.85,
          top_k: 40,
          max_tokens: 1500
        }
      );
      
      // 4. Structurer le résultat (dans une implémentation réelle, on parserait la réponse)
      const confidenceScore = 0.82; // Dans une vraie implémentation, ceci serait calculé
      const confidence = this.mapScoreToConfidenceLevel(confidenceScore);
      
      const kagAnalysis: KagAnalysis = {
        content: kagResponse.text,
        sourceType: 'KAG',
        confidenceScore, // Garder pour compatibilité
        confidence,      // Version typée pour le nouveau système
        factualityScore: 0.78,
        relevanceScore: 0.85,
        keyInsights: this.extractKeyInsights(kagResponse.text),
        knowledgeDomains: this.identifyKnowledgeDomains(kagResponse.text),
        timestamp: new Date(),
        processingTime: kagResponse.usage.processingTime
      };
      
      this.logger.debug('Analyse KAG terminée avec succès', {
        confidence: kagAnalysis.confidence,
        confidenceScore: kagAnalysis.confidenceScore,
        domainsCount: kagAnalysis.knowledgeDomains.length
      });
      
      return kagAnalysis;
      
    } catch (error) {
      this.logger.error('Erreur lors de l\'analyse KAG', { error: error.message });
      
      // Retourner une analyse minimale en cas d'erreur
      return {
        content: `Erreur d'analyse KAG: ${error.message}`,
        sourceType: 'KAG',
        confidenceScore: 0,
        confidence: 'LOW' as ConfidenceLevel,
        factualityScore: 0,
        relevanceScore: 0,
        keyInsights: [],
        knowledgeDomains: [],
        timestamp: new Date(),
        error: error.message,
        processingTime: 0
      };
    }
  }
  
  /**
   * Mappe un score numérique à un niveau de confiance typé
   * @param score Score entre 0 et 1
   * @returns Niveau de confiance
   */
  private mapScoreToConfidenceLevel(score: number): ConfidenceLevel {
    if (score >= 0.8) return 'HIGH';
    if (score >= 0.5) return 'MEDIUM';
    return 'LOW';
  }
  
  /**
   * Prépare le prompt KAG avec la requête
   * @param template Template de base
   * @param query Requête utilisateur
   * @returns Prompt complet
   */
  private prepareKagPrompt(template: string, query: UserQuery): string {
    // Utiliser fillTemplate du service de prompts si disponible
    return template.replace('{{query}}', query.text);
  }
  
  /**
   * Extrait les insights clés de l'analyse KAG
   * @param content Contenu de l'analyse
   * @returns Liste des insights clés
   */
  private extractKeyInsights(content: string): string[] {
    // Dans une implémentation réelle, on utiliserait NLP pour extraire les insights
    
    // Simulé
    return [
      'Connaissance factuelle sur le domaine commercial',
      'Principes de base du marketing stratégique',
      'Tendances générales du secteur'
    ];
  }
  
  /**
   * Identifie les domaines de connaissance dans l'analyse
   * @param content Contenu de l'analyse
   * @returns Liste des domaines de connaissance
   */
  private identifyKnowledgeDomains(content: string): string[] {
    // Dans une implémentation réelle, on utiliserait NLP pour extraire les domaines
    
    // Simulé
    return [
      'Commerce',
      'Marketing',
      'Stratégie d\'entreprise'
    ];
  }

  /**
   * Génère une analyse KAG pour une requête
   * @param query Requête utilisateur
   * @returns Analyse KAG
   */
  async generateAnalysis(query: UserQuery): Promise<KagAnalysis> {
    const startTime = Date.now();
    const text = typeof query === 'string' ? query : ('text' in query ? query.text : (query as any).content || '');
    
    this.logger.debug(`Génération d'une analyse KAG pour: "${text.substring(0, 50)}${text.length > 50 ? '...' : ''}"`, {
      queryLength: text.length
    });
    
    // Émettre un événement de début d'analyse KAG
    this.eventBus.emit({
      type: RagKagEventType.KAG_GENERATION_STARTED,
      source: 'KagEngineService',
      payload: { query: text }
    });

    try {
      // Générer l'analyse avec l'API
      const prompt = this.promptsService.getPromptTemplate(PromptTemplateType.KAG_ANALYSIS);
      const fullPrompt = prompt.replace('{{query}}', text);
      
      const apiResponse = await this.apiProviderFactory.generateResponse(ApiType.HOUSE_MODEL, fullPrompt, {
        temperature: 0.4,
        maxTokens: 1500
      });
      
      // Analyser la réponse pour extraire les métriques
      const confidence = this.evaluateConfidence(apiResponse.response);
      
      // Construire l'analyse KAG
      const analysis: KagAnalysis = {
        content: apiResponse.response,
        processingTime: Date.now() - startTime,
        confidence: this.mapConfidenceScore(confidence),
        confidenceScore: confidence,
        sourceType: 'KAG',
        factualityScore: 0.8, // Valeur par défaut
        relevanceScore: 0.85, // Valeur par défaut
        keyInsights: this.extractKeyInsights(apiResponse.response),
        knowledgeDomains: this.extractDomains(apiResponse.response),
        timestamp: new Date()
      };
      
      // Stocker l'analyse dans le graphe de connaissances
      this.storeAnalysisInGraph(text, analysis);
      
      // Émettre un événement de fin d'analyse KAG
      this.eventBus.emit({
        type: RagKagEventType.KAG_GENERATION_COMPLETED,
        source: 'KagEngineService',
        payload: {
          query: text,
          processingTime: analysis.processingTime,
          confidence: analysis.confidenceScore,
          domains: analysis.knowledgeDomains
        }
      });
      
      return analysis;
    } catch (error) {
      this.logger.error(`Erreur lors de la génération de l'analyse KAG: ${error.message}`, {
        error: error.stack,
        query: text
      });
      
      // Émettre un événement d'erreur
      this.eventBus.emit({
        type: RagKagEventType.QUERY_ERROR,
        source: 'KagEngineService',
        payload: {
          query: text,
          error: error.message
        }
      });
      
      return {
        content: "Désolé, une erreur s'est produite lors de l'analyse des connaissances.",
        processingTime: Date.now() - startTime,
        confidence: 'LOW',
        confidenceScore: 0.2,
        sourceType: 'KAG',
        error: error.message
      };
    }
  }

  /**
   * Évalue la confiance dans l'analyse générée
   * @param content Contenu de l'analyse
   * @returns Score de confiance (0-1)
   */
  private evaluateConfidence(content: string): number {
    // Approche similaire à celle utilisée dans RagEngine
    const positiveIndicators = [
      "selon les connaissances actuelles",
      "il est établi que",
      "les recherches montrent",
      "les experts s'accordent",
      "il est généralement admis"
    ];
    
    const uncertaintyIndicators = [
      "selon certains",
      "il est possible que",
      "peut-être",
      "il semble que",
      "certains pourraient argumenter"
    ];
    
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
    return 0.7 + (positiveScore * 0.3) - (uncertaintyScore * 0.3);
  }

  /**
   * Extrait les domaines de connaissances mentionnés
   * @param content Contenu de l'analyse
   * @returns Liste des domaines
   */
  private extractDomains(content: string): string[] {
    // Domaines potentiels à rechercher
    const potentialDomains = [
      "économie", "finance", "marketing", "commerce", "technologie", 
      "science", "médecine", "santé", "éducation", "histoire",
      "politique", "société", "culture", "art", "sport"
    ];
    
    const domains = new Set<string>();
    
    for (const domain of potentialDomains) {
      if (content.toLowerCase().includes(domain)) {
        domains.add(domain);
      }
    }
    
    return Array.from(domains);
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
   * Stocke l'analyse KAG dans le graphe de connaissances
   * @param query Requête analysée
   * @param analysis Analyse générée
   */
  private storeAnalysisInGraph(query: string, analysis: KagAnalysis): void {
    try {
      // Créer un nœud pour l'analyse
      const analysisNodeId = this.knowledgeGraph.addNode({
        label: `KAG Analysis: ${query.substring(0, 30)}${query.length > 30 ? '...' : ''}`,
        type: 'KAG_ANALYSIS',
        content: analysis.content,
        confidence: analysis.confidenceScore || 0.5,
        source: KnowledgeSource.KAG
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
          'HAS_KAG_ANALYSIS',
          analysisNodeId,
          analysis.confidenceScore || 0.5,
          {
            bidirectional: true,
            weight: 0.9
          }
        );
      }
      
      // Ajouter les domaines de connaissances comme nœuds
      if (analysis.knowledgeDomains) {
        for (const domain of analysis.knowledgeDomains) {
          this.knowledgeGraph.addFact(
            analysisNodeId,
            'COVERS_DOMAIN',
            {
              label: domain,
              type: 'KNOWLEDGE_DOMAIN',
              content: `Domaine de connaissance: ${domain}`,
              confidence: 0.8,
              source: KnowledgeSource.INFERENCE
            },
            0.8,
            {
              bidirectional: false,
              weight: 0.7
            }
          );
        }
      }
      
      this.logger.debug(`Analyse KAG stockée dans le graphe de connaissances`, {
        analysisId: analysisNodeId,
        confidence: analysis.confidenceScore
      });
    } catch (error) {
      this.logger.error(`Erreur lors du stockage de l'analyse KAG dans le graphe: ${error.message}`, {
        error: error.stack
      });
    }
  }
} 