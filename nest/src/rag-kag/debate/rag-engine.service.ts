import { Injectable, Inject } from '@nestjs/common';
import { LOGGER_TOKEN } from '../utils/logger-tokens';
import { ILogger } from '../utils/logger-tokens';
import { PromptsService, PromptTemplateType } from '../prompts/prompts.service';
import { ApiProviderFactory } from '../apis/api-provider-factory.service';
import { UserQuery, RagAnalysis, ConfidenceLevel } from '../types';

/**
 * Service pour l'analyse RAG (Retrieval Augmented Generation)
 */
@Injectable()
export class RagEngineService {
  constructor(
    @Inject(LOGGER_TOKEN) private readonly logger: ILogger,
    private readonly promptsService: PromptsService,
    private readonly apiProviderFactory: ApiProviderFactory
  ) {
    this.logger.info('Service RAG Engine initialisé');
  }
  
  /**
   * Effectue une recherche et analyse via RAG
   * @param query Requête utilisateur
   * @returns Analyse RAG
   */
  async retrieveAndAnalyze(query: UserQuery): Promise<RagAnalysis> {
    this.logger.debug('Analyse RAG de la requête', { queryId: query.sessionId });
    
    try {
      // 1. Dans une implémentation réelle, on rechercherait d'abord des documents pertinents
      const relevantDocuments = await this.simulateDocumentRetrieval(query);
      
      // 2. Obtenir le template de prompt RAG
      const ragPrompt = this.promptsService.getPromptTemplate(PromptTemplateType.RAG_ANALYSIS);
      
      // 3. Préparer le prompt avec la requête et les documents
      const fullPrompt = this.prepareRagPrompt(ragPrompt, query, relevantDocuments);
      
      // 4. Appeler le modèle via l'API
      const ragResponse = await this.apiProviderFactory.generateResponse(
        'deepseek', // Utiliser Deepseek pour l'analyse RAG
        fullPrompt,
        {
          temperature: 0.3,
          top_p: 0.85,
          top_k: 40,
          max_tokens: 1500
        }
      );
      
      // 5. Structurer le résultat
      const confidenceScore = 0.85;
      const confidence = this.mapScoreToConfidenceLevel(confidenceScore);
      
      const ragAnalysis: RagAnalysis = {
        content: ragResponse.text,
        sourceType: 'RAG',
        confidenceScore, // Garder pour compatibilité
        confidence,     // Version typée pour le nouveau système
        factualityScore: 0.92,
        relevanceScore: 0.88,
        retrievedDocuments: this.formatRetrievedDocuments(relevantDocuments),
        sourcesUsed: this.extractSourceReferences(relevantDocuments),
        sources: this.extractSourceReferences(relevantDocuments), // Pour compatibilité
        timestamp: new Date(),
        processingTime: ragResponse.usage.processingTime
      };
      
      this.logger.debug('Analyse RAG terminée avec succès', {
        confidence: ragAnalysis.confidence,
        confidenceScore: ragAnalysis.confidenceScore,
        documentCount: ragAnalysis.retrievedDocuments.length
      });
      
      return ragAnalysis;
      
    } catch (error) {
      this.logger.error('Erreur lors de l\'analyse RAG', { error: error.message });
      
      // Retourner une analyse minimale en cas d'erreur
      return {
        content: `Erreur d'analyse RAG: ${error.message}`,
        sourceType: 'RAG',
        confidenceScore: 0,
        confidence: 'LOW' as ConfidenceLevel,
        factualityScore: 0,
        relevanceScore: 0,
        retrievedDocuments: [],
        sourcesUsed: [],
        sources: [],
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
   * Simule la récupération de documents pertinents
   * @param query Requête utilisateur
   * @returns Documents pertinents
   */
  private async simulateDocumentRetrieval(query: UserQuery): Promise<any[]> {
    // Dans une vraie implémentation, on interrogerait une base de connaissances
    
    // Simulation
    return [
      {
        id: 'doc-1',
        title: 'Guide des stratégies commerciales',
        content: 'Contenu simulé sur les stratégies commerciales...',
        relevance: 0.92,
        source: 'Base de connaissances interne'
      },
      {
        id: 'doc-2',
        title: 'Tendances marketing 2023',
        content: 'Contenu simulé sur les tendances marketing...',
        relevance: 0.85,
        source: 'Rapport externe'
      },
      {
        id: 'doc-3',
        title: 'Analyse sectorielle',
        content: 'Contenu simulé sur l\'analyse sectorielle...',
        relevance: 0.78,
        source: 'Étude de marché'
      }
    ];
  }
  
  /**
   * Prépare le prompt RAG complet
   * @param template Template de base
   * @param query Requête utilisateur
   * @param documents Documents récupérés
   * @returns Prompt complet
   */
  private prepareRagPrompt(template: string, query: UserQuery, documents: any[]): string {
    // Dans une implémentation réelle, utiliser un système de templating
    
    // Format simplifié pour les documents
    const docsFormatted = documents
      .map(doc => `[${doc.id}] ${doc.title}: ${doc.content.substring(0, 100)}...`)
      .join('\n');
    
    return template
      .replace('{{query}}', query.text)
      .replace('{{documents}}', docsFormatted);
  }
  
  /**
   * Formate les documents récupérés
   * @param documents Documents bruts
   * @returns Documents formatés
   */
  private formatRetrievedDocuments(documents: any[]): any[] {
    return documents.map(doc => ({
      id: doc.id,
      title: doc.title,
      relevance: doc.relevance,
      source: doc.source
    }));
  }
  
  /**
   * Extrait les références aux sources
   * @param documents Documents utilisés
   * @returns Liste des sources
   */
  private extractSourceReferences(documents: any[]): string[] {
    return documents.map(doc => doc.source);
  }
} 