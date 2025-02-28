import { Injectable, Inject } from '@nestjs/common';
import { LOGGER_TOKEN } from '../utils/logger-tokens';
import { ILogger } from '../utils/logger-tokens';
import { PromptsService, PromptTemplateType } from '../prompts/prompts.service';
import { ApiProviderFactory } from '../apis/api-provider-factory.service';
import { UserQuery, KagAnalysis, ConfidenceLevel } from '../types';

/**
 * Service pour l'analyse KAG (Knowledge Augmented Generation)
 */
@Injectable()
export class KagEngineService {
  constructor(
    @Inject(LOGGER_TOKEN) private readonly logger: ILogger,
    private readonly promptsService: PromptsService,
    private readonly apiProviderFactory: ApiProviderFactory
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
} 