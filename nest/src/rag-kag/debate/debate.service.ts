import { Injectable, Inject } from '@nestjs/common';
import { LOGGER_TOKEN } from '../utils/logger-tokens';
import { ILogger } from '../utils/logger-tokens';
import { KagEngineService } from './kag-engine.service';
import { RagEngineService } from './rag-engine.service';
import { PromptsService, PromptTemplateType } from '../prompts/prompts.service';
import { ApiProviderFactory } from '../apis/api-provider-factory.service';
import { UserQuery, DebateInput, DebateResult, KagAnalysis, RagAnalysis, PoolType } from '../types';

/**
 * Service qui gère le débat entre les résultats KAG et RAG
 */
@Injectable()
export class DebateService {
  constructor(
    @Inject(LOGGER_TOKEN) private readonly logger: ILogger,
    private readonly kagEngine: KagEngineService,
    private readonly ragEngine: RagEngineService,
    private readonly promptsService: PromptsService,
    private readonly apiProviderFactory: ApiProviderFactory
  ) {
    this.logger.info('Service de débat initialisé');
  }
  
  /**
   * Génère une analyse KAG pour une requête utilisateur
   * @param query Requête utilisateur
   * @returns Analyse KAG
   */
  async generateKagAnalysis(query: UserQuery): Promise<KagAnalysis> {
    this.logger.debug('Génération de l\'analyse KAG', { queryId: query.sessionId });
    
    try {
      return this.kagEngine.analyzeQuery(query);
    } catch (error) {
      this.logger.error('Erreur lors de l\'analyse KAG', { error: error.message });
      throw error;
    }
  }
  
  /**
   * Génère une analyse RAG pour une requête utilisateur
   * @param query Requête utilisateur
   * @returns Analyse RAG
   */
  async generateRagAnalysis(query: UserQuery): Promise<RagAnalysis> {
    this.logger.debug('Génération de l\'analyse RAG', { queryId: query.sessionId });
    
    try {
      return this.ragEngine.retrieveAndAnalyze(query);
    } catch (error) {
      this.logger.error('Erreur lors de l\'analyse RAG', { error: error.message });
      throw error;
    }
  }
  
  /**
   * Facilite le débat entre les analyses KAG et RAG
   * @param debateInput Données d'entrée pour le débat
   * @returns Résultat du débat
   */
  async facilitateDebate(debateInput: DebateInput): Promise<DebateResult> {
    const { query, kagAnalysis, ragAnalysis, poolOutputs } = debateInput;
    const startTime = Date.now();
    
    this.logger.info('Démarrage du débat dialectique', { 
      queryId: query.sessionId 
    });
    
    try {
      // 1. Obtenir le prompt de débat
      const debatePrompt = this.promptsService.getPromptTemplate(PromptTemplateType.KAG_RAG_DEBATE);
      
      // 2. Préparer le prompt complet
      const fullPrompt = this.prepareFullDebatePrompt(debatePrompt, debateInput);
      
      // 3. Générer le débat via une API (simulé)
      // Dans une implémentation réelle, on utiliserait un LLM spécifique pour ce rôle
      const debateResponse = await this.apiProviderFactory.generateResponse(
        'google', // Modèle choisi pour le débat
        fullPrompt,
        {
          temperature: 0.7,
          top_p: 0.92,
          top_k: 60,
          max_tokens: 2000
        }
      );
      
      // 4. Analyser la réponse (dans une implémentation réelle, on ferait un parsing plus sophistiqué)
      const debateContent = debateResponse.text;
      
      // 5. Évaluer le consensus
      const consensusLevel = this.evaluateConsensusLevel(debateContent);
      
      // 6. Analyser les thèmes identifiés
      const identifiedThemes = this.extractThemes(debateContent);
      
      // 7. Calculer le temps de traitement
      const processingTime = Date.now() - startTime;
      
      // 8. Convertir les clés du poolOutputs en PoolType si nécessaire
      const poolsUsed = poolOutputs && this.convertPoolKeysToPoolTypes(
        Object.keys(poolOutputs).filter(key => 
          key !== 'errors' && 
          key !== 'timestamp' && 
          key !== 'query' && 
          poolOutputs[key]?.length > 0
        )
      );
      
      // 9. Préparer le résultat
      const result: DebateResult = {
        content: debateContent,
        hasConsensus: consensusLevel > 0.6,
        consensusLevel,
        identifiedThemes,
        processingTime,
        sourceMetrics: {
          kagConfidence: kagAnalysis.confidenceScore || this.mapConfidenceLevelToScore(kagAnalysis.confidence),
          ragConfidence: ragAnalysis.confidenceScore || this.mapConfidenceLevelToScore(ragAnalysis.confidence),
          poolsUsed: poolsUsed || [],
          sourcesUsed: ragAnalysis.sourcesUsed || ragAnalysis.sources || []
        },
        debateTimestamp: new Date()
      };
      
      this.logger.info('Débat terminé avec succès', { 
        consensusLevel,
        themeCount: identifiedThemes.length,
        processingTime
      });
      
      return result;
      
    } catch (error) {
      this.logger.error('Erreur lors du débat', { error: error.message });
      
      // Retourner un résultat minimal en cas d'erreur
      return {
        content: `Erreur lors du débat: ${error.message}`,
        hasConsensus: false,
        consensusLevel: 0,
        identifiedThemes: [],
        processingTime: Date.now() - startTime,
        sourceMetrics: {
          kagConfidence: kagAnalysis?.confidenceScore || this.mapConfidenceLevelToScore(kagAnalysis?.confidence),
          ragConfidence: ragAnalysis?.confidenceScore || this.mapConfidenceLevelToScore(ragAnalysis?.confidence),
          poolsUsed: []
        },
        debateTimestamp: new Date(),
        error: error.message
      };
    }
  }
  
  /**
   * Convertit les clés de pool en types de pool
   * @param keys Clés de pool (strings)
   * @returns Types de pool typés
   */
  private convertPoolKeysToPoolTypes(keys: string[]): PoolType[] {
    return keys.map(key => {
      switch (key.toUpperCase()) {
        case 'COMMERCIAL': return PoolType.COMMERCIAL;
        case 'MARKETING': return PoolType.MARKETING;
        case 'SECTORIEL': return PoolType.SECTORIEL;
        default: return null;
      }
    }).filter(Boolean) as PoolType[];
  }
  
  /**
   * Convertit un niveau de confiance typé en score numérique
   * @param level Niveau de confiance 
   * @returns Score de confiance (0-1)
   */
  private mapConfidenceLevelToScore(level: any): number {
    if (!level) return 0;
    
    switch (level) {
      case 'HIGH': return 0.9;
      case 'MEDIUM': return 0.6;
      case 'LOW': return 0.3;
      default: return 0.5;
    }
  }
  
  /**
   * Prépare le prompt complet pour le débat
   * @param template Template de base
   * @param input Données d'entrée
   * @returns Prompt complet
   */
  private prepareFullDebatePrompt(template: string, input: DebateInput): string {
    // Dans une implémentation réelle, on utiliserait un système de templating plus sophistiqué
    
    // Simulé - utiliser le remplacement de placeholder de PromptsService si disponible
    return template
      .replace('{{kagAnalysis}}', JSON.stringify(input.kagAnalysis))
      .replace('{{ragAnalysis}}', JSON.stringify(input.ragAnalysis))
      .replace('{{query}}', input.query.text)
      .replace('{{poolOutputs}}', JSON.stringify(input.poolOutputs));
  }
  
  /**
   * Évalue le niveau de consensus dans le débat
   * @param debateContent Contenu du débat
   * @returns Niveau de consensus (0-1)
   */
  private evaluateConsensusLevel(debateContent: string): number {
    // Dans une implémentation réelle, on utiliserait NLP pour analyser le texte
    
    // Approche simpliste: recherche de mots clés
    const consensusTerms = ['accord', 'consensus', 'alignement', 'converge', 'similaire'];
    const divergenceTerms = ['désaccord', 'divergence', 'conflit', 'oppose', 'contradiction'];
    
    const lowerContent = debateContent.toLowerCase();
    
    const consensusCount = consensusTerms.filter(term => lowerContent.includes(term)).length;
    const divergenceCount = divergenceTerms.filter(term => lowerContent.includes(term)).length;
    
    // Calculer un score approximatif
    if (consensusCount === 0 && divergenceCount === 0) return 0.5; // Neutre
    
    return consensusCount / (consensusCount + divergenceCount);
  }
  
  /**
   * Extrait les thèmes identifiés dans le débat
   * @param debateContent Contenu du débat
   * @returns Liste des thèmes
   */
  private extractThemes(debateContent: string): string[] {
    // Dans une implémentation réelle, on utiliserait des techniques d'extraction de thèmes NLP
    
    // Simulation simpliste
    return [
      'Données commerciales',
      'Stratégie marketing',
      'Tendances sectorielles'
    ];
  }
} 