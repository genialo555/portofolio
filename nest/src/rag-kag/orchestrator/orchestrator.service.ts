import { Injectable, Inject } from '@nestjs/common';
import { LOGGER_TOKEN, ILogger } from '../utils/logger-tokens';
import { RouterService } from './router.service';
import { OutputCollectorService } from './output-collector.service';
import { PoolManagerService } from '../pools/pool-manager.service';
import { DebateService } from '../debate/debate.service';
import { SynthesisService } from '../synthesis/synthesis.service';
import { QueryAnalyzerService } from '../core/query-analyzer.service';
import { EventBusService, RagKagEventType } from '../core/event-bus.service';
import { determineRelevantPools } from '../../legacy/prompts/meta-prompts/orchestrator';
import { CoordinationHandler } from '../../legacy/prompts/meta-prompts/handler';
import { 
  UserQuery, 
  FinalResponse, 
  PoolOutputs, 
  TargetPools,
  DebateResult,
  ExpertiseLevel,
  KagAnalysis, 
  RagAnalysis,
  DebateInput,
  ComponentStatus,
  ConfidenceLevel
} from '../types';
import { UserQuery as CommonUserQuery } from '../../types';
import { AgentType } from '../../legacy/types/agent.types';

interface ProcessingOptions {
  includeSuggestions?: boolean;
  maxLength?: number;
  prioritizeSpeed?: boolean;
  useLegacyRouter?: boolean;
  useAdvancedCoordination?: boolean;
  adaptiveExecution?: boolean;
  executionMode?: 'sequential' | 'parallel' | 'adaptive';
}

/**
 * Adapte une requête utilisateur du format commun au format RAG/KAG
 */
function adaptToRagKagUserQuery(query: CommonUserQuery | string): UserQuery {
  if (typeof query === 'string') {
    return { text: query, content: query };
  }
  
  return {
    ...query,
    content: query.text
  };
}

/**
 * Service d'orchestration des requêtes
 * Utilise l'EventBusService et le QueryAnalyzerService pour une coordination plus efficace
 * Intègre les fonctions legacy pour les cas où c'est nécessaire
 */
@Injectable()
export class OrchestratorService {
  private readonly legacyCoordinationHandler: CoordinationHandler;
  
  constructor(
    @Inject(LOGGER_TOKEN) private readonly logger: ILogger,
    private readonly router: RouterService,
    private readonly poolManager: PoolManagerService,
    private readonly debateService: DebateService,
    private readonly outputCollector: OutputCollectorService,
    private readonly synthesisService: SynthesisService,
    private readonly queryAnalyzer: QueryAnalyzerService,
    private readonly eventBus: EventBusService
  ) {
    this.legacyCoordinationHandler = new CoordinationHandler({
      maxRetries: 2,
      timeoutMs: 60000,
      enableCircuitBreaker: true,
      traceLevel: 'standard'
    });
    
    this.logger.info('OrchestratorService initialisé avec capacités de coordination avancées');
  }

  /**
   * Traite une requête utilisateur complète
   * @param query Requête utilisateur
   * @param expertiseLevel Niveau d'expertise
   * @param options Options supplémentaires
   * @returns Réponse finale structurée
   */
  async processQuery(
    query: UserQuery | CommonUserQuery | string, 
    expertiseLevel: ExpertiseLevel = 'INTERMEDIATE',
    options: ProcessingOptions = {}
  ): Promise<FinalResponse> {
    const startTime = Date.now();
    const adaptedQuery = typeof query === 'string' || 'text' in query 
      ? adaptToRagKagUserQuery(query)
      : query;
    
    const queryContent = typeof adaptedQuery === 'string' 
      ? adaptedQuery 
      : adaptedQuery.text || adaptedQuery.content || '';
    
    this.logger.info(`Traitement de la requête: "${queryContent.substring(0, 50)}${queryContent.length > 50 ? '...' : ''}"`, {
      expertiseLevel,
      prioritizeSpeed: options.prioritizeSpeed,
      useLegacyRouter: options.useLegacyRouter,
      useAdvancedCoordination: options.useAdvancedCoordination
    });
    
    this.eventBus.emit({
      type: RagKagEventType.QUERY_RECEIVED,
      source: 'OrchestratorService',
      payload: { 
        query: queryContent,
        expertiseLevel,
        options
      }
    });
    
    if (options.useAdvancedCoordination) {
      return this.processWithAdvancedCoordination(adaptedQuery, expertiseLevel, options);
    }

    try {
      const analysisResult = await this.queryAnalyzer.analyzeQuery(adaptedQuery);
      
      let targetPools: TargetPools;
      
      if (options.useLegacyRouter) {
        this.logger.debug('Utilisation du routeur legacy pour déterminer les pools cibles');
        const relevanceScores = determineRelevantPools(queryContent);
        
        targetPools = {
          commercial: relevanceScores[AgentType.COMMERCIAL] > 0.3,
          marketing: relevanceScores[AgentType.MARKETING] > 0.3,
          sectoriel: relevanceScores[AgentType.SECTORIEL] > 0.3,
          educational: relevanceScores[AgentType.EDUCATIONAL] > 0.3,
          primaryFocus: this.determinePrimaryFocusFromScores(relevanceScores)
        };
      } else {
        targetPools = await this.router.determineTargetPools(adaptedQuery);
      }

      this.logger.debug('Exécution des agents dans les pools cibles');
      const poolOutputs = await this.poolManager.executeAgents(targetPools, adaptedQuery);
      
      this.logger.debug('Collecte et traitement des sorties des pools');
      const processedOutputs = await this.outputCollector.collectAndProcess(poolOutputs);
      
      this.logger.debug('Démarrage du débat');
      const debateResult = await this.debateService.generateDebate(adaptedQuery, {
        includePoolOutputs: true,
        prioritizeSpeed: options.prioritizeSpeed
      });
      
      this.logger.debug('Génération de la synthèse finale');
      const finalResponse = await this.synthesisService.generateFinalResponse(
        adaptedQuery,
        debateResult,
        {
          expertiseLevel,
          includeSuggestions: options.includeSuggestions !== false,
          maxLength: options.maxLength
        }
      );
      
      this.eventBus.emit({
        type: RagKagEventType.QUERY_PROCESSED,
        source: 'OrchestratorService',
        payload: {
          query: queryContent,
          duration: Date.now() - startTime,
          poolsUsed: Object.entries(targetPools)
            .filter(([key, value]) => value === true && key !== 'primaryFocus')
            .map(([key]) => key.toUpperCase())
        }
      });
      
      return finalResponse;
    } catch (error) {
      this.logger.error(`Erreur lors du traitement de la requête: ${error.message}`, {
        error,
        query: queryContent
      });
      
      this.eventBus.emit({
        type: RagKagEventType.QUERY_ERROR,
        source: 'OrchestratorService',
        payload: {
          error,
          query: queryContent
        }
      });
      
      return {
        content: `Une erreur est survenue lors du traitement de votre requête : ${error.message}`,
        metaData: {
          sourceTypes: ['RAG', 'KAG'],
          confidenceLevel: 'LOW',
          processingTime: Date.now() - startTime,
          usedAgentCount: 0,
          expertiseLevel,
          topicsIdentified: []
        },
        error: error.message
      };
    }
  }
  
  /**
   * Adapte les résultats du legacy au format de réponse final
   */
  private adaptLegacyResultsToFinalResponse(
    results: any,
    expertiseLevel: ExpertiseLevel,
    startTime: number
  ): FinalResponse {
    // Extraire la réponse principale des résultats
    let content = '';
    let topicsIdentified: string[] = [];
    let confidenceLevel: ConfidenceLevel = 'MEDIUM';
    let usedAgentCount = 0;
    
    if (results && results.outputs && results.outputs.response) {
      content = results.outputs.response;
    } else if (results && results.synthesis) {
      content = results.synthesis;
    } else if (results && typeof results === 'string') {
      content = results;
    } else {
      content = 'Aucun résultat n\'a pu être généré avec la coordination avancée.';
      confidenceLevel = 'LOW';
    }
    
    // Extraire les thèmes identifiés si disponibles
    if (results && results.analysis && results.analysis.themes) {
      topicsIdentified = results.analysis.themes;
    } else if (results && results.metadata && results.metadata.themes) {
      topicsIdentified = results.metadata.themes;
    }
    
    // Déterminer le niveau de confiance
    if (results && results.metadata && results.metadata.confidence) {
      const confidence = parseFloat(results.metadata.confidence);
      if (confidence > 0.7) confidenceLevel = 'HIGH';
      else if (confidence > 0.4) confidenceLevel = 'MEDIUM';
      else confidenceLevel = 'LOW';
    }
    
    // Compter les agents utilisés
    if (results && results.metadata && results.metadata.agentsUsed) {
      usedAgentCount = results.metadata.agentsUsed;
    } else if (results && results.agentsUsed) {
      usedAgentCount = results.agentsUsed;
    }
    
    return {
      content,
      metaData: {
        sourceTypes: ['RAG', 'KAG', 'LEGACY'],
        confidenceLevel,
        processingTime: Date.now() - startTime,
        usedAgentCount,
        expertiseLevel,
        topicsIdentified
      },
      suggestedFollowUp: results && results.followUpQuestions 
        ? results.followUpQuestions 
        : []
    };
  }
  
  /**
   * Traite une requête avec le système de coordination avancé du legacy
   */
  private async processWithAdvancedCoordination(
    query: UserQuery,
    expertiseLevel: ExpertiseLevel,
    options: ProcessingOptions
  ): Promise<FinalResponse> {
    const startTime = Date.now();
    const queryText = typeof query === 'string' ? query : query.text || query.content || '';
    
    this.logger.info(`Traitement avec coordination avancée: "${queryText.substring(0, 50)}${queryText.length > 50 ? '...' : ''}"`, {
      executionMode: options.executionMode || 'adaptive'
    });
    
    try {
      // Récupérer l'état actuel du système pour la coordination
      const systemState = await this.getSystemState();
      
      // Créer le contexte de coordination pour le legacy handler
      // Convertir les types pour compatibilité
      const coordinationContext = {
        query: queryText,
        systemState,
        executionMode: this.mapExecutionMode(options.executionMode) as any,
        traceId: `query-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
        timeoutMs: 60000,
        priority: options.prioritizeSpeed ? 2 : 1
      };
      
      // Exécuter le système de coordination legacy
      const coordinationResults = await this.legacyCoordinationHandler.executeCoordination(coordinationContext);
      
      // Transformer les résultats en format de réponse finale
      const finalResponse = this.adaptLegacyResultsToFinalResponse(coordinationResults, expertiseLevel, startTime);
      
      // Émettre un événement de requête traitée
      this.eventBus.emit({
        type: RagKagEventType.QUERY_PROCESSED,
        source: 'OrchestratorService',
        payload: {
          query: queryText,
          duration: Date.now() - startTime,
          coordinationMode: options.executionMode || 'adaptive',
          isLegacy: true
        }
      });
      
      return finalResponse;
    } catch (error) {
      this.logger.error(`Erreur lors de la coordination avancée: ${error.message}`, {
        error,
        query: queryText
      });
      
      this.eventBus.emit({
        type: RagKagEventType.QUERY_ERROR,
        source: 'OrchestratorService',
        payload: {
          error,
          query: queryText,
          stage: 'advanced_coordination'
        }
      });
      
      return {
        content: `Une erreur est survenue lors de la coordination avancée: ${error.message}`,
        metaData: {
          sourceTypes: ['RAG', 'KAG'],
          confidenceLevel: 'LOW',
          processingTime: Date.now() - startTime,
          usedAgentCount: 0,
          expertiseLevel,
          topicsIdentified: []
        },
        error: error.message
      };
    }
  }
  
  private async getSystemState(): Promise<any> {
    return {
      components: {
        router: { status: ComponentStatus.READY },
        poolManager: { status: ComponentStatus.READY },
        debate: { status: ComponentStatus.READY },
        synthesis: { status: ComponentStatus.READY },
        queryAnalyzer: { status: ComponentStatus.READY }
      },
      resources: {
        memory: { usage: 0.5, available: true },
        cpu: { usage: 0.3, available: true },
        models: { usage: 0.4, available: true }
      },
      metrics: {
        lastResponseTimes: {
          router: 120,
          poolManager: 780,
          debate: 2100,
          synthesis: 520,
          queryAnalyzer: 350
        },
        errorRates: {
          router: 0.01,
          poolManager: 0.05,
          debate: 0.02,
          synthesis: 0.01,
          queryAnalyzer: 0.01
        }
      }
    };
  }
  
  private mapExecutionMode(mode?: string): 'sequential' | 'parallel' | 'adaptive' {
    if (!mode) return 'adaptive';
    
    switch (mode) {
      case 'sequential':
        return 'sequential';
      case 'parallel':
        return 'parallel';
      default:
        return 'adaptive';
    }
  }
  
  private determinePrimaryFocusFromScores(relevanceScores: Record<AgentType, number>): 'COMMERCIAL' | 'MARKETING' | 'SECTORIEL' | 'EDUCATIONAL' | undefined {
    let maxScore = 0;
    let primaryType: AgentType | undefined;
    
    for (const [type, score] of Object.entries(relevanceScores)) {
      if (score > maxScore) {
        maxScore = score;
        primaryType = type as AgentType;
      }
    }
    
    if (maxScore < 0.4) {
      return undefined;
    }
    
    return primaryType as 'COMMERCIAL' | 'MARKETING' | 'SECTORIEL' | 'EDUCATIONAL';
  }
} 