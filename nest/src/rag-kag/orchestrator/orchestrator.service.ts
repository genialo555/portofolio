import { Inject, Injectable, Optional, forwardRef } from '@nestjs/common';
import { LOGGER_TOKEN } from '../utils/logger-tokens';
import { ILogger } from '../utils/logger-tokens';
import { RouterService } from './router.service';
import { PoolManagerService } from '../pools/pool-manager.service';
import { DebateService } from '../debate/debate.service';
import { OutputCollectorService } from './output-collector.service';
import { SynthesisService } from '../synthesis/synthesis.service';
import { QueryAnalyzerService } from '../core/query-analyzer.service';
import { EventBusService, RagKagEventType } from '../core/event-bus.service';
import { UserQuery as RagKagUserQuery, ExpertiseLevel, TargetPools } from '../types';
import { UserQuery as CommonUserQuery } from '../../types';
import { AgentType } from '../../legacy/types/agent.types';
import { ComplexityAnalyzerService, ComplexityLevel } from '../utils/complexity-analyzer.service';
import { CoordinationHandler } from '../../legacy/prompts/meta-prompts/handler';
import { KagEngineService } from '../debate/kag-engine.service';
import { RagEngineService } from '../debate/rag-engine.service';
import { ApiProviderFactory } from '../apis/api-provider-factory.service';
import { determineRelevantPools } from '../../legacy/prompts/meta-prompts/orchestrator';
import { ResilienceService } from '../utils/resilience.service';

export enum ComponentStatus {
  READY = 'READY',
  INITIALIZING = 'INITIALIZING',
  PROCESSING = 'PROCESSING',
  COMPLETED = 'COMPLETED',
  ERROR = 'ERROR',
  IDLE = 'IDLE'
}

export enum ComponentType {
  QUERY_ANALYZER = 'QUERY_ANALYZER',
  KNOWLEDGE_RETRIEVER = 'KNOWLEDGE_RETRIEVER',
  RAG_ENGINE = 'RAG_ENGINE',
  KAG_ENGINE = 'KAG_ENGINE',
  DEBATE_PROTOCOL = 'DEBATE_PROTOCOL',
  SYNTHESIS = 'SYNTHESIS'
}

export interface SynthesisOptions {
  expertiseLevel: ExpertiseLevel;
  includeSuggestions: boolean;
  maxLength?: number;
  storeInGraph?: boolean;
}

interface ProcessingOptions {
  includeSuggestions?: boolean;
  maxLength?: number;
  prioritizeSpeed?: boolean;
  useLegacyRouter?: boolean;
  useAdvancedCoordination?: boolean;
  adaptiveExecution?: boolean;
  executionMode?: 'sequential' | 'parallel' | 'adaptive';
  useComplexityAnalyzer?: boolean;
}

function adaptToRagKagUserQuery(query: CommonUserQuery | string): RagKagUserQuery {
  if (typeof query === 'string') {
    return { text: query, content: query };
  } else if ('text' in query) {
    return { 
      text: query.text,
      content: query.text,
      contextInfo: query.contextInfo,
      domainHints: query.domainHints,
      timestamp: query.timestamp,
      sessionId: query.sessionId,
      userId: query.userId,
      metadata: query.metadata,
      userType: query.userType
    };
  }
  
  return query as unknown as RagKagUserQuery;
}

@Injectable()
export class OrchestratorService {
  private readonly legacyCoordinationHandler: CoordinationHandler;
  
  private apiProviderFactory?: ApiProviderFactory;
  private ragEngine?: RagEngineService;
  private kagEngine?: KagEngineService;

  constructor(
    @Inject(LOGGER_TOKEN) private readonly logger: ILogger,
    private readonly router: RouterService,
    private readonly poolManager: PoolManagerService,
    private readonly debateService: DebateService,
    private readonly outputCollector: OutputCollectorService,
    private readonly synthesisService: SynthesisService,
    private readonly queryAnalyzer: QueryAnalyzerService,
    private readonly eventBus: EventBusService,
    private readonly complexityAnalyzer: ComplexityAnalyzerService,
    private readonly resilienceService: ResilienceService,
    @Optional() @Inject(forwardRef(() => ApiProviderFactory)) apiProviderFactory?: ApiProviderFactory,
    @Optional() @Inject(forwardRef(() => KagEngineService)) kagEngine?: KagEngineService,
    @Optional() @Inject(forwardRef(() => RagEngineService)) ragEngine?: RagEngineService
  ) {
    this.legacyCoordinationHandler = new CoordinationHandler({
      maxRetries: 3,
      backoffFactor: 1.5,
      timeoutMs: 30000,
      enableCircuitBreaker: true,
      circuitBreakerThreshold: 5,
      enableCache: true,
      cacheTtlMs: 3600000,
      traceLevel: 'standard'
    });
    
    this.apiProviderFactory = apiProviderFactory;
    this.kagEngine = kagEngine;
    this.ragEngine = ragEngine;
    
    this.logger.log('OrchestratorService initialisé avec capacités de coordination avancées', {
      context: 'OrchestratorService'
    });
  }

  async processQuery(
    query: RagKagUserQuery | CommonUserQuery | string, 
    expertiseLevel: ExpertiseLevel = 'INTERMEDIATE',
    options: ProcessingOptions = {}
  ): Promise<any> {
    const startTime = Date.now();
    const adaptedQuery = adaptToRagKagUserQuery(query);
    
    const queryContent = adaptedQuery.content || '';
    
    this.eventBus.emit({
      type: RagKagEventType.QUERY_RECEIVED,
      source: 'OrchestratorService',
      payload: { query: adaptedQuery }
    });

    if (options.useAdvancedCoordination) {
      return this.processWithAdvancedCoordination(adaptedQuery, expertiseLevel, options);
    }
    
    try {
      if (options.useComplexityAnalyzer) {
        const complexity = await this.complexityAnalyzer.analyzeComplexity(adaptedQuery.content, {
          useCachedResults: true,
          quickAnalysis: options.prioritizeSpeed
        });
        
        this.logger.log(`Complexité de la requête: ${complexity.level} (score: ${complexity.score.toFixed(2)})`, {
          context: 'OrchestratorService'
        });
        
        switch (complexity.recommendedPipeline) {
          case 'simple':
            return await this.processSimpleQuery(adaptedQuery, expertiseLevel, options);
          case 'standard':
            return await this.processStandardQuery(adaptedQuery, expertiseLevel, options);
          case 'full':
          default:
            return await this.processFullQuery(adaptedQuery, expertiseLevel, options);
        }
      }
      
      return await this.processFullQuery(adaptedQuery, expertiseLevel, options);
    } catch (error) {
      this.logger.error(`Erreur lors du traitement de la requête: ${error.message}`, {
        context: 'OrchestratorService',
        error
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
        content: `Nous avons rencontré une erreur lors du traitement de votre requête: ${error.message}`,
        metaData: {
          sourceTypes: ['ERROR'],
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

  private adaptLegacyResultsToFinalResponse(
    results: any,
    expertiseLevel: string,
    startTime: number
  ): any {
    let content = '';
    let topicsIdentified: string[] = [];
    let confidenceLevel: string = 'MEDIUM';
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
    
    if (results && results.analysis && results.analysis.themes) {
      topicsIdentified = results.analysis.themes;
    } else if (results && results.metadata && results.metadata.themes) {
      topicsIdentified = results.metadata.themes;
    }
    
    if (results && results.metadata && results.metadata.confidence) {
      const confidence = parseFloat(results.metadata.confidence);
      if (confidence > 0.7) confidenceLevel = 'HIGH';
      else if (confidence > 0.4) confidenceLevel = 'MEDIUM';
      else confidenceLevel = 'LOW';
    }
    
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

  private async processWithAdvancedCoordination(
    query: RagKagUserQuery,
    expertiseLevel: ExpertiseLevel,
    options: ProcessingOptions
  ): Promise<any> {
    const startTime = Date.now();
    const queryText = query.content || '';
    
    this.logger.log(`Traitement avec coordination avancée: "${queryText.substring(0, 50)}${queryText.length > 50 ? '...' : ''}"`, {
      context: 'OrchestratorService'
    });
    
    try {
      const systemState = await this.getSystemState();
      
      const coordinationContext = {
        query: queryText,
        systemState,
        executionMode: this.mapExecutionMode(options.executionMode) as any,
        traceId: `query-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
        timeoutMs: 60000,
        priority: options.prioritizeSpeed ? 2 : 1
      };
      
      const coordinationResults = await this.legacyCoordinationHandler.executeCoordination(coordinationContext);
      
      const finalResponse = this.adaptLegacyResultsToFinalResponse(coordinationResults, expertiseLevel.toString(), startTime);
      
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
    
    Object.entries(relevanceScores).forEach(([type, score]) => {
      if (score > maxScore) {
        maxScore = score;
        primaryType = type as AgentType;
      }
    });
    
    if (maxScore < 0.4) {
      return undefined;
    }
    
    return primaryType as 'COMMERCIAL' | 'MARKETING' | 'SECTORIEL' | 'EDUCATIONAL';
  }

  private async processSimpleQuery(
    query: RagKagUserQuery,
    expertiseLevel: string,
    options: ProcessingOptions
  ): Promise<any> {
    const startTime = Date.now();
    this.logger.log(`Traitement de requête simple: ${query.content}`, {
      context: 'OrchestratorService'
    });
    
    try {
      const apiFact = this.apiProviderFactory || this.getApiProviderFactory();
      
      const response = await apiFact.generateResponse(
        'HOUSE_MODEL',
        `Réponds à la question suivante de manière ${this.adaptExpertiseLevel(expertiseLevel)}: ${query.content}`,
        { 
          temperature: 0.3,
          maxLength: options.maxLength || 500
        }
      );
      
      return {
        content: response.text || response.content || response,
        metaData: {
          sourceTypes: ['HOUSE_MODEL'],
          confidenceLevel: 'MEDIUM',
          processingTime: Date.now() - startTime,
          usedAgentCount: 1,
          expertiseLevel,
          topicsIdentified: []
        },
        suggestedFollowUp: []
      };
    } catch (error) {
      this.logger.error(`Erreur lors du traitement de la requête simple: ${error.message}`, {
        context: 'OrchestratorService',
        error
      });
      
      this.logger.log('Bascule vers le pipeline standard suite à une erreur', {
        context: 'OrchestratorService'
      });
      
      return this.processStandardQuery(query, expertiseLevel, options);
    }
  }

  private async processStandardQuery(
    query: RagKagUserQuery,
    expertiseLevel: string,
    options: ProcessingOptions
  ): Promise<any> {
    const startTime = Date.now();
    this.logger.log(`Traitement de requête standard: ${query.content}`, {
      context: 'OrchestratorService'
    });
    
    try {
      const queryAnalysis = await this.queryAnalyzer.analyzeQuery(query);
      
      const useRag = queryAnalysis.complexity > 0.5 || 
                    queryAnalysis.suggestedComponents.includes(ComponentType.RAG_ENGINE);
      
      let analysis;
      if (useRag) {
        const ragEng = this.ragEngine || this.getRagEngine();
        analysis = await ragEng.generateAnalysis(query);
      } else {
        const kagEng = this.kagEngine || this.getKagEngine();
        analysis = await kagEng.analyzeQuery(query);
      }
      
      const synthesisOptions = {
        expertiseLevel: expertiseLevel as ExpertiseLevel,
        includeSuggestions: options.includeSuggestions || false,
        maxLength: options.maxLength,
        storeInGraph: true
      };
      
      const simulatedDebateResult = {
        content: analysis.content,
        hasConsensus: true,
        identifiedThemes: [],
        processingTime: analysis.processingTime,
        sourceMetrics: {
          kagConfidence: useRag ? undefined : (analysis as any).confidenceScore,
          ragConfidence: useRag ? (analysis as any).confidenceScore : undefined,
          poolsUsed: [],
          sourcesUsed: useRag ? (analysis as any).sources || [] : []
        },
        consensusLevel: 1.0,
        debateTimestamp: new Date()
      };
      
      return await this.synthesisService.generateFinalResponse(
        query,
        simulatedDebateResult,
        synthesisOptions
      );
    } catch (error) {
      this.logger.error(`Erreur lors du traitement de la requête standard: ${error.message}`, {
        context: 'OrchestratorService',
        error
      });
      
      this.logger.log('Bascule vers le pipeline complet suite à une erreur', {
        context: 'OrchestratorService'
      });
      
      return this.processFullQuery(query, expertiseLevel, options);
    }
  }

  private async processFullQuery(
    query: RagKagUserQuery,
    expertiseLevel: string,
    options: ProcessingOptions
  ): Promise<any> {
    const startTime = Date.now();
    this.logger.log(`Traitement de requête complexe avec pipeline complet: ${query.content}`, {
      context: 'OrchestratorService'
    });
    
    try {
      const targetPools = options.useLegacyRouter
        ? this.determinePools(query)
        : await this.router.determineTargetPools(query);
      
      const poolOutputs = await this.poolManager.executeAgents(targetPools, query);
      
      const processedOutputs = await this.outputCollector.collectAndProcess(poolOutputs);
      
      const kagEng = this.kagEngine || this.getKagEngine();
      const kagAnalysis = await kagEng.analyzeQuery(query);
      
      const ragEng = this.ragEngine || this.getRagEngine();
      const ragAnalysis = await ragEng.generateAnalysis(query);
      
      const debateResult = await this.debateService.generateDebate(query, {
        includePoolOutputs: true,
        prioritizeSpeed: options.prioritizeSpeed
      });
      
      const synthesisOptions = {
        expertiseLevel: expertiseLevel as ExpertiseLevel,
        includeSuggestions: options.includeSuggestions !== false,
        maxLength: options.maxLength,
        storeInGraph: true
      };
      
      return await this.synthesisService.generateFinalResponse(
        query,
        debateResult,
        synthesisOptions
      );
    } catch (error) {
      this.logger.error(`Erreur lors du traitement de la requête complète: ${error.message}`, {
        context: 'OrchestratorService',
        error
      });
      
      return {
        content: `Nous avons rencontré une erreur lors du traitement de votre requête: ${error.message}`,
        metaData: {
          sourceTypes: ['ERROR'],
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

  private adaptExpertiseLevel(level: string): string {
    switch (level) {
      case 'BEGINNER': return 'simple et accessible';
      case 'INTERMEDIATE': return 'claire et informative';
      case 'ADVANCED': return 'détaillée et technique';
      default: return 'informative';
    }
  }

  private getApiProviderFactory(): ApiProviderFactory {
    if (this.apiProviderFactory) {
      return this.apiProviderFactory;
    }
    this.logger.warn('ApiProviderFactory not injected, attempting to create a new instance', {
      context: 'OrchestratorService'
    });
    throw new Error('ApiProviderFactory not available');
  }

  private getRagEngine(): RagEngineService {
    if (this.ragEngine) {
      return this.ragEngine;
    }
    this.logger.warn('RagEngine not injected, attempting to create a new instance', {
      context: 'OrchestratorService'
    });
    throw new Error('RagEngine not available');
  }

  private getKagEngine(): KagEngineService {
    if (this.kagEngine) {
      return this.kagEngine;
    }
    this.logger.warn('KagEngine not injected, attempting to create a new instance', {
      context: 'OrchestratorService'
    });
    throw new Error('KagEngine not available');
  }

  private determinePools(query: RagKagUserQuery): TargetPools {
    this.logger.log('Utilisation du routeur legacy pour déterminer les pools cibles', {
      context: 'OrchestratorService'
    });
    const relevanceScores = determineRelevantPools(query.content);
    
    return {
      commercial: relevanceScores[AgentType.COMMERCIAL] > 0.3,
      marketing: relevanceScores[AgentType.MARKETING] > 0.3,
      sectoriel: relevanceScores[AgentType.SECTORIEL] > 0.3,
      educational: relevanceScores[AgentType.EDUCATIONAL] > 0.3,
      primaryFocus: this.determinePrimaryFocusFromScores(relevanceScores)
    };
  }
} 