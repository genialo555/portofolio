import { Injectable, Inject, OnModuleInit, Optional } from '@nestjs/common';
import { LOGGER_TOKEN, ILogger } from '../utils/logger-tokens';
import { GoogleAiService } from './google-ai.service';
import { QwenAiService } from './qwen-ai.service';
import { DeepseekAiService } from './deepseek-ai.service';
import { HouseModelService } from './house-model.service';
import { ApiType } from '../types/index';
import { ResilienceService } from '../utils/resilience.service';
import { AnomalyDetectionService, AnomalyDetectionLevel } from '../../utils/anomaly-detection.service';
import { KnowledgeGraphService, KnowledgeSource, RelationType } from '../core/knowledge-graph.service';
import { EventBusService, RagKagEventType } from '../core/event-bus.service';

/**
 * Type pour les fournisseurs d'API
 */
export type ApiProvider = GoogleAiService | QwenAiService | DeepseekAiService | HouseModelService;

/**
 * Options pour les fournisseurs d'API
 */
export interface ApiProviderOptions {
  provider: ApiType | string;
  fallbackProvider?: ApiType | string;
  detectAnomalies?: boolean;
  anomalyDetectionLevel?: AnomalyDetectionLevel;
  usePerformanceMetrics?: boolean;
}

/**
 * Structure des métriques de performance d'un fournisseur d'API
 */
export interface ProviderPerformanceMetrics {
  provider: string;
  successRate: number;
  averageResponseTime: number;
  averageTokenCost: number;
  anomalyRate: number;
  lastUpdated: Date;
  callCount: number;
  successCount: number;
  failureCount: number;
  totalTokens: number;
  anomalyCount: number;
}

/**
 * Fabrique de fournisseurs d'API
 * Cette classe est responsable de la création et de la gestion des instances de différents fournisseurs d'API
 * Elle intègre des métriques de performance via le graphe de connaissances
 */
@Injectable()
export class ApiProviderFactory implements OnModuleInit {
  private readonly logger: ILogger;
  private performanceMetrics: Map<string, ProviderPerformanceMetrics> = new Map();
  private readonly metricsUpdateThreshold = 5; // Nombre d'appels après lequel on met à jour le graphe
  private readonly defaultMetrics: ProviderPerformanceMetrics = {
    provider: '',
    successRate: 1,
    averageResponseTime: 0,
    averageTokenCost: 0,
    anomalyRate: 0,
    lastUpdated: new Date(),
    callCount: 0,
    successCount: 0,
    failureCount: 0,
    totalTokens: 0,
    anomalyCount: 0
  };

  constructor(
    @Inject(LOGGER_TOKEN) logger: ILogger,
    private readonly googleAiService: GoogleAiService,
    private readonly qwenAiService: QwenAiService,
    private readonly deepseekAiService: DeepseekAiService,
    private readonly houseModelService: HouseModelService,
    private readonly resilienceService: ResilienceService,
    private readonly anomalyDetectionService: AnomalyDetectionService,
    @Optional() private readonly knowledgeGraph?: KnowledgeGraphService,
    @Optional() private readonly eventBus?: EventBusService
  ) {
    this.logger = logger;
    this.logger.info('Fabrique de fournisseurs d\'API initialisée');
  }
  
  /**
   * Initialisation du module
   */
  async onModuleInit() {
    // Charger les métriques de performance depuis le graphe de connaissances
    if (this.knowledgeGraph) {
      await this.loadPerformanceMetricsFromGraph();
    }
    
    // Émettre un événement d'initialisation
    if (this.eventBus) {
      this.eventBus.emit({
        type: RagKagEventType.SYSTEM_INIT,
        source: 'ApiProviderFactory',
        payload: {
          availableProviders: Object.values(ApiType),
          hasPerformanceMetrics: this.knowledgeGraph !== undefined
        }
      });
    }
  }
  
  /**
   * Charge les métriques de performance depuis le graphe de connaissances
   */
  private async loadPerformanceMetricsFromGraph() {
    try {
      this.logger.info('Chargement des métriques de performance depuis le graphe');
      
      const metricsNodes = await this.knowledgeGraph.search('API provider metrics', {
        nodeTypes: ['API_METRICS'],
        maxResults: 50
      });
      
      for (const node of metricsNodes.nodes) {
        const provider = node.metadata.provider;
        
        // Créer un objet de métriques à partir des données du nœud
        const metrics: ProviderPerformanceMetrics = {
          provider,
          successRate: node.metadata.successRate || 1,
          averageResponseTime: node.metadata.averageResponseTime || 0,
          averageTokenCost: node.metadata.averageTokenCost || 0,
          anomalyRate: node.metadata.anomalyRate || 0,
          lastUpdated: new Date(node.metadata.lastUpdated) || new Date(),
          callCount: node.metadata.callCount || 0,
          successCount: node.metadata.successCount || 0,
          failureCount: node.metadata.failureCount || 0,
          totalTokens: node.metadata.totalTokens || 0,
          anomalyCount: node.metadata.anomalyCount || 0
        };
        
        this.performanceMetrics.set(provider, metrics);
        this.logger.debug(`Métriques chargées pour ${provider}`, { metrics });
      }
      
      this.logger.info(`Métriques chargées pour ${this.performanceMetrics.size} fournisseurs`);
    } catch (error) {
      this.logger.error('Erreur lors du chargement des métriques de performance', { error });
    }
  }
  
  /**
   * Stocke les métriques de performance dans le graphe de connaissances
   */
  private async storePerformanceMetricsInGraph(provider: string) {
    if (!this.knowledgeGraph) return;
    
    try {
      const metrics = this.performanceMetrics.get(provider);
      if (!metrics) return;
      
      // Rechercher si un nœud existe déjà pour ce fournisseur
      const existingNodes = await this.knowledgeGraph.search(`API metrics ${provider}`, {
        nodeTypes: ['API_METRICS'],
        maxResults: 1
      });
      
      const now = new Date();
      
      if (existingNodes.nodes.length > 0) {
        // Mise à jour du nœud existant
        // Note: Dans une implémentation réelle, nous aurions besoin d'une API pour mettre à jour les nœuds
        this.logger.debug(`Nœud de métriques trouvé pour ${provider}, mise à jour requise`);
        
        // Dans une implémentation réelle, nous mettrions à jour le nœud existant
        // Ici, nous pourrions supprimer le nœud et en créer un nouveau, mais ce n'est pas optimal
        // Pour simuler une mise à jour, nous allons émettre un événement
        if (this.eventBus) {
          this.eventBus.emit({
            type: RagKagEventType.KNOWLEDGE_GRAPH_UPDATED,
            source: 'ApiProviderFactory',
            payload: {
              nodeType: 'API_METRICS',
              provider,
              metrics,
              operation: 'update'
            }
          });
        }
      } else {
        // Créer un nouveau nœud
        const nodeId = this.knowledgeGraph.addNode({
          label: `API Metrics: ${provider}`,
          type: 'API_METRICS',
          content: `Métriques de performance pour le fournisseur ${provider}`,
          confidence: 1,
          source: KnowledgeSource.SYSTEM,
          metadata: {
            provider,
            successRate: metrics.successRate,
            averageResponseTime: metrics.averageResponseTime,
            averageTokenCost: metrics.averageTokenCost,
            anomalyRate: metrics.anomalyRate,
            lastUpdated: now.toISOString(),
            callCount: metrics.callCount,
            successCount: metrics.successCount,
            failureCount: metrics.failureCount,
            totalTokens: metrics.totalTokens,
            anomalyCount: metrics.anomalyCount
          }
        });
        
        // Créer une relation avec le fournisseur d'API
        this.knowledgeGraph.addFact(
          nodeId,
          'MEASURES',
          {
            label: `API Provider: ${provider}`,
            type: 'API_PROVIDER',
            content: `Fournisseur d'API: ${provider}`,
            confidence: 1,
            source: KnowledgeSource.SYSTEM
          },
          1,
          { bidirectional: true, weight: 1 }
        );
        
        this.logger.debug(`Nœud de métriques créé pour ${provider}`, { nodeId });
        
        // Émettre un événement de création de nœud
        if (this.eventBus) {
          this.eventBus.emit({
            type: RagKagEventType.KNOWLEDGE_NODE_ADDED,
            source: 'ApiProviderFactory',
            payload: {
              nodeId,
              nodeType: 'API_METRICS',
              provider,
              metrics
            }
          });
        }
      }
    } catch (error) {
      this.logger.error(`Erreur lors du stockage des métriques pour ${provider}`, { error });
    }
  }

  /**
   * Récupère le fournisseur d'API en fonction du type
   * @param providerType Type de fournisseur d'API
   * @returns Le fournisseur d'API approprié
   */
  getProvider(providerType: ApiType | string): ApiProvider {
    this.logger.debug('Récupération du fournisseur d\'API', { providerType });

    switch (providerType) {
      case ApiType.GOOGLE_AI:
        return this.googleAiService;
      case ApiType.QWEN_AI:
        return this.qwenAiService;
      case ApiType.DEEPSEEK_AI:
        return this.deepseekAiService;
      case ApiType.HOUSE_MODEL:
        return this.houseModelService;
      default:
        this.logger.warn('Type de fournisseur non supporté, utilisation de Google AI par défaut', { providerType });
        return this.googleAiService;
    }
  }
  
  /**
   * Recommande le meilleur fournisseur d'API en fonction des métriques de performance
   * @param defaultProvider Fournisseur par défaut
   * @param options Options pour la recommandation
   * @returns Le fournisseur d'API recommandé
   */
  recommendProvider(
    defaultProvider: ApiType | string = ApiType.GOOGLE_AI,
    options: {
      prioritizeSpeed?: boolean;
      prioritizeCost?: boolean;
      prioritizeReliability?: boolean;
    } = {}
  ): ApiType | string {
    // Si pas de métriques, retourner le fournisseur par défaut
    if (this.performanceMetrics.size === 0) {
      return defaultProvider;
    }
    
    // Pondérations par défaut
    const weights = {
      speed: options.prioritizeSpeed ? 0.5 : 0.3,
      cost: options.prioritizeCost ? 0.5 : 0.2,
      reliability: options.prioritizeReliability ? 0.5 : 0.3,
      anomalyRate: 0.2
    };
    
    // Normaliser les poids
    const totalWeight = weights.speed + weights.cost + weights.reliability + weights.anomalyRate;
    const normalizedWeights = {
      speed: weights.speed / totalWeight,
      cost: weights.cost / totalWeight,
      reliability: weights.reliability / totalWeight,
      anomalyRate: weights.anomalyRate / totalWeight
    };
    
    // Calculer un score pour chaque fournisseur
    const scores: Record<string, number> = {};
    
    for (const [provider, metrics] of this.performanceMetrics.entries()) {
      // Ne considérer que les fournisseurs avec suffisamment d'appels
      if (metrics.callCount < 5) continue;
      
      // Calculer un score normalisé pour chaque métrique
      const speedScore = 1 - (metrics.averageResponseTime / 5000); // 0 = 5s, 1 = 0s
      const costScore = 1 - (metrics.averageTokenCost / 0.01); // 0 = 0.01$, 1 = 0$
      const reliabilityScore = metrics.successRate;
      const anomalyScore = 1 - metrics.anomalyRate;
      
      // Calculer le score final
      scores[provider] = (
        normalizedWeights.speed * speedScore +
        normalizedWeights.cost * costScore +
        normalizedWeights.reliability * reliabilityScore +
        normalizedWeights.anomalyRate * anomalyScore
      );
    }
    
    // Trouver le fournisseur avec le meilleur score
    let bestProvider = defaultProvider;
    let bestScore = -1;
    
    for (const [provider, score] of Object.entries(scores)) {
      if (score > bestScore) {
        bestScore = score;
        bestProvider = provider;
      }
    }
    
    this.logger.debug('Fournisseur recommandé', { 
      bestProvider, 
      bestScore, 
      scores,
      options
    });
    
    return bestProvider;
  }

  /**
   * Génère une réponse en utilisant le fournisseur d'API spécifié
   * @param provider Type de fournisseur d'API ou options
   * @param prompt Prompt à envoyer
   * @param options Options supplémentaires
   * @returns La réponse générée
   */
  async generateResponse(
    provider: ApiType | string | ApiProviderOptions,
    prompt: string,
    options: any = {}
  ): Promise<any> {
    // Déterminer le provider principal et le fallback
    const providerOptions = typeof provider === 'object' ? provider : { provider };
    const primaryProvider = providerOptions.provider;
    const fallbackProvider = providerOptions.fallbackProvider || ApiType.HOUSE_MODEL; // Par défaut, fallback vers le modèle local
    const detectAnomalies = providerOptions.detectAnomalies !== undefined ? providerOptions.detectAnomalies : true;
    const anomalyDetectionLevel = providerOptions.anomalyDetectionLevel || AnomalyDetectionLevel.MEDIUM_AND_ABOVE;
    const usePerformanceMetrics = providerOptions.usePerformanceMetrics !== undefined ? providerOptions.usePerformanceMetrics : true;
    
    // Utiliser les métriques de performance pour recommander un meilleur fournisseur si nécessaire
    let effectiveProvider = primaryProvider;
    if (usePerformanceMetrics && this.performanceMetrics.size > 0) {
      const recommendedProvider = this.recommendProvider(primaryProvider, {
        prioritizeSpeed: options.prioritizeSpeed,
        prioritizeCost: options.prioritizeCost,
        prioritizeReliability: options.prioritizeReliability
      });
      
      if (recommendedProvider !== primaryProvider) {
        this.logger.info('Utilisation d\'un fournisseur recommandé basé sur les métriques', {
          requestedProvider: primaryProvider,
          recommendedProvider
        });
        effectiveProvider = recommendedProvider;
      }
    }
    
    // Nom du service pour le circuit breaker
    const serviceName = `api-provider-${effectiveProvider}`;
    
    this.logger.info('Génération de réponse via un fournisseur API', { 
      effectiveProvider, 
      originalProvider: primaryProvider !== effectiveProvider ? primaryProvider : undefined,
      fallbackProvider,
      promptLength: prompt.length,
      detectAnomalies,
      anomalyDetectionLevel
    });
    
    // Initialiser ou récupérer les métriques pour ce fournisseur
    if (!this.performanceMetrics.has(effectiveProvider)) {
      this.performanceMetrics.set(effectiveProvider, {
        ...this.defaultMetrics,
        provider: effectiveProvider
      });
    }
    
    const startTime = Date.now();
    let hasAnomalies = false;
    let anomalyCount = 0;
    let tokenCount = 0;
    let success = false;
    
    // Fonction d'exécution principale
    const executionFn = async () => {
      const apiProvider = this.getProvider(effectiveProvider);
      const response = await apiProvider.generateResponse(prompt, options);
      
      // Enregistrer des métriques de base
      success = true;
      tokenCount = response.usage?.totalTokens || (prompt.length + this.extractContent(response).length) / 4;
      
      // Détection d'anomalies (si activée)
      if (detectAnomalies) {
        try {
          // Créer une structure similaire à PoolOutputs pour la détection d'anomalies
          const simulatedPoolOutput = {
            commercial: [],
            marketing: [],
            sectoriel: [],
            educational: [],
            timestamp: new Date(),
            // Ajouter la réponse comme si elle venait d'un agent
            [effectiveProvider.toString().toLowerCase()]: [{
              agentId: `${effectiveProvider}-agent`,
              agentName: `${effectiveProvider} Agent`,
              poolType: 'API',
              content: this.extractContent(response),
              confidence: 0.9,
              timestamp: Date.now(),
              processingTime: response.usage?.processingTime || 0,
              metadata: {
                model: response.model,
                api: effectiveProvider
              }
            }]
          };
          
          const anomalyReport = await this.anomalyDetectionService.detectAnomalies(
            simulatedPoolOutput,
            {
              detectionLevel: anomalyDetectionLevel,
              autoLog: true,
              throwOnHigh: false
            }
          );
          
          // Enrichir la réponse avec des informations sur les anomalies
          if (anomalyReport) {
            anomalyCount = anomalyReport.highPriorityAnomalies.length + 
                         anomalyReport.mediumPriorityAnomalies.length;
            hasAnomalies = anomalyCount > 0;
            
            if (hasAnomalies) {
              this.logger.warn('Anomalies détectées dans la réponse', {
                provider: effectiveProvider,
                anomalyCount
              });
              
              // Ajouter des informations sur les anomalies à la réponse
              const responseAny = response as any;
              responseAny.anomalyInfo = {
                detected: true,
                count: anomalyCount,
                reliability: anomalyReport.overallReliability,
                highPriority: anomalyReport.highPriorityAnomalies.map(a => ({
                  type: a.type,
                  description: a.description,
                  fragment: a.location.contentFragment
                }))
              };
            }
          }
        } catch (anomalyError) {
          this.logger.error('Erreur lors de la détection d\'anomalies', { anomalyError });
          // Ne pas bloquer le processus pour des erreurs de détection d'anomalies
        }
      }
      
      return response;
    };
    
    // Fonction de fallback
    const fallbackFn = async (error: Error) => {
      this.logger.warn(`Fallback vers ${fallbackProvider} suite à une erreur`, {
        originalProvider: effectiveProvider,
        error: error.message
      });
      
      // Mise à jour des métriques pour une exécution échouée
      this.updateMetrics(effectiveProvider, {
        success: false,
        responseTime: Date.now() - startTime,
        tokenCount: 0,
        hasAnomalies: false,
        anomalyCount: 0
      });
      
      try {
        const fallbackApiProvider = this.getProvider(fallbackProvider);
        // Ajouter une indication que c'est une réponse de fallback
        const fallbackOptions = {
          ...options,
          isFallback: true,
          originalProvider: effectiveProvider
        };
        
        const fallbackStartTime = Date.now();
        const fallbackResponse = await fallbackApiProvider.generateResponse(prompt, fallbackOptions);
        
        // Mise à jour des métriques pour le fournisseur de fallback
        const fallbackTokenCount = fallbackResponse.usage?.totalTokens || 
                                  (prompt.length + this.extractContent(fallbackResponse).length) / 4;
        
        this.updateMetrics(fallbackProvider, {
          success: true,
          responseTime: Date.now() - fallbackStartTime,
          tokenCount: fallbackTokenCount,
          hasAnomalies: false, // Pas de détection d'anomalies pour le fallback
          anomalyCount: 0
        });
        
        // Marquer la réponse comme provenant d'un fallback
        const responseAny = fallbackResponse as any;
        responseAny.fromFallback = true;
        responseAny.originalProvider = effectiveProvider;
        
        return fallbackResponse;
      } catch (fallbackError) {
        this.logger.error('Erreur également dans le fournisseur de fallback', {
          fallbackProvider,
          error: fallbackError
        });
        
        // Mise à jour des métriques pour une exécution échouée du fallback
        this.updateMetrics(fallbackProvider, {
          success: false,
          responseTime: Date.now() - startTime,
          tokenCount: 0,
          hasAnomalies: false,
          anomalyCount: 0
        });
        
        throw fallbackError;
      }
    };
    
    // Exécuter avec protection par circuit breaker
    try {
      const response = await this.resilienceService.executeWithCircuitBreaker(
        serviceName,
        executionFn,
        fallbackFn
      );
      
      // Mise à jour des métriques pour une exécution réussie
      if (success) {
        this.updateMetrics(effectiveProvider, {
          success: true,
          responseTime: Date.now() - startTime,
          tokenCount,
          hasAnomalies,
          anomalyCount
        });
      }
      
      return response;
    } catch (error) {
      this.logger.error('Erreur critique lors de la génération', {
        effectiveProvider,
        fallbackProvider,
        error
      });
      
      throw error;
    }
  }
  
  /**
   * Met à jour les métriques de performance pour un fournisseur d'API
   */
  private updateMetrics(
    provider: string, 
    data: {
      success: boolean;
      responseTime: number;
      tokenCount: number;
      hasAnomalies: boolean;
      anomalyCount: number;
    }
  ) {
    const metrics = this.performanceMetrics.get(provider) || {
      ...this.defaultMetrics,
      provider
    };
    
    // Incrémenter les compteurs
    metrics.callCount += 1;
    if (data.success) {
      metrics.successCount += 1;
    } else {
      metrics.failureCount += 1;
    }
    metrics.totalTokens += data.tokenCount;
    if (data.hasAnomalies) {
      metrics.anomalyCount += data.anomalyCount;
    }
    
    // Calculer les métriques moyennes
    metrics.successRate = metrics.successCount / metrics.callCount;
    metrics.averageResponseTime = (metrics.averageResponseTime * (metrics.callCount - 1) + data.responseTime) / metrics.callCount;
    metrics.averageTokenCost = metrics.totalTokens / metrics.callCount * 0.00001; // Coût approximatif par token
    metrics.anomalyRate = metrics.anomalyCount / metrics.callCount;
    metrics.lastUpdated = new Date();
    
    // Mettre à jour les métriques dans la map
    this.performanceMetrics.set(provider, metrics);
    
    // Si on a assez d'appels, mettre à jour le graphe
    if (metrics.callCount % this.metricsUpdateThreshold === 0 && this.knowledgeGraph) {
      this.storePerformanceMetricsInGraph(provider);
    }
    
    // Émettre un événement de métriques
    if (this.eventBus) {
      this.eventBus.emit({
        type: RagKagEventType.PERFORMANCE_METRIC,
        source: 'ApiProviderFactory',
        payload: {
          provider,
          metrics: {
            success: data.success,
            responseTime: data.responseTime,
            tokenCount: data.tokenCount,
            hasAnomalies: data.hasAnomalies,
            anomalyCount: data.anomalyCount
          },
          currentStats: {
            successRate: metrics.successRate,
            averageResponseTime: metrics.averageResponseTime,
            anomalyRate: metrics.anomalyRate,
            callCount: metrics.callCount
          }
        }
      });
    }
  }
  
  /**
   * Récupère les métriques de performance pour tous les fournisseurs
   */
  getPerformanceMetrics(): Record<string, ProviderPerformanceMetrics> {
    const result: Record<string, ProviderPerformanceMetrics> = {};
    
    for (const [provider, metrics] of this.performanceMetrics.entries()) {
      result[provider] = { ...metrics };
    }
    
    return result;
  }
  
  /**
   * Récupère les métriques de performance pour un fournisseur spécifique
   */
  getProviderMetrics(provider: string): ProviderPerformanceMetrics | null {
    const metrics = this.performanceMetrics.get(provider);
    return metrics ? { ...metrics } : null;
  }

  /**
   * Extrait le contenu textuel d'une réponse, quelle que soit sa structure
   * @param response Réponse d'un service API
   * @returns Contenu textuel de la réponse
   */
  private extractContent(response: any): string {
    if (typeof response === 'string') {
      return response;
    }
    
    if (response.response && typeof response.response === 'string') {
      return response.response;
    }
    
    if (response.text && typeof response.text === 'string') {
      return response.text;
    }
    
    return JSON.stringify(response);
  }
} 