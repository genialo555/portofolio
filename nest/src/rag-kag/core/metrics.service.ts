import { Injectable, Inject, OnModuleInit } from '@nestjs/common';
import { LOGGER_TOKEN, ILogger } from '../utils/logger-tokens';
import { EventBusService, RagKagEventType, EventListener } from './event-bus.service';
import { KnowledgeGraphService, KnowledgeSource, RelationType } from './knowledge-graph.service';
import { ApiType } from '../types';
import { v4 as uuidv4 } from 'uuid';

/**
 * Types de métriques supportés
 */
export enum MetricType {
  LATENCY = 'LATENCY',                   // Temps de réponse
  TOKENS = 'TOKENS',                     // Tokens utilisés
  SUCCESS_RATE = 'SUCCESS_RATE',         // Taux de succès
  ANOMALIES = 'ANOMALIES',               // Nombre d'anomalies
  CPU_USAGE = 'CPU_USAGE',               // Utilisation CPU
  MEMORY_USAGE = 'MEMORY_USAGE',         // Utilisation mémoire
  MODEL_ACCURACY = 'MODEL_ACCURACY',     // Précision des modèles
  CIRCUIT_BREAKER = 'CIRCUIT_BREAKER',   // État des circuit breakers
  API_CALLS = 'API_CALLS',               // Nombre d'appels API
  POOL_EXECUTIONS = 'POOL_EXECUTIONS',   // Exécutions des pools
  COST = 'COST'                          // Coût estimé
}

/**
 * Périodes d'agrégation disponibles
 */
export enum AggregationPeriod {
  MINUTE = 60 * 1000,
  HOUR = 60 * 60 * 1000,
  DAY = 24 * 60 * 60 * 1000,
  WEEK = 7 * 24 * 60 * 60 * 1000
}

/**
 * Opérations d'agrégation disponibles
 */
export enum AggregationOperation {
  AVG = 'AVG',
  MIN = 'MIN',
  MAX = 'MAX',
  SUM = 'SUM',
  COUNT = 'COUNT'
}

/**
 * Structure d'une métrique
 */
export interface Metric {
  id: string;                // Identifiant unique
  type: MetricType;          // Type de métrique
  value: number;             // Valeur de la métrique
  tags: Record<string, string>; // Tags pour la classification
  timestamp: number;         // Horodatage
  source: string;            // Source de la métrique
}

/**
 * Options pour la récupération des métriques
 */
export interface MetricsQueryOptions {
  startTime?: number;        // Début de la période
  endTime?: number;          // Fin de la période
  types?: MetricType[];      // Types de métriques
  tags?: Record<string, string>; // Filtrer par tags
  sources?: string[];        // Sources des métriques
  aggregation?: {
    period: AggregationPeriod;  // Période d'agrégation
    operation: AggregationOperation; // Opération d'agrégation
  };
  limit?: number;            // Nombre maximum de résultats
}

/**
 * Options pour les alertes
 */
export interface AlertRule {
  id: string;                // Identifiant unique
  metricType: MetricType;    // Type de métrique
  condition: {
    operator: '>' | '<' | '==' | '>=' | '<='; // Opérateur de comparaison
    threshold: number;       // Seuil
  };
  tags?: Record<string, string>; // Filtrer par tags
  windowPeriod: number;      // Période d'évaluation (ms)
  cooldown: number;          // Période de refroidissement après alerte (ms)
  severity: 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL'; // Sévérité de l'alerte
  description: string;       // Description de l'alerte
  enabled: boolean;          // Règle activée ou non
}

/**
 * Service pour la collecte et le stockage des métriques
 */
@Injectable()
export class MetricsService implements OnModuleInit {
  private metrics: Metric[] = [];
  private readonly retentionPeriod = 7 * 24 * 60 * 60 * 1000; // 7 jours
  private readonly persistenceInterval = 5 * 60 * 1000; // 5 minutes
  private readonly maxInMemoryMetrics = 10000;
  private alertRules: AlertRule[] = [];
  private alertHistory: Array<{
    ruleId: string;
    timestamp: number;
    value: number;
  }> = [];
  private subscriptionIds: string[] = [];
  
  constructor(
    @Inject(LOGGER_TOKEN) private readonly logger: ILogger,
    private readonly eventBus: EventBusService,
    private readonly knowledgeGraph: KnowledgeGraphService
  ) {}
  
  /**
   * Initialisation du service
   */
  async onModuleInit() {
    this.logger.info('Initialisation du service de métriques');
    
    // Création ou récupération du nœud racine pour les métriques
    await this.ensureMetricsRootNode();
    
    // Abonnement aux événements de métriques
    this.subscribeToEvents();
    
    // Initialisation des règles d'alerte par défaut
    this.initDefaultAlertRules();
    
    // Démarrage des tâches périodiques
    this.startPeriodicTasks();
    
    // Configure des alertes sur les métriques de performance clés
    this.createPerformanceAlerts();
    
    // Notifier que le service est prêt
    this.eventBus.emit({
      type: RagKagEventType.SYSTEM_INIT,
      source: 'MetricsService',
      payload: {
        metricTypes: Object.values(MetricType),
        alertRules: this.alertRules.length
      }
    });
  }
  
  /**
   * S'assure que le nœud racine pour les métriques existe dans le graphe
   */
  private async ensureMetricsRootNode() {
    try {
      // Rechercher si le nœud racine existe déjà
      const searchResults = await this.knowledgeGraph.search('metrics root', {
        nodeTypes: ['METRICS_ROOT'],
        maxResults: 1
      });
      
      if (searchResults.nodes.length === 0) {
        // Créer le nœud racine s'il n'existe pas
        const nodeId = this.knowledgeGraph.addNode({
          label: 'Metrics Root',
          type: 'METRICS_ROOT',
          content: 'Nœud racine pour toutes les métriques du système',
          confidence: 1,
          source: KnowledgeSource.SYSTEM,
          metadata: {
            createdAt: Date.now(),
            description: 'Racine de l\'arborescence des métriques système'
          }
        });
        
        this.logger.info('Nœud racine des métriques créé', { nodeId });
      } else {
        this.logger.debug('Nœud racine des métriques trouvé');
      }
    } catch (error) {
      this.logger.error('Erreur lors de la vérification du nœud racine des métriques', { error });
    }
  }
  
  /**
   * Abonnement aux événements système pour collecter les métriques
   */
  private subscribeToEvents() {
    // Métrique de performance
    const performanceMetricListener: EventListener = (event) => {
      if (event.payload) {
        this.processPerformanceMetric(event.source, event.payload);
      }
    };
    
    // Circuit breakers
    const circuitBreakerListener: EventListener = (event) => {
      if (event.payload && event.payload.serviceName) {
        this.recordMetric({
          type: MetricType.CIRCUIT_BREAKER,
          value: event.payload.newState === 'OPEN' ? 1 : 0,
          tags: {
            serviceName: event.payload.serviceName,
            state: event.payload.newState
          },
          source: event.source
        });
      }
    };
    
    // Exécution des agents/pools
    const poolExecutionListener: EventListener = (event) => {
      if (event.payload) {
        this.processPoolExecution(event.type, event.source, event.payload);
      }
    };
    
    // Évaluation des modèles
    const modelEvaluationListener: EventListener = (event) => {
      if (event.payload && event.payload.metrics) {
        this.processModelEvaluation(event.source, event.payload);
      }
    };
    
    // Anomalies détectées
    const anomalyListener: EventListener = (event) => {
      if (event.payload) {
        this.recordMetric({
          type: MetricType.ANOMALIES,
          value: event.payload.highPriorityAnomalies?.length || 0,
          tags: {
            severity: 'HIGH',
            source: event.source
          },
          source: event.source
        });
      }
    };
    
    // S'abonner aux événements
    this.subscriptionIds.push(
      this.eventBus.subscribe(RagKagEventType.PERFORMANCE_METRIC, performanceMetricListener),
      this.eventBus.subscribe(RagKagEventType.CIRCUIT_BREAKER_OPENED, circuitBreakerListener),
      this.eventBus.subscribe(RagKagEventType.CIRCUIT_BREAKER_CLOSED, circuitBreakerListener),
      this.eventBus.subscribe(RagKagEventType.POOL_EXECUTION_STARTED, poolExecutionListener),
      this.eventBus.subscribe(RagKagEventType.POOL_EXECUTION_COMPLETED, poolExecutionListener),
      this.eventBus.subscribe(RagKagEventType.MODEL_EVALUATION_COMPLETED, modelEvaluationListener),
      this.eventBus.subscribe(RagKagEventType.ANOMALY_DETECTED, anomalyListener)
    );
    
    this.logger.info(`Abonnement aux événements système pour la collecte des métriques: ${this.subscriptionIds.length} abonnements`);
  }
  
  /**
   * Initialise les règles d'alerte par défaut
   */
  private initDefaultAlertRules() {
    this.alertRules = [
      // Règles pour un appel API direct (modèles externes)
      {
        id: 'api-provider-latency',
        metricType: MetricType.LATENCY,
        condition: {
          operator: '>',
          threshold: 15000 // 15 secondes pour un appel API direct
        },
        tags: {
          source: 'ApiProvider'
        },
        windowPeriod: 3 * 60 * 1000, // 3 minutes
        cooldown: 5 * 60 * 1000, // 5 minutes
        severity: 'MEDIUM',
        description: 'Latence élevée pour un fournisseur d\'API',
        enabled: true
      },
      // Règle pour le traitement RAG ou KAG standard
      {
        id: 'high-latency',
        metricType: MetricType.LATENCY,
        condition: {
          operator: '>',
          threshold: 30000 // 30 secondes pour un traitement RAG ou KAG
        },
        windowPeriod: 3 * 60 * 1000, // 3 minutes
        cooldown: 5 * 60 * 1000, // 5 minutes
        severity: 'MEDIUM',
        description: 'Latence élevée détectée pour un traitement standard',
        enabled: true
      },
      // Règle pour les modules de débat (particulièrement exigeants)
      {
        id: 'debate-latency',
        metricType: MetricType.LATENCY,
        condition: {
          operator: '>',
          threshold: 90000 // 90 secondes (1min30) pour les modules de débat sophistiqués
        },
        tags: {
          component: 'debate'
        },
        windowPeriod: 5 * 60 * 1000, // 5 minutes
        cooldown: 10 * 60 * 1000, // 10 minutes
        severity: 'MEDIUM',
        description: 'Latence élevée pour un module de débat',
        enabled: true
      },
      // Règle pour le système complet (orchestration + débat + synthèse)
      {
        id: 'full-system-latency',
        metricType: MetricType.LATENCY,
        condition: {
          operator: '>',
          threshold: 120000 // 120 secondes (2 minutes) pour le traitement complet
        },
        tags: {
          component: 'orchestrator'
        },
        windowPeriod: 5 * 60 * 1000, // 5 minutes
        cooldown: 10 * 60 * 1000, // 10 minutes
        severity: 'MEDIUM',
        description: 'Latence élevée pour le système complet',
        enabled: true
      },
      // Règle pour la latence critique (vraiment problématique)
      {
        id: 'critical-latency',
        metricType: MetricType.LATENCY,
        condition: {
          operator: '>',
          threshold: 180000 // 180 secondes (3 minutes) - latence vraiment critique
        },
        windowPeriod: 5 * 60 * 1000, // 5 minutes
        cooldown: 15 * 60 * 1000, // 15 minutes
        severity: 'HIGH',
        description: 'Latence critique détectée',
        enabled: true
      },
      {
        id: 'circuit-breaker-open',
        metricType: MetricType.CIRCUIT_BREAKER,
        condition: {
          operator: '==',
          threshold: 1 // Circuit ouvert
        },
        windowPeriod: 1 * 60 * 1000, // 1 minute
        cooldown: 5 * 60 * 1000, // 5 minutes (réduit de 10 à 5 minutes)
        severity: 'HIGH',
        description: 'Circuit breaker ouvert',
        enabled: true
      },
      {
        id: 'high-anomaly-rate',
        metricType: MetricType.ANOMALIES,
        condition: {
          operator: '>',
          threshold: 3 // Plus de 3 anomalies (réduit de 5 à 3)
        },
        tags: {
          severity: 'HIGH'
        },
        windowPeriod: 5 * 60 * 1000, // 5 minutes (réduit de 10 à 5 minutes)
        cooldown: 10 * 60 * 1000, // 10 minutes (réduit de 30 à 10 minutes)
        severity: 'HIGH',
        description: 'Taux d\'anomalies élevé détecté',
        enabled: true
      },
      {
        id: 'low-success-rate',
        metricType: MetricType.SUCCESS_RATE,
        condition: {
          operator: '<',
          threshold: 0.9 // Moins de 90% de succès
        },
        windowPeriod: 3 * 60 * 1000, // 3 minutes
        cooldown: 5 * 60 * 1000, // 5 minutes
        severity: 'MEDIUM',
        description: 'Taux de succès anormalement bas',
        enabled: true
      }
    ];
    
    this.logger.info(`${this.alertRules.length} règles d'alerte par défaut initialisées`);
  }
  
  /**
   * Démarre les tâches périodiques (nettoyage, persistance, vérification des alertes)
   */
  private startPeriodicTasks() {
    // Nettoyage des métriques anciennes (toutes les heures)
    setInterval(() => {
      this.cleanupOldMetrics();
    }, 60 * 60 * 1000);
    
    // Persistance des métriques dans le graphe (toutes les 5 minutes)
    setInterval(() => {
      this.persistMetricsToGraph();
    }, this.persistenceInterval);
    
    // Vérification des règles d'alerte (toutes les minutes)
    setInterval(() => {
      this.checkAlertRules();
    }, 60 * 1000);
    
    this.logger.info('Tâches périodiques de gestion des métriques démarrées');
  }
  
  /**
   * Nettoie les métriques plus anciennes que la période de rétention
   */
  private cleanupOldMetrics() {
    const now = Date.now();
    const cutoffTime = now - this.retentionPeriod;
    
    const initialCount = this.metrics.length;
    this.metrics = this.metrics.filter(metric => metric.timestamp >= cutoffTime);
    const removedCount = initialCount - this.metrics.length;
    
    if (removedCount > 0) {
      this.logger.debug(`Nettoyage des métriques: ${removedCount} métriques supprimées`);
    }
  }
  
  /**
   * Persiste les métriques dans le graphe de connaissances
   */
  private async persistMetricsToGraph() {
    try {
      if (this.metrics.length === 0) return;
      
      // Récupérer le nœud racine des métriques
      const rootSearchResults = await this.knowledgeGraph.search('metrics root', {
        nodeTypes: ['METRICS_ROOT'],
        maxResults: 1
      });
      
      if (rootSearchResults.nodes.length === 0) {
        this.logger.error('Impossible de trouver le nœud racine des métriques pour la persistance');
        return;
      }
      
      const rootNodeId = rootSearchResults.nodes[0].id;
      const now = new Date();
      const dateStr = `${now.getFullYear()}-${String(now.getMonth() + 1).padStart(2, '0')}-${String(now.getDate()).padStart(2, '0')}`;
      
      // Regrouper les métriques par type
      const metricsByType: Record<string, Metric[]> = {};
      for (const metric of this.metrics) {
        if (!metricsByType[metric.type]) {
          metricsByType[metric.type] = [];
        }
        metricsByType[metric.type].push(metric);
      }
      
      // Créer ou mettre à jour les nœuds pour chaque type
      for (const [type, metricsOfType] of Object.entries(metricsByType)) {
        // Créer un nœud pour ce type et cette date
        const typeNodeId = this.knowledgeGraph.addNode({
          label: `Metrics ${type} ${dateStr}`,
          type: 'METRICS_COLLECTION',
          content: `Collection de métriques de type ${type} pour la journée ${dateStr}`,
          confidence: 1,
          source: KnowledgeSource.SYSTEM,
          metadata: {
            metricType: type,
            date: dateStr,
            count: metricsOfType.length,
            metrics: metricsOfType.slice(0, 1000) // Limiter à 1000 métriques pour éviter les problèmes de taille
          }
        });
        
        // Lier au nœud racine
        this.knowledgeGraph.addFact(
          rootNodeId,
          'HAS_METRICS',
          typeNodeId,
          1,
          { bidirectional: false, weight: 1 }
        );
      }
      
      // Calculer des agrégations pour certains types de métriques
      await this.persistAggregations(rootNodeId, dateStr);
      
      // Vider la liste de métriques
      this.metrics = [];
      
      this.logger.info('Persistance des métriques réalisée avec succès');
    } catch (error) {
      this.logger.error('Erreur lors de la persistance des métriques', { error });
    }
  }
  
  /**
   * Calcule et persiste des agrégations pour certains types de métriques
   */
  private async persistAggregations(rootNodeId: string, dateStr: string) {
    // Types de métriques à agréger
    const typesToAggregate = [
      MetricType.LATENCY,
      MetricType.SUCCESS_RATE,
      MetricType.API_CALLS,
      MetricType.TOKENS
    ];
    
    for (const type of typesToAggregate) {
      const metricsOfType = this.metrics.filter(m => m.type === type);
      
      if (metricsOfType.length === 0) continue;
      
      // Calculer des agrégations par source
      const aggregationsBySource: Record<string, {
        min: number;
        max: number;
        avg: number;
        count: number;
        sum: number;
      }> = {};
      
      for (const metric of metricsOfType) {
        const source = metric.source;
        
        if (!aggregationsBySource[source]) {
          aggregationsBySource[source] = {
            min: Infinity,
            max: -Infinity,
            avg: 0,
            count: 0,
            sum: 0
          };
        }
        
        const agg = aggregationsBySource[source];
        agg.min = Math.min(agg.min, metric.value);
        agg.max = Math.max(agg.max, metric.value);
        agg.sum += metric.value;
        agg.count += 1;
      }
      
      // Finaliser les moyennes
      for (const [source, agg] of Object.entries(aggregationsBySource)) {
        agg.avg = agg.sum / agg.count;
        
        // Créer un nœud pour cette agrégation
        const aggNodeId = this.knowledgeGraph.addNode({
          label: `${type} Aggregation ${source} ${dateStr}`,
          type: 'METRICS_AGGREGATION',
          content: `Agrégation des métriques de type ${type} pour ${source} le ${dateStr}`,
          confidence: 1,
          source: KnowledgeSource.SYSTEM,
          metadata: {
            metricType: type,
            source,
            date: dateStr,
            min: agg.min,
            max: agg.max,
            avg: agg.avg,
            count: agg.count,
            sum: agg.sum
          }
        });
        
        // Lier au nœud racine
        this.knowledgeGraph.addFact(
          rootNodeId,
          'HAS_AGGREGATION',
          aggNodeId,
          1,
          { bidirectional: false, weight: 1 }
        );
      }
    }
  }
  
  /**
   * Vérifie les règles d'alerte sur les métriques récentes
   */
  private checkAlertRules() {
    const now = Date.now();
    
    for (const rule of this.alertRules) {
      if (!rule.enabled) continue;
      
      // Vérifier si cette règle est en cooldown
      const lastAlert = this.alertHistory.find(a => a.ruleId === rule.id);
      if (lastAlert && now - lastAlert.timestamp < rule.cooldown) continue;
      
      // Filtrer les métriques par type et période
      const relevantMetrics = this.metrics.filter(metric => {
        if (metric.type !== rule.metricType) return false;
        
        // Vérifier la période
        if (now - metric.timestamp > rule.windowPeriod) return false;
        
        // Vérifier les tags
        if (rule.tags) {
          for (const [key, value] of Object.entries(rule.tags)) {
            if (metric.tags[key] !== value) return false;
          }
        }
        
        return true;
      });
      
      if (relevantMetrics.length === 0) continue;
      
      // Vérifier la condition
      const triggeringMetrics = relevantMetrics.filter(metric => {
        switch (rule.condition.operator) {
          case '>': return metric.value > rule.condition.threshold;
          case '<': return metric.value < rule.condition.threshold;
          case '==': return Math.abs(metric.value - rule.condition.threshold) < 0.001;
          case '>=': return metric.value >= rule.condition.threshold;
          case '<=': return metric.value <= rule.condition.threshold;
          default: return false;
        }
      });
      
      if (triggeringMetrics.length > 0) {
        // Règle déclenchée, émettre une alerte
        const latestValue = triggeringMetrics.reduce((max, metric) => 
          metric.timestamp > max.timestamp ? metric : max, triggeringMetrics[0]);
        
        this.emitAlert(rule, latestValue.value);
        
        // Enregistrer dans l'historique
        this.alertHistory.push({
          ruleId: rule.id,
          timestamp: now,
          value: latestValue.value
        });
        
        // Garder l'historique des alertes compact
        if (this.alertHistory.length > 100) {
          this.alertHistory.shift();
        }
      }
    }
  }
  
  /**
   * Émet une alerte via l'EventBus
   */
  private emitAlert(rule: AlertRule, value: number) {
    this.logger.warn(`Alerte déclenchée: ${rule.description}`, {
      ruleId: rule.id,
      metricType: rule.metricType,
      threshold: rule.condition.threshold,
      actualValue: value,
      severity: rule.severity
    });
    
    this.eventBus.emit({
      type: RagKagEventType.CUSTOM,
      source: 'MetricsService',
      payload: {
        eventType: 'METRICS_ALERT',
        alert: {
          id: uuidv4(),
          ruleId: rule.id,
          description: rule.description,
          metricType: rule.metricType,
          threshold: rule.condition.threshold,
          actualValue: value,
          severity: rule.severity,
          timestamp: Date.now()
        }
      }
    });
  }
  
  /**
   * Traite une métrique de performance
   */
  private processPerformanceMetric(source: string, payload: any) {
    // Latence
    if (payload.responseTime || payload.executionTime) {
      this.recordMetric({
        type: MetricType.LATENCY,
        value: payload.responseTime || payload.executionTime,
        tags: {
          source
        },
        source
      });
    }
    
    // Tokens
    if (payload.tokenCount || payload.tokensUsed) {
      this.recordMetric({
        type: MetricType.TOKENS,
        value: payload.tokenCount || payload.tokensUsed,
        tags: {
          source
        },
        source
      });
    }
    
    // Taux de succès
    if (payload.success !== undefined) {
      this.recordMetric({
        type: MetricType.SUCCESS_RATE,
        value: payload.success ? 1 : 0,
        tags: {
          source
        },
        source
      });
    }
    
    // Métriques spécifiques aux API
    if (source === 'ApiProviderFactory' && payload.provider) {
      if (payload.currentStats) {
        const stats = payload.currentStats;
        
        this.recordMetric({
          type: MetricType.SUCCESS_RATE,
          value: stats.successRate,
          tags: {
            provider: payload.provider,
            source: 'ApiProvider'
          },
          source: 'ApiProvider'
        });
        
        this.recordMetric({
          type: MetricType.LATENCY,
          value: stats.averageResponseTime,
          tags: {
            provider: payload.provider,
            source: 'ApiProvider'
          },
          source: 'ApiProvider'
        });
        
        if (stats.anomalyRate !== undefined) {
          this.recordMetric({
            type: MetricType.ANOMALIES,
            value: stats.anomalyRate,
            tags: {
              provider: payload.provider,
              source: 'ApiProvider'
            },
            source: 'ApiProvider'
          });
        }
      }
    }
    
    // Métriques des circuit breakers
    if (payload.circuitBreakers) {
      for (const [serviceName, status] of Object.entries(payload.circuitBreakers)) {
        // Typer correctement le status pour éviter les erreurs de linter
        const circuitStatus = status as { state: string, metrics: any };
        this.recordMetric({
          type: MetricType.CIRCUIT_BREAKER,
          value: circuitStatus.state === 'OPEN' ? 1 : (circuitStatus.state === 'HALF_OPEN' ? 0.5 : 0),
          tags: {
            serviceName,
            state: circuitStatus.state
          },
          source: source
        });
      }
    }
  }
  
  /**
   * Traite les métriques d'exécution des pools
   */
  private processPoolExecution(eventType: RagKagEventType, source: string, payload: any) {
    if (eventType === RagKagEventType.POOL_EXECUTION_COMPLETED && payload.poolOutputs) {
      const outputs = payload.poolOutputs;
      
      // Nombre d'agents par pool
      if (outputs.commercialCount !== undefined) {
        this.recordMetric({
          type: MetricType.POOL_EXECUTIONS,
          value: outputs.commercialCount,
          tags: {
            poolType: 'commercial'
          },
          source
        });
      }
      
      if (outputs.marketingCount !== undefined) {
        this.recordMetric({
          type: MetricType.POOL_EXECUTIONS,
          value: outputs.marketingCount,
          tags: {
            poolType: 'marketing'
          },
          source
        });
      }
      
      if (outputs.sectorialCount !== undefined) {
        this.recordMetric({
          type: MetricType.POOL_EXECUTIONS,
          value: outputs.sectorialCount,
          tags: {
            poolType: 'sectoriel'
          },
          source
        });
      }
      
      if (outputs.educationalCount !== undefined) {
        this.recordMetric({
          type: MetricType.POOL_EXECUTIONS,
          value: outputs.educationalCount,
          tags: {
            poolType: 'educational'
          },
          source
        });
      }
      
      // Erreurs
      if (outputs.errorCount !== undefined) {
        this.recordMetric({
          type: MetricType.SUCCESS_RATE,
          value: outputs.errorCount > 0 ? 0 : 1,
          tags: {
            component: 'pool_manager'
          },
          source
        });
      }
      
      // Temps d'exécution
      if (payload.executionTime) {
        this.recordMetric({
          type: MetricType.LATENCY,
          value: payload.executionTime,
          tags: {
            component: 'pool_manager'
          },
          source
        });
      }
    }
  }
  
  /**
   * Traite les métriques d'évaluation des modèles
   */
  private processModelEvaluation(source: string, payload: any) {
    if (payload.metrics) {
      const metrics = payload.metrics;
      const model = payload.model || payload.modelName;
      
      if (metrics.accuracy !== undefined) {
        this.recordMetric({
          type: MetricType.MODEL_ACCURACY,
          value: metrics.accuracy,
          tags: {
            model,
            metricType: 'accuracy'
          },
          source
        });
      }
      
      if (metrics.bleuScore !== undefined) {
        this.recordMetric({
          type: MetricType.MODEL_ACCURACY,
          value: metrics.bleuScore,
          tags: {
            model,
            metricType: 'bleu'
          },
          source
        });
      }
      
      if (metrics.semanticSimilarity !== undefined) {
        this.recordMetric({
          type: MetricType.MODEL_ACCURACY,
          value: metrics.semanticSimilarity,
          tags: {
            model,
            metricType: 'similarity'
          },
          source
        });
      }
    }
    
    if (payload.overallQuality !== undefined) {
      this.recordMetric({
        type: MetricType.MODEL_ACCURACY,
        value: payload.overallQuality,
        tags: {
          model: payload.model || payload.modelName,
          metricType: 'quality'
        },
        source
      });
    }
  }
  
  /**
   * Enregistre une nouvelle métrique
   */
  public recordMetric(params: {
    type: MetricType;
    value: number;
    tags: Record<string, string>;
    source: string;
    timestamp?: number;
  }): void {
    // Créer la métrique
    const metric: Metric = {
      id: uuidv4(),
      type: params.type,
      value: params.value,
      tags: params.tags,
      timestamp: params.timestamp || Date.now(),
      source: params.source
    };
    
    // Ajouter à la liste
    this.metrics.push(metric);
    
    // Vérifier si on doit nettoyer la mémoire
    if (this.metrics.length > this.maxInMemoryMetrics) {
      this.metrics = this.metrics.slice(-this.maxInMemoryMetrics);
    }
  }
  
  /**
   * Récupère les métriques selon des critères
   */
  public getMetrics(options: MetricsQueryOptions = {}): Metric[] {
    let filteredMetrics = [...this.metrics];
    
    // Filtrer par temps
    if (options.startTime !== undefined) {
      filteredMetrics = filteredMetrics.filter(m => m.timestamp >= options.startTime);
    }
    
    if (options.endTime !== undefined) {
      filteredMetrics = filteredMetrics.filter(m => m.timestamp <= options.endTime);
    }
    
    // Filtrer par type
    if (options.types && options.types.length > 0) {
      filteredMetrics = filteredMetrics.filter(m => options.types.includes(m.type));
    }
    
    // Filtrer par source
    if (options.sources && options.sources.length > 0) {
      filteredMetrics = filteredMetrics.filter(m => options.sources.includes(m.source));
    }
    
    // Filtrer par tags
    if (options.tags) {
      filteredMetrics = filteredMetrics.filter(m => {
        for (const [key, value] of Object.entries(options.tags)) {
          if (m.tags[key] !== value) return false;
        }
        return true;
      });
    }
    
    // Appliquer l'agrégation si demandée
    if (options.aggregation) {
      filteredMetrics = this.aggregateMetrics(
        filteredMetrics, 
        options.aggregation.period, 
        options.aggregation.operation
      );
    }
    
    // Limiter les résultats
    if (options.limit !== undefined && options.limit > 0) {
      filteredMetrics = filteredMetrics.slice(0, options.limit);
    }
    
    return filteredMetrics;
  }
  
  /**
   * Agrège des métriques selon une période et une opération
   */
  private aggregateMetrics(
    metrics: Metric[],
    period: AggregationPeriod,
    operation: AggregationOperation
  ): Metric[] {
    if (metrics.length === 0) return [];
    
    // Regrouper les métriques par période
    const groups: Record<string, Metric[]> = {};
    
    for (const metric of metrics) {
      const periodStart = Math.floor(metric.timestamp / period) * period;
      const key = `${metric.type}:${periodStart}:${JSON.stringify(metric.tags)}`;
      
      if (!groups[key]) {
        groups[key] = [];
      }
      
      groups[key].push(metric);
    }
    
    // Agréger chaque groupe
    const result: Metric[] = [];
    
    for (const [key, groupMetrics] of Object.entries(groups)) {
      let value: number;
      
      switch (operation) {
        case AggregationOperation.AVG:
          value = groupMetrics.reduce((sum, m) => sum + m.value, 0) / groupMetrics.length;
          break;
        case AggregationOperation.MIN:
          value = Math.min(...groupMetrics.map(m => m.value));
          break;
        case AggregationOperation.MAX:
          value = Math.max(...groupMetrics.map(m => m.value));
          break;
        case AggregationOperation.SUM:
          value = groupMetrics.reduce((sum, m) => sum + m.value, 0);
          break;
        case AggregationOperation.COUNT:
          value = groupMetrics.length;
          break;
        default:
          value = 0;
      }
      
      // Créer une métrique agrégée
      const [type, timestamp] = key.split(':');
      const tags = JSON.parse(key.split(':')[2]);
      
      result.push({
        id: uuidv4(),
        type: groupMetrics[0].type,
        value,
        tags: groupMetrics[0].tags,
        timestamp: parseInt(timestamp),
        source: 'aggregation'
      });
    }
    
    return result;
  }
  
  /**
   * Ajoute une règle d'alerte
   */
  public addAlertRule(rule: Omit<AlertRule, 'id'>): string {
    const id = uuidv4();
    this.alertRules.push({
      ...rule,
      id
    });
    
    this.logger.info(`Nouvelle règle d'alerte ajoutée: ${rule.description}`, { id });
    return id;
  }
  
  /**
   * Supprime une règle d'alerte
   */
  public removeAlertRule(id: string): boolean {
    const initialLength = this.alertRules.length;
    this.alertRules = this.alertRules.filter(rule => rule.id !== id);
    
    return this.alertRules.length < initialLength;
  }
  
  /**
   * Active ou désactive une règle d'alerte
   */
  public toggleAlertRule(id: string, enabled: boolean): boolean {
    const rule = this.alertRules.find(rule => rule.id === id);
    if (!rule) return false;
    
    rule.enabled = enabled;
    return true;
  }
  
  /**
   * Récupère toutes les règles d'alerte
   */
  public getAlertRules(): AlertRule[] {
    return [...this.alertRules];
  }
  
  /**
   * Récupère l'historique des alertes
   */
  public getAlertHistory(limit: number = 100): Array<{
    rule: AlertRule;
    timestamp: number;
    value: number;
  }> {
    return this.alertHistory
      .slice(-limit)
      .map(alert => ({
        rule: this.alertRules.find(rule => rule.id === alert.ruleId),
        timestamp: alert.timestamp,
        value: alert.value
      }))
      .filter(alert => alert.rule !== undefined);
  }
  
  /**
   * Nettoyage lors de la destruction du service
   */
  async onModuleDestroy() {
    // Se désabonner des événements
    for (const id of this.subscriptionIds) {
      this.eventBus.unsubscribe(id);
    }
    
    // Persistance finale des métriques
    await this.persistMetricsToGraph();
    
    this.logger.info('Service de métriques arrêté proprement');
  }

  /**
   * Configure des alertes sur les métriques de performance clés
   */
  private createPerformanceAlerts() {
    // Alerte si le temps de réponse moyen dépasse 5 secondes sur les 30 dernières minutes
    this.addAlertRule({
      metricType: MetricType.LATENCY,
      condition: {
        operator: '>=',
        threshold: 5000
      },
      windowPeriod: 30 * 60 * 1000,
      cooldown: 15 * 60 * 1000,
      severity: 'HIGH',
      description: "Le temps de réponse moyen est trop élevé",
      enabled: true
    });

    // Alerte si le taux d'erreur dépasse 10% sur les 60 dernières minutes 
    this.addAlertRule({
      metricType: MetricType.SUCCESS_RATE,
      condition: {
        operator: '<=',
        threshold: 0.9
      },
      windowPeriod: 60 * 60 * 1000,
      cooldown: 30 * 60 * 1000,
      severity: 'CRITICAL',
      description: "Le taux d'erreur est supérieur à 10%",
      enabled: true  
    });

    // Alerte si l'utilisation mémoire dépasse 80% 
    this.addAlertRule({
      metricType: MetricType.MEMORY_USAGE,
      condition: {
        operator: '>=',
        threshold: 0.8
      },
      windowPeriod: 5 * 60 * 1000,
      cooldown: 15 * 60 * 1000,
      severity: 'MEDIUM',
      description: "L'utilisation mémoire est supérieure à 80%",
      enabled: true
    });
  }
} 