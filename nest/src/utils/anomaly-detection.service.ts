import { Injectable, Inject, Optional } from '@nestjs/common';
import { PoolOutputs, AnomalyReport, AnomalySeverity, Anomaly } from '../types';
import { AnomalyDetectorService } from '../legacy/utils/anomalyDetector';
import { ILogger, LOGGER_TOKEN } from '../rag-kag/utils/logger-tokens';
import { KnowledgeGraphService, KnowledgeSource, RelationType } from '../rag-kag/core/knowledge-graph.service';
import { EventBusService, RagKagEventType } from '../rag-kag/core/event-bus.service';

/**
 * Niveaux de détection d'anomalies
 */
export enum AnomalyDetectionLevel {
  ALL = 'ALL',                 // Toutes les anomalies (HIGH, MEDIUM, LOW)
  MEDIUM_AND_ABOVE = 'MEDIUM', // Anomalies moyennes et critiques
  HIGH_ONLY = 'HIGH',          // Uniquement les anomalies critiques
  NONE = 'NONE'                // Désactiver la détection
}

/**
 * Options pour la détection d'anomalies
 */
export interface AnomalyDetectionOptions {
  detectionLevel?: AnomalyDetectionLevel;
  autoLog?: boolean;           // Logger automatiquement les anomalies détectées
  throwOnHigh?: boolean;       // Lever une exception sur anomalie critique
  alertThreshold?: number;     // Seuil de fiabilité pour déclencher une alerte (0-1)
  enrichResponse?: boolean;    // Enrichir la réponse avec info sur anomalies
  storeInGraph?: boolean;      // Stocker les anomalies dans le graphe de connaissances
  useAdvancedDetection?: boolean; // Utiliser les détecteurs avancés du legacy
}

/**
 * Exception levée en cas d'anomalie critique
 */
export class CriticalAnomalyException extends Error {
  constructor(
    message: string,
    public readonly anomalyReport: AnomalyReport
  ) {
    super(message);
    this.name = 'CriticalAnomalyException';
  }
}

/**
 * Service pour la détection d'anomalies dans les réponses
 * Intègre les capacités avancées du détecteur legacy
 */
@Injectable()
export class AnomalyDetectionService {
  private readonly logger: ILogger;
  private readonly defaultOptions: AnomalyDetectionOptions = {
    detectionLevel: AnomalyDetectionLevel.MEDIUM_AND_ABOVE,
    autoLog: true,
    throwOnHigh: false,
    alertThreshold: 0.6,
    enrichResponse: true,
    storeInGraph: true,
    useAdvancedDetection: true
  };

  constructor(
    private readonly detector: AnomalyDetectorService,
    @Inject(LOGGER_TOKEN) logger: ILogger,
    @Optional() private readonly knowledgeGraph?: KnowledgeGraphService,
    @Optional() private readonly eventBus?: EventBusService
  ) {
    this.logger = logger;
    this.logger.info('Service de détection d\'anomalies initialisé');
  }

  /**
   * Détecte les anomalies dans les outputs des pools
   * et les stocke optionnellement dans le graphe de connaissances
   * 
   * @param poolOutputs Outputs des pools
   * @param options Options de configuration
   * @returns Rapport d'anomalies
   */
  async detectAnomalies(
    poolOutputs: PoolOutputs, 
    options: AnomalyDetectionOptions = {}
  ): Promise<AnomalyReport> {
    const config = { ...this.defaultOptions, ...options };
    
    // Désactiver complètement la détection
    if (config.detectionLevel === AnomalyDetectionLevel.NONE) {
      return this.getEmptyReport();
    }
    
    // Émettre un événement de début de détection si EventBus est disponible
    if (this.eventBus) {
      this.eventBus.emit({
        type: RagKagEventType.ANOMALY_DETECTION_STARTED,
        source: 'AnomalyDetectionService',
        payload: { 
          detectionLevel: config.detectionLevel,
          useAdvancedDetection: config.useAdvancedDetection
        }
      });
    }
    
    // Exécuter la détection
    let report = await this.detector.detectAnomalies(poolOutputs);
    
    // Enrichir avec les détecteurs avancés du legacy si demandé
    if (config.useAdvancedDetection) {
      report = await this.enrichWithAdvancedDetection(report, poolOutputs);
    }
    
    // Filtrer selon le niveau de détection configuré
    const filteredReport = this.filterReport(report, config.detectionLevel);
    
    // Logger si nécessaire
    if (config.autoLog) {
      this.logAnomalies(filteredReport);
    }
    
    // Stocker dans le graphe de connaissances si demandé
    if (config.storeInGraph && this.knowledgeGraph) {
      this.storeAnomaliesInGraph(filteredReport, poolOutputs);
    }
    
    // Vérifier seuil d'alerte
    if (filteredReport.overallReliability < config.alertThreshold) {
      this.logger.warn(`Fiabilité en dessous du seuil d'alerte (${Math.round(filteredReport.overallReliability * 100)}%)`, {
        alertThreshold: config.alertThreshold,
        reliability: filteredReport.overallReliability
      });
      
      // Émettre un événement d'alerte
      if (this.eventBus) {
        this.eventBus.emit({
          type: RagKagEventType.ANOMALY_ALERT,
          source: 'AnomalyDetectionService',
          payload: {
            reliability: filteredReport.overallReliability,
            threshold: config.alertThreshold,
            anomalyCount: this.getTotalAnomalyCount(filteredReport)
          }
        });
      }
    }
    
    // Lever une exception si demandé et qu'il y a des anomalies critiques
    if (config.throwOnHigh && filteredReport.highPriorityAnomalies.length > 0) {
      const exception = new CriticalAnomalyException(
        `Anomalies critiques détectées (${filteredReport.highPriorityAnomalies.length})`,
        filteredReport
      );
      
      if (this.eventBus) {
        this.eventBus.emit({
          type: RagKagEventType.ANOMALY_CRITICAL,
          source: 'AnomalyDetectionService',
          payload: {
            anomalies: filteredReport.highPriorityAnomalies,
            exception
          }
        });
      }
      
      throw exception;
    }
    
    // Émettre un événement de fin de détection
    if (this.eventBus) {
      this.eventBus.emit({
        type: RagKagEventType.ANOMALY_DETECTION_COMPLETED,
        source: 'AnomalyDetectionService',
        payload: {
          reliability: filteredReport.overallReliability,
          anomalyCount: this.getTotalAnomalyCount(filteredReport)
        }
      });
    }
    
    return filteredReport;
  }

  /**
   * Enrichit le rapport d'anomalies avec des détections avancées spécifiques du legacy
   */
  private async enrichWithAdvancedDetection(
    baseReport: AnomalyReport, 
    poolOutputs: PoolOutputs
  ): Promise<AnomalyReport> {
    // Intégrer les détections avancées spécifiques
    const cognitiveAnomalies = this.detector.detectCognitiveBiases(poolOutputs);
    const statisticalAnomalies = this.detector.detectStatisticalErrors(poolOutputs);
    const citationAnomalies = this.detector.detectCitationIssues(poolOutputs);
    const methodologicalAnomalies = this.detector.detectMethodologicalFlaws(poolOutputs);
    
    // Ajouter les nouvelles anomalies au rapport en fonction de leur sévérité
    const highSeverity = [...cognitiveAnomalies, ...statisticalAnomalies, ...citationAnomalies, ...methodologicalAnomalies]
      .filter(anomaly => anomaly.severity === 'HIGH');
    
    const mediumSeverity = [...cognitiveAnomalies, ...statisticalAnomalies, ...citationAnomalies, ...methodologicalAnomalies]
      .filter(anomaly => anomaly.severity === 'MEDIUM');
    
    const lowSeverity = [...cognitiveAnomalies, ...statisticalAnomalies, ...citationAnomalies, ...methodologicalAnomalies]
      .filter(anomaly => anomaly.severity === 'LOW');
    
    // Créer un rapport enrichi
    const enrichedReport: AnomalyReport = {
      ...baseReport,
      highPriorityAnomalies: [...baseReport.highPriorityAnomalies, ...highSeverity],
      mediumPriorityAnomalies: [...baseReport.mediumPriorityAnomalies, ...mediumSeverity],
      lowPriorityAnomalies: [...baseReport.lowPriorityAnomalies || [], ...lowSeverity],
      // Améliorer le rapport avec les patterns identifiés par l'analyse avancée
      systemicPatterns: [...(baseReport.systemicPatterns || []), ...this.identifySystemicPatterns([
        ...highSeverity, ...mediumSeverity, ...lowSeverity
      ])]
    };
    
    // Recalculer la fiabilité globale
    enrichedReport.overallReliability = this.calculateOverallReliability(enrichedReport);
    
    return enrichedReport;
  }
  
  /**
   * Calcule la fiabilité globale basée sur le nombre et la sévérité des anomalies
   */
  private calculateOverallReliability(report: AnomalyReport): number {
    // Définir des poids pour chaque niveau de sévérité
    const weights = {
      high: 0.6, // Les anomalies critiques réduisent fortement la fiabilité
      medium: 0.3, // Les anomalies moyennes ont un impact modéré
      low: 0.1 // Les anomalies mineures ont un impact faible
    };
    
    // Calculer la réduction de fiabilité
    const highImpact = report.highPriorityAnomalies.length * weights.high;
    const mediumImpact = report.mediumPriorityAnomalies.length * weights.medium;
    const lowImpact = (report.lowPriorityAnomalies?.length || 0) * weights.low;
    
    // Calculer la fiabilité (1.0 = parfait, diminue avec les anomalies)
    const totalImpact = Math.min(1.0, (highImpact + mediumImpact + lowImpact) / 10);
    
    return Math.max(0, 1.0 - totalImpact);
  }
  
  /**
   * Identifie les patterns systémiques dans les anomalies détectées
   */
  private identifySystemicPatterns(anomalies: Anomaly[]): string[] {
    // Regrouper les anomalies par type
    const typeGroups = new Map<string, Anomaly[]>();
    
    for (const anomaly of anomalies) {
      if (!typeGroups.has(anomaly.type)) {
        typeGroups.set(anomaly.type, []);
      }
      typeGroups.get(anomaly.type)!.push(anomaly);
    }
    
    // Identifier les patterns pour chaque type
    const patterns: string[] = [];
    
    for (const [type, groupAnomalies] of typeGroups.entries()) {
      if (groupAnomalies.length >= 3) {
        patterns.push(`Pattern systémique détecté: multiples anomalies de type ${type} (${groupAnomalies.length})`);
      }
    }
    
    // Vérifier les anomalies par agent
    const agentGroups = new Map<string, Anomaly[]>();
    
    for (const anomaly of anomalies) {
      const agentId = anomaly.location.agentId;
      if (!agentGroups.has(agentId)) {
        agentGroups.set(agentId, []);
      }
      agentGroups.get(agentId)!.push(anomaly);
    }
    
    for (const [agentId, groupAnomalies] of agentGroups.entries()) {
      if (groupAnomalies.length >= 3) {
        patterns.push(`L'agent ${agentId} présente des anomalies systémiques (${groupAnomalies.length} anomalies)`);
      }
    }
    
    return patterns;
  }

  /**
   * Calcule le nombre total d'anomalies dans le rapport
   */
  private getTotalAnomalyCount(report: AnomalyReport): number {
    return report.highPriorityAnomalies.length + 
           report.mediumPriorityAnomalies.length + 
           (report.lowPriorityAnomalies?.length || 0);
  }

  /**
   * Filtre un rapport d'anomalies selon le niveau de détection
   */
  private filterReport(
    report: AnomalyReport, 
    level: AnomalyDetectionLevel
  ): AnomalyReport {
    if (level === AnomalyDetectionLevel.ALL) {
      return report;
    }
    
    // Copie du rapport pour éviter de modifier l'original
    const filteredReport = { ...report };
    
    if (level === AnomalyDetectionLevel.HIGH_ONLY) {
      // Conserver uniquement les anomalies critiques
      filteredReport.mediumPriorityAnomalies = [];
      filteredReport.minorIssues = [];
    } else if (level === AnomalyDetectionLevel.MEDIUM_AND_ABOVE) {
      // Conserver les anomalies critiques et moyennes
      filteredReport.minorIssues = [];
    }
    
    return filteredReport;
  }

  /**
   * Crée un rapport d'anomalies vide
   */
  private getEmptyReport(): AnomalyReport {
    return {
      highPriorityAnomalies: [],
      mediumPriorityAnomalies: [],
      minorIssues: [],
      lowPriorityAnomalies: [],
      overallReliability: 1.0,
      report: 'Détection d\'anomalies désactivée',
      systemicPatterns: []
    };
  }

  /**
   * Génère des logs pour les anomalies détectées
   */
  private logAnomalies(report: AnomalyReport): void {
    const highCount = report.highPriorityAnomalies.length;
    const mediumCount = report.mediumPriorityAnomalies.length;
    const lowCount = report.minorIssues.length;
    
    // Log général
    this.logger.info(`Détection d'anomalies: ${highCount} critiques, ${mediumCount} moyennes, ${lowCount} mineures`, {
      highCount,
      mediumCount,
      lowCount,
      reliability: report.overallReliability
    });
    
    // Log des anomalies critiques
    if (highCount > 0) {
      this.logger.warn('Anomalies critiques détectées', {
        anomalies: report.highPriorityAnomalies.map(a => ({
          type: a.type,
          description: a.description,
          location: `Agent ${a.location.agentId} (${a.location.poolType})`,
          fragment: a.location.contentFragment
        }))
      });
    }
    
    // Log des patterns systémiques
    if (report.systemicPatterns && report.systemicPatterns.length > 0) {
      this.logger.warn('Patterns systémiques identifiés', { 
        patterns: report.systemicPatterns 
      });
    }
  }

  /**
   * Formatte le rapport d'anomalies pour inclusion dans la réponse
   */
  formatReportForResponse(report: AnomalyReport): any {
    return {
      reliability: report.overallReliability,
      issuesDetected: report.highPriorityAnomalies.length + 
                      report.mediumPriorityAnomalies.length + 
                      report.minorIssues.length,
      criticalIssues: report.highPriorityAnomalies.length,
      summary: report.report || 'Rapport d\'anomalies non disponible',
      systemicPatterns: report.systemicPatterns || []
    };
  }

  /**
   * Stocke les anomalies dans le graphe de connaissances
   * @param report Rapport d'anomalies
   * @param poolOutputs Outputs des pools
   */
  private storeAnomaliesInGraph(report: AnomalyReport, poolOutputs: PoolOutputs): void {
    if (!this.knowledgeGraph) return;
    
    try {
      // Créer un nœud pour le rapport d'anomalies
      const reportNodeId = this.knowledgeGraph.addNode({
        label: `Anomaly Report: ${new Date().toISOString()}`,
        type: 'ANOMALY_REPORT',
        content: report.report || 'Rapport d\'anomalies',
        confidence: report.overallReliability,
        source: KnowledgeSource.SYSTEM
      });
      
      // Stocker les anomalies critiques
      this.storeAnomaliesOfType(
        report.highPriorityAnomalies, 
        'HIGH', 
        reportNodeId
      );
      
      // Stocker les anomalies moyennes
      this.storeAnomaliesOfType(
        report.mediumPriorityAnomalies, 
        'MEDIUM', 
        reportNodeId
      );
      
      // Stocker les anomalies mineures
      this.storeAnomaliesOfType(
        report.minorIssues, 
        'LOW', 
        reportNodeId
      );
      
      // Stocker les patterns systémiques
      if (report.systemicPatterns && report.systemicPatterns.length > 0) {
        for (const pattern of report.systemicPatterns) {
          this.knowledgeGraph.addFact(
            reportNodeId,
            'HAS_SYSTEMIC_PATTERN',
            {
              label: `Pattern: ${pattern.substring(0, 30)}${pattern.length > 30 ? '...' : ''}`,
              type: 'SYSTEMIC_PATTERN',
              content: pattern,
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
      
      // Lier le rapport à la requête si elle existe dans les outputs
      if (poolOutputs.query) {
        const queryText = typeof poolOutputs.query === 'string' 
          ? poolOutputs.query 
          : (poolOutputs.query as any).text || (poolOutputs.query as any).content || '';
        
        if (queryText) {
          // Rechercher le nœud de requête existant
          const searchResults = this.knowledgeGraph.search(queryText, {
            nodeTypes: ['QUERY'],
            maxResults: 1,
            maxDepth: 0
          });
          
          if (searchResults.nodes.length > 0) {
            const queryNodeId = searchResults.nodes[0].id;
            
            // Lier le rapport d'anomalies à la requête
            this.knowledgeGraph.addFact(
              queryNodeId,
              'HAS_ANOMALY_REPORT',
              reportNodeId,
              0.9,
              {
                bidirectional: true,
                weight: 0.8
              }
            );
          }
        }
      }
      
      this.logger.debug('Anomalies stockées dans le graphe de connaissances', {
        reportId: reportNodeId,
        anomalyCount: report.highPriorityAnomalies.length + 
                     report.mediumPriorityAnomalies.length + 
                     report.minorIssues.length
      });
    } catch (error) {
      this.logger.error(`Erreur lors du stockage des anomalies dans le graphe: ${error.message}`, {
        error: error.stack
      });
    }
  }

  /**
   * Stocke un groupe d'anomalies d'un niveau spécifique dans le graphe
   * @param anomalies Liste des anomalies
   * @param severity Niveau de sévérité
   * @param reportNodeId ID du nœud de rapport parent
   */
  private storeAnomaliesOfType(
    anomalies: Anomaly[], 
    severity: string, 
    reportNodeId: string
  ): void {
    if (!this.knowledgeGraph) return;
    
    for (const anomaly of anomalies) {
      // Créer un nœud pour l'anomalie
      const anomalyNodeId = this.knowledgeGraph.addNode({
        label: `${severity} Anomaly: ${anomaly.type}`,
        type: 'ANOMALY',
        content: anomaly.description,
        confidence: 0.9,
        source: KnowledgeSource.INFERENCE,
        metadata: {
          severity,
          type: anomaly.type,
          agentId: anomaly.location.agentId,
          poolType: anomaly.location.poolType
        }
      });
      
      // Lier l'anomalie au rapport
      this.knowledgeGraph.addFact(
        reportNodeId,
        'CONTAINS_ANOMALY',
        anomalyNodeId,
        0.9,
        {
          bidirectional: false,
          weight: 0.9
        }
      );
    }
  }
}

/**
 * Fonction décorateur pour la détection automatique d'anomalies
 * Utilisation: @DetectAnomalies() sur une méthode de classe qui renvoie PoolOutputs
 * Exemple d'utilisation:
 * 
 * @DetectAnomalies({ 
 *   detectionLevel: AnomalyDetectionLevel.HIGH_ONLY,
 *   throwOnHigh: true
 * })
 * async processPoolOutputs(outputs: PoolOutputs): Promise<ProcessedResult> {
 *   // Traitement...
 * }
 */
export function DetectAnomalies(options: AnomalyDetectionOptions = {}) {
  return function (
    target: any,
    propertyKey: string,
    descriptor: PropertyDescriptor
  ) {
    const originalMethod = descriptor.value;
    
    descriptor.value = async function(...args: any[]) {
      const result = await originalMethod.apply(this, args);
      
      // Vérifier si le résultat ou un argument est un PoolOutputs
      const poolOutputs = result.poolOutputs || result;
      
      // Vérifier que le service AnomalyDetectionService est disponible
      if (!this.anomalyDetectionService) {
        console.warn('AnomalyDetectionService n\'est pas disponible dans cette classe.');
        return result;
      }
      
      // Détecter les anomalies
      try {
        const anomalyReport = await this.anomalyDetectionService.detectAnomalies(
          poolOutputs,
          options
        );
        
        // Enrichir la réponse si demandé
        if (options.enrichResponse !== false && result.response) {
          result.response.anomalyAnalysis = 
            this.anomalyDetectionService.formatReportForResponse(anomalyReport);
        }
      } catch (error) {
        // Ne pas bloquer si une erreur survient dans la détection
        console.error('Erreur lors de la détection d\'anomalies:', error);
      }
      
      return result;
    };
    
    return descriptor;
  };
} 