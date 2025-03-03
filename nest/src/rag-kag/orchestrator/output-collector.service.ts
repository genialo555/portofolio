import { Injectable, Inject } from '@nestjs/common';
import { LOGGER_TOKEN, ILogger } from '../utils/logger-tokens';
import { PoolOutputs } from '../../types';
import { AgentOutput } from '../types';
import { AnomalyDetectionService, DetectAnomalies, AnomalyDetectionLevel } from '../../utils/anomaly-detection.service';

/**
 * Service de collecte et traitement des sorties des pools d'agents
 */
@Injectable()
export class OutputCollectorService {
  private readonly logger: ILogger;
  
  constructor(
    @Inject(LOGGER_TOKEN) logger: ILogger,
    private readonly anomalyDetectionService: AnomalyDetectionService
  ) {
    this.logger = logger;
    this.logger.info('Service de collecte des outputs initialisé');
  }
  
  /**
   * Collecte et traite les sorties des pools d'agents
   * @param poolOutputs Outputs bruts des pools d'agents
   * @returns Outputs traités et structurés
   */
  @DetectAnomalies({
    detectionLevel: AnomalyDetectionLevel.ALL,
    enrichResponse: true
  })
  async collectAndProcess(poolOutputs: PoolOutputs): Promise<PoolOutputs> {
    this.logger.debug('Traitement des sorties des pools');
    
    // Détection d'anomalies
    try {
      const anomalyReport = await this.anomalyDetectionService.detectAnomalies(
        poolOutputs,
        { 
          detectionLevel: AnomalyDetectionLevel.MEDIUM_AND_ABOVE,
          autoLog: true 
        }
      );
      
      // Si la fiabilité est très basse, on le signale
      if (anomalyReport.overallReliability < 0.4) {
        this.logger.warn('Fiabilité des résultats très basse, considérer une requête de clarification', {
          reliability: anomalyReport.overallReliability,
          highPriorityAnomalies: anomalyReport.highPriorityAnomalies.length
        });
      }
    } catch (error) {
      this.logger.error('Erreur lors de la détection d\'anomalies', { error });
    }
    
    // Comptabiliser les résultats
    const commercialCount = poolOutputs.commercial?.length || 0;
    const marketingCount = poolOutputs.marketing?.length || 0;
    const sectorialCount = poolOutputs.sectoriel?.length || 0;
    const totalCount = commercialCount + marketingCount + sectorialCount;
    
    // Enrichir les métadonnées mais retourner le même format de données
    const enhancedPoolOutputs: PoolOutputs = {
      commercial: poolOutputs.commercial || [],
      marketing: poolOutputs.marketing || [],
      sectoriel: poolOutputs.sectoriel || [],
      educational: poolOutputs.educational || [],
      timestamp: new Date(),
      query: poolOutputs.query
    };
    
    this.logger.debug('Traitement des sorties terminé', { 
      totalAgents: totalCount,
      commercialCount, 
      marketingCount,
      sectorialCount
    });
    
    return enhancedPoolOutputs;
  }
  
  /**
   * Filtre les résultats avec un seuil de confiance élevé
   * @param results Résultats à filtrer
   * @returns Résultats filtrés
   */
  private filterHighConfidence(results: any[]): any[] {
    const CONFIDENCE_THRESHOLD = 0.75;
    return results.filter(r => r.confidence >= CONFIDENCE_THRESHOLD);
  }
  
  /**
   * Traite les résultats d'un domaine spécifique
   * @param domainResults Résultats du domaine
   * @param domainType Type de domaine
   * @returns Résultats traités
   */
  private processDomainResults(domainResults: any[], domainType: string): any {
    if (!domainResults || domainResults.length === 0) {
      return {
        present: false,
        agentCount: 0,
        insights: []
      };
    }
    
    // Trier par confiance décroissante
    const sortedResults = [...domainResults].sort((a, b) => b.confidence - a.confidence);
    
    return {
      present: true,
      agentCount: domainResults.length,
      insights: sortedResults.map(r => ({
        content: r.content || r.response,
        confidence: r.confidence,
        agentName: r.agentName,
        metrics: r.metrics
      })),
      topInsight: sortedResults[0]?.content || sortedResults[0]?.response,
      averageConfidence: this.calculateAverageConfidence(domainResults)
    };
  }
  
  /**
   * Calcule la confiance moyenne d'un ensemble de résultats
   * @param results Résultats à analyser
   * @returns Confiance moyenne
   */
  private calculateAverageConfidence(results: any[]): number {
    if (!results || results.length === 0) return 0;
    
    const sum = results.reduce((acc, r) => acc + (r.confidence || 0), 0);
    return sum / results.length;
  }
  
  /**
   * Extrait les insights combinés de tous les résultats
   * @param allResults Tous les résultats des agents
   * @returns Insights combinés
   */
  private extractCombinedInsights(allResults: any[]): string[] {
    // Dans une implémentation réelle, on utiliserait des techniques plus
    // sophistiquées comme l'analyse NLP, la déduplication sémantique, etc.
    
    // Simulation simplifiée: prendre les 3 meilleurs résultats par confiance
    const topResults = [...allResults]
      .sort((a, b) => b.confidence - a.confidence)
      .slice(0, 3);
    
    return topResults.map(r => r.content || r.response);
  }
} 