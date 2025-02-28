import { Injectable } from '@nestjs/common';
import { PoolOutputs, AnomalyReport, AnomalySeverity } from '../types';
import { AnomalyDetectorService } from './anomalyDetector';
import { Logger } from './logger';

/**
 * Niveau minimal d'anomalie à détecter
 */
export enum AnomalyDetectionLevel {
  ALL = 'ALL',                 // Toutes les anomalies (HIGH, MEDIUM, LOW)
  MEDIUM_AND_ABOVE = 'MEDIUM', // Anomalies moyennes et critiques
  HIGH_ONLY = 'HIGH',          // Uniquement les anomalies critiques
  NONE = 'NONE'                // Désactiver la détection
}

/**
 * Options de configuration pour la détection d'anomalies
 */
export interface AnomalyDetectionOptions {
  detectionLevel?: AnomalyDetectionLevel;
  autoLog?: boolean;           // Logger automatiquement les anomalies détectées
  throwOnHigh?: boolean;       // Lever une exception sur anomalie critique
  alertThreshold?: number;     // Seuil de fiabilité pour déclencher une alerte (0-1)
  enrichResponse?: boolean;    // Enrichir la réponse avec info sur anomalies
}

/**
 * Exception levée quand des anomalies critiques sont détectées
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
 * Service de gestion de la détection d'anomalies
 * Wrapper autour du détecteur d'anomalies pour faciliter son utilisation 
 * et son intégration dans l'architecture
 */
@Injectable()
export class AnomalyDetectionService {
  private readonly logger: Logger;
  private readonly defaultOptions: AnomalyDetectionOptions = {
    detectionLevel: AnomalyDetectionLevel.MEDIUM_AND_ABOVE,
    autoLog: true,
    throwOnHigh: false,
    alertThreshold: 0.6,  // Alerte si fiabilité < 60%
    enrichResponse: true
  };

  constructor(private readonly detector: AnomalyDetectorService) {
    this.logger = new Logger({
      level: 2, // INFO
      colorize: true,
      timestamp: true
    });
    // Définir le tag du logger manuellement
    (this.logger as any).logPrefix = 'AnomalyDetection';
  }

  /**
   * Détecte les anomalies dans les outputs des pools
   * avec des options personnalisées
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
    
    // Exécuter la détection
    const report = await this.detector.detectAnomalies(poolOutputs);
    
    // Filtrer selon le niveau de détection configuré
    const filteredReport = this.filterReport(report, config.detectionLevel);
    
    // Logger si nécessaire
    if (config.autoLog) {
      this.logAnomalies(filteredReport);
    }
    
    // Vérifier seuil d'alerte
    if (filteredReport.overallReliability < config.alertThreshold) {
      this.logger.warn(`Fiabilité en dessous du seuil d'alerte (${Math.round(filteredReport.overallReliability * 100)}%)`, {
        alertThreshold: config.alertThreshold,
        reliability: filteredReport.overallReliability
      });
    }
    
    // Lever une exception si des anomalies critiques sont présentes et l'option est activée
    if (config.throwOnHigh && filteredReport.highPriorityAnomalies.length > 0) {
      const message = `${filteredReport.highPriorityAnomalies.length} anomalies critiques détectées`;
      this.logger.error(message, { 
        anomalyCount: filteredReport.highPriorityAnomalies.length 
      });
      throw new CriticalAnomalyException(message, filteredReport);
    }
    
    return filteredReport;
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