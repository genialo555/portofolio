import { Module, Global } from '@nestjs/common';
import { AnomalyDetectorService } from '../legacy/utils/anomalyDetector';
import { AnomalyDetectionService } from './anomaly-detection.service';

/**
 * Module pour la détection d'anomalies
 * Intègre le détecteur d'anomalies dans l'architecture NestJS
 */
@Global()  // Rendre le module global pour partager ses providers
@Module({
  providers: [
    AnomalyDetectorService,
    AnomalyDetectionService
  ],
  exports: [
    AnomalyDetectorService,
    AnomalyDetectionService
  ],
})
export class AnomalyDetectorModule {} 