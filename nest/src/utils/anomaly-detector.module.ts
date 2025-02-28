import { Module } from '@nestjs/common';
import { AnomalyDetectorService } from './anomalyDetector';
import { AnomalyDetectionService } from './anomaly-detection.service';

/**
 * Module pour la détection d'anomalies
 * Intègre le détecteur d'anomalies dans l'architecture NestJS
 */
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