import { Global, Module } from '@nestjs/common';
import { ResilienceService } from '../utils/resilience.service';
import { AnomalyDetectionService } from '../../utils/anomaly-detection.service';
import { AnomalyDetectorService } from '../../legacy/utils/anomalyDetector';
import { LoggerModule } from '../utils/logger.module';
import { CoreModule } from '../core/core.module';
import { GoogleAiService } from '../apis/google-ai.service';
import { QwenAiService } from '../apis/qwen-ai.service';
import { DeepseekAiService } from '../apis/deepseek-ai.service';
import { HouseModelService } from '../apis/house-model.service';
import { ModelTrainingService } from '../apis/model-training.service';
import { ModelEvaluationService } from '../apis/model-evaluation.service';
import { ModelUtilsService } from '../apis/model-utils.service';
import { TokenizerService } from '../apis/tokenizer.service';
import { PythonApiService } from '../apis/python-api.service';

/**
 * Module commun qui regroupe tous les services partagés
 * Solution au problème des dépendances circulaires
 */
@Global()
@Module({
  imports: [
    LoggerModule,
    CoreModule,
  ],
  providers: [
    // Services de résilience
    ResilienceService,
    
    // Services de détection d'anomalies
    AnomalyDetectionService,
    AnomalyDetectorService,
    
    // Services d'API - Ces services ne sont pas fournis par CoreModule
    GoogleAiService,
    QwenAiService,
    DeepseekAiService,
    
    // Services de modèles locaux
    HouseModelService,
    ModelTrainingService,
    ModelEvaluationService,
    ModelUtilsService,
    TokenizerService,
    
    // Service d'intégration Python
    PythonApiService,
  ],
  exports: [
    // Services de résilience
    ResilienceService,
    
    // Services de détection d'anomalies
    AnomalyDetectionService,
    AnomalyDetectorService,
    
    // Services d'API
    GoogleAiService,
    QwenAiService,
    DeepseekAiService,
    
    // Services de modèles locaux
    HouseModelService,
    ModelTrainingService,
    ModelEvaluationService,
    ModelUtilsService,
    TokenizerService,
    
    // Service d'intégration Python
    PythonApiService,
    
    // Re-export des modules importés
    CoreModule,
    LoggerModule,
  ]
})
export class CommonModule {} 