import { Module } from '@nestjs/common';
import { LoggerProvider, LOGGER_TOKEN } from './utils/logger-tokens';
import { OrchestratorService } from './orchestrator/orchestrator.service';
import { RouterService } from './orchestrator/router.service';
import { OutputCollectorService } from './orchestrator/output-collector.service';
import { PoolManagerService } from './pools/pool-manager.service';
import { CommercialPoolService } from './pools/commercial-pool.service';
import { MarketingPoolService } from './pools/marketing-pool.service';
import { SectorialPoolService } from './pools/sectorial-pool.service';
import { EducationalPoolService } from './pools/educational-pool.service';
import { DebateService } from './debate/debate.service';
import { KagEngineService } from './debate/kag-engine.service';
import { RagEngineService } from './debate/rag-engine.service';
import { SynthesisService } from './synthesis/synthesis.service';
import { PromptsService } from './prompts/prompts.service';
import { TypesModule } from './types/types.module';
import { RagKagController } from './controllers/rag-kag.controller';
import { ApiProviderFactory } from './apis/api-provider-factory.service';
import { GoogleAiService } from './apis/google-ai.service';
import { QwenAiService } from './apis/qwen-ai.service';
import { DeepseekAiService } from './apis/deepseek-ai.service';
import { HouseModelService } from './apis/house-model.service';
import { ModelTrainingService } from './apis/model-training.service';
import { ModelUtilsService } from './apis/model-utils.service';
import { TokenizerService } from './apis/tokenizer.service';
import { ModelEvaluationService } from './apis/model-evaluation.service';
import { ResilienceService } from './utils/resilience.service';
import { AgentFactoryService } from './agents/agent-factory.service';
import { AnomalyDetectorModule } from '../utils/anomaly-detector.module';
import { ScheduleModule } from '@nestjs/schedule';
import { CoreModule } from './core/core.module';
import { AutoTestService } from './testing/auto-test.service';

/**
 * Module principal du système RAG/KAG
 */
@Module({
  imports: [
    TypesModule,
    AnomalyDetectorModule,
    CoreModule, // Module pour les services de base (EventBus, KnowledgeGraph)
    ScheduleModule.forRoot() // Pour les tâches périodiques
  ],
  controllers: [
    RagKagController
  ],
  providers: [
    // Logger
    LoggerProvider,
    
    // Résilience
    ResilienceService,
    
    // APIs
    ApiProviderFactory,
    GoogleAiService,
    QwenAiService,
    DeepseekAiService,
    HouseModelService,
    ModelTrainingService,
    ModelUtilsService, // Service d'utilités pour les modèles TensorFlow
    TokenizerService, // Service de tokenization
    ModelEvaluationService, // Service d'évaluation de modèles
    
    // Agents
    AgentFactoryService,
    
    // Orchestration
    OrchestratorService,
    RouterService,
    OutputCollectorService,
    
    // Pools
    PoolManagerService,
    CommercialPoolService,
    MarketingPoolService,
    SectorialPoolService,
    EducationalPoolService,
    
    // Débat
    DebateService,
    KagEngineService,
    RagEngineService,
    
    // Synthèse
    SynthesisService,
    
    // Prompts
    PromptsService,
    AutoTestService,
  ],
  exports: [
    OrchestratorService,
    DebateService,
    SynthesisService,
    PromptsService,
    ApiProviderFactory,
    AgentFactoryService,
    HouseModelService,
    ModelTrainingService,
    EducationalPoolService,
    ModelUtilsService,
    TokenizerService,
    ModelEvaluationService,
    ResilienceService,
    AutoTestService
  ]
})
export class RagKagModule {} 