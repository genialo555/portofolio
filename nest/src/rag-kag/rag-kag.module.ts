import { Module } from '@nestjs/common';
import { LoggerProvider, LOGGER_TOKEN } from './utils/logger-tokens';
import { OrchestratorService } from './orchestrator/orchestrator.service';
import { RouterService } from './orchestrator/router.service';
import { OutputCollectorService } from './orchestrator/output-collector.service';
import { PoolManagerService } from './pools/pool-manager.service';
import { CommercialPoolService } from './pools/commercial-pool.service';
import { MarketingPoolService } from './pools/marketing-pool.service';
import { SectorialPoolService } from './pools/sectorial-pool.service';
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
import { AgentFactoryService } from './agents/agent-factory.service';
import { AnomalyDetectorModule } from '../utils/anomaly-detector.module';

/**
 * Module principal du système RAG/KAG
 */
@Module({
  imports: [
    TypesModule,
    AnomalyDetectorModule
  ],
  controllers: [
    RagKagController
  ],
  providers: [
    // Logger
    LoggerProvider,
    
    // APIs
    ApiProviderFactory,
    GoogleAiService,
    QwenAiService,
    DeepseekAiService,
    
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
    
    // Débat
    DebateService,
    KagEngineService,
    RagEngineService,
    
    // Synthèse
    SynthesisService,
    
    // Prompts
    PromptsService
  ],
  exports: [
    OrchestratorService,
    DebateService,
    SynthesisService,
    PromptsService,
    ApiProviderFactory,
    AgentFactoryService
  ]
})
export class RagKagModule {} 