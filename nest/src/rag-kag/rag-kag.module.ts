import { Module } from '@nestjs/common';
import { LoggerProvider } from './utils/logger-tokens';
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
import { AgentFactoryService } from './agents/agent-factory.service';
import { ScheduleModule } from '@nestjs/schedule';
import { AutoTestService } from './testing/auto-test.service';
import { CommonModule } from './common/common.module';
import { ComplexityAnalyzerService } from './utils/complexity-analyzer.service';

/**
 * Module principal du système RAG/KAG
 * Utilise maintenant CommonModule pour éviter les dépendances circulaires
 */
@Module({
  imports: [
    TypesModule,
    ScheduleModule.forRoot(), // Pour les tâches périodiques
    CommonModule, // Module qui regroupe tous les services partagés
  ],
  controllers: [
    RagKagController
  ],
  providers: [
    // Logger
    LoggerProvider,
    
    // Services de complexité
    ComplexityAnalyzerService,
    
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
    AgentFactoryService,
    EducationalPoolService,
    AutoTestService,
    ComplexityAnalyzerService,
  ]
})
export class RagKagModule {} 