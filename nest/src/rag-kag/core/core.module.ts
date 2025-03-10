import { Module, OnModuleInit } from '@nestjs/common';
import { EventBusService } from './event-bus.service';
import { KnowledgeGraphService } from './knowledge-graph.service';
import { QueryAnalyzerService } from './query-analyzer.service';
import { MetricsService } from './metrics.service';
import { LoggerModule } from '../utils/logger.module';
import { KnowledgeVerifierService } from './knowledge-verifier.service';
import { ApiProviderFactory } from '../apis/api-provider-factory.service';
import { ApisModule } from '../apis/apis.module';
import { Inject, Injectable } from '@nestjs/common';
import { LOGGER_TOKEN } from '../utils/logger-tokens';
import { ILogger } from '../utils/logger-tokens';

/**
 * Initialise les connexions entre services du Core
 */
@Injectable()
export class CoreInitializer implements OnModuleInit {
  constructor(
    @Inject(LOGGER_TOKEN) private readonly logger: ILogger,
    private readonly knowledgeGraph: KnowledgeGraphService,
    private readonly knowledgeVerifier: KnowledgeVerifierService
  ) {}

  onModuleInit() {
    this.logger.log('Initialisation des services Core', { context: 'CoreInitializer' });
    
    // Connecter le vérifieur au graphe de connaissances
    this.knowledgeGraph.setKnowledgeVerifier(this.knowledgeVerifier);
    
    // Activer la vérification automatique par défaut
    this.knowledgeGraph.enableAutoVerification(true, 'STANDARD');
    
    this.logger.log('Services Core initialisés avec succès', { context: 'CoreInitializer' });
  }
}

/**
 * Module principal contenant les services fondamentaux du système RAG/KAG
 */
@Module({
  imports: [
    LoggerModule,
    ApisModule
  ],
  providers: [
    EventBusService,
    KnowledgeGraphService,
    QueryAnalyzerService,
    MetricsService,
    KnowledgeVerifierService,
    ApiProviderFactory,
    CoreInitializer
  ],
  exports: [
    EventBusService,
    KnowledgeGraphService,
    QueryAnalyzerService,
    MetricsService,
    KnowledgeVerifierService,
    ApiProviderFactory
  ]
})
export class CoreModule {} 