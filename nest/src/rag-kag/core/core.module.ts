import { Module } from '@nestjs/common';
import { EventBusService } from './event-bus.service';
import { KnowledgeGraphService } from './knowledge-graph.service';
import { QueryAnalyzerService } from './query-analyzer.service';
import { MetricsService } from './metrics.service';
import { LoggerModule } from '../utils/logger.module';

/**
 * Module pour les services de base (core) du système RAG/KAG
 * Inclut les services fondamentaux comme le bus d'événements et le graphe de connaissances
 */
@Module({
  imports: [LoggerModule],
  providers: [EventBusService, KnowledgeGraphService, QueryAnalyzerService, MetricsService],
  exports: [EventBusService, KnowledgeGraphService, QueryAnalyzerService, MetricsService]
})
export class CoreModule {} 