import { Injectable, Inject, Optional } from '@nestjs/common';
import { LOGGER_TOKEN, ILogger } from '../utils/logger-tokens';
import { CommercialPoolService } from './commercial-pool.service';
import { MarketingPoolService } from './marketing-pool.service';
import { SectorialPoolService } from './sectorial-pool.service';
import { EducationalPoolService } from './educational-pool.service';
import { UserQuery, TargetPools, PoolOutputs } from '../types';
import { EventBusService, RagKagEventType } from '../core/event-bus.service';

/**
 * Service de gestion des pools d'agents
 */
@Injectable()
export class PoolManagerService {
  private readonly logger: ILogger;
  
  constructor(
    @Inject(LOGGER_TOKEN) logger: ILogger,
    private readonly commercialPool: CommercialPoolService,
    private readonly marketingPool: MarketingPoolService,
    private readonly sectorialPool: SectorialPoolService,
    private readonly educationalPool: EducationalPoolService,
    @Optional() private readonly eventBus?: EventBusService
  ) {
    this.logger = logger;
    this.logger.info('Pool Manager initialisé');
    
    // Émettre un événement d'initialisation
    if (this.eventBus) {
      this.eventBus.emit({
        type: RagKagEventType.SYSTEM_INIT,
        source: 'PoolManagerService',
        payload: {
          poolsAvailable: ['commercial', 'marketing', 'sectoriel', 'educational']
        }
      });
    }
  }
  
  /**
   * Exécute les agents des pools ciblés en parallèle
   * @param targetPools Configuration des pools à utiliser
   * @param query Requête utilisateur
   * @returns Résultats combinés des différents pools
   */
  async executeAgents(targetPools: TargetPools, query: UserQuery): Promise<PoolOutputs> {
    this.logger.info('Exécution des agents dans les pools ciblés', { 
      targetPools,
      queryId: query.sessionId
    });
    
    // Émettre un événement de début d'exécution
    if (this.eventBus) {
      this.eventBus.emit({
        type: RagKagEventType.POOL_EXECUTION_STARTED,
        source: 'PoolManagerService',
        payload: {
          targetPools,
          query: {
            id: query.sessionId,
            content: query.content,
            preferences: query.preferences
          }
        }
      });
    }
    
    try {
      // Exécuter les pools en parallèle selon la configuration
      const poolPromises = [];
      
      if (targetPools.commercial) {
        this.emitPoolEvent(RagKagEventType.AGENT_EXECUTION_STARTED, 'commercial', query);
        poolPromises.push(this.commercialPool.executeAgents(query)
          .then(results => {
            this.emitPoolEvent(RagKagEventType.AGENT_EXECUTION_COMPLETED, 'commercial', query, { resultCount: results.length });
            return { pool: 'commercial', results };
          })
          .catch(error => {
            this.emitPoolEvent(RagKagEventType.QUERY_ERROR, 'commercial', query, { error: error.message });
            this.logger.error('Échec du pool commercial', { error: error.message });
            return { pool: 'commercial', results: [], error: error.message };
          }));
      }
      
      if (targetPools.marketing) {
        this.emitPoolEvent(RagKagEventType.AGENT_EXECUTION_STARTED, 'marketing', query);
        poolPromises.push(this.marketingPool.executeAgents(query)
          .then(results => {
            this.emitPoolEvent(RagKagEventType.AGENT_EXECUTION_COMPLETED, 'marketing', query, { resultCount: results.length });
            return { pool: 'marketing', results };
          })
          .catch(error => {
            this.emitPoolEvent(RagKagEventType.QUERY_ERROR, 'marketing', query, { error: error.message });
            this.logger.error('Échec du pool marketing', { error: error.message });
            return { pool: 'marketing', results: [], error: error.message };
          }));
      }
      
      if (targetPools.sectoriel) {
        this.emitPoolEvent(RagKagEventType.AGENT_EXECUTION_STARTED, 'sectoriel', query);
        poolPromises.push(this.sectorialPool.executeAgents(query)
          .then(results => {
            this.emitPoolEvent(RagKagEventType.AGENT_EXECUTION_COMPLETED, 'sectoriel', query, { resultCount: results.length });
            return { pool: 'sectoriel', results };
          })
          .catch(error => {
            this.emitPoolEvent(RagKagEventType.QUERY_ERROR, 'sectoriel', query, { error: error.message });
            this.logger.error('Échec du pool sectoriel', { error: error.message });
            return { pool: 'sectoriel', results: [], error: error.message };
          }));
      }
      
      if (targetPools.educational) {
        this.emitPoolEvent(RagKagEventType.AGENT_EXECUTION_STARTED, 'educational', query);
        poolPromises.push(this.educationalPool.processQuery(query.content, {
            timeout: query.preferences?.maxResponseTime,
            adaptationLevel: query.preferences?.educationalLevel as any
          })
          .then(results => {
            this.emitPoolEvent(RagKagEventType.AGENT_EXECUTION_COMPLETED, 'educational', query, { resultCount: results.length });
            return { pool: 'educational', results };
          })
          .catch(error => {
            this.emitPoolEvent(RagKagEventType.QUERY_ERROR, 'educational', query, { error: error.message });
            this.logger.error('Échec du pool éducatif', { error: error.message });
            return { pool: 'educational', results: [], error: error.message };
          }));
      }
      
      // Attendre que tous les pools terminent leur exécution
      const poolResults = await Promise.all(poolPromises);
      
      // Transformer les résultats dans le format attendu
      const poolOutputs: PoolOutputs = {
        commercial: poolResults.find(p => p.pool === 'commercial')?.results || [],
        marketing: poolResults.find(p => p.pool === 'marketing')?.results || [],
        sectoriel: poolResults.find(p => p.pool === 'sectoriel')?.results || [],
        educational: poolResults.find(p => p.pool === 'educational')?.results || [],
        errors: poolResults.filter(p => p.error).map(p => p.error),
        timestamp: new Date(),
        query
      };
      
      this.logger.info('Exécution des pools terminée', {
        commercialCount: poolOutputs.commercial.length,
        marketingCount: poolOutputs.marketing.length,
        sectorialCount: poolOutputs.sectoriel.length,
        educationalCount: poolOutputs.educational.length,
        errorCount: poolOutputs.errors.length
      });
      
      // Émettre un événement de fin d'exécution
      if (this.eventBus) {
        this.eventBus.emit({
          type: RagKagEventType.POOL_EXECUTION_COMPLETED,
          source: 'PoolManagerService',
          payload: {
            poolOutputs: {
              commercialCount: poolOutputs.commercial.length,
              marketingCount: poolOutputs.marketing.length,
              sectorialCount: poolOutputs.sectoriel.length,
              educationalCount: poolOutputs.educational.length,
              errorCount: poolOutputs.errors.length
            },
            queryId: query.sessionId,
            executionTime: Date.now() - poolOutputs.timestamp.getTime()
          }
        });
      }
      
      return poolOutputs;
      
    } catch (error) {
      this.logger.error('Erreur globale dans l\'exécution des pools', { error: error.message });
      
      // Émettre un événement d'erreur
      if (this.eventBus) {
        this.eventBus.emit({
          type: RagKagEventType.QUERY_ERROR,
          source: 'PoolManagerService',
          payload: {
            error: error.message,
            queryId: query.sessionId,
            targetPools
          }
        });
      }
      
      throw error;
    }
  }
  
  /**
   * Émet un événement lié à l'exécution d'un pool
   * @param eventType Type d'événement
   * @param poolType Type de pool
   * @param query Requête utilisateur
   * @param additionalData Données supplémentaires
   */
  private emitPoolEvent(
    eventType: RagKagEventType,
    poolType: string,
    query: UserQuery,
    additionalData: Record<string, any> = {}
  ): void {
    if (!this.eventBus) return;
    
    this.eventBus.emit({
      type: eventType,
      source: `PoolManagerService:${poolType}`,
      payload: {
        poolType,
        queryId: query.sessionId,
        ...additionalData
      }
    });
  }
} 