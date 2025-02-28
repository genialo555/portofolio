import { Injectable, Inject } from '@nestjs/common';
import { LOGGER_TOKEN, ILogger } from '../utils/logger-tokens';
import { CommercialPoolService } from './commercial-pool.service';
import { MarketingPoolService } from './marketing-pool.service';
import { SectorialPoolService } from './sectorial-pool.service';
import { UserQuery, TargetPools, PoolOutputs } from '../types';

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
    private readonly sectorialPool: SectorialPoolService
  ) {
    this.logger = logger;
    this.logger.info('Pool Manager initialisé');
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
    
    try {
      // Exécuter les pools en parallèle selon la configuration
      const poolPromises = [];
      
      if (targetPools.commercial) {
        poolPromises.push(this.commercialPool.executeAgents(query)
          .then(results => ({ pool: 'commercial', results }))
          .catch(error => {
            this.logger.error('Échec du pool commercial', { error: error.message });
            return { pool: 'commercial', results: [], error: error.message };
          }));
      }
      
      if (targetPools.marketing) {
        poolPromises.push(this.marketingPool.executeAgents(query)
          .then(results => ({ pool: 'marketing', results }))
          .catch(error => {
            this.logger.error('Échec du pool marketing', { error: error.message });
            return { pool: 'marketing', results: [], error: error.message };
          }));
      }
      
      if (targetPools.sectoriel) {
        poolPromises.push(this.sectorialPool.executeAgents(query)
          .then(results => ({ pool: 'sectoriel', results }))
          .catch(error => {
            this.logger.error('Échec du pool sectoriel', { error: error.message });
            return { pool: 'sectoriel', results: [], error: error.message };
          }));
      }
      
      // Attendre que tous les pools terminent leur exécution
      const poolResults = await Promise.all(poolPromises);
      
      // Transformer les résultats dans le format attendu
      const poolOutputs: PoolOutputs = {
        commercial: poolResults.find(p => p.pool === 'commercial')?.results || [],
        marketing: poolResults.find(p => p.pool === 'marketing')?.results || [],
        sectoriel: poolResults.find(p => p.pool === 'sectoriel')?.results || [],
        errors: poolResults.filter(p => p.error).map(p => p.error)
      };
      
      this.logger.info('Exécution des pools terminée', {
        commercialCount: poolOutputs.commercial.length,
        marketingCount: poolOutputs.marketing.length,
        sectorialCount: poolOutputs.sectoriel.length,
        errorCount: poolOutputs.errors.length
      });
      
      return poolOutputs;
      
    } catch (error) {
      this.logger.error('Erreur globale dans l\'exécution des pools', { error: error.message });
      throw error;
    }
  }
} 