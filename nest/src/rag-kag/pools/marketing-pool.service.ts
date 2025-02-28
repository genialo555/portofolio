import { Injectable, Inject } from '@nestjs/common';
import { LOGGER_TOKEN, ILogger } from '../utils/logger-tokens';
import { AgentFactoryService } from '../agents/agent-factory.service';
import { PromptsService, PromptTemplateType } from '../prompts/prompts.service';
import { UserQuery, PoolType, AgentResult } from '../types';

/**
 * Service gérant le pool d'agents marketing
 */
@Injectable()
export class MarketingPoolService {
  private readonly logger: ILogger;

  constructor(
    @Inject(LOGGER_TOKEN) logger: ILogger,
    private readonly agentFactory: AgentFactoryService,
    private readonly promptsService: PromptsService
  ) {
    this.logger = logger;
    this.logger.info('Service de pool marketing initialisé');
  }
  
  /**
   * Exécute les agents du pool marketing sur une requête utilisateur
   * @param query Requête à traiter
   * @returns Résultats des agents marketing
   */
  async executeAgents(query: UserQuery): Promise<AgentResult[]> {
    this.logger.info('Exécution des agents marketing', { queryId: query.sessionId });
    
    try {
      // 1. Obtenir les agents configurés pour ce pool
      const agents = await this.agentFactory.getAgentsForPool(PoolType.MARKETING);
      
      if (!agents || agents.length === 0) {
        this.logger.warn('Aucun agent marketing disponible');
        return [];
      }
      
      this.logger.debug(`${agents.length} agents marketing disponibles`);
      
      // 2. Exécuter tous les agents en parallèle
      const agentResults = await Promise.all(
        agents.map(async (agent) => {
          try {
            // Préparer le prompt avec le contexte de la requête
            const finalPrompt = query.text;
            
            // Exécution de l'agent via la fabrique d'agents
            return this.agentFactory.executeAgent(agent.id, finalPrompt);
            
          } catch (error) {
            this.logger.error(`Erreur avec l'agent ${agent.name}`, { error });
            
            // Créer un résultat d'erreur formaté comme AgentResult
            return {
              agentId: agent.id,
              agentName: agent.name || 'Agent inconnu',
              poolType: PoolType.MARKETING,
              content: '',
              timestamp: Date.now(),
              processingTime: 0,
              error: error instanceof Error ? error.message : 'Erreur inconnue'
            };
          }
        })
      );
      
      this.logger.info('Traitement des agents marketing terminé', {
        successCount: agentResults.filter(r => !r.error).length,
        totalCount: agentResults.length
      });
      
      return agentResults;
      
    } catch (error) {
      this.logger.error('Erreur dans l\'exécution du pool marketing', { 
        error
      });
      throw error;
    }
  }
} 