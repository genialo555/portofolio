import { Injectable, Inject } from '@nestjs/common';
import { LOGGER_TOKEN, ILogger } from '../utils/logger-tokens';
import { AgentFactoryService } from '../agents/agent-factory.service';
import { AgentResponse, PoolType, ApiType } from '../types';
import { educationalAgents } from '../../legacy/config/poolConfig';

/**
 * Service gérant le pool d'agents éducatifs
 * Ce pool utilise principalement le modèle maison (HouseModel) optimisé pour l'enseignement 
 * et contient différents types d'agents spécialisés dans la pédagogie
 */
@Injectable()
export class EducationalPoolService {
  private readonly logger: ILogger;
  private readonly agentIds: string[];

  constructor(
    @Inject(LOGGER_TOKEN) logger: ILogger,
    private readonly agentFactory: AgentFactoryService
  ) {
    this.logger = logger;
    this.agentIds = educationalAgents.map(agent => agent.id);
    this.logger.info('Service du pool éducatif initialisé', { agentCount: this.agentIds.length });
  }

  /**
   * Traite une requête avec tous les agents du pool éducatif
   * @param query Requête à traiter
   * @param options Options de traitement
   * @returns Réponses des agents du pool
   */
  async processQuery(
    query: string,
    options: {
      timeout?: number;
      maxAgents?: number;
      confidenceThreshold?: number;
      adaptationLevel?: 'simple' | 'interactive' | 'advanced' | 'junior';
    } = {}
  ): Promise<AgentResponse[]> {
    const startTime = Date.now();
    const maxAgents = options.maxAgents || this.agentIds.length;
    
    this.logger.info('Traitement de requête par le pool éducatif', {
      queryLength: query.length,
      maxAgents,
      options
    });

    // Sélection des agents basée sur les options (adaptation au niveau pédagogique)
    let selectedAgentIds = [...this.agentIds];
    if (options.adaptationLevel) {
      switch (options.adaptationLevel) {
        case 'simple':
          selectedAgentIds = [this.findAgentIdByName('Agent Éducatif Général')];
          break;
        case 'interactive':
          selectedAgentIds = [this.findAgentIdByName('Agent Éducatif Interactif')];
          break;
        case 'advanced':
          selectedAgentIds = [this.findAgentIdByName('Agent Éducatif Spécialisé')];
          break;
        case 'junior':
          selectedAgentIds = [this.findAgentIdByName('Agent Éducatif Junior')];
          break;
      }
    }

    // Limitation du nombre d'agents si nécessaire
    if (selectedAgentIds.length > maxAgents) {
      selectedAgentIds = selectedAgentIds.slice(0, maxAgents);
    }

    try {
      // Création des promesses pour chaque agent
      const agentPromises = selectedAgentIds.map(agentId => {
        return this.agentFactory.createAgent({
          id: agentId,
          poolType: PoolType.EDUCATIONAL,
          api: ApiType.HOUSE_MODEL
        }).then(agent => {
          return agent.processQuery(query);
        });
      });

      // Exécution parallèle avec timeout éventuel
      let responses: AgentResponse[];
      if (options.timeout) {
        const timeoutPromise = new Promise<AgentResponse[]>((resolve) => {
          setTimeout(() => resolve([]), options.timeout);
        });
        responses = await Promise.race([Promise.all(agentPromises), timeoutPromise]);
      } else {
        responses = await Promise.all(agentPromises);
      }

      // Filtrage par seuil de confiance si nécessaire
      if (options.confidenceThreshold) {
        responses = responses.filter(r => r.confidence >= options.confidenceThreshold);
      }

      const processingTime = Date.now() - startTime;
      this.logger.info('Requête traitée par le pool éducatif', {
        agentCount: responses.length,
        processingTime
      });

      return responses;
    } catch (error) {
      this.logger.error('Erreur lors du traitement de la requête par le pool éducatif', { error });
      throw error;
    }
  }

  /**
   * Trouve l'ID d'un agent par son nom
   * @param name Nom de l'agent
   * @returns ID de l'agent ou premier ID si non trouvé
   */
  private findAgentIdByName(name: string): string {
    const agent = educationalAgents.find(a => a.name === name);
    return agent ? agent.id : this.agentIds[0];
  }
} 