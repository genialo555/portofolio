import { Injectable, Inject } from '@nestjs/common';
import { LOGGER_TOKEN, ILogger } from '../utils/logger-tokens';
import { ApiProviderFactory } from '../apis/api-provider-factory.service';
import { AgentResult, PoolType, ApiType } from '../types/index';
import { AgentResponse } from '../../types/agent.types';

/**
 * Interface pour la définition d'un agent
 */
export interface Agent {
  id: string;
  name: string;
  poolType: PoolType;
  api: ApiType;
  parameters: {
    temperature: number;
    top_p: number;
    top_k: number;
    max_tokens: number;
    repetition_penalty?: number;
  };
  promptTemplate: string;
}

/**
 * Service de fabrique d'agents
 * Ce service est responsable de la gestion des agents et de leur exécution
 */
@Injectable()
export class AgentFactoryService {
  private readonly logger: ILogger;
  
  // Catalogue simulé d'agents disponibles
  private readonly agentCatalog: Agent[] = [
    {
      id: 'comm-1',
      name: 'Agent Commercial Principal',
      poolType: PoolType.COMMERCIAL,
      api: ApiType.GOOGLE_AI,
      parameters: {
        temperature: 0.7,
        top_p: 0.95,
        top_k: 40,
        max_tokens: 1000
      },
      promptTemplate: 'En tant qu\'expert commercial, analyse cette requête: {{query}}'
    },
    {
      id: 'comm-2',
      name: 'Agent Marketing Stratégique',
      poolType: PoolType.COMMERCIAL,
      api: ApiType.QWEN_AI,
      parameters: {
        temperature: 0.8,
        top_p: 0.9,
        top_k: 50,
        max_tokens: 1200
      },
      promptTemplate: 'Fais une analyse commerciale approfondie de: {{query}}'
    },
    {
      id: 'mark-1',
      name: 'Agent Marketing Créatif',
      poolType: PoolType.MARKETING,
      api: ApiType.QWEN_AI,
      parameters: {
        temperature: 0.9,
        top_p: 0.95,
        top_k: 60,
        max_tokens: 1500
      },
      promptTemplate: 'Analyse cette requête du point de vue marketing: {{query}}'
    },
    {
      id: 'mark-2',
      name: 'Agent Marketing Analytique',
      poolType: PoolType.MARKETING,
      api: ApiType.DEEPSEEK_AI,
      parameters: {
        temperature: 0.6,
        top_p: 0.85,
        top_k: 30,
        max_tokens: 1000,
        repetition_penalty: 1.1
      },
      promptTemplate: 'Fournis une analyse marketing basée sur les données pour: {{query}}'
    },
    {
      id: 'sect-1',
      name: 'Agent Sectoriel Finance',
      poolType: PoolType.SECTORIEL,
      api: ApiType.GOOGLE_AI,
      parameters: {
        temperature: 0.5,
        top_p: 0.8,
        top_k: 40,
        max_tokens: 1200
      },
      promptTemplate: 'Analyse cette requête avec un focus sur le secteur financier: {{query}}'
    },
    {
      id: 'sect-2',
      name: 'Agent Sectoriel Tech',
      poolType: PoolType.SECTORIEL,
      api: ApiType.DEEPSEEK_AI,
      parameters: {
        temperature: 0.7,
        top_p: 0.9,
        top_k: 50,
        max_tokens: 1500
      },
      promptTemplate: 'Analyse cette requête avec une perspective technologique: {{query}}'
    }
  ];

  constructor(
    @Inject(LOGGER_TOKEN) logger: ILogger,
    private readonly apiProviderFactory: ApiProviderFactory
  ) {
    this.logger = logger;
    this.logger.info('Service de fabrique d\'agents initialisé');
  }

  /**
   * Récupère les agents disponibles pour un pool spécifique
   * @param poolType Type de pool
   * @returns Liste des agents du pool
   */
  getAgentsForPool(poolType: PoolType): Agent[] {
    this.logger.debug('Récupération des agents pour le pool', { poolType });
    return this.agentCatalog.filter(agent => agent.poolType === poolType);
  }

  /**
   * Récupère un agent par son ID
   * @param agentId ID de l'agent
   * @returns L'agent correspondant ou undefined
   */
  getAgentById(agentId: string): Agent | undefined {
    this.logger.debug('Récupération de l\'agent par ID', { agentId });
    return this.agentCatalog.find(agent => agent.id === agentId);
  }

  /**
   * Exécute un prompt sur un agent spécifique
   * @param agentId ID de l'agent à utiliser
   * @param prompt Prompt à exécuter
   * @returns Résultat de l'exécution
   */
  async executeAgent(agentId: string, prompt: string): Promise<AgentResult> {
    const startTime = Date.now();
    this.logger.info('Exécution de l\'agent', { agentId, promptLength: prompt.length });
    
    try {
      const agent = this.getAgentById(agentId);
      
      if (!agent) {
        throw new Error(`Agent non trouvé: ${agentId}`);
      }
      
      // Préparer le prompt final en remplaçant les placeholders
      const finalPrompt = agent.promptTemplate.replace('{{query}}', prompt);
      
      // Exécuter via le fournisseur d'API approprié
      const response = await this.apiProviderFactory.generateResponse(
        agent.api,
        finalPrompt,
        agent.parameters
      );
      
      const endTime = Date.now();
      
      // Construire le résultat
      const result: AgentResult = {
        agentId: agent.id,
        agentName: agent.name,
        poolType: agent.poolType,
        content: response.response,
        timestamp: endTime,
        processingTime: endTime - startTime,
        metadata: {
          api: agent.api,
          temperature: agent.parameters.temperature,
          tokensUsed: response.usage.totalTokens
        }
      };
      
      this.logger.info('Agent exécuté avec succès', { 
        agentId, 
        processingTime: result.processingTime 
      });
      
      return result;
    } catch (error) {
      const endTime = Date.now();
      
      this.logger.error('Erreur lors de l\'exécution de l\'agent', { 
        agentId, 
        error 
      });
      
      return {
        agentId,
        agentName: 'Unknown',
        poolType: PoolType.COMMERCIAL, // Valeur par défaut
        content: '',
        timestamp: endTime,
        processingTime: endTime - startTime,
        error: error.message || 'Erreur inconnue lors de l\'exécution de l\'agent'
      };
    }
  }

  /**
   * Crée un agent à partir de son ID
   * @param params Paramètres pour la création de l'agent
   * @returns Un agent prêt à traiter des requêtes
   */
  async createAgent(params: {
    id: string;
    poolType: PoolType;
    api: ApiType;
  }): Promise<{
    processQuery: (query: string) => Promise<AgentResponse>;
  }> {
    const { id, poolType, api } = params;
    this.logger.debug('Création d\'un agent', { id, poolType, api });
    
    // Dans une implémentation réelle, on récupérerait la configuration de l'agent depuis une source
    // et on construirait l'agent en fonction de ses spécificités
    
    return {
      processQuery: async (query: string): Promise<AgentResponse> => {
        const startTime = Date.now();
        
        try {
          // Récupérer la configuration de l'agent depuis le catalogue
          const agentConfig = this.agentCatalog.find(a => a.id === id);
          
          if (!agentConfig) {
            throw new Error(`Agent non trouvé: ${id}`);
          }
          
          // Préparer le prompt
          const prompt = agentConfig.promptTemplate.replace('{{query}}', query);
          
          // Appeler le service d'API approprié
          const response = await this.apiProviderFactory.generateResponse(
            api,
            prompt,
            agentConfig.parameters
          );
          
          const processingTime = Date.now() - startTime;
          
          // Construire la réponse
          return {
            agentId: id,
            content: response.response || response.content,
            confidence: 0.8, // Valeur par défaut ou calculée selon la réponse
            metadata: {
              processingTime,
              tokenCount: response.usage?.totalTokens || 0,
              modelUsed: response.modelUsed || api,
              poolType
            }
          };
        } catch (error) {
          this.logger.error('Erreur lors du traitement de la requête par l\'agent', { id, error });
          
          // Retourner une réponse d'erreur
          return {
            agentId: id,
            content: 'Une erreur est survenue lors du traitement de la requête.',
            confidence: 0,
            metadata: {
              processingTime: Date.now() - startTime,
              tokenCount: 0,
              error: error.message
            }
          };
        }
      }
    };
  }
} 