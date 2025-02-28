import { Injectable, Inject } from '@nestjs/common';
import { LOGGER_TOKEN, ILogger } from '../utils/logger-tokens';
import { ApiUsageMetrics } from '../types/index';

/**
 * Service pour interagir avec l'API Qwen AI
 */
@Injectable()
export class QwenAiService {
  private readonly logger: ILogger;

  constructor(
    @Inject(LOGGER_TOKEN) logger: ILogger
  ) {
    this.logger = logger;
    this.logger.info('Service Qwen AI initialisé');
  }

  /**
   * Génère une réponse à partir de l'API Qwen AI
   * @param prompt Prompt à envoyer à l'API
   * @param options Options pour la génération
   * @returns Réponse générée et métriques d'utilisation
   */
  async generateResponse(
    prompt: string,
    options: {
      temperature?: number;
      maxTokens?: number;
      topP?: number;
      topK?: number;
      repetitionPenalty?: number;
    } = {}
  ): Promise<{ response: string; usage: ApiUsageMetrics }> {
    const startTime = Date.now();
    
    this.logger.info("Envoi d'une requête à Qwen AI", {
      promptLength: prompt.length,
      options
    });

    // Simulation d'un appel à l'API Qwen AI
    // À remplacer par un vrai appel API
    const delay = Math.random() * 1500 + 800;
    await new Promise(resolve => setTimeout(resolve, delay));

    const responseText = `Ceci est une réponse simulée de l'API Qwen AI pour le prompt : "${prompt.substring(0, 50)}..."`;
    
    const endTime = Date.now();
    const processingTime = endTime - startTime;

    // Simulation des métriques d'utilisation
    const usage: ApiUsageMetrics = {
      promptTokens: Math.ceil(prompt.length / 4),
      completionTokens: Math.ceil(responseText.length / 4),
      totalTokens: Math.ceil((prompt.length + responseText.length) / 4),
      processingTime,
      cost: Math.ceil((prompt.length + responseText.length) / 4) * 0.000007
    };

    this.logger.info('Réponse reçue de Qwen AI', { 
      responseLength: responseText.length, 
      processingTime, 
      usage 
    });

    return {
      response: responseText,
      usage
    };
  }
} 