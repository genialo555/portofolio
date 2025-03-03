import { Injectable, Inject } from '@nestjs/common';
import { LOGGER_TOKEN, ILogger } from '../utils/logger-tokens';
import { ResilienceService } from '../utils/resilience.service';
import { ApiUsageMetrics } from '../types/index';

/**
 * Service pour interagir avec l'API Qwen AI
 */
@Injectable()
export class QwenAiService {
  private readonly logger: ILogger;
  private readonly apiKey: string;
  private readonly serviceName = 'qwen-ai';

  constructor(
    @Inject(LOGGER_TOKEN) logger: ILogger,
    private readonly resilienceService: ResilienceService
  ) {
    this.logger = logger;
    // Dans une implémentation réelle, récupérer depuis l'environnement
    this.apiKey = process.env.QWEN_API_KEY || 'dummy-api-key';
    this.logger.info('Service Qwen AI initialisé');
  }

  /**
   * Génère une réponse à partir de l'API Qwen AI
   * @param prompt Texte d'entrée
   * @param options Options supplémentaires
   * @returns Réponse générée et métriques
   */
  async generateResponse(
    prompt: string,
    options: { 
      temperature?: number;
      maxTokens?: number;
      model?: string;
    } = {}
  ): Promise<{ response: string; usage: ApiUsageMetrics; model: string }> {
    this.logger.info('Demande de génération à Qwen AI', {
      promptLength: prompt.length
    });

    const executionFn = async () => {
      // Simuler un appel API pour le moment
      const model = options.model || 'qwen-72b';
      const delay = 1000 + Math.random() * 2000; // 1-3 secondes
      
      // Simuler un échec aléatoire pour tester le circuit breaker (15% de chance)
      if (Math.random() < 0.15) {
        throw new Error('Erreur simulée de l\'API Qwen AI');
      }
      
      await new Promise(resolve => setTimeout(resolve, delay));
      
      return {
        response: `Réponse générée par Qwen AI (${model}): "${prompt.substring(0, 30)}..."`,
        usage: {
          promptTokens: Math.ceil(prompt.length / 4),
          completionTokens: 120,
          totalTokens: Math.ceil(prompt.length / 4) + 120,
          processingTime: delay,
          cost: 0.015
        },
        model
      };
    };
    
    // Fallback en cas d'échec
    const fallbackFn = async (error: Error) => {
      this.logger.warn('Fallback pour Qwen AI activé', { error: error.message });
      
      return {
        response: `Réponse de secours (Qwen AI non disponible): "${prompt.substring(0, 20)}..."`,
        usage: {
          promptTokens: Math.ceil(prompt.length / 4),
          completionTokens: 40,
          totalTokens: Math.ceil(prompt.length / 4) + 40,
          processingTime: 100,
          cost: 0.001
        },
        model: 'fallback-model'
      };
    };
    
    // Exécuter avec protection du circuit breaker
    try {
      return await this.resilienceService.executeWithCircuitBreaker(
        this.serviceName,
        executionFn,
        fallbackFn
      );
    } catch (error) {
      this.logger.error('Erreur critique avec Qwen AI', { error });
      throw error;
    }
  }
} 