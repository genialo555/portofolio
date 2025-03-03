import { Injectable, Inject } from '@nestjs/common';
import { LOGGER_TOKEN, ILogger } from '../utils/logger-tokens';
import { ResilienceService } from '../utils/resilience.service';
import { ApiUsageMetrics } from '../types/index';

/**
 * Service pour interagir avec l'API DeepSeek AI
 */
@Injectable()
export class DeepseekAiService {
  private readonly logger: ILogger;
  private readonly apiKey: string;
  private readonly serviceName = 'deepseek-ai';

  constructor(
    @Inject(LOGGER_TOKEN) logger: ILogger,
    private readonly resilienceService: ResilienceService
  ) {
    this.logger = logger;
    // Dans une implémentation réelle, récupérer depuis l'environnement
    this.apiKey = process.env.DEEPSEEK_API_KEY || 'dummy-api-key';
    this.logger.info('Service DeepSeek AI initialisé');
  }

  /**
   * Génère une réponse à partir de l'API DeepSeek AI
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
    this.logger.info('Demande de génération à DeepSeek AI', {
      promptLength: prompt.length
    });

    const executionFn = async () => {
      // Simuler un appel API pour le moment
      const model = options.model || 'deepseek-67b';
      const delay = 1200 + Math.random() * 2500; // 1.2-3.7 secondes
      
      // Simuler un échec aléatoire pour tester le circuit breaker (10% de chance)
      if (Math.random() < 0.1) {
        throw new Error('Erreur simulée de l\'API DeepSeek AI');
      }
      
      await new Promise(resolve => setTimeout(resolve, delay));
      
      return {
        response: `Réponse générée par DeepSeek AI (${model}): "${prompt.substring(0, 30)}..."`,
        usage: {
          promptTokens: Math.ceil(prompt.length / 4),
          completionTokens: 180,
          totalTokens: Math.ceil(prompt.length / 4) + 180,
          processingTime: delay,
          cost: 0.018
        },
        model
      };
    };
    
    // Fallback en cas d'échec
    const fallbackFn = async (error: Error) => {
      this.logger.warn('Fallback pour DeepSeek AI activé', { error: error.message });
      
      return {
        response: `Réponse de secours (DeepSeek AI non disponible): "${prompt.substring(0, 20)}..."`,
        usage: {
          promptTokens: Math.ceil(prompt.length / 4),
          completionTokens: 45,
          totalTokens: Math.ceil(prompt.length / 4) + 45,
          processingTime: 120,
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
      this.logger.error('Erreur critique avec DeepSeek AI', { error });
      throw error;
    }
  }
} 