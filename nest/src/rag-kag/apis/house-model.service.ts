import { Injectable, Inject } from '@nestjs/common';
import { LOGGER_TOKEN, ILogger } from '../utils/logger-tokens';
import { ResilienceService } from '../utils/resilience.service';
import { ApiUsageMetrics } from '../types/index';
import axios from 'axios'; // Import Axios for making HTTP requests

/**
 * Service pour interagir avec les modèles maison
 */
@Injectable()
export class HouseModelService {
  private readonly logger: ILogger;
  private readonly serviceName = 'house-model';

  constructor(
    @Inject(LOGGER_TOKEN) logger: ILogger,
    private readonly resilienceService: ResilienceService
  ) {
    this.logger = logger;
    this.logger.info('Service de modèles maison initialisé');
  }

  /**
   * Génère une réponse à partir d'un modèle maison
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
    this.logger.info('Demande de génération à un modèle maison', {
      promptLength: prompt.length
    });

    const executionFn = async () => {
      const model = options.model || 'phi-3-mini';
      
      // TODO: Valider les entrées et gérer les erreurs
      const response = await this.callPythonApi(prompt, model, options);
      
      return {
        response: response.data.text,
        usage: response.data.usage,
        model
      };
    };
    
    // Fallback en cas d'échec
    const fallbackFn = async (error: Error) => {
      this.logger.warn('Fallback pour le modèle maison activé', { error: error.message });
      
      return {
        response: `Réponse de secours (modèle maison non disponible): "${prompt.substring(0, 20)}..."`,
        usage: {
          promptTokens: Math.ceil(prompt.length / 4),
          completionTokens: 50,
          totalTokens: Math.ceil(prompt.length / 4) + 50,
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
      this.logger.error('Erreur critique avec le modèle maison', { error });
      throw error;
    }
  }
  
  /**
   * Appelle l'API Python pour générer une réponse à partir d'un modèle
   * @param prompt Texte d'entrée 
   * @param model Nom du modèle à utiliser
   * @param options Options supplémentaires
   * @returns Réponse de l'API
   */
  private async callPythonApi(prompt: string, model: string, options: any): Promise<any> {
    // TODO: Définir une interface TypeScript pour la requête et la réponse
    const requestData = {
      prompt,
      model,
      ...options
    };
    
    // TODO: Gérer les erreurs et la validation de la réponse
    const response = await axios.post('http://localhost:5000/generate', requestData);
    
    return response;
  }
} 