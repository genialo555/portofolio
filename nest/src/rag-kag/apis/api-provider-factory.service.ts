import { Injectable, Inject } from '@nestjs/common';
import { LOGGER_TOKEN, ILogger } from '../utils/logger-tokens';
import { GoogleAiService } from './google-ai.service';
import { QwenAiService } from './qwen-ai.service';
import { DeepseekAiService } from './deepseek-ai.service';
import { ApiType } from '../types/index';

/**
 * Type pour les fournisseurs d'API
 */
export type ApiProvider = GoogleAiService | QwenAiService | DeepseekAiService;

/**
 * Options pour les fournisseurs d'API
 */
export interface ApiProviderOptions {
  provider: ApiType | string;
}

/**
 * Fabrique de fournisseurs d'API
 * Cette classe est responsable de la création et de la gestion des instances de différents fournisseurs d'API
 */
@Injectable()
export class ApiProviderFactory {
  private readonly logger: ILogger;

  constructor(
    @Inject(LOGGER_TOKEN) logger: ILogger,
    private readonly googleAiService: GoogleAiService,
    private readonly qwenAiService: QwenAiService,
    private readonly deepseekAiService: DeepseekAiService
  ) {
    this.logger = logger;
    this.logger.info('Fabrique de fournisseurs d\'API initialisée');
  }

  /**
   * Récupère le fournisseur d'API en fonction du type
   * @param providerType Type de fournisseur d'API
   * @returns Le fournisseur d'API approprié
   */
  getProvider(providerType: ApiType | string): ApiProvider {
    this.logger.debug('Récupération du fournisseur d\'API', { providerType });

    switch (providerType) {
      case ApiType.GOOGLE_AI:
        return this.googleAiService;
      case ApiType.QWEN_AI:
        return this.qwenAiService;
      case ApiType.DEEPSEEK_AI:
        return this.deepseekAiService;
      default:
        this.logger.warn('Type de fournisseur non supporté, utilisation de Google AI par défaut', { providerType });
        return this.googleAiService;
    }
  }

  /**
   * Génère une réponse en utilisant le fournisseur d'API spécifié
   * @param provider Type de fournisseur d'API ou options
   * @param prompt Prompt à envoyer
   * @param options Options supplémentaires
   * @returns La réponse générée
   */
  async generateResponse(
    provider: ApiType | string | ApiProviderOptions,
    prompt: string,
    options: any = {}
  ): Promise<any> {
    try {
      const providerType = typeof provider === 'object' ? provider.provider : provider;
      const apiProvider = this.getProvider(providerType);
      
      this.logger.info('Génération de réponse via un fournisseur API', { 
        providerType, 
        promptLength: prompt.length 
      });
      
      return await apiProvider.generateResponse(prompt, options);
    } catch (error) {
      this.logger.error('Erreur lors de la génération de réponse', { error });
      throw error;
    }
  }
} 