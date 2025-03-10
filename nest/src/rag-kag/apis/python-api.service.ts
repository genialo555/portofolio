import { Injectable, Inject } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import { LOGGER_TOKEN, ILogger } from '../utils/logger-tokens';
import axios, { AxiosInstance, AxiosRequestConfig } from 'axios';
import { ResilienceService } from '../utils/resilience.service';

/**
 * Interface pour les options de la requête d'entraînement
 */
export interface TrainingRequestOptions {
  model: string;
  epochs?: number;
  batchSize?: number;
  learningRate?: number;
  validationSplit?: number;
  maxExamples?: number;
  useCache?: boolean;
  saveToDisk?: boolean;
  outputDirectory?: string;
}

/**
 * Interface pour les options de la requête d'inférence
 */
export interface InferenceRequestOptions {
  temperature?: number;
  maxTokens?: number;
  topP?: number;
  topK?: number;
  repetitionPenalty?: number;
  stop?: string[];
}

/**
 * Interface pour les résultats de l'entraînement
 */
export interface TrainingResult {
  success: boolean;
  model: string;
  trainedExamples: number;
  epochs: number;
  accuracy: number;
  loss: number;
  validationAccuracy?: number;
  validationLoss?: number;
  trainingTime: number;
  modelSize: number;
  timestamp: string;
  message?: string;
  checkpointPath?: string;
  metrics?: Record<string, any>;
}

/**
 * Interface pour les résultats de l'inférence
 */
export interface InferenceResult {
  text: string;
  logprobs?: number[];
  tokensUsed: number;
  generationTime: number;
  model: string;
}

/**
 * Service pour l'intégration avec l'API Python
 */
@Injectable()
export class PythonApiService {
  private readonly httpClient: AxiosInstance;
  private readonly baseUrl: string;
  private readonly apiKey: string;
  private isApiAvailable: boolean = false;
  private readonly retryCount: number = 3;
  private readonly timeoutMs: number = 30000;

  constructor(
    @Inject(LOGGER_TOKEN) private readonly logger: ILogger,
    private readonly configService: ConfigService,
    private readonly resilienceService: ResilienceService
  ) {
    this.baseUrl = this.configService.get<string>('PYTHON_API_URL') || 'http://localhost:5000';
    this.apiKey = this.configService.get<string>('PYTHON_API_KEY') || 'dev-key';
    
    // Configuration du client HTTP
    const config: AxiosRequestConfig = {
      baseURL: this.baseUrl,
      timeout: this.timeoutMs,
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${this.apiKey}`
      }
    };
    
    this.httpClient = axios.create(config);
    
    // Vérification de la disponibilité de l'API au démarrage
    this.checkApiAvailability();
    
    this.logger.info(`Service PythonApiService initialisé avec l'URL: ${this.baseUrl}`);
  }
  
  /**
   * Vérifie la disponibilité de l'API Python
   */
  private async checkApiAvailability(): Promise<void> {
    try {
      const response = await this.httpClient.get('/health');
      
      if (response.status === 200 && response.data?.status === 'ok') {
        this.isApiAvailable = true;
        this.logger.info('API Python disponible et opérationnelle', {
          version: response.data?.version || 'unknown',
          models: response.data?.models || []
        });
      } else {
        this.isApiAvailable = false;
        this.logger.warn('API Python disponible mais renvoie un statut non-OK', {
          status: response.data?.status,
          message: response.data?.message
        });
      }
    } catch (error) {
      this.isApiAvailable = false;
      this.logger.error('API Python inaccessible', {
        error: error.message,
        baseUrl: this.baseUrl
      });
    }
  }
  
  /**
   * Entraîne un modèle via l'API Python
   * @param modelName Nom du modèle à entraîner
   * @param options Options d'entraînement
   * @returns Résultat de l'entraînement
   */
  public async trainModel(modelName: string, options?: Partial<TrainingRequestOptions>): Promise<TrainingResult> {
    if (!this.isApiAvailable) {
      await this.checkApiAvailability();
      
      if (!this.isApiAvailable) {
        throw new Error('API Python inaccessible pour l\'entraînement du modèle');
      }
    }
    
    // Préparation des données de la requête
    const requestData = {
      model: modelName,
      ...options
    };
    
    // Utilisation du service de résilience pour gérer les erreurs et retries
    return this.resilienceService.executeWithRetry(
      async () => {
        const startTime = Date.now();
        this.logger.info(`Début de l'entraînement du modèle ${modelName} via l'API Python`, { options });
        
        try {
          const response = await this.httpClient.post<TrainingResult>('/train', requestData);
          
          this.logger.info(`Entraînement du modèle ${modelName} terminé avec succès`, {
            trainingTime: Date.now() - startTime,
            accuracy: response.data.accuracy,
            examples: response.data.trainedExamples
          });
          
          return response.data;
        } catch (error) {
          this.logger.error(`Erreur lors de l'entraînement du modèle ${modelName}`, {
            error: error.message,
            stack: error.stack,
            requestData
          });
          
          // Transformer l'erreur en résultat d'entraînement échoué
          return {
            success: false,
            model: modelName,
            trainedExamples: 0,
            epochs: 0,
            accuracy: 0,
            loss: 0,
            trainingTime: Date.now() - startTime,
            modelSize: 0,
            timestamp: new Date().toISOString(),
            message: `Erreur: ${error.message}`
          };
        }
      },
      {
        maxRetries: this.retryCount,
        retryCondition: (error) => {
          // Ne pas réessayer en cas d'erreurs 4xx
          if (error.response && error.response.status >= 400 && error.response.status < 500) {
            return false;
          }
          return true;
        },
        onRetry: (error, attempt) => {
          this.logger.warn(`Tentative ${attempt}/${this.retryCount} pour l'entraînement du modèle ${modelName}`, {
            error: error.message
          });
        }
      }
    );
  }
  
  /**
   * Génère une inférence à partir d'un modèle via l'API Python
   * @param prompt Texte d'entrée
   * @param modelName Nom du modèle à utiliser
   * @param options Options d'inférence
   * @returns Résultat de l'inférence
   */
  public async generateInference(prompt: string, modelName: string, options?: Partial<InferenceRequestOptions>): Promise<InferenceResult> {
    if (!this.isApiAvailable) {
      await this.checkApiAvailability();
      
      if (!this.isApiAvailable) {
        throw new Error('API Python inaccessible pour la génération');
      }
    }
    
    // Préparation des données de la requête
    const requestData = {
      prompt,
      model: modelName,
      ...options
    };
    
    // Utilisation du service de résilience pour gérer les erreurs et retries
    return this.resilienceService.executeWithRetry(
      async () => {
        const startTime = Date.now();
        this.logger.debug(`Génération via le modèle ${modelName}`, { 
          promptLength: prompt.length,
          options 
        });
        
        try {
          const response = await this.httpClient.post<InferenceResult>('/generate', requestData);
          
          return {
            ...response.data,
            generationTime: Date.now() - startTime
          };
        } catch (error) {
          this.logger.error(`Erreur lors de la génération avec le modèle ${modelName}`, {
            error: error.message,
            promptLength: prompt.length
          });
          
          throw error;
        }
      },
      {
        maxRetries: this.retryCount,
        retryCondition: (error) => {
          // Ne pas réessayer en cas d'erreurs 4xx
          if (error.response && error.response.status >= 400 && error.response.status < 500) {
            return false;
          }
          return true;
        }
      }
    );
  }
  
  /**
   * Récupère les métriques d'un modèle
   * @param modelName Nom du modèle
   * @returns Métriques du modèle
   */
  public async getModelMetrics(modelName: string): Promise<any> {
    try {
      const response = await this.httpClient.get(`/models/${modelName}/metrics`);
      return response.data;
    } catch (error) {
      this.logger.error(`Erreur lors de la récupération des métriques du modèle ${modelName}`, {
        error: error.message
      });
      
      return {
        success: false,
        message: `Erreur: ${error.message}`
      };
    }
  }
  
  /**
   * Vérifie si l'API Python est disponible
   * @returns true si l'API est disponible, false sinon
   */
  public isAvailable(): boolean {
    return this.isApiAvailable;
  }
} 