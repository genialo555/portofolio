import { Controller, Post, Body, Get, Param, Inject } from '@nestjs/common';
import { ApiTags, ApiOperation, ApiResponse, ApiParam } from '@nestjs/swagger';
import { ModelTrainingService } from '../apis/model-training.service';
import { LOGGER_TOKEN, ILogger } from '../utils/logger-tokens';
import { ModelEvaluationService } from '../apis/model-evaluation.service';
import { ResilienceService } from '../utils/resilience.service';
import { ApiProviderFactory } from '../apis/api-provider-factory.service';
import { AnomalyDetectionService, AnomalyDetectionLevel } from '../../utils/anomaly-detection.service';

/**
 * DTO pour la requête de l'utilisateur
 */
export class QueryRequestDto {
  query: string;
  contextInfo?: any;
  expertiseLevel?: 'BEGINNER' | 'INTERMEDIATE' | 'ADVANCED';
  useSimplifiedProcess?: boolean;
  detectAnomalies?: boolean;
  anomalyDetectionLevel?: AnomalyDetectionLevel;
  resilience?: {
    enableCircuitBreaker?: boolean;
    fallbackProvider?: string;
  };
}

/**
 * DTO pour la réponse
 */
export interface FinalResponse {
  content: string;
  metaData: {
    sourceTypes: string[];
    confidenceLevel: string;
    processingTime: number;
    usedAgentCount: number;
    expertiseLevel: string;
    topicsIdentified: string[];
  };
  suggestedFollowUp?: string[];
  error?: string;
}
/**
 * Contrôleur principal pour les requêtes RAG/KAG
 */
@Controller('api/rag-kag')
@ApiTags('rag-kag')
export class RagKagController {
  constructor(
    @Inject(LOGGER_TOKEN) private readonly logger: ILogger,
    private readonly modelTrainingService: ModelTrainingService,
    private readonly modelEvaluationService: ModelEvaluationService,
    private readonly resilienceService: ResilienceService,
    private readonly apiProviderFactory: ApiProviderFactory,
    private readonly anomalyDetectionService: AnomalyDetectionService
  ) {}

  /**
   * Endpoint pour traiter une requête utilisateur
   * @param queryRequest Requête de l'utilisateur
   * @returns Résultat du traitement
   */
  @Post('query')
  @ApiOperation({ summary: 'Traiter une requête utilisateur' })
  @ApiResponse({ 
    status: 200, 
    description: 'La requête a été traitée avec succès',
    type: Object 
  })
  async processQuery(@Body() queryRequest: QueryRequestDto): Promise<FinalResponse> {
    // Réponse simulée
    console.log(`Requête reçue: ${queryRequest.query}`);
    return {
      content: `Réponse à la question: ${queryRequest.query}`,
      metaData: {
        sourceTypes: ['KAG', 'RAG'],
        confidenceLevel: 'HIGH',
        processingTime: 500,
        usedAgentCount: 3,
        expertiseLevel: queryRequest.expertiseLevel || 'INTERMEDIATE',
        topicsIdentified: ['IA', 'RAG', 'KAG'],
      },
      suggestedFollowUp: [
        'Comment fonctionne le débat entre KAG et RAG?',
        'Quels sont les avantages du système hybride?'
      ]
    };
  }

  /**
   * Endpoint de santé pour vérifier que le service est opérationnel
   * @returns Message de statut
   */
  @Get('health')
  @ApiOperation({ summary: 'Vérifier l\'état du service' })
  @ApiResponse({ 
    status: 200, 
    description: 'Le service est opérationnel',
    schema: {
      type: 'object',
      properties: {
        status: { type: 'string' },
        timestamp: { type: 'string', format: 'date-time' },
        apis: { 
          type: 'array',
          items: { type: 'string' }
        }
      }
    }
  })
  async checkHealth() {
    return {
      status: 'ok',
      timestamp: new Date(),
      apis: ['query', 'health']
    };
  }

  /**
   * Endpoint pour forcer l'entraînement d'un modèle spécifique
   * @param modelName Nom du modèle à entraîner
   * @returns Résultat de l'entraînement
   */
  @Post('train/:modelName')
  @ApiOperation({ summary: 'Forcer l\'entraînement d\'un modèle distillé' })
  @ApiParam({
    name: 'modelName',
    description: 'Nom du modèle à entraîner (phi-3-mini, llama-3-8b, mistral-7b-fr)',
    type: String,
    required: true
  })
  @ApiResponse({ 
    status: 200, 
    description: 'L\'entraînement du modèle a été déclenché',
    schema: {
      type: 'object',
      properties: {
        success: { type: 'boolean' },
        message: { type: 'string' },
        modelName: { type: 'string' },
        timestamp: { type: 'string', format: 'date-time' }
      }
    }
  })
  async trainModel(@Param('modelName') modelName: string) {
    this.logger.info(`Demande d'entraînement pour le modèle ${modelName}`);
    
    try {
      const success = await this.modelTrainingService.forceTrainModel(modelName);
      
      if (success) {
        return {
          success: true,
          message: `L'entraînement du modèle ${modelName} a été déclenché avec succès`,
          modelName,
          timestamp: new Date()
        };
      } else {
        return {
          success: false,
          message: `Impossible de lancer l'entraînement du modèle ${modelName} (opération déjà en cours ou données insuffisantes)`,
          modelName,
          timestamp: new Date()
        };
      }
    } catch (error) {
      this.logger.error(`Erreur lors de la demande d'entraînement du modèle ${modelName}`, { error });
      return {
        success: false,
        message: `Erreur lors de la demande d'entraînement: ${error.message || 'Erreur inconnue'}`,
        modelName,
        timestamp: new Date()
      };
    }
  }

  /**
   * Endpoint pour récupérer les statistiques d'entraînement
   * @returns Statistiques d'entraînement des modèles
   */
  @Get('train/stats')
  @ApiOperation({ summary: 'Récupérer les statistiques d\'entraînement des modèles' })
  @ApiResponse({ 
    status: 200, 
    description: 'Statistiques d\'entraînement récupérées avec succès',
    schema: {
      type: 'object'
    }
  })
  async getTrainingStats() {
    try {
      return {
        ...this.modelTrainingService.getTrainingStatistics(),
        timestamp: new Date()
      };
    } catch (error) {
      this.logger.error(`Erreur lors de la récupération des statistiques d'entraînement`, { error });
      return {
        success: false,
        message: `Erreur lors de la récupération des statistiques: ${error.message || 'Erreur inconnue'}`,
        timestamp: new Date()
      };
    }
  }

  /**
   * Récupère les métriques d'évaluation des modèles
   */
  @Get('evaluation/stats')
  async getEvaluationStats() {
    this.logger.info('Récupération des métriques d\'évaluation des modèles');
    
    try {
      const evaluationStats = this.modelEvaluationService.getAllModelsEvaluationStats();
      
      return {
        status: 'success',
        data: evaluationStats
      };
    } catch (error) {
      this.logger.error('Erreur lors de la récupération des métriques d\'évaluation', { error });
      
      return {
        status: 'error',
        message: 'Une erreur est survenue lors de la récupération des métriques d\'évaluation',
        error: error.message
      };
    }
  }
  
  /**
   * Force l'évaluation d'un modèle spécifique
   */
  @Post('evaluation/:modelName')
  async evaluateModel(@Param('modelName') modelName: string) {
    this.logger.info(`Demande d'évaluation forcée du modèle ${modelName}`);
    
    try {
      // Vérifier si le modèle est valide
      const validModels = ['phi-3-mini', 'llama-3-8b', 'mistral-7b-fr'];
      
      if (!validModels.includes(modelName)) {
        return {
          status: 'error',
          message: `Le modèle ${modelName} n'est pas un modèle valide pour l'évaluation`
        };
      }
      
      // Lancer l'évaluation
      const evaluation = await this.modelEvaluationService.evaluateModel(modelName);
      
      return {
        status: 'success',
        message: `Évaluation du modèle ${modelName} terminée`,
        data: {
          modelName: evaluation.modelName,
          timestamp: evaluation.timestamp,
          metrics: evaluation.metrics,
          comparisonToTeacher: evaluation.comparisonToTeacher
        }
      };
    } catch (error) {
      this.logger.error(`Erreur lors de l'évaluation du modèle ${modelName}`, { error });
      
      return {
        status: 'error',
        message: `Une erreur est survenue lors de l'évaluation du modèle ${modelName}`,
        error: error.message
      };
    }
  }
  
  /**
   * Récupère les domaines recommandés pour un modèle spécifique
   */
  @Get('evaluation/:modelName/reliability')
  async getModelReliability(@Param('modelName') modelName: string) {
    this.logger.info(`Récupération de la fiabilité du modèle ${modelName}`);
    
    try {
      // Vérifier si le modèle est valide
      const validModels = ['phi-3-mini', 'llama-3-8b', 'mistral-7b-fr'];
      
      if (!validModels.includes(modelName)) {
        return {
          status: 'error',
          message: `Le modèle ${modelName} n'est pas un modèle valide pour l'évaluation`
        };
      }
      
      const reliability = this.modelEvaluationService.isModelReliable(modelName);
      
      return {
        status: 'success',
        data: reliability
      };
    } catch (error) {
      this.logger.error(`Erreur lors de la récupération de la fiabilité du modèle ${modelName}`, { error });
      
      return {
        status: 'error',
        message: `Une erreur est survenue lors de la récupération de la fiabilité du modèle ${modelName}`,
        error: error.message
      };
    }
  }

  /**
   * Récupère l'état des circuit breakers
   */
  @Get('resilience/status')
  async getCircuitBreakersStatus() {
    this.logger.info('Récupération de l\'état des circuit breakers');
    
    try {
      const status = this.resilienceService.getAllCircuitBreakersStatus();
      
      return {
        status: 'success',
        data: status
      };
    } catch (error) {
      this.logger.error('Erreur lors de la récupération de l\'état des circuit breakers', { error });
      
      return {
        status: 'error',
        message: 'Une erreur est survenue lors de la récupération de l\'état des circuit breakers',
        error: error.message
      };
    }
  }
  
  /**
   * Réinitialise un circuit breaker spécifique
   */
  @Post('resilience/reset/:serviceName')
  async resetCircuitBreaker(@Param('serviceName') serviceName: string) {
    this.logger.info(`Réinitialisation du circuit breaker pour le service ${serviceName}`);
    
    try {
      const result = this.resilienceService.resetCircuitBreaker(serviceName);
      
      if (result) {
        return {
          status: 'success',
          message: `Circuit breaker pour ${serviceName} réinitialisé avec succès`
        };
      } else {
        return {
          status: 'error',
          message: `Circuit breaker pour ${serviceName} introuvable`
        };
      }
    } catch (error) {
      this.logger.error(`Erreur lors de la réinitialisation du circuit breaker pour ${serviceName}`, { error });
      
      return {
        status: 'error',
        message: `Une erreur est survenue lors de la réinitialisation du circuit breaker pour ${serviceName}`,
        error: error.message
      };
    }
  }

  /**
   * Exécute une requête directement sur un fournisseur API avec détection d'anomalies
   */
  @Post('query/direct')
  async directQuery(
    @Body() request: { 
      provider: string; 
      prompt: string; 
      options?: any;
      detectAnomalies?: boolean;
      anomalyDetectionLevel?: AnomalyDetectionLevel;
      fallbackProvider?: string;
    }
  ) {
    this.logger.info('Requête directe à un fournisseur API', {
      provider: request.provider,
      promptLength: request.prompt.length,
      detectAnomalies: request.detectAnomalies,
      anomalyDetectionLevel: request.anomalyDetectionLevel
    });
    
    try {
      const providerOptions = {
        provider: request.provider,
        fallbackProvider: request.fallbackProvider,
        detectAnomalies: request.detectAnomalies !== undefined ? request.detectAnomalies : true,
        anomalyDetectionLevel: request.anomalyDetectionLevel || AnomalyDetectionLevel.MEDIUM_AND_ABOVE
      };
      
      const response = await this.apiProviderFactory.generateResponse(
        providerOptions,
        request.prompt,
        request.options || {}
      );
      
      return {
        status: 'success',
        data: response,
        provider: request.provider,
        // Si le fournisseur a changé en raison d'un fallback
        actualProvider: (response as any).fromFallback ? (response as any).originalProvider : request.provider
      };
    } catch (error) {
      this.logger.error('Erreur lors de la requête directe', { error });
      
      return {
        status: 'error',
        message: `Erreur lors de la requête directe: ${error.message}`,
        provider: request.provider
      };
    }
  }

  /**
   * Analyse une réponse pour détecter les anomalies
   */
  @Post('anomalies/detect')
  async detectAnomaliesInResponse(
    @Body() request: { 
      content: string; 
      source?: string;
      level?: AnomalyDetectionLevel;
    }
  ) {
    this.logger.info('Demande de détection d\'anomalies', {
      contentLength: request.content.length,
      source: request.source,
      level: request.level
    });
    
    try {
      // Créer une structure similaire à PoolOutputs pour la détection d'anomalies
      const simulatedPoolOutput = {
        commercial: [],
        marketing: [],
        sectoriel: [],
        educational: [],
        timestamp: new Date(),
        // Ajouter la réponse comme si elle venait d'un agent
        analysis: [{
          agentId: 'analysis-agent',
          agentName: 'Analysis Agent',
          poolType: 'ANALYSIS',
          content: request.content,
          confidence: 0.9,
          timestamp: Date.now(),
          processingTime: 0,
          metadata: {
            source: request.source || 'unknown'
          }
        }]
      };
      
      const anomalyReport = await this.anomalyDetectionService.detectAnomalies(
        simulatedPoolOutput,
        {
          detectionLevel: request.level || AnomalyDetectionLevel.ALL,
          autoLog: true
        }
      );
      
      // Formater le rapport pour la réponse
      const formattedReport = {
        reliability: anomalyReport.overallReliability,
        anomalies: {
          high: anomalyReport.highPriorityAnomalies.map(a => ({
            type: a.type,
            description: a.description,
            fragment: a.location.contentFragment,
            suggestedResolution: a.suggestedResolution
          })),
          medium: anomalyReport.mediumPriorityAnomalies.map(a => ({
            type: a.type,
            description: a.description,
            fragment: a.location.contentFragment,
            suggestedResolution: a.suggestedResolution
          })),
          low: anomalyReport.minorIssues.map(a => ({
            type: a.type,
            description: a.description,
            fragment: a.location.contentFragment,
            suggestedResolution: a.suggestedResolution
          }))
        },
        systemicPatterns: anomalyReport.systemicPatterns,
        summary: anomalyReport.report
      };
      
      return {
        status: 'success',
        data: formattedReport
      };
    } catch (error) {
      this.logger.error('Erreur lors de la détection d\'anomalies', { error });
      
      return {
        status: 'error',
        message: `Erreur lors de la détection d'anomalies: ${error.message}`
      };
    }
  }

  /**
   * Récupère des informations détaillées sur l'état de santé du système et sa résilience
   */
  @Get('health/detailed')
  async getDetailedHealth() {
    this.logger.info('Récupération de l\'état de santé détaillé du système');
    
    try {
      // Récupérer l'état des circuit breakers
      const circuitBreakersStatus = this.resilienceService.getAllCircuitBreakersStatus();
      
      // Calculer des statistiques globales sur l'état des circuit breakers
      const totalCircuitBreakers = Object.keys(circuitBreakersStatus).length;
      const openCircuitBreakers = Object.values(circuitBreakersStatus)
        .filter(cb => cb.state === 'OPEN')
        .length;
      const halfOpenCircuitBreakers = Object.values(circuitBreakersStatus)
        .filter(cb => cb.state === 'HALF_OPEN')
        .length;
      
      // Récupérer des informations sur les modèles
      const modelStatus = {
        teacher: 'deepseek-r1',
        distilled: this.modelTrainingService.getTrainingStatistics()
      };
      
      // Récupérer des statistiques d'évaluation
      const evaluationStats = this.modelEvaluationService.getAllModelsEvaluationStats();
      
      return {
        status: 'success',
        data: {
          timestamp: new Date(),
          resilience: {
            circuitBreakers: {
              total: totalCircuitBreakers,
              open: openCircuitBreakers,
              halfOpen: halfOpenCircuitBreakers,
              closed: totalCircuitBreakers - openCircuitBreakers - halfOpenCircuitBreakers,
              details: circuitBreakersStatus
            }
          },
          models: modelStatus,
          evaluation: evaluationStats
        }
      };
    } catch (error) {
      this.logger.error('Erreur lors de la récupération de l\'état de santé détaillé', { error });
      
      return {
        status: 'error',
        message: `Erreur lors de la récupération de l'état de santé détaillé: ${error.message}`
      };
    }
  }
} 