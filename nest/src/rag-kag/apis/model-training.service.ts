import { Injectable, Inject, OnModuleInit, Optional } from '@nestjs/common';
import { Cron } from '@nestjs/schedule';
import { LOGGER_TOKEN, ILogger } from '../utils/logger-tokens';
import { HouseModelService } from './house-model.service';
import { ModelEvaluationService } from './model-evaluation.service';
import { EventBusService, RagKagEventType } from '../core/event-bus.service';
import { KnowledgeGraphService, KnowledgeSource } from '../core/knowledge-graph.service';
import { PythonApiService } from './python-api.service';

/**
 * Service pour la gestion de l'entraînement des modèles
 * Intégré avec EventBus et KnowledgeGraph pour traçabilité et persistance
 */
@Injectable()
export class ModelTrainingService implements OnModuleInit {
  private readonly logger: ILogger;
  private readonly distilledModels = ['phi-3-mini', 'llama-3-8b', 'mistral-7b-fr'];
  private isTrainingInProgress = false;
  private trainingStats = new Map<string, { lastTraining: Date, examples: number, accuracy?: number, loss?: number }>();
  
  constructor(
    @Inject(LOGGER_TOKEN) logger: ILogger,
    private readonly houseModelService: HouseModelService,
    private readonly modelEvaluationService: ModelEvaluationService,
    @Optional() private readonly pythonApiService?: PythonApiService,
    @Optional() private readonly eventBus?: EventBusService,
    @Optional() private readonly knowledgeGraph?: KnowledgeGraphService
  ) {
    this.logger = logger;
  }
  
  /**
   * Initialisation du service
   */
  async onModuleInit() {
    this.logger.info('Initialisation du service d\'entraînement des modèles');
    
    // Initialisation des statistiques d'entraînement
    for (const model of this.distilledModels) {
      this.trainingStats.set(model, {
        lastTraining: new Date(0), // Date très ancienne pour forcer un premier entraînement
        examples: 0
      });
    }
    
    // Émettre un événement d'initialisation
    if (this.eventBus) {
      this.eventBus.emit({
        type: RagKagEventType.SYSTEM_INIT,
        source: 'ModelTrainingService',
        payload: {
          distilledModels: this.distilledModels
        }
      });
    }
    
    // Charger les statistiques d'entraînement existantes depuis le graphe de connaissances
    if (this.knowledgeGraph) {
      await this.loadTrainingStatsFromGraph();
    }
  }
  
  /**
   * Charge les statistiques d'entraînement depuis le graphe de connaissances
   */
  private async loadTrainingStatsFromGraph(): Promise<void> {
    try {
      // Rechercher les nœuds d'entraînement dans le graphe
      for (const model of this.distilledModels) {
        const searchResults = this.knowledgeGraph.search(`training ${model}`, {
          nodeTypes: ['MODEL_TRAINING'],
          maxResults: 1,
          sortByRelevance: true
        });
        
        if (searchResults.nodes.length > 0) {
          const trainingNode = searchResults.nodes[0];
          
          if (trainingNode.metadata) {
            const meta = trainingNode.metadata;
            
            // Mettre à jour les statistiques depuis le graphe
            this.trainingStats.set(model, {
              lastTraining: new Date(meta.timestamp || 0),
              examples: meta.examples || 0,
              accuracy: meta.accuracy,
              loss: meta.loss
            });
            
            this.logger.info(`Statistiques d'entraînement chargées depuis le graphe pour ${model}`, {
              lastTraining: new Date(meta.timestamp || 0),
              examples: meta.examples || 0
            });
          }
        }
      }
    } catch (error) {
      this.logger.error(`Erreur lors du chargement des statistiques d'entraînement depuis le graphe`, {
        error: error.message,
        stack: error.stack
      });
    }
  }
  
  /**
   * Tâche planifiée pour l'entraînement périodique des modèles (chaque 12 heures)
   */
  @Cron('0 0 */12 * * *')
  async scheduledTraining() {
    this.logger.info('Démarrage de l\'entraînement périodique des modèles');
    
    // Émettre un événement de début d'entraînement périodique
    if (this.eventBus) {
      this.eventBus.emit({
        type: RagKagEventType.MODEL_TRAINING_STARTED,
        source: 'ModelTrainingService',
        payload: {
          type: 'scheduled',
          modelCount: this.distilledModels.length
        }
      });
    }
    
    if (this.isTrainingInProgress) {
      this.logger.warn('Un entraînement est déjà en cours, entraînement périodique ignoré');
      
      // Émettre un événement d'annulation
      if (this.eventBus) {
        this.eventBus.emit({
          type: RagKagEventType.CUSTOM,
          source: 'ModelTrainingService',
          payload: {
            eventType: 'TRAINING_SKIPPED',
            reason: 'already_in_progress'
          }
        });
      }
      
      return;
    }
    
    await this.checkAndTrainModels();
    
    // Émettre un événement de fin d'entraînement périodique
    if (this.eventBus) {
      this.eventBus.emit({
        type: RagKagEventType.MODEL_TRAINING_COMPLETED,
        source: 'ModelTrainingService',
        payload: {
          type: 'scheduled',
          modelCount: this.distilledModels.length,
          isTrainingInProgress: this.isTrainingInProgress
        }
      });
    }
  }
  
  /**
   * Tâche planifiée pour l'évaluation périodique des modèles (chaque 24 heures)
   */
  @Cron('0 0 0 * * *')
  async scheduledEvaluation() {
    this.logger.info('Démarrage de l\'évaluation périodique des modèles');
    
    // Émettre un événement de début d'évaluation
    if (this.eventBus) {
      this.eventBus.emit({
        type: RagKagEventType.MODEL_EVALUATION_STARTED,
        source: 'ModelTrainingService',
        payload: {
          type: 'scheduled',
          modelCount: this.distilledModels.length
        }
      });
    }
    
    if (this.isTrainingInProgress) {
      this.logger.warn('Un entraînement est en cours, évaluation périodique reportée');
      
      // Émettre un événement d'annulation
      if (this.eventBus) {
        this.eventBus.emit({
          type: RagKagEventType.CUSTOM,
          source: 'ModelTrainingService',
          payload: {
            eventType: 'EVALUATION_SKIPPED',
            reason: 'training_in_progress'
          }
        });
      }
      
      return;
    }
    
    try {
      await this.modelEvaluationService.evaluateAllModels();
      this.logger.info('Évaluation périodique des modèles terminée avec succès');
      
      // Émettre un événement de fin d'évaluation
      if (this.eventBus) {
        this.eventBus.emit({
          type: RagKagEventType.MODEL_EVALUATION_COMPLETED,
          source: 'ModelTrainingService',
          payload: {
            type: 'scheduled',
            modelCount: this.distilledModels.length,
            success: true
          }
        });
      }
    } catch (error) {
      this.logger.error('Erreur lors de l\'évaluation périodique des modèles', { error });
      
      // Émettre un événement d'erreur
      if (this.eventBus) {
        this.eventBus.emit({
          type: RagKagEventType.QUERY_ERROR,
          source: 'ModelTrainingService',
          payload: {
            operation: 'evaluation',
            error: error.message
          }
        });
      }
    }
  }
  
  /**
   * Vérifie et entraîne les modèles si nécessaire
   */
  public async checkAndTrainModels() {
    try {
      this.isTrainingInProgress = true;
      this.logger.info('Vérification des modèles pour entraînement');
      
      for (const model of this.distilledModels) {
        try {
          const stats = this.trainingStats.get(model);
          
          // Vérifier si le dernier entraînement date de plus de 24 heures
          const lastTraining = stats.lastTraining;
          const now = new Date();
          const hoursSinceLastTraining = (now.getTime() - lastTraining.getTime()) / (1000 * 60 * 60);
          
          if (hoursSinceLastTraining >= 24) {
            this.logger.info(`Le modèle ${model} n'a pas été entraîné depuis plus de 24 heures, démarrage de l'entraînement`);
            
            // Émettre un événement de début d'entraînement spécifique
            if (this.eventBus) {
              this.eventBus.emit({
                type: RagKagEventType.MODEL_TRAINING_STARTED,
                source: 'ModelTrainingService',
                payload: {
                  model,
                  hoursSinceLastTraining: Math.round(hoursSinceLastTraining)
                }
              });
            }
            
            // TODO: Implement new training logic for the Python API
            // const result = await this.houseModelService.finetuneDistilledModel(model);
            // 
            // if (result.success) {
            //   const updatedStats = {
            //     lastTraining: now,
            //     examples: result.trainedExamples || 0,
            //     accuracy: result.accuracy,
            //     loss: result.loss
            //   };
            //   
            //   this.trainingStats.set(model, updatedStats);
            //   
            //   // Stocker les résultats dans le graphe de connaissances
            //   if (this.knowledgeGraph) {
            //     this.storeTrainingResultInGraph(model, result, updatedStats);
            //   }
            //   
            //   this.logger.info(`Entraînement du modèle ${model} terminé avec succès`, {
            //     examples: result.trainedExamples,
            //     accuracy: result.accuracy,
            //     loss: result.loss
            //   });
            //   
            //   // Émettre un événement de succès
            //   if (this.eventBus) {
            //     this.eventBus.emit({
            //       type: RagKagEventType.MODEL_TRAINING_COMPLETED,
            //       source: 'ModelTrainingService',
            //       payload: {
            //         model,
            //         success: true,
            //         examples: result.trainedExamples,
            //         accuracy: result.accuracy,
            //         loss: result.loss
            //       }
            //     });
            //   }
            // } else {
            //   this.logger.warn(`Échec de l'entraînement du modèle ${model}`, {
            //     reason: result.message
            //   });
            //   
            //   // Émettre un événement d'échec
            //   if (this.eventBus) {
            //     this.eventBus.emit({
            //       type: RagKagEventType.CUSTOM,
            //       source: 'ModelTrainingService',
            //       payload: {
            //         eventType: 'MODEL_TRAINING_FAILED',
            //         model,
            //         reason: result.message
            //       }
            //     });
            //   }
          } else {
            this.logger.info(`Le modèle ${model} a été entraîné récemment (il y a ${hoursSinceLastTraining.toFixed(1)} heures), entraînement ignoré`);
            
            // Émettre un événement d'ignorance
            if (this.eventBus) {
              this.eventBus.emit({
                type: RagKagEventType.CUSTOM,
                source: 'ModelTrainingService',
                payload: {
                  eventType: 'MODEL_TRAINING_SKIPPED',
                  model,
                  hoursSinceLastTraining: Math.round(hoursSinceLastTraining)
                }
              });
            }
          }
        } catch (error) {
          this.logger.error(`Erreur lors de l'entraînement du modèle ${model}`, { error });
          
          // Émettre un événement d'erreur
          if (this.eventBus) {
            this.eventBus.emit({
              type: RagKagEventType.QUERY_ERROR,
              source: 'ModelTrainingService',
              payload: {
                operation: 'training',
                model,
                error: error.message
              }
            });
          }
        }
      }
    } finally {
      this.isTrainingInProgress = false;
    }
  }
  
  /**
   * Stocke les résultats d'entraînement dans le graphe de connaissances
   */
  private storeTrainingResultInGraph(
    model: string,
    result: any,
    stats: { lastTraining: Date, examples: number, accuracy?: number, loss?: number }
  ): void {
    try {
      // Créer un nœud pour le résultat d'entraînement
      const trainingNodeId = this.knowledgeGraph.addNode({
        label: `Training: ${model} ${stats.lastTraining.toISOString()}`,
        type: 'MODEL_TRAINING',
        content: `Résultat d'entraînement pour le modèle ${model}: ${result.trainedExamples} exemples, précision ${result.accuracy}, perte ${result.loss}`,
        confidence: 0.9,
        source: KnowledgeSource.SYSTEM,
        metadata: {
          model,
          timestamp: stats.lastTraining.getTime(),
          examples: stats.examples,
          accuracy: stats.accuracy,
          loss: stats.loss
        }
      });
      
      // Créer ou récupérer un nœud pour le modèle
      const modelNodeResults = this.knowledgeGraph.search(model, {
        nodeTypes: ['MODEL'],
        maxResults: 1
      });
      
      let modelNodeId: string;
      
      if (modelNodeResults.nodes.length > 0) {
        modelNodeId = modelNodeResults.nodes[0].id;
      } else {
        // Créer un nœud pour le modèle s'il n'existe pas encore
        modelNodeId = this.knowledgeGraph.addNode({
          label: `Model: ${model}`,
          type: 'MODEL',
          content: `Modèle distillé: ${model}`,
          confidence: 0.95,
          source: KnowledgeSource.SYSTEM
        });
      }
      
      // Lier le résultat d'entraînement au modèle
      this.knowledgeGraph.addFact(
        modelNodeId,
        'HAS_TRAINING_RESULT',
        trainingNodeId,
        0.9,
        {
          bidirectional: true,
          weight: 0.8
        }
      );
      
      this.logger.debug(`Résultat d'entraînement stocké dans le graphe de connaissances pour ${model}`, {
        trainingNodeId,
        modelNodeId
      });
    } catch (error) {
      this.logger.error(`Erreur lors du stockage des résultats d'entraînement dans le graphe: ${error.message}`, {
        error: error.stack
      });
    }
  }
  
  /**
   * Force l'entraînement d'un modèle spécifique
   * @returns boolean - true si l'entraînement est lancé avec succès, false sinon
   */
  public async forceTrainModel(modelName: string): Promise<boolean> {
    if (!this.distilledModels.includes(modelName)) {
      throw new Error(`Le modèle ${modelName} n'est pas un modèle distillé valide`);
    }
    
    try {
      // Si le service Python API n'est pas disponible
      if (!this.pythonApiService) {
        this.logger.warn(`Service Python API non disponible pour l'entraînement du modèle ${modelName}`);
        return false;
      }
      
      // Vérifier si le service est disponible
      if (!this.pythonApiService.isAvailable()) {
        this.logger.warn(`API Python non disponible pour l'entraînement du modèle ${modelName}`);
        return false;
      }
      
      this.logger.info(`Démarrage de l'entraînement forcé du modèle ${modelName} via l'API Python`);
      
      // Si un entraînement est déjà en cours
      if (this.isTrainingInProgress) {
        this.logger.warn(`Impossible de lancer l'entraînement: un autre entraînement est déjà en cours`);
        return false;
      }
      
      this.isTrainingInProgress = true;
      
      // Émettre un événement de début d'entraînement
      if (this.eventBus) {
        this.eventBus.emit({
          type: RagKagEventType.MODEL_TRAINING_STARTED,
          source: 'ModelTrainingService',
          payload: {
            type: 'forced',
            model: modelName
          }
        });
      }
      
      try {
        // Appel de l'API Python pour l'entraînement
        const result = await this.pythonApiService.trainModel(modelName, {
          epochs: 5,
          batchSize: 32,
          learningRate: 5e-5,
          validationSplit: 0.1,
          maxExamples: 1000,
          saveToDisk: true
        });
        
        // Traitement du résultat
        if (result.success) {
          this.logger.info(`Modèle ${modelName} entraîné avec succès`, {
            examples: result.trainedExamples,
            accuracy: result.accuracy,
            loss: result.loss
          });
          
          // Stockage des résultats dans le graphe de connaissances
          this.storeTrainingResultInGraph(modelName, result, {
            lastTraining: new Date(),
            examples: result.trainedExamples || 0,
            accuracy: result.accuracy,
            loss: result.loss
          });
          
          // Mise à jour des statistiques locales
          this.trainingStats.set(modelName, {
            lastTraining: new Date(),
            examples: result.trainedExamples || 0,
            accuracy: result.accuracy,
            loss: result.loss
          });
          
          // Émettre un événement de fin d'entraînement réussi
          if (this.eventBus) {
            this.eventBus.emit({
              type: RagKagEventType.MODEL_TRAINING_COMPLETED,
              source: 'ModelTrainingService',
              payload: {
                type: 'forced',
                model: modelName,
                success: true,
                stats: {
                  accuracy: result.accuracy,
                  loss: result.loss,
                  examples: result.trainedExamples
                }
              }
            });
          }
          
          this.isTrainingInProgress = false;
          return true;
        } else {
          this.logger.warn(`Échec de l'entraînement forcé du modèle ${modelName}`, {
            message: result.message
          });
          
          // Émettre un événement d'échec d'entraînement
          if (this.eventBus) {
            this.eventBus.emit({
              type: RagKagEventType.MODEL_TRAINING_FAILED,
              source: 'ModelTrainingService',
              payload: {
                type: 'forced',
                model: modelName,
                reason: result.message || 'Erreur inconnue'
              }
            });
          }
          
          this.isTrainingInProgress = false;
          return false;
        }
      } catch (error) {
        this.logger.error(`Erreur lors de l'entraînement forcé du modèle ${modelName}`, { error });
        
        // Émettre un événement d'erreur
        if (this.eventBus) {
          this.eventBus.emit({
            type: RagKagEventType.QUERY_ERROR,
            source: 'ModelTrainingService',
            payload: {
              operation: 'force_train_model',
              model: modelName,
              error: error.message
            }
          });
        }
        
        this.isTrainingInProgress = false;
        return false;
      }
    } catch (error) {
      this.logger.error(`Erreur lors de l'entraînement forcé du modèle ${modelName}`, { error });
      return false;
    }
  }
  
  /**
   * Récupère les statistiques d'entraînement
   */
  public getTrainingStatistics() {
    const stats = {};
    
    for (const model of this.distilledModels) {
      const modelStats = this.trainingStats.get(model);
      
      if (modelStats) {
        // Récupérer également les statistiques d'évaluation
        const evaluationStats = this.modelEvaluationService.getModelEvaluationStats(model);
        const reliability = this.modelEvaluationService.isModelReliable(model);
        
        stats[model] = {
          training: {
            lastTraining: modelStats.lastTraining,
            examples: modelStats.examples,
            accuracy: modelStats.accuracy,
            loss: modelStats.loss
          },
          evaluation: {
            comparisonWithTeacher: evaluationStats.comparisonWithTeacher,
            reliability: reliability.isReliable,
            reliabilityScore: reliability.reliabilityScore,
            recommendedDomains: reliability.recommendedDomains
          }
        };
      }
    }
    
    return {
      models: stats,
      isTrainingInProgress: this.isTrainingInProgress
    };
  }
} 
