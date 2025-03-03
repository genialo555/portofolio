import { Injectable, Inject, OnModuleInit, Logger, forwardRef, Optional } from '@nestjs/common';
import { LOGGER_TOKEN, ILogger } from '../utils/logger-tokens';
import { ApiUsageMetrics } from '../types/index';
import { ModelUtilsService } from './model-utils.service';
import { ModelEvaluationService } from './model-evaluation.service';
import { EventBusService, RagKagEventType } from '../core/event-bus.service';
import { KnowledgeGraphService, KnowledgeSource, KnowledgeNode } from '../core/knowledge-graph.service';
import * as fs from 'fs';

/**
 * Service pour interagir avec les modèles d'IA en local (modèle maison)
 * Intégration des modèles open-source hébergés sur l'infrastructure locale
 * Utilise EventBus pour tracer l'utilisation des modèles et le processus d'apprentissage
 * 
 * Cette implémentation concerne les modèles:
 * - Modèles distillés légers (Phi-3-mini, TinyLlama, etc.)
 * - Modèles spécialisés pour l'enseignement
 */
@Injectable()
export class HouseModelService implements OnModuleInit {
  private readonly availableModels = {
    'phi-3-mini': {
      description: 'Modèle léger et rapide pour requêtes simples',
      contextSize: 4096,
      speedFactor: 0.5, // Plus rapide
      capabilities: ['texte', 'qa', 'synthèse'],
      strength: 'rapidité',
      cost: 'faible'
    },
    'llama-3-8b': {
      description: 'Modèle généraliste équilibré',
      contextSize: 8192,
      speedFactor: 1.0, // Référence
      capabilities: ['texte', 'qa', 'synthèse', 'raisonnement'],
      strength: 'polyvalence',
      cost: 'modéré'
    },
    'mistral-7b-fr': {
      description: 'Modèle spécialisé pour le français',
      contextSize: 8192,
      speedFactor: 1.2,
      capabilities: ['texte', 'qa', 'synthèse', 'français'],
      strength: 'français',
      cost: 'modéré'
    },
    'deepseek-r1': {
      description: 'Modèle spécialisé pour l\'enseignement et le contenu éducatif',
      contextSize: 16384,
      speedFactor: 2.0, // Plus lent mais plus précis
      capabilities: ['texte', 'qa', 'raisonnement', 'éducation', 'pédagogie'],
      strength: 'contenu éducatif',
      cost: 'élevé'
    }
  };

  private learningExamples: Array<{
    prompt: string;
    response: string;
    modelUsed: string;
    timestamp: Date;
    category: string;
  }> = [];

  constructor(
    @Inject(LOGGER_TOKEN) private readonly logger: ILogger,
    private readonly modelUtilsService: ModelUtilsService,
    @Inject(forwardRef(() => ModelEvaluationService)) private readonly modelEvaluationService?: ModelEvaluationService,
    @Optional() private readonly eventBus?: EventBusService,
    @Optional() private readonly knowledgeGraph?: KnowledgeGraphService
  ) {
    this.logger.info("Service de modèle maison initialisé", {
      modelsAvailable: Object.keys(this.availableModels)
    });
  }

  /**
   * Initialisation du module
   */
  async onModuleInit() {
    this.logger.info('Initialisation du service de modèle maison');
    // Initialiser le référentiel d'apprentissage
    await this.initLearningRepository();
    
    // Émettre un événement de démarrage
    if (this.eventBus) {
      this.eventBus.emit({
        type: RagKagEventType.SYSTEM_INIT,
        source: 'HouseModelService',
        payload: { 
          availableModels: Object.keys(this.availableModels)
        }
      });
    }
  }

  /**
   * Initialise le système de stockage des exemples d'apprentissage
   */
  private async initLearningRepository() {
    if (this.knowledgeGraph) {
      await this.loadLearningExamplesFromGraph();
    } else {
      this.logger.warn('Le graphe de connaissances n\'est pas disponible, utilisation du stockage en mémoire pour les exemples d\'apprentissage');
    }
  }

  private async loadLearningExamplesFromGraph(): Promise<void> {
    try {
      const exampleNodes = await this.knowledgeGraph.search('learning example', {
        nodeTypes: ['LEARNING_EXAMPLE'],
        maxDepth: 0,
        maxResults: 1000
      });
      
      this.learningExamples = exampleNodes.nodes.map(node => ({
        prompt: node.metadata.prompt,
        response: node.metadata.response,
        modelUsed: node.metadata.modelUsed,
        timestamp: new Date(node.metadata.timestamp),
        category: node.metadata.category
      }));
      
      this.logger.info(`${this.learningExamples.length} exemples d'apprentissage chargés depuis le graphe de connaissances`);
    } catch (error) {
      this.logger.error('Erreur lors du chargement des exemples d\'apprentissage depuis le graphe de connaissances', { error });
    }
  }

  /**
   * Sélectionne le meilleur modèle pour une requête donnée
   */
  private selectBestModel(
    prompt: string,
    options?: { 
      forceModel?: string; 
      prioritizeFast?: boolean;
      languageHint?: 'french' | 'english' | 'other';
      educationalContent?: boolean;
    }
  ): string {
    // Si un modèle est spécifiquement demandé et qu'il existe, l'utiliser
    if (options?.forceModel && this.availableModels[options.forceModel]) {
      return options.forceModel;
    }
    
    // Si c'est du contenu éducatif, privilégier DeepSeek R1 
    if (options?.educationalContent || 
      prompt.toLowerCase().includes('apprendre') || 
      prompt.toLowerCase().includes('enseigner') ||
      prompt.toLowerCase().includes('explique') ||
      prompt.toLowerCase().includes('cours') ||
      prompt.toLowerCase().includes('éducation')) {
      return 'deepseek-r1';
    }
    
    // Si le prompt est en français, privilégier Mistral-7B-FR
    if (options?.languageHint === 'french' || 
        this.isFrenchContent(prompt)) {
      
      // Vérifier si Mistral-7B-FR est suffisamment fiable d'après les évaluations
      if (this.modelEvaluationService) {
        const reliability = this.modelEvaluationService.isModelReliable('mistral-7b-fr');
        if (reliability.isReliable) {
          return 'mistral-7b-fr';
        }
      }
      
      return 'deepseek-r1'; // Fallback si Mistral n'est pas assez fiable
    }
    
    // Si on privilégie la vitesse, utiliser Phi-3-mini
    if (options?.prioritizeFast || prompt.length < 100) {
      // Vérifier si Phi-3-mini est suffisamment fiable d'après les évaluations
      if (this.modelEvaluationService) {
        const reliability = this.modelEvaluationService.isModelReliable('phi-3-mini');
        if (reliability.isReliable) {
          return 'phi-3-mini';
        }
      }
    }
    
    // Vérifier si Llama-3-8B est fiable pour ce type de requête
    if (this.modelEvaluationService) {
      const llamaReliability = this.modelEvaluationService.isModelReliable('llama-3-8b');
      
      if (llamaReliability.isReliable) {
        // Vérifier si le domaine de la requête correspond aux domaines recommandés pour le modèle
        const promptDomains = this.identifyDomains(prompt);
        const hasMatchingDomain = llamaReliability.recommendedDomains.some(domain => 
          promptDomains.includes(domain)
        );
        
        if (hasMatchingDomain) {
          return 'llama-3-8b';
        }
      }
    }
    
    // Par défaut, utiliser DeepSeek R1 qui est le modèle le plus puissant
    return 'deepseek-r1';
  }
  
  /**
   * Détermine si le contenu est probablement en français
   */
  private isFrenchContent(text: string): boolean {
    // Liste simplifiée de mots français caractéristiques
    const frenchKeywords = [
      'le', 'la', 'les', 'un', 'une', 'des', 'du', 'est', 'sont', 'être',
      'avoir', 'faire', 'dire', 'voir', 'pouvoir', 'vouloir', 'aller', 'venir',
      'puis', 'donc', 'ainsi', 'comme', 'mais', 'ou', 'où', 'et', 'car', 'cela',
      'voilà', 'très', 'bien', 'bon', 'bonjour', 'merci', 'beaucoup', 'peu'
    ];
    
    // Analyser les mots du prompt
    const words = text.toLowerCase()
      .replace(/[.,;:!?()]/g, '')
      .split(/\s+/);
    
    // Compter les mots français
    const frenchWordCount = words.filter(word => frenchKeywords.includes(word)).length;
    
    // Si plus de 10% des mots sont reconnus comme français, considérer le texte comme français
    return frenchWordCount / words.length > 0.1;
  }
  
  /**
   * Identifie les domaines probables d'une requête
   */
  private identifyDomains(prompt: string): string[] {
    const lowercasePrompt = prompt.toLowerCase();
    const domains = [];
    
    // Domaines simples basés sur des mots-clés
    const domainKeywords = {
      'science': ['physique', 'chimie', 'biologie', 'équation', 'théorie', 'scientifique'],
      'technologie': ['informatique', 'logiciel', 'programmation', 'code', 'algorithme', 'internet'],
      'business': ['entreprise', 'management', 'économie', 'finance', 'marketing', 'stratégie'],
      'environnement': ['écologie', 'climat', 'pollution', 'durable', 'énergie', 'recycler'],
      'santé': ['médecine', 'santé', 'maladie', 'traitement', 'symptôme', 'diagnostic']
    };
    
    // Vérifier chaque domaine
    for (const [domain, keywords] of Object.entries(domainKeywords)) {
      for (const keyword of keywords) {
        if (lowercasePrompt.includes(keyword)) {
          domains.push(domain);
          break;
        }
      }
    }
    
    return domains;
  }

  /**
   * Génère une réponse à partir d'un texte en utilisant un modèle local
   */
  async generateResponse(prompt: string, options?: {
    modelName?: string;
    maxLength?: number;
    temperature?: number;
    educationalContent?: boolean;
  }): Promise<{
    text: string;
    model: string;
    timings: {
      total: number;
    };
    usage: ApiUsageMetrics;
    fromLearningExample?: boolean;
  }> {
    const start = Date.now();
    let text = '';
    let modelName = options?.modelName || this.selectBestModel(prompt, options);
    let fromLearningExample = false;
    
    try {
      // Vérifier s'il existe un exemple d'apprentissage similaire
      const similarExample = await this.findSimilarLearningExample(prompt, modelName);
      
      if (similarExample) {
        this.logger.info(`Exemple d'apprentissage similaire trouvé pour ${modelName}`, {
          promptLength: prompt.length,
          exampleLength: similarExample.prompt.length,
          similarity: this.cosineSimilarity(
            this.simpleHash(this.extractKeywords(prompt).join(' ')),
            this.simpleHash(this.extractKeywords(similarExample.prompt).join(' '))
          )
        });
        
        text = similarExample.response;
        fromLearningExample = true;
        
        // Émettre un événement d'utilisation d'exemple
        if (this.eventBus) {
          this.eventBus.emit({
            type: RagKagEventType.CUSTOM,
            source: 'HouseModelService',
            payload: { 
              eventType: 'LEARNING_EXAMPLE_USED',
              model: modelName,
              category: similarExample.category,
              exampleAge: Date.now() - similarExample.timestamp.getTime()
            }
          });
        }
      } else {
        // Si c'est un modèle distillé, essayons de charger le modèle TensorFlow
        if (modelName !== 'deepseek-r1') {
          try {
            const model = await this.modelUtilsService.loadModel(modelName);
            
            if (model) {
              this.logger.info(`Modèle TensorFlow trouvé pour ${modelName}, génération...`);
              text = await this.modelUtilsService.predictText(model, prompt, modelName, options?.maxLength || 512);
            } else {
              this.logger.warn(`Aucun modèle TensorFlow trouvé pour ${modelName}, utilisation du mode simulé`);
              text = this.simulateModelGeneration(prompt, modelName);
            }
          } catch (error) {
            this.logger.error(`Erreur de chargement/prédiction pour ${modelName}`, {
              error: error.message
            });
            text = this.simulateModelGeneration(prompt, modelName);
          }
        } else {
          // Pour DeepSeek R1, utiliser la simulation (ou l'API si disponible)
          text = this.simulateModelGeneration(prompt, modelName);
        }
      }
    } catch (error) {
      this.logger.error(`Erreur lors de la génération avec ${modelName}`, {
        error: error.message,
        stack: error.stack
      });
      
      // Émettre un événement d'erreur
      if (this.eventBus) {
        this.eventBus.emit({
          type: RagKagEventType.QUERY_ERROR,
          source: 'HouseModelService',
          payload: { 
            model: modelName,
            error: error.message
          }
        });
      }
      
      // Fallback pour garantir une réponse
      text = `Désolé, je n'ai pas pu générer une réponse avec le modèle ${modelName}. Erreur: ${error.message}`;
    }
    
    const end = Date.now();
    const processingTime = end - start;
    
    // Calculer l'utilisation des tokens (simplifiée)
    const promptTokens = prompt.length / 4; // Approximation grossière
    const completionTokens = text.length / 4; // Approximation grossière
    
    // Métriques d'utilisation
    const usage: ApiUsageMetrics = {
      promptTokens: Math.ceil(promptTokens),
      completionTokens: Math.ceil(completionTokens),
      totalTokens: Math.ceil(promptTokens + completionTokens),
      processingTime
    };
    
    // Si c'est le modèle enseignant (DeepSeek R1), créer un exemple d'apprentissage
    if (modelName === 'deepseek-r1' && !fromLearningExample) {
      const category = this.identifyDomains(prompt).join(',') || 'general';
      await this.createLearningExample(prompt, text, modelName, category);
      
      // Émettre un événement de création d'exemple
      if (this.eventBus) {
        this.eventBus.emit({
          type: RagKagEventType.CUSTOM,
          source: 'HouseModelService',
          payload: { 
            eventType: 'LEARNING_EXAMPLE_CREATED',
            model: modelName,
            category,
            promptLength: prompt.length,
            responseLength: text.length
          }
        });
      }
    }
    
    // Émettre un événement de fin de génération
    if (this.eventBus) {
      this.eventBus.emit({
        type: RagKagEventType.MODEL_TRAINING_COMPLETED,
        source: 'HouseModelService',
        payload: { 
          model: modelName,
          processingTime,
          fromLearningExample,
          tokenUsage: usage.totalTokens
        }
      });
    }
    
    return {
      text,
      model: modelName,
      timings: {
        total: processingTime
      },
      usage,
      fromLearningExample
    };
  }

  /**
   * Crée un exemple d'apprentissage à partir d'une requête et sa réponse
   */
  private async createLearningExample(prompt: string, response: string, modelUsed: string, category: string) {
    const example = {
      prompt,
      response,
      modelUsed,
      timestamp: new Date(),
      category
    };
    
    if (this.knowledgeGraph) {
      await this.storeLearningExampleInGraph(example);
    } else {
      this.learningExamples.push(example);
      this.saveLearningExamples();
    }
  }

  private async storeLearningExampleInGraph(example: {
    prompt: string;
    response: string;
    modelUsed: string;
    timestamp: Date;
    category: string;
  }): Promise<void> {
    try {
      const exampleNodeId = this.knowledgeGraph.addNode({
        label: `Learning Example ${example.timestamp.toISOString()}`,
        type: 'LEARNING_EXAMPLE',
        content: example.prompt,
        confidence: 1,
        source: KnowledgeSource.SYSTEM,
        metadata: {
          prompt: example.prompt,
          response: example.response,
          modelUsed: example.modelUsed,
          timestamp: example.timestamp.toISOString(),
          category: example.category
        }
      });
      
      this.knowledgeGraph.addFact(
        exampleNodeId,
        'GENERATED_BY',
        {
          label: example.modelUsed,
          type: 'MODEL',
          content: example.modelUsed,
          confidence: 1,
          source: KnowledgeSource.SYSTEM
        },
        1,
        { bidirectional: false, weight: 1 }
      );
      
      this.logger.debug(`Exemple d'apprentissage stocké dans le graphe de connaissances`, {
        nodeId: exampleNodeId,
        prompt: example.prompt,
        model: example.modelUsed
      });
    } catch (error) {
      this.logger.error('Erreur lors du stockage de l\'exemple d\'apprentissage dans le graphe de connaissances', { error });
    }
  }

  /**
   * Trouve un exemple d'apprentissage similaire à la requête donnée
   */
  private async findSimilarLearningExample(prompt: string, modelName: string) {
    if (this.knowledgeGraph) {
      return this.findSimilarLearningExampleInGraph(prompt, modelName);
    } else {
      return this.findSimilarLearningExampleInMemory(prompt, modelName);
    }
  }

  private async findSimilarLearningExampleInGraph(prompt: string, modelName: string): Promise<{
    prompt: string;
    response: string;
    modelUsed: string;
    timestamp: Date;
    category: string;
  } | null> {
    try {
      const results = await this.knowledgeGraph.search(prompt, {
        nodeTypes: ['LEARNING_EXAMPLE'],
        maxDepth: 0,
        maxResults: 1,
        sortByRelevance: true,
        filter: (node: KnowledgeNode) => node.metadata.modelUsed === modelName
      });
      
      if (results.nodes.length > 0) {
        const exampleNode = results.nodes[0];
        return {
          prompt: exampleNode.metadata.prompt,
          response: exampleNode.metadata.response,
          modelUsed: exampleNode.metadata.modelUsed,
          timestamp: new Date(exampleNode.metadata.timestamp),
          category: exampleNode.metadata.category
        };
      }
      
      return null;
    } catch (error) {
      this.logger.error('Erreur lors de la recherche d\'un exemple d\'apprentissage similaire dans le graphe de connaissances', { error });
      return null;
    }
  }

  private findSimilarLearningExampleInMemory(prompt: string, modelName: string) {
    // Implémentation simplifiée: recherche de mots-clés communs
    const promptKeywords = this.extractKeywords(prompt);
    
    return this.learningExamples.find(example => {
      // Vérifier si au moins 60% des mots-clés correspondent
      const exampleKeywords = this.extractKeywords(example.prompt);
      const commonKeywords = promptKeywords.filter(kw => exampleKeywords.includes(kw));
      
      return (commonKeywords.length / promptKeywords.length) > 0.6;
    });
  }

  /**
   * Extrait les mots-clés significatifs d'un texte
   */
  private extractKeywords(text: string): string[] {
    // Implémentation simplifiée
    const stopWords = ['le', 'la', 'les', 'un', 'une', 'des', 'et', 'ou', 'de', 'à', 'en', 'dans', 'sur'];
    return text
      .toLowerCase()
      .replace(/[.,?!;:()]/g, '')
      .split(/\s+/)
      .filter(word => word.length > 3 && !stopWords.includes(word));
  }

  /**
   * Sauvegarde les exemples d'apprentissage dans un stockage persistant
   */
  private saveLearningExamples() {
    if (!this.knowledgeGraph) {
      // Sauvegarder les exemples en mémoire dans un fichier JSON
      fs.writeFileSync('learning_examples.json', JSON.stringify(this.learningExamples, null, 2));
    }
  }

  /**
   * Génère une réponse simulée à partir d'un texte en utilisant un modèle local
   */
  private simulateModelGeneration(prompt: string, modelName: string): string {
    // Implémentation simplifiée de la génération simulée
    return `[${modelName}] Réponse simulée pour: "${prompt.substring(0, 30)}..."`;
  }

  /**
   * Fine-tune un modèle distillé avec TensorFlow.js
   * @param modelName Nom du modèle à fine-tuner
   */
  async finetuneDistilledModel(modelName: string): Promise<{
    success: boolean;
    message: string;
    trainedExamples?: number;
    accuracy?: number;
    loss?: number;
  }> {
    try {
      this.logger.info(`Début du fine-tuning du modèle ${modelName}`);
      
      // Filtrer les exemples pertinents pour ce modèle
      const relevantExamples = this.learningExamples.filter(ex => 
        ex.modelUsed === 'deepseek-r1' && 
        !ex.category.includes('specific') && 
        ex.prompt.length > 10 && 
        ex.response.length > 20
      );
      
      // Vérifier si nous avons assez d'exemples
      if (relevantExamples.length < 5) {
        this.logger.warn(`Pas assez d'exemples pour entraîner ${modelName}`, {
          count: relevantExamples.length
        });
        return {
          success: false,
          message: `Pas assez d'exemples (${relevantExamples.length}/5 requis)`,
        };
      }
      
      this.logger.info(`Préparation des données pour l'entraînement de ${modelName}`, {
        examplesCount: relevantExamples.length
      });
      
      // Préparer les données d'entraînement
      const trainingPrompts = relevantExamples.map(ex => ex.prompt);
      const trainingResponses = relevantExamples.map(ex => ex.response);
      
      // Charger ou créer le modèle
      let model;
      const vocabSize = 5000; // Taille de vocabulaire simplifiée pour la démonstration
      
      // Vérifier si le modèle existe déjà
      if (this.modelUtilsService.modelExists(modelName)) {
        this.logger.info(`Chargement du modèle existant ${modelName}`);
        model = await this.modelUtilsService.loadModel(modelName);
      } else {
        this.logger.info(`Création d'un nouveau modèle ${modelName}`);
        model = await this.modelUtilsService.createSimpleModel(
          modelName,
          vocabSize,
          vocabSize
        );
      }
      
      if (!model) {
        return {
          success: false,
          message: `Impossible de charger ou de créer le modèle ${modelName}`,
        };
      }
      
      // Créer le dataset pour l'entraînement
      const { dataset } = this.modelUtilsService.createDataset(
        trainingPrompts,
        trainingResponses
      );
      
      // Entraîner le modèle
      this.logger.info(`Début de l'entraînement du modèle ${modelName}`);
      const tf = this.modelUtilsService.getTensorflow();
      
      // Configuration pour l'entraînement
      const epochs = 10;
      const trainingHistory = await model.fitDataset(dataset, {
        epochs,
        callbacks: {
          onEpochEnd: (epoch, logs) => {
            this.logger.info(`Epoch ${epoch + 1}/${epochs}`, {
              loss: logs.loss.toFixed(4),
              accuracy: logs.accuracy ? logs.accuracy.toFixed(4) : 'N/A'
            });
          }
        }
      });
      
      // Sauvegarder le modèle entraîné
      const modelSavePath = `file://./models/${modelName}`;
      await model.save(modelSavePath);
      
      this.logger.info(`Modèle ${modelName} entraîné et sauvegardé avec succès`, {
        path: modelSavePath,
        examples: relevantExamples.length
      });
      
      const finalLoss = trainingHistory.history.loss[trainingHistory.history.loss.length - 1];
      const finalAccuracy = trainingHistory.history.accuracy 
        ? trainingHistory.history.accuracy[trainingHistory.history.accuracy.length - 1]
        : undefined;
      
      return {
        success: true,
        message: `Modèle ${modelName} entraîné avec succès`,
        trainedExamples: relevantExamples.length,
        accuracy: finalAccuracy,
        loss: finalLoss
      };
    } catch (error) {
      this.logger.error(`Erreur lors du fine-tuning du modèle ${modelName}`, { error });
      return {
        success: false,
        message: `Erreur: ${error.message}`,
      };
    }
  }

  /**
   * Calcule la similarité cosinus entre deux nombres
   * @param a Premier nombre
   * @param b Second nombre
   * @returns Similarité cosinus (entre 0 et 1)
   */
  private cosineSimilarity(a: number, b: number): number {
    const dotProduct = a * b;
    const normA = Math.abs(a);
    const normB = Math.abs(b);
    return dotProduct / (normA * normB);
  }

  /**
   * Génère un hash simple à partir d'un texte
   * @param text Texte à hacher
   * @returns Hash numérique
   */
  private simpleHash(text: string): number {
    let hash = 0;
    for (let i = 0; i < text.length; i++) {
      const char = text.charCodeAt(i);
      hash = (hash << 5) - hash + char;
      hash = hash & hash; // Convertir en entier 32 bits
    }
    return hash;
  }
} 