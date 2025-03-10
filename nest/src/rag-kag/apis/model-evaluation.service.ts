import { Injectable, Inject, OnModuleInit, forwardRef, Optional } from '@nestjs/common';
import { LOGGER_TOKEN, ILogger } from '../utils/logger-tokens';
import { HouseModelService } from './house-model.service';
import { ModelUtilsService } from './model-utils.service';
import { EventBusService, RagKagEventType } from '../core/event-bus.service';
import { KnowledgeGraphService, KnowledgeSource } from '../core/knowledge-graph.service';
import * as fs from 'fs';
import * as path from 'path';

interface ModelEvaluation {
  modelName: string;
  timestamp: Date;
  metrics: {
    accuracy: number;
    bleuScore: number;
    rougeScore: number;
    semanticSimilarity: number;
    averageDivergence: number;
    responseTime: number;
  };
  comparisonToTeacher: {
    semanticSimilarity: number;
    contentCoverage: number;
    styleAlignment: number;
    overallQuality: number;
  };
  evaluationSamples: Array<{
    prompt: string;
    teacherResponse: string;
    modelResponse: string;
    metrics: {
      similarity: number;
      contentCoverage: number;
    };
  }>;
}

interface EvaluationConfig {
  minSamples: number;
  targetModels: string[];
  evaluationFrequency: number; // en heures
  semanticThreshold: number; // seuil de similarité sémantique acceptable
}

/**
 * Service pour l'évaluation des modèles distillés
 * Permet de mesurer la qualité des modèles distillés par rapport au modèle enseignant
 * Intégré avec EventBus et KnowledgeGraph pour traçabilité et persistance
 */
@Injectable()
export class ModelEvaluationService implements OnModuleInit {
  private readonly evaluationsDir = './evaluations';
  private evaluationHistory: Map<string, ModelEvaluation[]> = new Map();
  private evaluationConfig: EvaluationConfig = {
    minSamples: 10,
    targetModels: ['phi-3-mini', 'llama-3-8b', 'mistral-7b-fr'],
    evaluationFrequency: 12,
    semanticThreshold: 0.7
  };
  
  constructor(
    @Inject(LOGGER_TOKEN) private readonly logger: ILogger,
    @Inject(forwardRef(() => HouseModelService)) private readonly houseModelService: HouseModelService,
    private readonly modelUtilsService: ModelUtilsService,
    @Optional() private readonly eventBus?: EventBusService,
    @Optional() private readonly knowledgeGraph?: KnowledgeGraphService
  ) {}
  
  async onModuleInit() {
    this.logger.info('Initialisation du service d\'évaluation des modèles');
    this.ensureDirectories();
    
    // Charger l'historique d'évaluation depuis le graphe de connaissances s'il est disponible
    if (this.knowledgeGraph) {
      await this.loadEvaluationsFromGraph();
    } else {
      await this.loadEvaluationHistory();
    }
    
    // Émettre un événement d'initialisation
    if (this.eventBus) {
      this.eventBus.emit({
        type: RagKagEventType.MODEL_EVALUATION_INITIALIZED,
        source: 'ModelEvaluationService',
        payload: {
          targetModels: this.evaluationConfig.targetModels,
          frequency: this.evaluationConfig.evaluationFrequency
        }
      });
    }
  }
  
  /**
   * Crée les répertoires nécessaires
   */
  private ensureDirectories() {
    if (!fs.existsSync(this.evaluationsDir)) {
      fs.mkdirSync(this.evaluationsDir, { recursive: true });
      this.logger.info('Répertoire d\'évaluations créé');
    }
  }
  
  /**
   * Charge l'historique des évaluations
   */
  private async loadEvaluationHistory() {
    try {
      for (const model of this.evaluationConfig.targetModels) {
        const filePath = path.join(this.evaluationsDir, `${model}_evaluations.json`);
        
        if (fs.existsSync(filePath)) {
          const data = fs.readFileSync(filePath, 'utf8');
          const evaluations = JSON.parse(data) as ModelEvaluation[];
          this.evaluationHistory.set(model, evaluations);
          
          this.logger.info(`Historique d'évaluation chargé pour ${model}`, {
            evaluationCount: evaluations.length
          });
        } else {
          this.evaluationHistory.set(model, []);
        }
      }
    } catch (error) {
      this.logger.error(`Erreur lors du chargement de l'historique d'évaluation`, {
        error: error.message,
        stack: error.stack
      });
    }
  }
  
  /**
   * Charge les évaluations depuis le graphe de connaissances
   */
  private async loadEvaluationsFromGraph(): Promise<void> {
    if (!this.knowledgeGraph) {
      this.logger.warn('Le graphe de connaissances n\'est pas disponible, impossible de charger l\'historique d\'évaluation');
      return;
    }
    
    try {
      // Rechercher les nœuds d'évaluation dans le graphe
      const evaluationNodes = await this.knowledgeGraph.search('model evaluation', {
        nodeTypes: ['MODEL_EVALUATION'],
        maxDepth: 0,
        maxResults: 100
      });
      
      // Convertir les nœuds en objets ModelEvaluation
      const evaluations: ModelEvaluation[] = evaluationNodes.nodes.map(node => ({
        modelName: node.metadata.modelName,
        timestamp: new Date(node.metadata.timestamp),
        metrics: node.metadata.metrics,
        comparisonToTeacher: node.metadata.comparisonToTeacher,
        evaluationSamples: node.metadata.evaluationSamples
      }));
      
      // Mettre à jour l'historique d'évaluation
      for (const evaluation of evaluations) {
        const { modelName } = evaluation;
        if (!this.evaluationHistory.has(modelName)) {
          this.evaluationHistory.set(modelName, []);
        }
        this.evaluationHistory.get(modelName)!.push(evaluation);
      }
      
      this.logger.info(`Historique d'évaluation chargé depuis le graphe de connaissances: ${evaluations.length} évaluations`);
    } catch (error) {
      this.logger.error('Erreur lors du chargement de l\'historique d\'évaluation depuis le graphe de connaissances', { error });
    }
  }
  
  /**
   * Génère un jeu de données d'évaluation
   */
  private async generateEvaluationDataset(sampleSize: number = 10): Promise<Array<{prompt: string, teacherResponse: string}>> {
    this.logger.info(`Génération d'un jeu de données d'évaluation de ${sampleSize} échantillons`);
    
    // Prompts d'évaluation prédéfinis (pour la démonstration)
    const evaluationPrompts = [
      "Explique-moi comment fonctionne un moteur à combustion interne",
      "Quels sont les avantages et inconvénients des énergies renouvelables ?",
      "Comment fonctionnent les algorithmes de machine learning ?",
      "Quelles sont les causes principales du changement climatique ?",
      "Comment peut-on améliorer la sécurité informatique d'une entreprise ?",
      "Explique-moi le principe de la relativité d'Einstein",
      "Qu'est-ce que l'apprentissage par renforcement en IA ?",
      "Quels sont les fondements de l'économie circulaire ?",
      "Comment fonctionne un réseau de neurones artificiels ?",
      "Quelles sont les techniques de négociation les plus efficaces ?",
      "Explique le fonctionnement d'une blockchain",
      "Comment créer une stratégie marketing digitale efficace ?",
      "Quels sont les principes de la comptabilité en partie double ?",
      "Comment fonctionne le système immunitaire humain ?",
      "Quelles sont les meilleures pratiques de développement Agile ?"
    ];
    
    // Sélectionner des prompts aléatoires du jeu prédéfini
    const selectedPrompts = [...evaluationPrompts].sort(() => 0.5 - Math.random()).slice(0, sampleSize);
    
    // Générer les réponses du modèle enseignant
    const dataset = [];
    
    for (const prompt of selectedPrompts) {
      try {
        this.logger.info(`Génération de la réponse de référence pour: "${prompt.substring(0, 30)}..."`);
        
        // Appeler DeepSeek R1 pour obtenir une réponse de référence
        const teacherResponse = await this.houseModelService.generateResponse(prompt, {
          model: 'deepseek-r1',
          temperature: 0.7
        });
        
        dataset.push({
          prompt,
          teacherResponse: teacherResponse.response
        });
        
        // Attendre un peu entre les générations pour éviter de surcharger l'API
        await new Promise(resolve => setTimeout(resolve, 500));
      } catch (error) {
        this.logger.error(`Erreur lors de la génération de la réponse pour: "${prompt.substring(0, 30)}..."`, { error });
      }
    }
    
    this.logger.info(`Jeu de données d'évaluation généré avec ${dataset.length} échantillons`);
    return dataset;
  }
  
  /**
   * Calcule la similarité cosinus entre deux vecteurs
   */
  private cosineSimilarity(vecA: number[], vecB: number[]): number {
    if (vecA.length !== vecB.length) {
      throw new Error('Les vecteurs doivent avoir la même dimension');
    }
    
    let dotProduct = 0;
    let normA = 0;
    let normB = 0;
    
    for (let i = 0; i < vecA.length; i++) {
      dotProduct += vecA[i] * vecB[i];
      normA += vecA[i] * vecA[i];
      normB += vecB[i] * vecB[i];
    }
    
    if (normA === 0 || normB === 0) {
      return 0;
    }
    
    return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
  }
  
  /**
   * Calcule un score BLEU simplifié
   */
  private calculateBleuScore(reference: string, hypothesis: string): number {
    // Version très simplifiée de BLEU pour la démonstration
    const refWords = reference.toLowerCase().split(/\s+/);
    const hypWords = hypothesis.toLowerCase().split(/\s+/);
    
    let matches = 0;
    for (const word of hypWords) {
      if (refWords.includes(word)) {
        matches++;
      }
    }
    
    // Pénalité de brièveté
    const brevityPenalty = Math.exp(Math.min(0, 1 - refWords.length / hypWords.length));
    
    return brevityPenalty * (matches / hypWords.length);
  }
  
  /**
   * Calcule un score ROUGE simplifié
   */
  private calculateRougeScore(reference: string, hypothesis: string): number {
    // Version simplifiée de ROUGE-L pour la démonstration
    const refWords = reference.toLowerCase().split(/\s+/);
    const hypWords = hypothesis.toLowerCase().split(/\s+/);
    
    // Calculer la plus longue sous-séquence commune
    const lcsMatrix = Array(refWords.length + 1).fill(0).map(() => Array(hypWords.length + 1).fill(0));
    
    for (let i = 1; i <= refWords.length; i++) {
      for (let j = 1; j <= hypWords.length; j++) {
        if (refWords[i - 1] === hypWords[j - 1]) {
          lcsMatrix[i][j] = lcsMatrix[i - 1][j - 1] + 1;
        } else {
          lcsMatrix[i][j] = Math.max(lcsMatrix[i - 1][j], lcsMatrix[i][j - 1]);
        }
      }
    }
    
    const lcsLength = lcsMatrix[refWords.length][hypWords.length];
    
    // Calculer précision, rappel et F1
    const precision = lcsLength / hypWords.length;
    const recall = lcsLength / refWords.length;
    
    if (precision + recall === 0) {
      return 0;
    }
    
    // F1 score (moyenne harmonique)
    return 2 * precision * recall / (precision + recall);
  }
  
  /**
   * Calcule la similarité sémantique entre deux textes
   * Utilise une approche simplifiée basée sur les mots-clés pour la démonstration
   */
  private calculateSemanticSimilarity(reference: string, hypothesis: string): number {
    // Extraire les mots significatifs (simulé)
    const getKeywords = (text: string): string[] => {
      return text.toLowerCase()
        .replace(/[.,;:!?()]/g, '')
        .split(/\s+/)
        .filter(word => word.length > 3)
        .filter(word => !['avec', 'pour', 'dans', 'cette', 'comme', 'plus'].includes(word));
    };
    
    const refKeywords = getKeywords(reference);
    const hypKeywords = getKeywords(hypothesis);
    
    // Compter les intersections
    const intersection = hypKeywords.filter(word => refKeywords.includes(word));
    
    // Calculer Jaccard
    const union = new Set([...refKeywords, ...hypKeywords]);
    
    return intersection.length / union.size;
  }
  
  /**
   * Réalise l'évaluation d'un modèle distillé par rapport au modèle enseignant
   */
  async evaluateModel(modelName: string, sampleSize: number = 10): Promise<ModelEvaluation> {
    this.logger.info(`Début de l'évaluation du modèle ${modelName}`);
    
    // Émettre un événement de début d'évaluation
    if (this.eventBus) {
      this.eventBus.emit({
        type: RagKagEventType.MODEL_EVALUATION_STARTED,
        source: 'ModelEvaluationService',
        payload: {
          model: modelName,
          sampleSize
        }
      });
    }
    
    // Vérifier si le modèle est disponible
    const modelExists = this.modelUtilsService.modelExists(modelName);
    if (!modelExists) {
      this.logger.warn(`Le modèle ${modelName} n'existe pas encore, création d'une évaluation de base`);
      
      const baseEvaluation = {
        modelName,
        timestamp: new Date(),
        metrics: {
          accuracy: 0,
          bleuScore: 0,
          rougeScore: 0,
          semanticSimilarity: 0,
          averageDivergence: 1,
          responseTime: 0
        },
        comparisonToTeacher: {
          semanticSimilarity: 0,
          contentCoverage: 0,
          styleAlignment: 0,
          overallQuality: 0
        },
        evaluationSamples: []
      };
      
      // Émettre un événement d'échec d'évaluation
      if (this.eventBus) {
        this.eventBus.emit({
          type: RagKagEventType.CUSTOM,
          source: 'ModelEvaluationService',
          payload: {
            eventType: 'MODEL_EVALUATION_SKIPPED',
            model: modelName,
            reason: 'model_not_exists'
          }
        });
      }
      
      return baseEvaluation;
    }
    
    // Générer un jeu de données d'évaluation
    const evaluationDataset = await this.generateEvaluationDataset(sampleSize);
    
    // Évaluer chaque échantillon
    const samples = [];
    let totalBleu = 0;
    let totalRouge = 0;
    let totalSemantic = 0;
    let totalResponseTime = 0;
    
    for (const sample of evaluationDataset) {
      try {
        const startTime = Date.now();
        
        const teacherResponse = sample.teacherResponse;
        
        const modelResponse = await this.houseModelService.generateResponse(sample.prompt, {
          temperature: 0.7,
          maxTokens: 100,
          model: modelName
        });
        
        const responseTime = Date.now() - startTime;
        totalResponseTime += responseTime;
        
        // Calculer les métriques
        const bleu = this.calculateBleuScore(teacherResponse, modelResponse.response);
        const rouge = this.calculateRougeScore(teacherResponse, modelResponse.response);
        const semantic = this.calculateSemanticSimilarity(teacherResponse, modelResponse.response);
        
        totalBleu += bleu;
        totalRouge += rouge;
        totalSemantic += semantic;
        
        // Ajouter l'échantillon aux résultats
        samples.push({
          prompt: sample.prompt,
          teacherResponse,
          modelResponse: modelResponse.response,
          metrics: {
            similarity: semantic,
            contentCoverage: (bleu + rouge) / 2
          }
        });
        
        this.logger.info(`Évaluation de l'échantillon ${samples.length}/${evaluationDataset.length}`, {
          prompt: sample.prompt.substring(0, 30) + '...',
          bleu: bleu.toFixed(2),
          rouge: rouge.toFixed(2),
          semantic: semantic.toFixed(2)
        });
        
        // Émettre un événement pour chaque échantillon évalué
        if (this.eventBus) {
          this.eventBus.emit({
            type: RagKagEventType.CUSTOM,
            source: 'ModelEvaluationService',
            payload: {
              eventType: 'SAMPLE_EVALUATED',
              model: modelName,
              metrics: {
                bleu,
                rouge,
                semantic,
                responseTime
              },
              sampleIndex: samples.length,
              totalSamples: evaluationDataset.length
            }
          });
        }
      } catch (error) {
        this.logger.error(`Erreur lors de l'évaluation de l'échantillon`, { error, prompt: sample.prompt.substring(0, 30) + '...' });
        
        // Émettre un événement d'erreur
        if (this.eventBus) {
          this.eventBus.emit({
            type: RagKagEventType.QUERY_ERROR,
            source: 'ModelEvaluationService',
            payload: {
              model: modelName,
              operation: 'sample_evaluation',
              error: error.message
            }
          });
        }
      }
    }
    
    // Calculer les moyennes
    const sampleCount = samples.length;
    const avgBleu = sampleCount > 0 ? totalBleu / sampleCount : 0;
    const avgRouge = sampleCount > 0 ? totalRouge / sampleCount : 0;
    const avgSemantic = sampleCount > 0 ? totalSemantic / sampleCount : 0;
    const avgResponseTime = sampleCount > 0 ? totalResponseTime / sampleCount : 0;
    
    // Modèle d'évaluation complet
    const evaluation: ModelEvaluation = {
      modelName,
      timestamp: new Date(),
      metrics: {
        accuracy: (avgBleu + avgRouge) / 2,
        bleuScore: avgBleu,
        rougeScore: avgRouge,
        semanticSimilarity: avgSemantic,
        averageDivergence: 1 - avgSemantic,
        responseTime: avgResponseTime
      },
      comparisonToTeacher: {
        semanticSimilarity: avgSemantic,
        contentCoverage: avgBleu,
        styleAlignment: avgRouge,
        overallQuality: (avgSemantic + avgBleu + avgRouge) / 3
      },
      evaluationSamples: samples
    };
    
    // Sauvegarder l'évaluation
    await this.saveEvaluation(modelName, evaluation);
    
    // Stocker l'évaluation dans le graphe de connaissances
    if (this.knowledgeGraph) {
      await this.storeEvaluationInGraph(modelName, evaluation);
    }
    
    // Émettre un événement de fin d'évaluation
    if (this.eventBus) {
      this.eventBus.emit({
        type: RagKagEventType.MODEL_EVALUATION_COMPLETED,
        source: 'ModelEvaluationService',
        payload: {
          model: modelName,
          metrics: {
            accuracy: evaluation.metrics.accuracy,
            bleuScore: evaluation.metrics.bleuScore,
            semanticSimilarity: evaluation.metrics.semanticSimilarity,
            overallQuality: evaluation.comparisonToTeacher.overallQuality
          },
          sampleCount: sampleCount,
          timestamp: evaluation.timestamp
        }
      });
    }
    
    this.logger.info(`Évaluation du modèle ${modelName} terminée`, {
      overallQuality: evaluation.comparisonToTeacher.overallQuality.toFixed(2),
      samples: sampleCount
    });
    
    return evaluation;
  }
  
  /**
   * Sauvegarde une évaluation
   */
  private async saveEvaluation(modelName: string, evaluation: ModelEvaluation) {
    try {
      // Récupérer l'historique existant
      let history = this.evaluationHistory.get(modelName) || [];
      
      // Ajouter la nouvelle évaluation
      history.push(evaluation);
      
      // Limiter la taille de l'historique (garder les 10 dernières évaluations)
      if (history.length > 10) {
        history = history.slice(history.length - 10);
      }
      
      // Mettre à jour le cache
      this.evaluationHistory.set(modelName, history);
      
      // Sauvegarder dans le fichier
      const filePath = path.join(this.evaluationsDir, `${modelName}_evaluations.json`);
      fs.writeFileSync(filePath, JSON.stringify(history, null, 2));
      
      this.logger.info(`Évaluation sauvegardée pour ${modelName}`);
    } catch (error) {
      this.logger.error(`Erreur lors de la sauvegarde de l'évaluation pour ${modelName}`, { error });
    }
  }
  
  /**
   * Stocke une évaluation dans le graphe de connaissances
   */
  private async storeEvaluationInGraph(modelName: string, evaluation: ModelEvaluation): Promise<void> {
    if (!this.knowledgeGraph) {
      this.logger.warn('Le graphe de connaissances n\'est pas disponible, impossible de stocker l\'évaluation');
      return;
    }
    
    try {
      // Créer un nœud pour l'évaluation
      const evaluationNodeId = this.knowledgeGraph.addNode({
        label: `Evaluation ${modelName} ${evaluation.timestamp.toISOString()}`,
        type: 'MODEL_EVALUATION',
        content: JSON.stringify(evaluation),
        confidence: 1,
        source: KnowledgeSource.SYSTEM,
        metadata: {
          modelName,
          timestamp: evaluation.timestamp.toISOString(),
          metrics: evaluation.metrics,
          comparisonToTeacher: evaluation.comparisonToTeacher,
          evaluationSamples: evaluation.evaluationSamples
        }
      });
      
      // Lier l'évaluation au modèle
      this.knowledgeGraph.addFact(
        evaluationNodeId,
        'EVALUATES',
        {
          label: modelName,
          type: 'MODEL',
          content: modelName,
          confidence: 1,
          source: KnowledgeSource.SYSTEM
        },
        1,
        { bidirectional: false, weight: 1 }
      );
      
      this.logger.debug(`Évaluation de ${modelName} stockée dans le graphe de connaissances`, {
        nodeId: evaluationNodeId,
        timestamp: evaluation.timestamp
      });
    } catch (error) {
      this.logger.error('Erreur lors du stockage de l\'évaluation dans le graphe de connaissances', { error });
    }
  }
  
  /**
   * Récupère les statistiques d'évaluation pour un modèle
   */
  getModelEvaluationStats(modelName: string): {
    latestEvaluation: ModelEvaluation | null;
    progressTrend: {
      accuracy: number[];
      semanticSimilarity: number[];
      timestamps: Date[];
    };
    comparisonWithTeacher: number;
  } {
    const history = this.evaluationHistory.get(modelName) || [];
    
    if (history.length === 0) {
      return {
        latestEvaluation: null,
        progressTrend: {
          accuracy: [],
          semanticSimilarity: [],
          timestamps: []
        },
        comparisonWithTeacher: 0
      };
    }
    
    // Récupérer la dernière évaluation
    const latestEvaluation = history[history.length - 1];
    
    // Calculer les tendances
    const accuracy = history.map(e => e.metrics.accuracy);
    const semanticSimilarity = history.map(e => e.metrics.semanticSimilarity);
    const timestamps = history.map(e => e.timestamp);
    
    // Comparaison avec le modèle enseignant (valeur entre 0 et 1)
    const comparisonWithTeacher = latestEvaluation.comparisonToTeacher.overallQuality;
    
    return {
      latestEvaluation,
      progressTrend: {
        accuracy,
        semanticSimilarity,
        timestamps
      },
      comparisonWithTeacher
    };
  }
  
  /**
   * Récupère les statistiques pour tous les modèles
   */
  getAllModelsEvaluationStats() {
    const stats = {};
    
    for (const modelName of this.evaluationConfig.targetModels) {
      stats[modelName] = this.getModelEvaluationStats(modelName);
    }
    
    return stats;
  }
  
  /**
   * Vérifie si un modèle est prêt à remplacer le modèle enseignant pour certaines requêtes
   */
  isModelReliable(modelName: string): {
    isReliable: boolean;
    reliabilityScore: number;
    recommendedDomains: string[];
  } {
    const stats = this.getModelEvaluationStats(modelName);
    
    if (!stats.latestEvaluation) {
      return {
        isReliable: false,
        reliabilityScore: 0,
        recommendedDomains: []
      };
    }
    
    // Un modèle est considéré fiable si sa similarité sémantique avec le modèle enseignant
    // dépasse le seuil défini dans la configuration
    const isReliable = stats.comparisonWithTeacher >= this.evaluationConfig.semanticThreshold;
    
    // Domaines recommandés (à adapter selon les résultats réels)
    const recommendedDomains = [];
    
    // Analyser les échantillons pour déterminer les forces du modèle
    const latestSamples = stats.latestEvaluation.evaluationSamples;
    
    // Regrouper les échantillons par domaine (simulé)
    const domainKeywords = {
      'science': ['moteur', 'physique', 'relativité', 'combustion', 'climatique'],
      'technologie': ['algorithmes', 'machine learning', 'informatique', 'sécurité', 'réseau', 'neurones', 'blockchain'],
      'business': ['négociation', 'marketing', 'stratégie', 'comptabilité', 'économie'],
      'environnement': ['énergies', 'renouvelables', 'climat', 'climatique', 'circulaire'],
      'santé': ['immunitaire', 'humain', 'santé']
    };
    
    // Analyser chaque domaine
    const domainScores = {};
    
    // Initialiser les scores à 0
    for (const domain of Object.keys(domainKeywords)) {
      domainScores[domain] = 0;
    }
    
    // Calculer les scores par domaine
    for (const sample of latestSamples) {
      const prompt = sample.prompt.toLowerCase();
      
      for (const [domain, keywords] of Object.entries(domainKeywords)) {
        for (const keyword of keywords) {
          if (prompt.includes(keyword)) {
            domainScores[domain] += sample.metrics.similarity;
            break;
          }
        }
      }
    }
    
    // Normaliser les scores
    let totalScore = 0;
    for (const domain of Object.keys(domainScores)) {
      totalScore += domainScores[domain];
    }
    
    // Sélectionner les domaines où le modèle est le plus performant
    if (totalScore > 0) {
      for (const [domain, score] of Object.entries(domainScores)) {
        const normalizedScore = score as number / totalScore;
        if (normalizedScore > 0.2) { // Seuil arbitraire pour considérer un domaine comme recommandé
          recommendedDomains.push(domain);
        }
      }
    }
    
    return {
      isReliable,
      reliabilityScore: stats.comparisonWithTeacher,
      recommendedDomains
    };
  }
  
  /**
   * Déclenche l'évaluation périodique de tous les modèles
   */
  async evaluateAllModels() {
    this.logger.info('Démarrage de l\'évaluation périodique de tous les modèles');
    
    // Émettre un événement de début d'évaluation de tous les modèles
    if (this.eventBus) {
      this.eventBus.emit({
        type: RagKagEventType.MODEL_EVALUATION_STARTED,
        source: 'ModelEvaluationService',
        payload: {
          type: 'periodic',
          models: this.evaluationConfig.targetModels,
          sampleSize: this.evaluationConfig.minSamples
        }
      });
    }
    
    const results = {};
    const errors = [];
    
    for (const modelName of this.evaluationConfig.targetModels) {
      try {
        const evaluation = await this.evaluateModel(modelName, this.evaluationConfig.minSamples);
        results[modelName] = {
          accuracy: evaluation.metrics.accuracy,
          overallQuality: evaluation.comparisonToTeacher.overallQuality
        };
      } catch (error) {
        this.logger.error(`Erreur lors de l'évaluation périodique du modèle ${modelName}`, { error });
        errors.push({
          model: modelName,
          error: error.message
        });
      }
    }
    
    // Émettre un événement de fin d'évaluation de tous les modèles
    if (this.eventBus) {
      this.eventBus.emit({
        type: RagKagEventType.MODEL_EVALUATION_COMPLETED,
        source: 'ModelEvaluationService',
        payload: {
          type: 'periodic',
          results,
          errors,
          timestamp: new Date()
        }
      });
    }
    
    this.logger.info('Évaluation périodique de tous les modèles terminée', {
      successes: Object.keys(results).length,
      errors: errors.length
    });
    
    return {
      results,
      errors
    };
  }
  
  async scheduledEvaluation() {
    const results: Record<string, ModelEvaluation> = {};
    const errors: string[] = [];

    for (const modelName of this.evaluationConfig.targetModels) {
      try {
        const evaluation = await this.evaluateModel(modelName);
        results[modelName] = evaluation;
        
        // Stocker le résultat dans le graphe de connaissances
        if (this.knowledgeGraph) {
          await this.storeEvaluationInGraph(modelName, evaluation);
        }
      } catch (error) {
        this.logger.error(`Erreur lors de l'évaluation périodique de ${modelName}`, { error });
        errors.push(`${modelName}: ${error.message}`);
      }
    }
    
    // Émettre un événement pour l'évaluation périodique
    if (this.eventBus) {
      this.eventBus.emit({
        type: RagKagEventType.MODEL_EVALUATION_COMPLETED,
        source: 'ModelEvaluationService',
        payload: {
          type: 'periodic',
          results,
          errors,
          timestamp: new Date()
        }
      });
    }
    
    this.logger.info('Évaluation périodique de tous les modèles terminée', {
      successes: Object.keys(results).length,
      errors: errors.length
    });
    
    return {
      results,
      errors
    };
  }
  
  // TODO: Update this method to work with the new Python API integration
  // async forceEvaluateModel(modelName: string) {
  //   const evaluation = await this.evaluateModel(modelName);
  //   await this.saveEvaluation(modelName, evaluation);
  //   
  //   this.logger.info(`Évaluation forcée du modèle ${modelName} terminée`, {
  //     accuracy: evaluation.metrics.accuracy,
  //     bleuScore: evaluation.metrics.bleuScore
  //   });
  //   
  //   return evaluation;
  // }
} 