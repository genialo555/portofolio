import { Inject, Injectable, Optional, forwardRef } from '@nestjs/common';
import { LOGGER_TOKEN } from './logger-tokens';
import { ILogger } from './logger-tokens';
import { KnowledgeGraphService } from '../core/knowledge-graph.service';
import { ApiProviderFactory } from '../apis/api-provider-factory.service';
import { HouseModelService } from '../apis/house-model.service';
import { UserQuery } from '../types';
import { ModuleRef } from '@nestjs/core';

/**
 * Niveau de complexité d'une requête
 */
export enum ComplexityLevel {
  SIMPLE = 'SIMPLE',          // Requête simple, un seul modèle suffit
  STANDARD = 'STANDARD',      // Requête standard, nécessite RAG ou KAG
  COMPLEX = 'COMPLEX'         // Requête complexe, nécessite le pipeline complet
}

/**
 * Représente le résultat de l'analyse de complexité d'une requête
 */
export interface ComplexityAnalysisResult {
  level: ComplexityLevel;     // Niveau de complexité global
  score: number;              // Score de complexité (0-1)
  requiresFullDebate: boolean; // Si un débat complet RAG/KAG est nécessaire
  recommendedPipeline: 'simple' | 'standard' | 'full'; // Pipeline recommandé
  factors: {                  // Facteurs qui ont influencé la décision
    queryLength: number;      // Longueur de la requête
    domainSpecificity: number; // Spécificité du domaine
    factualNature: number;    // Nature factuelle vs. opinion
    technicalLevel: number;   // Niveau technique
    multiPartQuery: boolean;  // Si la requête contient plusieurs sous-questions
    patternMatch?: number;    // Correspondance avec des patterns connus
    requiresRetrieval: number; // Nécessité de récupérer des informations externes
  };
  cachedResult: boolean;      // Si le résultat provient du cache
  processingTime: number;     // Temps de traitement en ms
}

/**
 * Options pour l'analyse de complexité
 */
export interface ComplexityAnalysisOptions {
  useCachedResults?: boolean; // Utiliser les résultats en cache si disponibles
  quickAnalysis?: boolean;    // Utiliser une analyse rapide (moins précise)
  maxProcessingTime?: number; // Temps de traitement maximum en ms
  threshold?: {              // Seuils personnalisés pour la classification
    simple: number;          // Seuil pour les requêtes simples (0-1)
    complex: number;         // Seuil pour les requêtes complexes (0-1)
  };
}

/**
 * Service d'analyse de complexité des requêtes
 * Utilise Phi-4 comme agent de détection couplé à un algorithme K-means
 */
@Injectable()
export class ComplexityAnalyzerService {
  private readonly complexityCache: Map<string, ComplexityAnalysisResult> = new Map();
  private readonly cacheSize = 1000; // Taille maximale du cache
  private kMeansModel: any = null; // Modèle K-means pour la classification

  constructor(
    @Inject(LOGGER_TOKEN) private readonly logger: ILogger,
    private readonly houseModelService: HouseModelService,
    private readonly apiProviderFactory: ApiProviderFactory,
    @Optional() private readonly knowledgeGraph?: KnowledgeGraphService
  ) {
    this.initializeKMeansModel();
  }

  /**
   * Initialise le modèle K-means à partir des données du graphe de connaissances
   * ou crée un modèle de base si le graphe n'est pas disponible
   */
  private async initializeKMeansModel(): Promise<void> {
    if (this.knowledgeGraph) {
      // TODO: Charger des exemples du graphe de connaissances pour entraîner le modèle K-means
      this.logger.log('Initialisation du modèle K-means à partir du graphe de connaissances', {
        context: 'ComplexityAnalyzerService'
      });
    } else {
      // Création d'un modèle K-means basique avec 3 clusters (simple, standard, complexe)
      this.kMeansModel = {
        predict: (features: number[]): number => {
          // Implémentation simplifiée de K-means, à remplacer par une vraie implémentation
          const sum = features.reduce((a, b) => a + b, 0);
          const avg = sum / features.length;
          if (avg < 0.3) return 0; // Simple
          if (avg < 0.7) return 1; // Standard
          return 2; // Complexe
        }
      };
      this.logger.log('Modèle K-means basique initialisé', {
        context: 'ComplexityAnalyzerService'
      });
    }
  }

  /**
   * Analyse la complexité d'une requête pour déterminer le pipeline approprié
   * 
   * @param query La requête à analyser
   * @param options Options d'analyse
   * @returns Résultat de l'analyse de complexité
   */
  async analyzeComplexity(
    query: UserQuery | string,
    options: ComplexityAnalysisOptions = {}
  ): Promise<ComplexityAnalysisResult> {
    const startTime = Date.now();
    const queryText = typeof query === 'string' ? query : query.content;
    const queryHash = this.hashString(queryText);

    // Vérifier le cache si activé
    if (options.useCachedResults !== false && this.complexityCache.has(queryHash)) {
      const cachedResult = this.complexityCache.get(queryHash)!;
      return {
        ...cachedResult,
        cachedResult: true,
        processingTime: Date.now() - startTime
      };
    }

    try {
      // Analyse rapide basée sur des heuristiques simples si demandée
      if (options.quickAnalysis) {
        return await this.performQuickAnalysis(queryText, options, startTime);
      }

      // Analyse complète utilisant Phi-4 et K-means
      return await this.performFullAnalysis(queryText, options, startTime);
    } catch (error) {
      this.logger.error(`Erreur lors de l'analyse de complexité: ${error.message}`, {
        context: 'ComplexityAnalyzerService',
        error
      });
      
      // En cas d'erreur, retourner une analyse par défaut basée sur des heuristiques
      return this.performFallbackAnalysis(queryText, startTime);
    }
  }

  /**
   * Effectue une analyse rapide basée sur des heuristiques simples
   */
  private async performQuickAnalysis(
    queryText: string,
    options: ComplexityAnalysisOptions,
    startTime: number
  ): Promise<ComplexityAnalysisResult> {
    // Calcul des facteurs de base
    const factors = this.calculateBasicFactors(queryText);
    
    // Classification basée sur des règles simples
    const score = this.calculateComplexityScore(factors);
    const level = this.classifyComplexity(score, options.threshold);
    const queryHash = this.hashString(queryText);
    
    const result: ComplexityAnalysisResult = {
      level,
      score,
      requiresFullDebate: level === ComplexityLevel.COMPLEX,
      recommendedPipeline: this.getPipelineRecommendation(level),
      factors,
      cachedResult: false,
      processingTime: Date.now() - startTime
    };
    
    // Mettre en cache le résultat
    this.cacheResult(queryHash, result);
    
    return result;
  }

  /**
   * Effectue une analyse complète utilisant Phi-4 et K-means
   */
  private async performFullAnalysis(
    queryText: string,
    options: ComplexityAnalysisOptions,
    startTime: number
  ): Promise<ComplexityAnalysisResult> {
    // Calcul des facteurs de base
    const basicFactors = this.calculateBasicFactors(queryText);
    const queryHash = this.hashString(queryText);
    
    // Utiliser Phi-4 pour une analyse plus sophistiquée
    const phi4Analysis = await this.getPhi4Analysis(queryText, options.maxProcessingTime);
    
    // Combiner les facteurs de base avec l'analyse de Phi-4
    const factors = {
      ...basicFactors,
      ...phi4Analysis.factors
    };
    
    // Préparer les caractéristiques pour le modèle K-means
    const features = this.prepareKMeansFeatures(factors);
    
    // Prédire la classe avec K-means
    const clusterIndex = this.kMeansModel.predict(features);
    
    // Convertir l'index de cluster en niveau de complexité
    const level = this.mapClusterToComplexityLevel(clusterIndex);
    
    // Calculer le score global
    const score = this.calculateComplexityScore(factors);
    
    const result: ComplexityAnalysisResult = {
      level,
      score,
      requiresFullDebate: level === ComplexityLevel.COMPLEX,
      recommendedPipeline: this.getPipelineRecommendation(level),
      factors,
      cachedResult: false,
      processingTime: Date.now() - startTime
    };
    
    // Mettre en cache le résultat
    this.cacheResult(queryHash, result);
    
    return result;
  }

  /**
   * Analyse de repli en cas d'erreur
   */
  private performFallbackAnalysis(queryText: string, startTime: number): ComplexityAnalysisResult {
    const factors = this.calculateBasicFactors(queryText);
    const score = this.calculateComplexityScore(factors);
    const level = this.classifyComplexity(score);
    
    return {
      level,
      score,
      requiresFullDebate: level === ComplexityLevel.COMPLEX,
      recommendedPipeline: this.getPipelineRecommendation(level),
      factors,
      cachedResult: false,
      processingTime: Date.now() - startTime
    };
  }

  /**
   * Calcule les facteurs de base pour l'analyse de complexité
   */
  private calculateBasicFactors(queryText: string): ComplexityAnalysisResult['factors'] {
    const words = queryText.split(/\s+/).filter(w => w.length > 0);
    const technicalTerms = [
      'architecture', 'système', 'algorithme', 'optimisation', 'performance',
      'latence', 'scalabilité', 'framework', 'infrastructure', 'microservices',
      'machine learning', 'deep learning', 'neural', 'statistical', 'probabilistic',
      'économétrie', 'qualitatif', 'quantitatif', 'méthodologie', 'paradigme'
    ];
    
    // Détection des termes techniques
    const technicalTermCount = words.filter(word => 
      technicalTerms.some(term => word.toLowerCase().includes(term.toLowerCase()))
    ).length;
    
    // Compter les points d'interrogation pour détecter les requêtes multi-parties
    const questionMarkCount = (queryText.match(/\?/g) || []).length;
    
    return {
      queryLength: words.length / 50, // Normalisé, 1.0 = 50 mots
      domainSpecificity: 0.5, // Valeur par défaut, à améliorer
      factualNature: 0.5, // Valeur par défaut, à améliorer
      technicalLevel: Math.min(1, technicalTermCount / 5), // Normalisé, 1.0 = 5+ termes techniques
      multiPartQuery: questionMarkCount > 1,
      requiresRetrieval: 0.5 // Valeur par défaut, à améliorer
    };
  }

  /**
   * Effectue une analyse avec Phi-4
   */
  private async getPhi4Analysis(
    queryText: string,
    maxProcessingTime?: number
  ): Promise<{ classification: string; factors: Partial<ComplexityAnalysisResult['factors']> }> {
    try {
      const startTime = Date.now();
      
      const response = await this.houseModelService.generateResponse(queryText, {
        // TODO: Adjust model parameters as needed
        temperature: 0.3,
        maxTokens: 100,
        model: 'phi-3-mini', // On suppose que Phi-4 est disponible sous ce nom
      });
      
      const processingTime = Date.now() - startTime;
      
      // Extraire la classification et les facteurs du JSON dans la réponse
      const jsonMatch = response.response.match(/\{[\s\S]*\}/);
      
      if (jsonMatch) {
        const json = JSON.parse(jsonMatch[0]);
        return {
          classification: json.classification,
          factors: {
            queryLength: json.factors?.queryLength,
            domainSpecificity: json.factors?.domainSpecificity,
            factualNature: json.factors?.factualNature,
            technicalLevel: json.factors?.technicalLevel,
            multiPartQuery: json.factors?.multiPartQuery,
            requiresRetrieval: json.factors?.requiresRetrieval
          }
        };
      } else {
        this.logger.warn('Réponse Phi-4 invalide, classification par défaut utilisée', {
          response: response.response
        });
        
        return {
          classification: 'STANDARD',
          factors: {}
        };
      }
    } catch (error) {
      this.logger.error('Erreur lors de l\'analyse Phi-4', { error });
      
      // Fallback à une classification par défaut en cas d'erreur
      return {
        classification: 'STANDARD',
        factors: {}
      };
    }
  }

  /**
   * Prépare les caractéristiques pour le modèle K-means
   */
  private prepareKMeansFeatures(factors: ComplexityAnalysisResult['factors']): number[] {
    return [
      factors.queryLength,
      factors.domainSpecificity,
      factors.factualNature,
      factors.technicalLevel,
      factors.multiPartQuery ? 1 : 0,
      factors.requiresRetrieval
    ];
  }

  /**
   * Convertit l'index de cluster en niveau de complexité
   */
  private mapClusterToComplexityLevel(clusterIndex: number): ComplexityLevel {
    switch (clusterIndex) {
      case 0: return ComplexityLevel.SIMPLE;
      case 1: return ComplexityLevel.STANDARD;
      case 2: 
      default: return ComplexityLevel.COMPLEX;
    }
  }

  /**
   * Calcule un score de complexité global basé sur les facteurs
   */
  private calculateComplexityScore(factors: ComplexityAnalysisResult['factors']): number {
    // Pondération des facteurs
    const weights = {
      queryLength: 0.15,
      domainSpecificity: 0.2,
      factualNature: 0.15,
      technicalLevel: 0.2,
      multiPartQuery: 0.1,
      requiresRetrieval: 0.2
    };
    
    let score = 
      factors.queryLength * weights.queryLength +
      factors.domainSpecificity * weights.domainSpecificity +
      factors.factualNature * weights.factualNature +
      factors.technicalLevel * weights.technicalLevel +
      (factors.multiPartQuery ? weights.multiPartQuery : 0) +
      factors.requiresRetrieval * weights.requiresRetrieval;
    
    // Normaliser le score entre 0 et 1
    return Math.min(1, Math.max(0, score));
  }

  /**
   * Classifie le niveau de complexité en fonction du score
   */
  private classifyComplexity(
    score: number,
    thresholds?: { simple: number; complex: number }
  ): ComplexityLevel {
    const simpleThreshold = thresholds?.simple ?? 0.3;
    const complexThreshold = thresholds?.complex ?? 0.7;
    
    if (score < simpleThreshold) return ComplexityLevel.SIMPLE;
    if (score < complexThreshold) return ComplexityLevel.STANDARD;
    return ComplexityLevel.COMPLEX;
  }

  /**
   * Détermine le pipeline recommandé en fonction du niveau de complexité
   */
  private getPipelineRecommendation(level: ComplexityLevel): 'simple' | 'standard' | 'full' {
    switch (level) {
      case ComplexityLevel.SIMPLE: return 'simple';
      case ComplexityLevel.STANDARD: return 'standard';
      case ComplexityLevel.COMPLEX: return 'full';
    }
  }

  /**
   * Met en cache le résultat d'analyse
   */
  private cacheResult(queryHash: string, result: ComplexityAnalysisResult): void {
    // Si le cache atteint sa taille maximale, supprimer l'entrée la plus ancienne
    if (this.complexityCache.size >= this.cacheSize) {
      const oldestKey = this.complexityCache.keys().next().value;
      this.complexityCache.delete(oldestKey);
    }
    
    this.complexityCache.set(queryHash, result);
  }

  /**
   * Génère un hash pour une chaîne de caractères
   */
  private hashString(str: string): string {
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
      const char = str.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash; // Conversion en entier 32 bits
    }
    return hash.toString(36);
  }
} 