import { Inject, Injectable, Optional } from '@nestjs/common';
import { LOGGER_TOKEN } from '../utils/logger-tokens';
import { ILogger } from '../utils/logger-tokens';
import { ApiProviderFactory } from '../apis/api-provider-factory.service';
import { ApiType } from '../types';
import { KnowledgeGraphService, KnowledgeNode, KnowledgeSource, RelationType } from './knowledge-graph.service';
import { EventBusService, RagKagEventType } from './event-bus.service';

export enum VerificationLevel {
  STRICT = 'STRICT',       // Vérification la plus rigoureuse, pour les faits critiques
  STANDARD = 'STANDARD',   // Vérification standard, équilibrée
  RELAXED = 'RELAXED',     // Vérification légère, pour les connaissances moins critiques
}

export enum VerificationMethod {
  API_CONSENSUS = 'API_CONSENSUS',         // Consensus entre plusieurs API
  INTERNAL_DEBATE = 'INTERNAL_DEBATE',     // Débat interne (pilpoul)
  KNOWLEDGE_GRAPH = 'KNOWLEDGE_GRAPH',     // Vérification contre le graphe existant
  EXTERNAL_SOURCE = 'EXTERNAL_SOURCE',     // Vérification par source externe
  TEMPORAL_CONSISTENCY = 'TEMPORAL_CONSISTENCY', // Cohérence temporelle
}

export interface VerificationResult {
  isVerified: boolean;
  confidenceScore: number; // 0-1
  methods: VerificationMethod[];
  apiConsensus?: {
    agreementLevel: number; // 0-1
    responses: Array<{
      provider: string;
      agrees: boolean;
      confidence: number;
      reasoning: string;
    }>;
  };
  internalDebate?: {
    perspectives: Array<{
      position: 'SUPPORT' | 'OPPOSE' | 'NEUTRAL';
      reasoning: string;
      confidence: number;
    }>;
    resolution: string;
    consensusLevel: number; // 0-1
  };
  knowledgeGraphCheck?: {
    compatibilityScore: number; // 0-1
    contradictions: Array<{
      nodeId: string;
      claim: string;
      contradictionExplanation: string;
    }>;
    supportingNodes: string[];
  };
  metadata: {
    verificationTime: number; // en ms
    apiCallCount: number;
    verificationLevel: VerificationLevel;
    // Champs additionnels optionnels
    fromCache?: boolean;
    originalVerificationTime?: number;
    cachedClaimSimilarity?: number;
    quickCheck?: boolean;
    adaptive?: boolean;
    progressiveLevels?: string[];
    error?: string;
    [key: string]: any; // Permet d'ajouter des métadonnées supplémentaires
  };
}

export interface KnowledgeClaim {
  claim: string;
  source: KnowledgeSource;
  domain: string;
  confidence: number;
  metadata?: Record<string, any>;
  temporalContext?: {
    validFrom?: Date;
    validUntil?: Date;
    isTimeSensitive: boolean;
  };
}

interface DebatePerspective {
  role: string;
  stance: 'SUPPORT' | 'OPPOSE' | 'NEUTRAL' | 'CHALLENGE';
  priority: number;
  prompt: string;
}

/**
 * Service responsable de la vérification des connaissances avant leur intégration au graphe
 * Utilise une approche multi-méthodes: consensus API, débat interne, cohérence avec le graphe
 */
@Injectable()
export class KnowledgeVerifierService {
  // Perspectives pour le débat interne (pilpoul)
  private readonly debatePerspectives: DebatePerspective[] = [
    {
      role: 'Défenseur factuel',
      stance: 'SUPPORT',
      priority: 1,
      prompt: 'Analysez cette affirmation en cherchant des preuves qui la soutiennent. Concentrez-vous sur les faits vérifiables, les statistiques, et les sources crédibles qui appuient cette affirmation. Présentez un argument factuel en faveur de cette affirmation.'
    },
    {
      role: 'Critique sceptique',
      stance: 'OPPOSE',
      priority: 1,
      prompt: 'Examinez cette affirmation avec un regard critique. Identifiez les faiblesses potentielles, les manques de preuves, ou les erreurs de raisonnement. Remettez en question les suppositions et présentez des contre-arguments ou des contre-exemples si possible.'
    },
    {
      role: 'Analyste méthodologique',
      stance: 'CHALLENGE',
      priority: 2,
      prompt: 'Évaluez la méthodologie derrière cette affirmation. Est-elle basée sur des méthodes scientifiques valides? Y a-t-il des problèmes statistiques? Les conclusions sont-elles proportionnelles aux preuves? Concentrez-vous sur la rigueur méthodologique plutôt que sur le contenu lui-même.'
    },
    {
      role: 'Intégrateur de contexte',
      stance: 'NEUTRAL',
      priority: 2,
      prompt: 'Placez cette affirmation dans son contexte plus large. Quelles sont les circonstances importantes qui l\'entourent? Quels facteurs historiques, culturels ou situationnels pourraient influencer l\'interprétation de cette affirmation? Ne prenez pas position, mais enrichissez la compréhension contextuelle.'
    },
    {
      role: 'Vérificateur de cohérence',
      stance: 'CHALLENGE',
      priority: 3,
      prompt: 'Vérifiez la cohérence interne de cette affirmation. Est-elle logiquement solide? Contient-elle des contradictions? Est-elle compatible avec d\'autres connaissances établies dans ce domaine? Concentrez-vous uniquement sur la cohérence logique.'
    }
  ];

  // Configurations par niveau de vérification
  private readonly verificationConfigs = {
    [VerificationLevel.STRICT]: {
      minApiConsensus: 0.8,
      minInternalConsensus: 0.7,
      minGraphCompatibility: 0.9,
      requiredMethods: [
        VerificationMethod.API_CONSENSUS,
        VerificationMethod.INTERNAL_DEBATE,
        VerificationMethod.KNOWLEDGE_GRAPH
      ],
      apiProviders: [ApiType.GOOGLE_AI, ApiType.QWEN_AI, ApiType.DEEPSEEK_AI, ApiType.HOUSE_MODEL],
      perspectiveCount: 5,
      confidenceThreshold: 0.85
    },
    [VerificationLevel.STANDARD]: {
      minApiConsensus: 0.7,
      minInternalConsensus: 0.6,
      minGraphCompatibility: 0.7,
      requiredMethods: [
        VerificationMethod.API_CONSENSUS,
        VerificationMethod.INTERNAL_DEBATE
      ],
      apiProviders: [ApiType.GOOGLE_AI, ApiType.HOUSE_MODEL],
      perspectiveCount: 3,
      confidenceThreshold: 0.7
    },
    [VerificationLevel.RELAXED]: {
      minApiConsensus: 0.6,
      minInternalConsensus: 0.5,
      minGraphCompatibility: 0.5,
      requiredMethods: [
        VerificationMethod.INTERNAL_DEBATE
      ],
      apiProviders: [ApiType.HOUSE_MODEL],
      perspectiveCount: 2,
      confidenceThreshold: 0.6
    }
  };

  // Cache de vérifications pour optimiser les performances
  private verificationCache: Map<string, {
    result: VerificationResult;
    claim: string;
    timestamp: number;
    embeddings?: number[];
  }> = new Map();

  // Taille maximale du cache
  private readonly maxCacheSize: number = 1000;
  
  // Durée de validité du cache en ms (24h par défaut)
  private readonly cacheTTL: number = 24 * 60 * 60 * 1000;

  constructor(
    @Inject(LOGGER_TOKEN) private readonly logger: ILogger,
    private readonly apiProviderFactory: ApiProviderFactory,
    @Optional() private readonly knowledgeGraph?: KnowledgeGraphService,
    @Optional() private readonly eventBus?: EventBusService
  ) {}

  /**
   * Vérifie une affirmation avec une approche adaptative qui optimise les performances
   * en utilisant une stratégie progressive (rapide -> approfondie) selon les besoins
   * @param claim L'affirmation à vérifier
   * @param options Options de vérification
   * @returns Résultat détaillé de la vérification
   */
  async verifyClaimAdaptive(
    claim: KnowledgeClaim,
    options: {
      useCache?: boolean;
      cacheSimilarityThreshold?: number;
      forceLevel?: VerificationLevel;
      timeConstraint?: number;
    } = {}
  ): Promise<VerificationResult> {
    const startTime = Date.now();
    
    // Paramètres par défaut
    const useCache = options.useCache !== false;
    const cacheSimilarityThreshold = options.cacheSimilarityThreshold || 0.85;
    const timeConstraint = options.timeConstraint || 10000; // 10s par défaut
    
    this.logger.log(`Démarrage de la vérification adaptative: "${claim.claim.substring(0, 50)}..."`, {
      context: 'KnowledgeVerifierService',
      adaptive: true
    });

    // 1. Vérifier le cache si activé
    if (useCache) {
      const cachedResult = await this.getCachedVerification(claim.claim, cacheSimilarityThreshold);
      if (cachedResult) {
        this.logger.log(`Utilisation du résultat en cache pour une affirmation similaire`, {
          context: 'KnowledgeVerifierService',
          cacheHit: true,
          similarity: cachedResult.similarity
        });
        
        // Adapter les métadonnées pour refléter l'utilisation du cache
        const adaptedResult: VerificationResult = {
          ...cachedResult.result,
          metadata: {
            ...cachedResult.result.metadata,
            fromCache: true,
            originalVerificationTime: cachedResult.result.metadata.verificationTime,
            cachedClaimSimilarity: cachedResult.similarity,
            verificationTime: Date.now() - startTime
          }
        };
        
        return adaptedResult;
      }
    }
    
    // Si un niveau spécifique est forcé, utiliser la vérification standard
    if (options.forceLevel) {
      const result = await this.verifyClaim(claim, options.forceLevel);
      
      // Mettre en cache le résultat
      if (useCache) {
        this.cacheVerificationResult(claim.claim, result);
      }
      
      return result;
    }

    // 2. Vérification rapide pour détecter les cas évidents
    const quickResult = await this.performQuickCheck(claim);
    const processingTime = Date.now() - startTime;
    
    // Si la vérification rapide est très confiante ou si on a une contrainte de temps forte
    if (quickResult.confidenceScore > 0.9 || quickResult.confidenceScore < 0.15 || 
        (processingTime > timeConstraint * 0.3)) {
      
      // Finaliser le résultat rapide
      const finalResult = this.finalizeQuickResult(quickResult, VerificationLevel.RELAXED);
      
      // Mettre en cache si nécessaire
      if (useCache) {
        this.cacheVerificationResult(claim.claim, finalResult);
      }
      
      this.logger.log(`Vérification rapide suffisante (${finalResult.isVerified ? 'VALIDÉE' : 'REJETÉE'}) - score: ${finalResult.confidenceScore.toFixed(2)}`, {
        context: 'KnowledgeVerifierService',
        verificationTime: processingTime
      });
      
      return finalResult;
    }

    // 3. Vérification intermédiaire si le résultat rapide est incertain
    // Utiliser un débat interne limité (2-3 perspectives)
    const perspectives = this.selectPerspectives(3);
    const internalDebate = await this.performInternalDebate(claim, perspectives);
    const intermediateProcessingTime = Date.now() - startTime;
    
    // Si le débat donne un signal clair ou si on approche de la contrainte de temps
    if (Math.abs(internalDebate.consensusLevel - 0.5) > 0.3 || 
        (intermediateProcessingTime > timeConstraint * 0.7)) {
      
      // Combiner les résultats rapides et du débat interne
      const combinedResult: Partial<VerificationResult> = {
        ...quickResult,
        internalDebate,
        methods: [...quickResult.methods, VerificationMethod.INTERNAL_DEBATE]
      };
      
      // Calculer le score de confiance
      const confidenceScore = this.calculateLimitedConfidence(
        quickResult.confidenceScore,
        internalDebate.consensusLevel
      );
      
      // Finaliser le résultat intermédiaire
      const finalResult: VerificationResult = {
        ...combinedResult as any,
        isVerified: confidenceScore >= this.verificationConfigs[VerificationLevel.STANDARD].confidenceThreshold,
        confidenceScore,
        metadata: {
          verificationTime: intermediateProcessingTime,
          apiCallCount: quickResult.metadata.apiCallCount + perspectives.length + 1, // +1 pour la synthèse
          verificationLevel: VerificationLevel.STANDARD,
          adaptive: true
        }
      };
      
      // Mettre en cache si nécessaire
      if (useCache) {
        this.cacheVerificationResult(claim.claim, finalResult);
      }
      
      this.logger.log(`Vérification intermédiaire terminée (${finalResult.isVerified ? 'VALIDÉE' : 'REJETÉE'}) - score: ${finalResult.confidenceScore.toFixed(2)}`, {
        context: 'KnowledgeVerifierService',
        verificationTime: intermediateProcessingTime
      });
      
      return finalResult;
    }

    // 4. Vérification complète pour les cas complexes
    this.logger.log(`Passage à une vérification complète pour une affirmation complexe`, {
      context: 'KnowledgeVerifierService',
      processingTime: intermediateProcessingTime
    });
    
    const fullResult = await this.verifyClaim(claim, VerificationLevel.STRICT);
    
    // Mettre en cache le résultat complet
    if (useCache) {
      this.cacheVerificationResult(claim.claim, fullResult);
    }
    
    return {
      ...fullResult,
      metadata: {
        ...fullResult.metadata,
        adaptive: true,
        progressiveLevels: ['QUICK', 'INTERMEDIATE', 'FULL']
      }
    };
  }
  
  /**
   * Effectue une vérification rapide de l'affirmation avec un seul modèle
   */
  private async performQuickCheck(claim: KnowledgeClaim): Promise<Partial<VerificationResult>> {
    const startTime = Date.now();
    
    try {
      // Utiliser un seul modèle (de préférence local pour la vitesse)
      const provider = ApiType.HOUSE_MODEL;
      const verificationPrompt = this.createVerificationPrompt(claim.claim);
      
      const response = await this.apiProviderFactory.generateResponse(
        provider,
        verificationPrompt,
        { temperature: 0.1 }
      );
      
      const analysis = this.parseVerificationResponse(response.text || response.response);
      
      // Créer un résultat partiel
      const result: Partial<VerificationResult> = {
        methods: [VerificationMethod.API_CONSENSUS],
        confidenceScore: analysis.confidence * (analysis.agrees ? 1 : 0),
        apiConsensus: {
          agreementLevel: analysis.agrees ? 1 : 0,
          responses: [{
            provider: provider.toString(),
            agrees: analysis.agrees,
            confidence: analysis.confidence,
            reasoning: analysis.reasoning
          }]
        },
        metadata: {
          verificationTime: Date.now() - startTime,
          apiCallCount: 1,
          verificationLevel: VerificationLevel.RELAXED
        }
      };
      
      return result;
    } catch (error) {
      this.logger.error(`Échec de la vérification rapide`, {
        context: 'KnowledgeVerifierService',
        error: error.message
      });
      
      // En cas d'erreur, renvoyer un résultat neutre
      return {
        methods: [VerificationMethod.API_CONSENSUS],
        confidenceScore: 0.5,
        apiConsensus: {
          agreementLevel: 0.5,
          responses: []
        },
        metadata: {
          verificationTime: Date.now() - startTime,
          apiCallCount: 0,
          verificationLevel: VerificationLevel.RELAXED,
          error: error.message
        }
      };
    }
  }
  
  /**
   * Finalise un résultat rapide en un résultat complet
   */
  private finalizeQuickResult(
    quickResult: Partial<VerificationResult>,
    level: VerificationLevel
  ): VerificationResult {
    const config = this.verificationConfigs[level];
    
    return {
      ...quickResult as any,
      isVerified: quickResult.confidenceScore >= config.confidenceThreshold,
      methods: quickResult.methods || [],
      metadata: {
        ...quickResult.metadata,
        quickCheck: true
      }
    };
  }
  
  /**
   * Calcule un score de confiance à partir des résultats limités
   */
  private calculateLimitedConfidence(quickScore: number, debateConsensus: number): number {
    // Donner plus de poids au débat interne (60%) qu'à la vérification rapide (40%)
    return (quickScore * 0.4) + (debateConsensus * 0.6);
  }

  /**
   * Recherche une vérification similaire dans le cache
   */
  private async getCachedVerification(
    claim: string,
    similarityThreshold: number = 0.85
  ): Promise<{ result: VerificationResult; similarity: number } | null> {
    // Nettoyer le cache des entrées expirées
    this.cleanCache();
    
    if (this.verificationCache.size === 0) {
      return null;
    }
    
    // Recherche exacte d'abord (optimisation)
    const normalizedClaim = this.normalizeClaimForCache(claim);
    if (this.verificationCache.has(normalizedClaim)) {
      const cached = this.verificationCache.get(normalizedClaim);
      return {
        result: cached.result,
        similarity: 1.0
      };
    }
    
    // Recherche par similarité sémantique
    try {
      // Générer l'embedding pour la requête
      const claimEmbedding = await this.generateEmbedding(claim);
      
      // Trouver l'entrée la plus similaire
      let bestMatch: { entry: any; similarity: number } = { entry: null, similarity: 0 };
      
      for (const [key, entry] of this.verificationCache.entries()) {
        // Générer l'embedding s'il n'existe pas déjà
        if (!entry.embeddings) {
          entry.embeddings = await this.generateEmbedding(entry.claim);
        }
        
        // Calculer la similarité
        const similarity = this.cosineSimilarity(claimEmbedding, entry.embeddings);
        
        if (similarity > bestMatch.similarity) {
          bestMatch = { entry, similarity };
        }
      }
      
      // Retourner le meilleur match si au-dessus du seuil
      if (bestMatch.similarity >= similarityThreshold) {
        return {
          result: bestMatch.entry.result,
          similarity: bestMatch.similarity
        };
      }
    } catch (error) {
      this.logger.error(`Erreur lors de la recherche dans le cache`, {
        context: 'KnowledgeVerifierService',
        error: error.message
      });
    }
    
    return null;
  }
  
  /**
   * Ajoute un résultat de vérification au cache
   */
  private cacheVerificationResult(claim: string, result: VerificationResult): void {
    // Si le cache est plein, supprimer l'entrée la plus ancienne
    if (this.verificationCache.size >= this.maxCacheSize) {
      let oldestKey: string = null;
      let oldestTime = Date.now();
      
      for (const [key, entry] of this.verificationCache.entries()) {
        if (entry.timestamp < oldestTime) {
          oldestTime = entry.timestamp;
          oldestKey = key;
        }
      }
      
      if (oldestKey) {
        this.verificationCache.delete(oldestKey);
      }
    }
    
    // Normaliser la claim pour le cache
    const normalizedClaim = this.normalizeClaimForCache(claim);
    
    // Ajouter au cache
    this.verificationCache.set(normalizedClaim, {
      result,
      claim,
      timestamp: Date.now()
    });
    
    this.logger.log(`Résultat de vérification mis en cache`, {
      context: 'KnowledgeVerifierService',
      claimLength: claim.length
    });
  }
  
  /**
   * Nettoie le cache des entrées expirées
   */
  private cleanCache(): void {
    const now = Date.now();
    let expiredCount = 0;
    
    for (const [key, entry] of this.verificationCache.entries()) {
      if (now - entry.timestamp > this.cacheTTL) {
        this.verificationCache.delete(key);
        expiredCount++;
      }
    }
    
    if (expiredCount > 0) {
      this.logger.debug(`${expiredCount} entrées expirées supprimées du cache`, {
        context: 'KnowledgeVerifierService'
      });
    }
  }
  
  /**
   * Normalise une affirmation pour le cache (supprime les espaces, ponctuation, etc.)
   */
  private normalizeClaimForCache(claim: string): string {
    return claim
      .toLowerCase()
      .replace(/\s+/g, ' ')
      .replace(/[^\w\s]/g, '')
      .trim();
  }
  
  /**
   * Génère un embedding pour une affirmation
   * Note: Cette fonction simule la génération d'embeddings, à remplacer par une vraie implémentation
   */
  private async generateEmbedding(text: string): Promise<number[]> {
    // Simplification: Utiliser le modèle local pour générer un embedding
    // Dans une implémentation réelle, utiliser un modèle d'embedding dédié
    try {
      // Simulation d'embeddings - à remplacer par une vraie implémentation
      const hashCode = (s: string) => {
        let h = 0;
        for (let i = 0; i < s.length; i++) {
          h = Math.imul(31, h) + s.charCodeAt(i) | 0;
        }
        return h;
      };
      
      // Générer un vecteur pseudo-aléatoire déterministe basé sur le contenu
      const words = text.toLowerCase().split(/\W+/).filter(w => w.length > 0);
      const vector: number[] = new Array(20).fill(0);
      
      for (let i = 0; i < words.length; i++) {
        const word = words[i];
        const hash = hashCode(word);
        
        // Répartir l'influence du mot sur plusieurs dimensions
        for (let j = 0; j < 3; j++) {
          const idx = Math.abs((hash + j * 37) % vector.length);
          vector[idx] += (hash % 10) / 10;
        }
      }
      
      // Normaliser le vecteur
      const magnitude = Math.sqrt(vector.reduce((sum, val) => sum + val * val, 0));
      return vector.map(v => magnitude > 0 ? v / magnitude : 0);
    } catch (error) {
      this.logger.error(`Erreur lors de la génération d'embedding`, {
        context: 'KnowledgeVerifierService',
        error: error.message
      });
      
      // Retourner un vecteur par défaut en cas d'erreur
      return new Array(20).fill(0.1);
    }
  }
  
  /**
   * Calcule la similarité cosinus entre deux vecteurs
   */
  private cosineSimilarity(a: number[], b: number[]): number {
    if (a.length !== b.length) {
      throw new Error('Les vecteurs doivent avoir la même dimension');
    }
    
    let dotProduct = 0;
    let normA = 0;
    let normB = 0;
    
    for (let i = 0; i < a.length; i++) {
      dotProduct += a[i] * b[i];
      normA += a[i] * a[i];
      normB += b[i] * b[i];
    }
    
    normA = Math.sqrt(normA);
    normB = Math.sqrt(normB);
    
    if (normA === 0 || normB === 0) {
      return 0;
    }
    
    return dotProduct / (normA * normB);
  }

  /**
   * Vérifie une affirmation de connaissance avant son intégration au graphe
   * @param claim L'affirmation à vérifier
   * @param level Niveau de rigueur de la vérification
   * @returns Résultat détaillé de la vérification
   */
  async verifyClaim(
    claim: KnowledgeClaim,
    level: VerificationLevel = VerificationLevel.STANDARD
  ): Promise<VerificationResult> {
    const startTime = Date.now();
    this.logger.log(`Démarrage de la vérification: "${claim.claim.substring(0, 100)}..."`, {
      context: 'KnowledgeVerifierService',
      level
    });

    const config = this.verificationConfigs[level];
    let apiCallCount = 0;
    const methods: VerificationMethod[] = [];
    const result: Partial<VerificationResult> = {
      metadata: {
        verificationLevel: level,
        verificationTime: 0,
        apiCallCount: 0
      }
    };

    // 1. Vérification par consensus API
    if (config.requiredMethods.includes(VerificationMethod.API_CONSENSUS)) {
      const apiConsensus = await this.performApiConsensusCheck(claim, config.apiProviders);
      result.apiConsensus = apiConsensus;
      methods.push(VerificationMethod.API_CONSENSUS);
      apiCallCount += apiConsensus.responses.length;
    }

    // 2. Débat interne (pilpoul)
    if (config.requiredMethods.includes(VerificationMethod.INTERNAL_DEBATE)) {
      const internalDebate = await this.performInternalDebate(
        claim,
        this.selectPerspectives(config.perspectiveCount)
      );
      result.internalDebate = internalDebate;
      methods.push(VerificationMethod.INTERNAL_DEBATE);
      apiCallCount += config.perspectiveCount;
    }

    // 3. Vérification contre le graphe de connaissances existant
    if (config.requiredMethods.includes(VerificationMethod.KNOWLEDGE_GRAPH) && this.knowledgeGraph) {
      const graphCheck = await this.performKnowledgeGraphCheck(claim);
      result.knowledgeGraphCheck = graphCheck;
      methods.push(VerificationMethod.KNOWLEDGE_GRAPH);
    }

    // Calcul du score de confiance global
    const confidenceScore = this.calculateOverallConfidence(result, config);
    
    // Détermination de la validation finale
    const isVerified = confidenceScore >= config.confidenceThreshold;

    const verificationTime = Date.now() - startTime;
    
    // Assemblage du résultat final
    const finalResult: VerificationResult = {
      ...result as any,
      isVerified,
      confidenceScore,
      methods,
      metadata: {
        verificationTime,
        apiCallCount,
        verificationLevel: level
      }
    };

    this.logger.log(`Vérification terminée: ${isVerified ? 'VALIDÉE' : 'REJETÉE'} (score: ${confidenceScore.toFixed(2)})`, {
      context: 'KnowledgeVerifierService',
      verificationTime,
      confidenceScore
    });

    // Émission d'un événement de vérification
    this.emitVerificationEvent(claim, finalResult);

    return finalResult;
  }

  /**
   * Effectue une vérification auprès de plusieurs API et analyse leur consensus
   */
  private async performApiConsensusCheck(
    claim: KnowledgeClaim,
    providers: ApiType[]
  ): Promise<VerificationResult['apiConsensus']> {
    const verificationPrompt = this.createVerificationPrompt(claim.claim);
    const responses: any[] = [];

    for (const provider of providers) {
      try {
        const response = await this.apiProviderFactory.generateResponse(
          provider,
          verificationPrompt,
          { temperature: 0.2 }
        );
        
        const analysis = this.parseVerificationResponse(response.text || response.response);
        responses.push({
          provider: provider.toString(),
          agrees: analysis.agrees,
          confidence: analysis.confidence,
          reasoning: analysis.reasoning
        });
      } catch (error) {
        this.logger.error(`Échec de la vérification avec ${provider}`, {
          context: 'KnowledgeVerifierService',
          error: error.message
        });
      }
    }

    // Calcul du niveau d'accord
    const agreementLevel = responses.length > 0
      ? responses.filter(r => r.agrees).length / responses.length
      : 0;

    return {
      agreementLevel,
      responses
    };
  }

  /**
   * Crée un prompt de vérification adapté pour les API LLM
   */
  private createVerificationPrompt(claim: string): string {
    return `Je dois vérifier l'exactitude factuelle de l'affirmation suivante:
"${claim}"

Ton rôle est d'analyser cette affirmation pour déterminer:
1. Si l'affirmation est factuelle et véridique
2. Ton niveau de confiance dans cette évaluation (de 0 à 1)
3. Ton raisonnement détaillé

Réponds UNIQUEMENT dans ce format JSON:
{
  "agrees": true/false,
  "confidence": [valeur entre 0 et 1],
  "reasoning": "Ton analyse détaillée ici"
}

Sois rigoureux, objectif et précis dans ton évaluation.`;
  }

  /**
   * Parse la réponse d'une API de vérification
   */
  private parseVerificationResponse(response: string): { agrees: boolean; confidence: number; reasoning: string } {
    try {
      // Essai de parser directement comme JSON
      const jsonMatch = response.match(/\{[\s\S]*\}/);
      if (jsonMatch) {
        const parsed = JSON.parse(jsonMatch[0]);
        if (typeof parsed.agrees === 'boolean' && typeof parsed.confidence === 'number') {
          return {
            agrees: parsed.agrees,
            confidence: Math.max(0, Math.min(1, parsed.confidence)), // Clamp entre 0 et 1
            reasoning: parsed.reasoning || ""
          };
        }
      }
    } catch (e) {
      // Échec du parsing JSON, utiliser analyse heuristique
    }

    // Analyse heuristique si le parsing JSON échoue
    const agrees = response.toLowerCase().includes('vrai') || 
                   response.toLowerCase().includes('correcte') || 
                   response.toLowerCase().includes('exacte');
    const confidenceMatch = response.match(/confiance[^\d]*(\d+(?:\.\d+)?)/i);
    const confidence = confidenceMatch ? parseFloat(confidenceMatch[1]) / 10 : 0.5;

    return {
      agrees,
      confidence: Math.max(0, Math.min(1, confidence)),
      reasoning: response.substring(0, 500) // Limiter la taille
    };
  }

  /**
   * Sélectionne un ensemble de perspectives pour le débat interne
   */
  private selectPerspectives(count: number): DebatePerspective[] {
    // Toujours inclure au moins une perspective SUPPORT et une OPPOSE
    const essential = this.debatePerspectives.filter(p => 
      p.stance === 'SUPPORT' || p.stance === 'OPPOSE'
    ).slice(0, 2);
    
    // Compléter avec d'autres perspectives
    const remaining = this.debatePerspectives
      .filter(p => p.stance !== 'SUPPORT' && p.stance !== 'OPPOSE')
      .sort((a, b) => a.priority - b.priority);
    
    return [...essential, ...remaining].slice(0, count);
  }

  /**
   * Effectue un débat interne (pilpoul) sur l'affirmation
   */
  private async performInternalDebate(
    claim: KnowledgeClaim,
    perspectives: DebatePerspective[]
  ): Promise<VerificationResult['internalDebate']> {
    const debatePerspectives: Array<{
      position: 'SUPPORT' | 'OPPOSE' | 'NEUTRAL';
      reasoning: string;
      confidence: number;
    }> = [];

    for (const perspective of perspectives) {
      const prompt = `${perspective.prompt}

Affirmation à analyser: "${claim.claim}"

${perspective.role}: Analysez cette affirmation selon votre perspective spécifique.
Concentrez-vous sur votre rôle unique et fournissez une analyse détaillée.
Indiquez votre position finale (SUPPORT/OPPOSE/NEUTRAL) et votre niveau de confiance de 0 à 1.

Répondez dans ce format:
Position: [SUPPORT/OPPOSE/NEUTRAL]
Confiance: [valeur entre 0 et 1]
Analyse:
[Votre analyse détaillée ici]`;

      try {
        const response = await this.apiProviderFactory.generateResponse(
          ApiType.HOUSE_MODEL,
          prompt,
          { temperature: 0.3 }
        );

        const content = response.text || response.response;
        const positionMatch = content.match(/Position:\s*(SUPPORT|OPPOSE|NEUTRAL)/i);
        const confidenceMatch = content.match(/Confiance:\s*(\d+(?:\.\d+)?)/i);
        const analysisMatch = content.match(/Analyse:\s*([\s\S]+)/i);

        debatePerspectives.push({
          position: (positionMatch?.[1] || this.inferPosition(content)) as any,
          reasoning: analysisMatch?.[1] || content,
          confidence: confidenceMatch ? parseFloat(confidenceMatch[1]) : 0.5
        });
      } catch (error) {
        this.logger.error(`Échec de l'analyse de perspective: ${perspective.role}`, {
          context: 'KnowledgeVerifierService',
          error: error.message
        });
      }
    }

    // Synthèse du débat
    const debateResolutionPrompt = `J'ai organisé un débat interne sur l'affirmation suivante:
"${claim.claim}"

Différentes perspectives ont été exprimées:
${debatePerspectives.map((p, i) => `
Perspective ${i+1} (${p.position}, confiance: ${p.confidence}):
${p.reasoning.substring(0, 300)}...`).join('\n')}

Agis comme un modérateur impartial qui doit synthétiser ce débat.
1. Résume les principaux points de chaque côté
2. Identifie les zones de consensus et de désaccord
3. Évalue le niveau de consensus global (de 0 à 1)
4. Fournis une résolution finale

Format de réponse:
{
  "resolution": "Résumé de la résolution finale",
  "consensusLevel": [valeur entre 0 et 1]
}`;

    let resolution = "Impossible de déterminer une résolution finale";
    let consensusLevel = 0.5;

    try {
      const resolutionResponse = await this.apiProviderFactory.generateResponse(
        ApiType.HOUSE_MODEL,
        debateResolutionPrompt,
        { temperature: 0.1 }
      );

      const content = resolutionResponse.text || resolutionResponse.response;
      
      // Tenter de parser comme JSON
      try {
        const jsonMatch = content.match(/\{[\s\S]*\}/);
        if (jsonMatch) {
          const parsed = JSON.parse(jsonMatch[0]);
          resolution = parsed.resolution || resolution;
          consensusLevel = typeof parsed.consensusLevel === 'number' 
            ? parsed.consensusLevel 
            : this.calculateConsensusLevel(debatePerspectives);
        }
      } catch (e) {
        // Si le parsing échoue, extraire manuellement
        const resolutionMatch = content.match(/résolution:?\s*([^]*?)(?:\n\n|\n?consensusLevel|$)/i);
        resolution = resolutionMatch?.[1]?.trim() || resolution;
        
        const consensusMatch = content.match(/consensus[^:]*:\s*(\d+(?:\.\d+)?)/i);
        consensusLevel = consensusMatch 
          ? parseFloat(consensusMatch[1]) 
          : this.calculateConsensusLevel(debatePerspectives);
      }
    } catch (error) {
      this.logger.error(`Échec de la synthèse du débat interne`, {
        context: 'KnowledgeVerifierService',
        error: error.message
      });
      
      // Calcul de secours du consensus
      consensusLevel = this.calculateConsensusLevel(debatePerspectives);
    }

    return {
      perspectives: debatePerspectives,
      resolution,
      consensusLevel
    };
  }

  /**
   * Infère la position à partir du contenu textuel
   */
  private inferPosition(content: string): 'SUPPORT' | 'OPPOSE' | 'NEUTRAL' {
    const lowerContent = content.toLowerCase();
    
    // Mots-clés indiquant le support
    const supportKeywords = ['vrai', 'correct', 'exact', 'prouvé', 'validé', 'confirmé', 'soutiens'];
    const supportScore = supportKeywords.reduce(
      (score, keyword) => score + (lowerContent.includes(keyword) ? 1 : 0), 0
    );
    
    // Mots-clés indiquant l'opposition
    const opposeKeywords = ['faux', 'incorrect', 'inexact', 'erroné', 'trompeur', 'contredit', 'réfute'];
    const opposeScore = opposeKeywords.reduce(
      (score, keyword) => score + (lowerContent.includes(keyword) ? 1 : 0), 0
    );
    
    // Mots-clés indiquant la neutralité
    const neutralKeywords = ['incertain', 'mitigé', 'nuancé', 'complexe', 'contexte', 'dépend'];
    const neutralScore = neutralKeywords.reduce(
      (score, keyword) => score + (lowerContent.includes(keyword) ? 1 : 0), 0
    );
    
    // Déterminer la position en fonction des scores
    if (supportScore > opposeScore && supportScore > neutralScore) {
      return 'SUPPORT';
    } else if (opposeScore > supportScore && opposeScore > neutralScore) {
      return 'OPPOSE';
    } else {
      return 'NEUTRAL';
    }
  }

  /**
   * Calcule le niveau de consensus dans le débat interne
   */
  private calculateConsensusLevel(perspectives: Array<{
    position: 'SUPPORT' | 'OPPOSE' | 'NEUTRAL';
    confidence: number;
  }>): number {
    if (perspectives.length === 0) return 0.5;
    
    // Compter les perspectives par position
    const counts = {
      SUPPORT: 0,
      OPPOSE: 0,
      NEUTRAL: 0
    };
    
    // Somme pondérée par la confiance
    const weightedCounts = {
      SUPPORT: 0,
      OPPOSE: 0,
      NEUTRAL: 0
    };
    
    let totalWeight = 0;
    
    for (const p of perspectives) {
      counts[p.position]++;
      weightedCounts[p.position] += p.confidence;
      totalWeight += p.confidence;
    }
    
    // Si tout le monde est d'accord, consensus parfait
    if (counts.SUPPORT === perspectives.length || 
        counts.OPPOSE === perspectives.length || 
        counts.NEUTRAL === perspectives.length) {
      return 1;
    }
    
    // Si c'est un mélange de positions, calculer le consensus basé sur la dominance
    const dominantPosition = Object.entries(weightedCounts)
      .sort((a, b) => b[1] - a[1])[0][0];
    
    // Pourcentage pondéré de la position dominante
    const consensusLevel = totalWeight > 0 
      ? weightedCounts[dominantPosition] / totalWeight 
      : 0.5;
    
    // Bonus si toutes les positions sont soit SUPPORT soit OPPOSE (pas de NEUTRAL)
    const hasClarity = counts.NEUTRAL === 0;
    
    return hasClarity 
      ? Math.min(1, consensusLevel * 1.2) // Bonus de 20% pour la clarté
      : consensusLevel;
  }

  /**
   * Vérifie la cohérence de l'affirmation avec le graphe de connaissances existant
   */
  private async performKnowledgeGraphCheck(claim: KnowledgeClaim): Promise<VerificationResult['knowledgeGraphCheck']> {
    if (!this.knowledgeGraph) {
      return {
        compatibilityScore: 0.5,
        contradictions: [],
        supportingNodes: []
      };
    }

    // Recherche des nœuds pertinents pour cette affirmation
    const searchResults = this.knowledgeGraph.search(claim.claim, {
      maxDepth: 2,
      minConfidence: 0.5,
      sortByRelevance: true,
      maxResults: 20
    });

    // Analyse des contradictions potentielles
    const contradictions: Array<{
      nodeId: string;
      claim: string;
      contradictionExplanation: string;
    }> = [];

    const supportingNodes: string[] = [];

    let compatibilityScore = 0.5; // Score neutre par défaut

    if (searchResults.nodes.length > 0) {
      const relevantNodes = searchResults.nodes;
      
      // Vérifier les contradictions avec chaque nœud pertinent
      for (const node of relevantNodes) {
        const isContradictory = await this.checkContradiction(claim.claim, node);
        
        if (isContradictory.contradicts) {
          contradictions.push({
            nodeId: node.id,
            claim: node.content,
            contradictionExplanation: isContradictory.explanation
          });
        } else if (isContradictory.supports) {
          supportingNodes.push(node.id);
        }
      }
      
      // Calculer le score de compatibilité
      const totalNodes = relevantNodes.length;
      const contradictionRatio = contradictions.length / totalNodes;
      const supportRatio = supportingNodes.length / totalNodes;
      
      // Formule: plus de support et moins de contradictions = meilleur score
      compatibilityScore = 0.5 + (supportRatio * 0.5) - (contradictionRatio * 0.5);
    }

    return {
      compatibilityScore,
      contradictions,
      supportingNodes
    };
  }

  /**
   * Vérifie si deux affirmations sont contradictoires
   */
  private async checkContradiction(
    claim: string, 
    node: KnowledgeNode
  ): Promise<{ contradicts: boolean; supports: boolean; explanation: string }> {
    const contradictionPrompt = `J'ai deux affirmations et je dois déterminer si elles sont contradictoires ou compatibles:

Affirmation 1: "${claim}"
Affirmation 2: "${node.content}"

Analyse ces deux affirmations et détermine:
1. Si elles se contredisent directement
2. Si elles se soutiennent mutuellement
3. Si elles sont simplement indépendantes

Réponds UNIQUEMENT au format JSON:
{
  "contradicts": true/false,
  "supports": true/false,
  "explanation": "Ton explication détaillée ici"
}`;

    try {
      const response = await this.apiProviderFactory.generateResponse(
        ApiType.HOUSE_MODEL,
        contradictionPrompt,
        { temperature: 0.1 }
      );

      const content = response.text || response.response;
      
      // Essai de parser comme JSON
      try {
        const jsonMatch = content.match(/\{[\s\S]*\}/);
        if (jsonMatch) {
          const parsed = JSON.parse(jsonMatch[0]);
          return {
            contradicts: Boolean(parsed.contradicts),
            supports: Boolean(parsed.supports),
            explanation: parsed.explanation || ""
          };
        }
      } catch (e) {
        // Parsing JSON échoué
      }
      
      // Analyse heuristique si le parsing échoue
      return {
        contradicts: content.toLowerCase().includes('contradict') || content.toLowerCase().includes('contradictoires'),
        supports: content.toLowerCase().includes('support') || content.toLowerCase().includes('soutien'),
        explanation: content.substring(0, 300)
      };
    } catch (error) {
      this.logger.error(`Échec de l'analyse de contradiction`, {
        context: 'KnowledgeVerifierService',
        error: error.message
      });
      
      // Résultat par défaut en cas d'erreur
      return {
        contradicts: false,
        supports: false,
        explanation: "Impossible d'analyser la relation entre ces affirmations due à une erreur."
      };
    }
  }

  /**
   * Calcule le score de confiance global à partir des différentes vérifications
   */
  private calculateOverallConfidence(
    result: Partial<VerificationResult>,
    config: typeof this.verificationConfigs[VerificationLevel]
  ): number {
    const scores: number[] = [];
    const weights: number[] = [];
    
    // API Consensus
    if (result.apiConsensus) {
      const apiScore = result.apiConsensus.agreementLevel;
      scores.push(apiScore);
      weights.push(0.4); // 40% du score total
    }
    
    // Débat interne
    if (result.internalDebate) {
      const debateScore = result.internalDebate.consensusLevel;
      scores.push(debateScore);
      weights.push(0.4); // 40% du score total
    }
    
    // Graphe de connaissances
    if (result.knowledgeGraphCheck) {
      const graphScore = result.knowledgeGraphCheck.compatibilityScore;
      scores.push(graphScore);
      weights.push(0.2); // 20% du score total
    }
    
    // S'il n'y a aucun score, renvoyer 0
    if (scores.length === 0) return 0;
    
    // Calcul de la moyenne pondérée
    const totalWeight = weights.reduce((sum, w) => sum + w, 0);
    const weightedSum = scores.reduce((sum, score, i) => sum + (score * weights[i]), 0);
    
    return totalWeight > 0 ? weightedSum / totalWeight : 0;
  }

  /**
   * Émet un événement sur le résultat de la vérification
   */
  private emitVerificationEvent(claim: KnowledgeClaim, result: VerificationResult): void {
    if (!this.eventBus) return;
    
    this.eventBus.emit({
      type: RagKagEventType.KNOWLEDGE_VERIFICATION_COMPLETED,
      source: 'KnowledgeVerifierService',
      payload: {
        claim,
        verificationResult: result,
        timestamp: Date.now()
      }
    });
  }

  /**
   * Ajoute l'affirmation vérifiée au graphe de connaissances
   */
  async addVerifiedClaimToGraph(
    claim: KnowledgeClaim,
    verificationResult: VerificationResult
  ): Promise<string | null> {
    if (!this.knowledgeGraph || !verificationResult.isVerified) {
      return null;
    }
    
    try {
      // Création du nœud de connaissance
      const nodeId = this.knowledgeGraph.addNode({
        label: claim.domain,
        type: 'VERIFIED_KNOWLEDGE',
        content: claim.claim,
        confidence: verificationResult.confidenceScore,
        source: claim.source,
        metadata: {
          ...claim.metadata,
          verification: {
            methods: verificationResult.methods,
            apiConsensus: verificationResult.apiConsensus?.agreementLevel,
            internalDebateConsensus: verificationResult.internalDebate?.consensusLevel,
            graphCompatibility: verificationResult.knowledgeGraphCheck?.compatibilityScore,
            verificationLevel: verificationResult.metadata.verificationLevel,
            verificationTime: verificationResult.metadata.verificationTime
          },
          temporalContext: claim.temporalContext
        }
      });
      
      // Création des liens avec les nœuds supportant cette connaissance
      if (verificationResult.knowledgeGraphCheck?.supportingNodes) {
        for (const supportingNodeId of verificationResult.knowledgeGraphCheck.supportingNodes) {
          this.knowledgeGraph.addEdge({
            sourceId: nodeId,
            targetId: supportingNodeId,
            type: RelationType.SUPPORTED_BY,
            weight: 0.8,
            bidirectional: false,
            confidence: verificationResult.confidenceScore
          });
        }
      }
      
      this.logger.log(`Connaissance vérifiée ajoutée au graphe avec ID: ${nodeId}`, {
        context: 'KnowledgeVerifierService'
      });
      
      return nodeId;
    } catch (error) {
      this.logger.error(`Échec de l'ajout de la connaissance vérifiée au graphe`, {
        context: 'KnowledgeVerifierService',
        error: error.message
      });
      
      return null;
    }
  }

  /**
   * Place l'affirmation en quarantaine pour une vérification ultérieure
   */
  async quarantineClaim(
    claim: KnowledgeClaim,
    reason: string,
    expirationDays: number = 7
  ): Promise<string | null> {
    if (!this.knowledgeGraph) {
      return null;
    }
    
    try {
      // Création d'un nœud de quarantaine
      const nodeId = this.knowledgeGraph.addNode({
        label: 'QUARANTINE',
        type: 'UNVERIFIED_KNOWLEDGE',
        content: claim.claim,
        confidence: claim.confidence * 0.5, // Réduire la confiance
        source: claim.source,
        metadata: {
          ...claim.metadata,
          quarantine: {
            reason,
            timestamp: Date.now(),
            expirationTimestamp: Date.now() + (expirationDays * 24 * 60 * 60 * 1000),
            domain: claim.domain
          },
          temporalContext: claim.temporalContext
        }
      });
      
      this.logger.log(`Connaissance mise en quarantaine avec ID: ${nodeId}`, {
        context: 'KnowledgeVerifierService',
        reason
      });
      
      return nodeId;
    } catch (error) {
      this.logger.error(`Échec de la mise en quarantaine`, {
        context: 'KnowledgeVerifierService',
        error: error.message
      });
      
      return null;
    }
  }
} 