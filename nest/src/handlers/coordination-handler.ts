import { v4 as uuidv4 } from 'uuid';
import { Logger } from '../utils/logger';

/**
 * Types d'exécution possibles pour le système de coordination
 */
export enum ExecutionMode {
  SEQUENTIAL = 'SEQUENTIAL',   // Exécution séquentielle des composants
  PARALLEL = 'PARALLEL',       // Exécution parallèle maximale
  ADAPTIVE = 'ADAPTIVE'        // Mode adaptatif basé sur la charge et complexité
}

/**
 * Niveaux de priorité pour les composants du système
 */
export enum ComponentPriority {
  CRITICAL = 'CRITICAL',     // Composants essentiels au fonctionnement
  HIGH = 'HIGH',             // Composants à haute priorité
  NORMAL = 'NORMAL',         // Priorité standard
  LOW = 'LOW'                // Composants non-critiques
}

/**
 * État d'un composant dans le pipeline de coordination
 */
export enum ComponentStatus {
  IDLE = 'IDLE',             // Pas encore démarré
  PENDING = 'PENDING',       // En attente de dépendances
  RUNNING = 'RUNNING',       // En cours d'exécution
  COMPLETED = 'COMPLETED',   // Exécution terminée avec succès
  FAILED = 'FAILED',         // Échec d'exécution
  SKIPPED = 'SKIPPED'        // Exécution ignorée
}

/**
 * Interface pour les options d'exécution de la coordination
 */
export interface ExecutionOptions {
  mode: ExecutionMode;                // Mode d'exécution à utiliser
  timeout?: number;                   // Timeout global en millisecondes
  maxRetries?: number;                // Nombre maximum de tentatives en cas d'échec
  backoffFactor?: number;             // Facteur pour le backoff exponentiel
  abortOnComponentFailure?: boolean;  // Arrêter tout en cas d'échec d'un composant
  useCircuitBreaker?: boolean;        // Activer les circuit breakers
  includeMetrics?: boolean;           // Collecter les métriques détaillées
  resourceConstraints?: {             // Contraintes de ressources
    maxParallelExecutions?: number;   // Nombre maximum d'exécutions parallèles
    maxMemoryUsage?: number;          // Utilisation mémoire maximale (MB)
  };
}

/**
 * Interface pour un composant exécutable dans le système de coordination
 */
export interface CoordinationComponent {
  id: string;                         // Identifiant unique du composant
  name: string;                       // Nom lisible du composant
  description?: string;               // Description du rôle du composant
  execute: (context: CoordinationContext) => Promise<any>; // Fonction d'exécution
  dependencies?: string[];            // IDs des composants dont celui-ci dépend
  priority: ComponentPriority;        // Priorité d'exécution du composant
  timeout?: number;                   // Timeout spécifique à ce composant
  retryable?: boolean;                // Si le composant peut être réessayé en cas d'échec
  status: ComponentStatus;            // État actuel du composant
  result?: any;                       // Résultat de l'exécution si disponible
  error?: Error;                      // Erreur si l'exécution a échoué
  startTime?: number;                 // Timestamp de début d'exécution
  endTime?: number;                   // Timestamp de fin d'exécution
}

/**
 * Interface pour le contexte d'exécution de la coordination
 */
export interface CoordinationContext {
  traceId: string;                    // ID unique pour tracer l'exécution complète
  requestId: string;                  // ID de la requête utilisateur
  query: string;                      // Requête utilisateur originale
  userContext?: any;                  // Contexte spécifique à l'utilisateur
  startTime: number;                  // Timestamp de début d'exécution
  components: Map<string, CoordinationComponent>; // Composants disponibles
  results: Map<string, any>;          // Résultats intermédiaires et finaux
  options: ExecutionOptions;          // Options d'exécution
  metadata: Map<string, any>;         // Métadonnées additionnelles
  systemLoad?: number;                // Charge système actuelle (0-1)
  complexityScore?: number;           // Score de complexité de la requête (0-1)
  logger: Logger;                     // Logger pour cette exécution
}

/**
 * Interface pour les métriques d'exécution
 */
export interface ExecutionMetrics {
  totalDuration: number;               // Durée totale de l'exécution en ms
  componentMetrics: {                  // Métriques par composant
    [componentId: string]: {
      duration: number;                // Durée d'exécution
      status: ComponentStatus;         // Statut final
      retries: number;                 // Nombre de tentatives
    }
  };
  resourceUtilization: {               // Utilisation des ressources
    peakMemoryUsage: number;           // Pic d'utilisation mémoire
    averageCpuUsage: number;           // Utilisation CPU moyenne
  };
  bottlenecks: string[];               // Composants identifiés comme goulots
  optimizationSuggestions: string[];   // Suggestions d'optimisation
}

/**
 * Résultat final d'une exécution de coordination
 */
export interface CoordinationResult {
  success: boolean;                    // Si l'exécution a réussi
  traceId: string;                     // ID de trace pour l'exécution
  requestId: string;                   // ID de la requête
  startTime: number;                   // Timestamp de début
  endTime: number;                     // Timestamp de fin
  duration: number;                    // Durée totale en ms
  result: any;                         // Résultat final agrégé
  componentResults: Map<string, any>;  // Résultats par composant
  failedComponents: string[];          // IDs des composants en échec
  metrics?: ExecutionMetrics;          // Métriques détaillées si activées
  executionPath: string[];             // Séquence d'exécution réelle
}

/**
 * Gestionnaire principal de coordination pour le système RAG/KAG
 */
export class CoordinationHandler {
  private logger: Logger;
  private defaultOptions: ExecutionOptions;

  /**
   * Crée une nouvelle instance du gestionnaire de coordination
   * @param logger Instance du logger à utiliser
   */
  constructor(logger: Logger) {
    this.logger = logger;
    
    // Configuration par défaut
    this.defaultOptions = {
      mode: ExecutionMode.ADAPTIVE,
      timeout: 30000,
      maxRetries: 2,
      backoffFactor: 1.5,
      abortOnComponentFailure: false,
      useCircuitBreaker: true,
      includeMetrics: true,
      resourceConstraints: {
        maxParallelExecutions: 5,
        maxMemoryUsage: 1024
      }
    };
  }

  /**
   * Exécute un workflow de coordination complet
   * @param query Requête utilisateur
   * @param components Liste des composants à exécuter
   * @param userContext Contexte utilisateur
   * @param options Options d'exécution personnalisées
   * @returns Résultat de l'exécution du workflow
   */
  public async execute(
    query: string,
    components: CoordinationComponent[],
    userContext?: any,
    options?: Partial<ExecutionOptions>
  ): Promise<CoordinationResult> {
    const startTime = Date.now();
    const traceId = uuidv4();
    const requestId = uuidv4();
    
    // Fusionner les options par défaut avec les options personnalisées
    const mergedOptions: ExecutionOptions = {
      ...this.defaultOptions,
      ...options
    };

    this.logger.info(`[Coordination:${traceId}] Démarrage exécution, mode=${mergedOptions.mode}, composants=${components.length}`);

    // Initialiser le contexte d'exécution
    const context: CoordinationContext = {
      traceId,
      requestId,
      query,
      userContext,
      startTime,
      components: new Map(),
      results: new Map(),
      options: mergedOptions,
      metadata: new Map(),
      logger: this.logger,
    };

    // Enregistrer les composants dans le contexte
    components.forEach(component => {
      context.components.set(component.id, {
        ...component,
        status: ComponentStatus.IDLE
      });
    });

    // Évaluer la complexité de la requête
    context.complexityScore = this.evaluateComplexity(query, components);
    
    // Simuler la charge système (à remplacer par une mesure réelle)
    context.systemLoad = Math.random() * 0.5; // 0-0.5 pour simuler
    
    this.logger.debug(`[Coordination:${traceId}] Complexité: ${context.complexityScore}, Charge: ${context.systemLoad}`);

    // Déterminer le mode d'exécution optimal si en mode adaptatif
    if (mergedOptions.mode === ExecutionMode.ADAPTIVE) {
      const mode = this.determineOptimalExecutionMode(context);
      context.options.mode = mode;
      this.logger.debug(`[Coordination:${traceId}] Mode adaptatif sélectionné: ${mode}`);
    }

    let result: CoordinationResult;
    
    try {
      // Exécuter selon le mode choisi
      if (context.options.mode === ExecutionMode.PARALLEL) {
        result = await this.executeParallel(context);
      } else {
        result = await this.executeSequential(context);
      }
      
      this.logger.info(`[Coordination:${traceId}] Exécution terminée avec succès en ${result.duration}ms`);
      return result;
      
    } catch (error) {
      const endTime = Date.now();
      this.logger.error(`[Coordination:${traceId}] Échec de l'exécution: ${error.message}`);
      
      // Construire un résultat d'échec
      const failedComponents = Array.from(context.components.values())
        .filter(c => c.status === ComponentStatus.FAILED)
        .map(c => c.id);
      
      return {
        success: false,
        traceId,
        requestId,
        startTime,
        endTime,
        duration: endTime - startTime,
        result: null,
        componentResults: context.results,
        failedComponents,
        executionPath: this.extractExecutionPath(context)
      };
    }
  }

  /**
   * Exécute les composants en mode séquentiel
   * @param context Contexte d'exécution
   * @returns Résultat de l'exécution
   */
  private async executeSequential(context: CoordinationContext): Promise<CoordinationResult> {
    const { components, logger, traceId } = context;
    const executionPath: string[] = [];
    const startTime = context.startTime;
    
    logger.debug(`[Coordination:${traceId}] Démarrage exécution séquentielle`);
    
    // Déterminer l'ordre d'exécution basé sur les dépendances et priorités
    const executionOrder = this.calculateExecutionOrder(Array.from(components.values()));
    
    // Exécuter les composants dans l'ordre déterminé
    for (const componentId of executionOrder) {
      const component = components.get(componentId);
      
      // Vérifier si toutes les dépendances sont satisfaites
      if (component.dependencies && component.dependencies.length > 0) {
        const unsatisfiedDeps = component.dependencies.filter(depId => {
          const dep = components.get(depId);
          return !dep || dep.status !== ComponentStatus.COMPLETED;
        });
        
        if (unsatisfiedDeps.length > 0) {
          logger.warn(`[Coordination:${traceId}] Composant ${componentId} ignoré: dépendances non satisfaites: ${unsatisfiedDeps.join(', ')}`);
          component.status = ComponentStatus.SKIPPED;
          continue;
        }
      }
      
      // Exécuter le composant
      logger.debug(`[Coordination:${traceId}] Exécution composant: ${component.name} (${componentId})`);
      executionPath.push(componentId);
      
      component.status = ComponentStatus.RUNNING;
      component.startTime = Date.now();
      
      try {
        // Exécution avec gestion du timeout
        const timeoutValue = component.timeout || context.options.timeout;
        let result;
        
        if (timeoutValue) {
          result = await Promise.race([
            component.execute(context),
            new Promise((_, reject) => 
              setTimeout(() => reject(new Error(`Timeout after ${timeoutValue}ms`)), timeoutValue)
            )
          ]);
        } else {
          result = await component.execute(context);
        }
        
        // Enregistrer le succès
        component.status = ComponentStatus.COMPLETED;
        component.endTime = Date.now();
        component.result = result;
        context.results.set(componentId, result);
        
        logger.debug(`[Coordination:${traceId}] Composant ${componentId} terminé en ${component.endTime - component.startTime}ms`);
        
      } catch (error) {
        // Gérer l'échec
        component.status = ComponentStatus.FAILED;
        component.endTime = Date.now();
        component.error = error;
        
        logger.error(`[Coordination:${traceId}] Échec composant ${componentId}: ${error.message}`);
        
        // Gérer les échecs selon la configuration
        if (context.options.abortOnComponentFailure) {
          throw new Error(`Exécution avortée suite à l'échec du composant ${componentId}: ${error.message}`);
        }
      }
    }
    
    // Finaliser et assembler le résultat
    const endTime = Date.now();
    const duration = endTime - startTime;
    
    const finalResult = this.aggregateResults(context);
    const failedComponents = Array.from(components.values())
      .filter(c => c.status === ComponentStatus.FAILED)
      .map(c => c.id);
    
    // Collecter les métriques si activé
    let metrics: ExecutionMetrics | undefined;
    if (context.options.includeMetrics) {
      metrics = this.collectMetrics(context, duration);
    }
    
    return {
      success: failedComponents.length === 0,
      traceId: context.traceId,
      requestId: context.requestId,
      startTime,
      endTime,
      duration,
      result: finalResult,
      componentResults: context.results,
      failedComponents,
      metrics,
      executionPath
    };
  }
  
  /**
   * Exécute les composants en mode parallèle
   * @param context Contexte d'exécution
   * @returns Résultat de l'exécution
   */
  private async executeParallel(context: CoordinationContext): Promise<CoordinationResult> {
    const { components, logger, traceId } = context;
    const executionPath: string[] = [];
    const startTime = context.startTime;
    
    logger.debug(`[Coordination:${traceId}] Démarrage exécution parallèle`);
    
    // Créer des niveaux d'exécution basés sur les dépendances
    const executionLevels = this.calculateExecutionLevels(Array.from(components.values()));
    
    // Exécuter chaque niveau en parallèle
    for (let levelIndex = 0; levelIndex < executionLevels.length; levelIndex++) {
      const level = executionLevels[levelIndex];
      const levelComponents = level.map(id => components.get(id));
      
      logger.debug(`[Coordination:${traceId}] Exécution niveau ${levelIndex+1}/${executionLevels.length} avec ${level.length} composants`);
      
      // Préparer les promesses pour tous les composants de ce niveau
      const executionPromises = levelComponents.map(async component => {
        // Vérifier si toutes les dépendances sont satisfaites
        if (component.dependencies && component.dependencies.length > 0) {
          const unsatisfiedDeps = component.dependencies.filter(depId => {
            const dep = components.get(depId);
            return !dep || dep.status !== ComponentStatus.COMPLETED;
          });
          
          if (unsatisfiedDeps.length > 0) {
            logger.warn(`[Coordination:${traceId}] Composant ${component.id} ignoré: dépendances non satisfaites`);
            component.status = ComponentStatus.SKIPPED;
            return;
          }
        }
        
        // Exécuter le composant
        executionPath.push(component.id);
        component.status = ComponentStatus.RUNNING;
        component.startTime = Date.now();
        
        try {
          // Exécution avec gestion du timeout
          const timeoutValue = component.timeout || context.options.timeout;
          let result;
          
          if (timeoutValue) {
            result = await Promise.race([
              component.execute(context),
              new Promise((_, reject) => 
                setTimeout(() => reject(new Error(`Timeout after ${timeoutValue}ms`)), timeoutValue)
              )
            ]);
          } else {
            result = await component.execute(context);
          }
          
          // Enregistrer le succès
          component.status = ComponentStatus.COMPLETED;
          component.endTime = Date.now();
          component.result = result;
          context.results.set(component.id, result);
          
          logger.debug(`[Coordination:${traceId}] Composant ${component.id} terminé en ${component.endTime - component.startTime}ms`);
          
        } catch (error) {
          // Gérer l'échec
          component.status = ComponentStatus.FAILED;
          component.endTime = Date.now();
          component.error = error;
          
          logger.error(`[Coordination:${traceId}] Échec composant ${component.id}: ${error.message}`);
          
          // Lever une exception si configuré pour aborter
          if (context.options.abortOnComponentFailure) {
            throw error;
          }
        }
      });
      
      // Attendre que tous les composants du niveau soient terminés
      try {
        await Promise.all(executionPromises);
      } catch (error) {
        if (context.options.abortOnComponentFailure) {
          throw new Error(`Exécution avortée au niveau ${levelIndex+1}: ${error.message}`);
        }
      }
    }
    
    // Finaliser et assembler le résultat
    const endTime = Date.now();
    const duration = endTime - startTime;
    
    const finalResult = this.aggregateResults(context);
    const failedComponents = Array.from(components.values())
      .filter(c => c.status === ComponentStatus.FAILED)
      .map(c => c.id);
    
    // Collecter les métriques si activé
    let metrics: ExecutionMetrics | undefined;
    if (context.options.includeMetrics) {
      metrics = this.collectMetrics(context, duration);
    }
    
    return {
      success: failedComponents.length === 0,
      traceId: context.traceId,
      requestId: context.requestId,
      startTime,
      endTime,
      duration,
      result: finalResult,
      componentResults: context.results,
      failedComponents,
      metrics,
      executionPath
    };
  }

  /**
   * Agrège les résultats de tous les composants
   * @param context Contexte d'exécution
   * @returns Résultat final agrégé
   */
  private aggregateResults(context: CoordinationContext): any {
    // Cette implémentation dépendra fortement de la structure des données
    // Version simple qui combine les résultats des composants complétés
    const aggregatedResult: any = {
      query: context.query,
      processedBy: [],
      data: {},
      metadata: {
        executionTime: Date.now() - context.startTime,
        traceId: context.traceId
      }
    };
    
    // Récupérer les résultats de tous les composants terminés
    Array.from(context.components.values())
      .filter(c => c.status === ComponentStatus.COMPLETED)
      .forEach(component => {
        const result = component.result;
        
        aggregatedResult.processedBy.push({
          id: component.id,
          name: component.name,
          executionTime: component.endTime - component.startTime
        });
        
        // Intégrer le résultat dans la structure agrégée
        if (result && typeof result === 'object') {
          // Si le résultat a une propriété 'data', la fusionner
          if (result.data) {
            aggregatedResult.data = {
              ...aggregatedResult.data,
              [component.id]: result.data
            };
          } else {
            // Sinon, ajouter tout le résultat
            aggregatedResult.data[component.id] = result;
          }
          
          // Fusionner les métadonnées si présentes
          if (result.metadata) {
            aggregatedResult.metadata[component.id] = result.metadata;
          }
        } else if (result !== undefined) {
          // Pour les résultats simples (non-objets)
          aggregatedResult.data[component.id] = result;
        }
      });
      
    return aggregatedResult;
  }
  
  /**
   * Évalue la complexité d'une requête en fonction de divers facteurs
   * @param query Requête à évaluer
   * @param components Composants disponibles
   * @returns Score de complexité (0-1)
   */
  private evaluateComplexity(query: string, components: CoordinationComponent[]): number {
    // Facteurs de complexité basiques
    const length = Math.min(query.length / 200, 1); // Normaliser la longueur
    const wordCount = query.split(/\s+/).length;
    const wordCountNormalized = Math.min(wordCount / 30, 1);
    
    // Complexité basée sur le nombre de composants nécessaires
    const componentComplexity = Math.min(components.length / 10, 1);
    
    // Complexité basée sur les dépendances entre composants
    let dependencyComplexity = 0;
    if (components.length > 0) {
      const totalPossibleDeps = components.length * (components.length - 1);
      let actualDeps = 0;
      
      components.forEach(comp => {
        if (comp.dependencies) {
          actualDeps += comp.dependencies.length;
        }
      });
      
      dependencyComplexity = totalPossibleDeps > 0 ? Math.min(actualDeps / totalPossibleDeps * 2, 1) : 0;
    }
    
    // Mots clés indiquant une requête complexe
    const complexityKeywords = ['comparer', 'analyser', 'synthétiser', 'évaluer', 'prédire', 
                               'multiples', 'détaillé', 'approfondi', 'sophistiqué'];
    const keywordMatches = complexityKeywords.filter(keyword => 
      query.toLowerCase().includes(keyword.toLowerCase())
    ).length;
    const keywordComplexity = Math.min(keywordMatches / 3, 1);
    
    // Pondération et calcul du score final
    const weights = {
      length: 0.1,
      wordCount: 0.15,
      componentComplexity: 0.3,
      dependencyComplexity: 0.25,
      keywordComplexity: 0.2
    };
    
    const complexityScore = 
      length * weights.length +
      wordCountNormalized * weights.wordCount +
      componentComplexity * weights.componentComplexity +
      dependencyComplexity * weights.dependencyComplexity +
      keywordComplexity * weights.keywordComplexity;
    
    return Math.min(Math.max(complexityScore, 0.1), 1);
  }
  
  /**
   * Détermine le mode d'exécution optimal basé sur le contexte
   * @param context Contexte d'exécution
   * @returns Mode d'exécution recommandé
   */
  private determineOptimalExecutionMode(context: CoordinationContext): ExecutionMode {
    const { complexityScore = 0.5, systemLoad = 0.5 } = context;
    
    // Stratégie simple basée sur la complexité et la charge
    // - Haute complexité + faible charge → parallèle
    // - Faible complexité ou haute charge → séquentiel
    
    const parallelThreshold = 0.6; // Seuil de complexité pour parallélisme
    const loadThreshold = 0.7;    // Seuil de charge pour exécution séquentielle
    
    if (complexityScore > parallelThreshold && systemLoad < loadThreshold) {
      return ExecutionMode.PARALLEL;
    } else {
      return ExecutionMode.SEQUENTIAL;
    }
  }
  
  /**
   * Calcule l'ordre d'exécution optimal basé sur les dépendances et priorités
   * @param components Liste de composants à ordonner
   * @returns Liste ordonnée d'IDs de composants
   */
  private calculateExecutionOrder(components: CoordinationComponent[]): string[] {
    // Trier d'abord par priorité (du plus important au moins important)
    const priorityOrder = {
      [ComponentPriority.CRITICAL]: 0,
      [ComponentPriority.HIGH]: 1,
      [ComponentPriority.NORMAL]: 2,
      [ComponentPriority.LOW]: 3
    };
    
    const sortedByPriority = [...components].sort((a, b) => 
      priorityOrder[a.priority] - priorityOrder[b.priority]
    );
    
    // Construire un graphe de dépendances
    const graph: Map<string, string[]> = new Map();
    const inDegree: Map<string, number> = new Map();
    
    // Initialiser les structures
    sortedByPriority.forEach(component => {
      graph.set(component.id, []);
      inDegree.set(component.id, 0);
    });
    
    // Remplir le graphe de dépendances
    sortedByPriority.forEach(component => {
      if (component.dependencies && component.dependencies.length > 0) {
        component.dependencies.forEach(depId => {
          // Vérifier que la dépendance existe
          if (graph.has(depId)) {
            graph.get(depId).push(component.id);
            inDegree.set(component.id, inDegree.get(component.id) + 1);
          }
        });
      }
    });
    
    // Tri topologique (algorithme de Kahn)
    const result: string[] = [];
    const queue: string[] = [];
    
    // Ajouter les nœuds sans dépendances à la file
    sortedByPriority.forEach(component => {
      if (inDegree.get(component.id) === 0) {
        queue.push(component.id);
      }
    });
    
    // Traiter la file
    while (queue.length > 0) {
      const current = queue.shift();
      result.push(current);
      
      // Mettre à jour les degrés entrants pour les voisins
      const neighbors = graph.get(current);
      for (const neighbor of neighbors) {
        inDegree.set(neighbor, inDegree.get(neighbor) - 1);
        
        // Si toutes les dépendances sont satisfaites, ajouter à la file
        if (inDegree.get(neighbor) === 0) {
          queue.push(neighbor);
        }
      }
    }
    
    // Vérifier s'il y a un cycle dans le graphe
    if (result.length !== components.length) {
      throw new Error("Cycle détecté dans les dépendances des composants");
    }
    
    return result;
  }
  
  /**
   * Calcule les niveaux d'exécution pour le mode parallèle
   * @param components Liste de composants
   * @returns Liste de niveaux avec les IDs de composants pouvant s'exécuter en parallèle
   */
  private calculateExecutionLevels(components: CoordinationComponent[]): string[][] {
    // Construire un graphe de dépendances inverse
    const dependsOn: Map<string, Set<string>> = new Map();
    
    // Initialiser les dépendances
    components.forEach(component => {
      dependsOn.set(component.id, new Set());
      
      if (component.dependencies && component.dependencies.length > 0) {
        component.dependencies.forEach(depId => {
          if (components.some(c => c.id === depId)) {
            dependsOn.get(component.id).add(depId);
          }
        });
      }
    });
    
    // Calculer les niveaux par itération
    const levels: string[][] = [];
    const remaining = new Set(components.map(c => c.id));
    
    while (remaining.size > 0) {
      // Trouver les composants sans dépendances
      const currentLevel = Array.from(remaining).filter(id => 
        dependsOn.get(id).size === 0
      );
      
      // Si aucun composant ne peut être exécuté, il y a un cycle
      if (currentLevel.length === 0) {
        throw new Error("Cycle détecté dans les dépendances des composants");
      }
      
      // Ajouter le niveau actuel
      levels.push(currentLevel);
      
      // Mettre à jour les dépendances restantes
      currentLevel.forEach(id => {
        remaining.delete(id);
        
        // Supprimer ce composant des dépendances des autres
        remaining.forEach(remainingId => {
          dependsOn.get(remainingId).delete(id);
        });
      });
    }
    
    return levels;
  }
  
  /**
   * Collecte les métriques détaillées de l'exécution
   * @param context Contexte d'exécution
   * @param totalDuration Durée totale en ms
   * @returns Métriques complètes
   */
  private collectMetrics(context: CoordinationContext, totalDuration: number): ExecutionMetrics {
    const componentMetrics: any = {};
    let peakExecutionTime = 0;
    const bottlenecks: string[] = [];
    
    // Collecter les métriques par composant
    Array.from(context.components.values()).forEach(component => {
      if (component.startTime && component.endTime) {
        const duration = component.endTime - component.startTime;
        componentMetrics[component.id] = {
          duration,
          status: component.status,
          retries: 0 // À implémenter plus tard
        };
        
        // Identifier les bottlenecks potentiels (composants qui prennent >20% du temps total)
        if (duration > totalDuration * 0.2) {
          bottlenecks.push(component.id);
          peakExecutionTime = Math.max(peakExecutionTime, duration);
        }
      }
    });
    
    // Générer des suggestions d'optimisation
    const optimizationSuggestions: string[] = [];
    
    if (bottlenecks.length > 0) {
      optimizationSuggestions.push(`Optimiser les composants à longue durée: ${bottlenecks.join(', ')}`);
    }
    
    if (context.options.mode === ExecutionMode.SEQUENTIAL && totalDuration > 5000) {
      optimizationSuggestions.push("Considérer le mode parallèle pour réduire la latence totale");
    }
    
    // Construire le résultat final
    return {
      totalDuration,
      componentMetrics,
      resourceUtilization: {
        peakMemoryUsage: 0, // À remplacer par des mesures réelles
        averageCpuUsage: 0  // À remplacer par des mesures réelles
      },
      bottlenecks,
      optimizationSuggestions
    };
  }
  
  /**
   * Extrait le chemin d'exécution réel à partir du contexte
   * @param context Contexte d'exécution
   * @returns Liste des IDs de composants dans l'ordre d'exécution
   */
  private extractExecutionPath(context: CoordinationContext): string[] {
    // Cette méthode doit être implémentée pour suivre l'ordre réel d'exécution
    // Actuellement, nous retournons simplement les composants terminés triés par heure de début
    return Array.from(context.components.values())
      .filter(c => c.startTime !== undefined)
      .sort((a, b) => (a.startTime || 0) - (b.startTime || 0))
      .map(c => c.id);
  }
} 