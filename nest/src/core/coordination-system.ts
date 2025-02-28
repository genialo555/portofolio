import { v4 as uuidv4 } from 'uuid';
import { CoordinationHandler, ExecutionMode, ExecutionOptions, ComponentPriority } from '../handlers/coordination-handler';
import { ComponentRegistry, ComponentType } from '../components/registry';
import { Logger, LogLevel } from '../utils/logger';

/**
 * Configuration du système de coordination
 */
export interface CoordinationSystemConfig {
  logLevel?: LogLevel;                  // Niveau de journalisation
  defaultExecutionMode?: ExecutionMode; // Mode d'exécution par défaut
  autoRegisterComponents?: boolean;     // Enregistrer automatiquement les composants de base
  defaultTimeout?: number;              // Timeout par défaut en ms
  enableCircuitBreaker?: boolean;       // Activer les circuit breakers
  enableMetrics?: boolean;              // Activer les métriques détaillées
  maxParallelExecutions?: number;       // Exécutions parallèles maximum
}

/**
 * Options pour une requête
 */
export interface QueryOptions {
  executionMode?: ExecutionMode;        // Mode d'exécution
  timeout?: number;                     // Timeout en ms
  abortOnFailure?: boolean;             // Arrêter en cas d'échec
  includeDebugInfo?: boolean;           // Inclure les infos de débogage
  useComponents?: string[];             // IDs des composants à utiliser
  excludeComponents?: string[];         // IDs des composants à exclure
  userContext?: any;                    // Contexte utilisateur
  enableAnomalyDetection?: boolean;     // Activer la détection d'anomalies
  traceId?: string;                     // ID de trace fourni
}

/**
 * Résultat d'une requête traitée
 */
export interface QueryResult {
  success: boolean;                     // Succès de l'exécution
  traceId: string;                      // ID de trace
  result: any;                          // Résultat principal
  error?: string;                       // Message d'erreur si échec
  duration: number;                     // Durée totale en ms
  debugInfo?: any;                      // Informations de débogage (si activé)
  anomalies?: any[];                    // Anomalies détectées (si activé)
  componentResults?: Record<string, any>; // Résultats par composant
}

/**
 * Système de coordination global pour l'architecture RAG/KAG
 */
export class CoordinationSystem {
  private logger: Logger;
  private registry: ComponentRegistry;
  private handler: CoordinationHandler;
  private defaultConfig: CoordinationSystemConfig;

  /**
   * Crée une instance du système de coordination
   * @param config Configuration du système
   */
  constructor(config: CoordinationSystemConfig = {}) {
    // Configuration par défaut
    this.defaultConfig = {
      logLevel: LogLevel.INFO,
      defaultExecutionMode: ExecutionMode.ADAPTIVE,
      autoRegisterComponents: true,
      defaultTimeout: 30000,
      enableCircuitBreaker: true,
      enableMetrics: true,
      maxParallelExecutions: 5,
      ...config
    };

    // Initialiser le logger
    this.logger = new Logger({
      level: this.defaultConfig.logLevel,
      colorize: true,
      timestamp: true,
      includeTraceId: true,
      outputToConsole: true
    });

    // Initialiser le registre et le gestionnaire
    this.registry = new ComponentRegistry(this.logger);
    this.handler = new CoordinationHandler(this.logger);

    // Enregistrer les composants de base si demandé
    if (this.defaultConfig.autoRegisterComponents) {
      this.registerBaseComponents();
    }

    this.logger.info('Système de coordination initialisé');
  }

  /**
   * Enregistre les composants de base du système
   */
  private registerBaseComponents(): void {
    // Cette méthode enregistrerait les composants essentiels
    // Ce sont des placeholders - dans une implémentation réelle, 
    // vous importeriez les implémentations réelles des composants
    
    // Composant d'analyse de requête
    this.registry.register({
      type: ComponentType.QUERY_ANALYZER,
      name: 'Analyseur de Requête Standard',
      description: 'Analyse la requête utilisateur pour déterminer les besoins',
      version: '1.0.0',
      priority: ComponentPriority.CRITICAL,
      executeFunction: async (context) => {
        // Implémentation réelle à fournir
        this.logger.debug(`[${context.traceId}] Analyse de requête: "${context.query.substring(0, 50)}..."`);
        return {
          queryType: 'information',
          complexityScore: 0.6,
          domains: ['general'],
          keywords: context.query.split(' ').filter(w => w.length > 4)
        };
      },
      isEnabled: true
    });
    
    // Composant de sélection de pools
    this.registry.register({
      type: ComponentType.POOL_SELECTOR,
      name: 'Sélecteur de Pools',
      description: 'Détermine quels pools d\'agents activer',
      version: '1.0.0',
      priority: ComponentPriority.HIGH,
      dependencies: [], // Dans une implémentation réelle, dépendrait de l'analyseur
      executeFunction: async (context) => {
        // Implémentation réelle à fournir
        this.logger.debug(`[${context.traceId}] Sélection des pools d'agents`);
        return {
          selectedPools: ['commercial', 'marketing'],
          poolRelevanceScores: {
            commercial: 0.8,
            marketing: 0.6,
            sectoral: 0.3
          }
        };
      },
      isEnabled: true
    });
    
    // Composant de formatage de sortie
    this.registry.register({
      type: ComponentType.OUTPUT_FORMATTER,
      name: 'Formateur de Sortie',
      description: 'Formate les résultats pour l\'utilisateur',
      version: '1.0.0',
      priority: ComponentPriority.NORMAL,
      dependencies: [], // Dépendrait des agents et autres composants dans une implémentation réelle
      executeFunction: async (context) => {
        // Implémentation réelle à fournir
        this.logger.debug(`[${context.traceId}] Formatage des résultats`);
        return {
          formattedOutput: `Réponse à la requête: "${context.query}"`,
          format: 'text'
        };
      },
      isEnabled: true
    });
    
    // Détecteur d'anomalies
    this.registry.register({
      type: ComponentType.ANOMALY_DETECTOR,
      name: 'Détecteur d\'Anomalies',
      description: 'Identifie les anomalies dans les outputs des agents',
      version: '1.0.0',
      priority: ComponentPriority.LOW, // Exécuté en dernier
      dependencies: [], // Dépendrait des composants de génération de réponse
      executeFunction: async (context) => {
        // Implémentation réelle à fournir
        this.logger.debug(`[${context.traceId}] Détection d'anomalies`);
        return {
          anomaliesDetected: false,
          confidenceScore: 0.95
        };
      },
      isEnabled: true
    });
    
    this.logger.info('Composants de base enregistrés');
  }

  /**
   * Traite une requête utilisateur
   * @param query Requête à traiter
   * @param options Options de traitement
   * @returns Résultat de la requête
   */
  public async processQuery(query: string, options: QueryOptions = {}): Promise<QueryResult> {
    const startTime = Date.now();
    const traceId = options.traceId || uuidv4();
    
    this.logger.info(`[${traceId}] Traitement de requête: "${query.substring(0, 50)}..."`);
    
    try {
      // Déterminer les composants à utiliser
      let components = this.registry.buildExecutableComponents();
      
      // Filtrer selon les options
      if (options.useComponents && options.useComponents.length > 0) {
        components = components.filter(c => options.useComponents.includes(c.id));
      }
      
      if (options.excludeComponents && options.excludeComponents.length > 0) {
        components = components.filter(c => !options.excludeComponents.includes(c.id));
      }
      
      // Vérifier qu'il y a des composants à exécuter
      if (components.length === 0) {
        throw new Error('Aucun composant disponible pour traiter la requête');
      }
      
      // Préparer les options d'exécution
      const executionOptions: ExecutionOptions = {
        mode: options.executionMode || this.defaultConfig.defaultExecutionMode,
        timeout: options.timeout || this.defaultConfig.defaultTimeout,
        abortOnComponentFailure: options.abortOnFailure || false,
        useCircuitBreaker: this.defaultConfig.enableCircuitBreaker,
        includeMetrics: this.defaultConfig.enableMetrics,
        resourceConstraints: {
          maxParallelExecutions: this.defaultConfig.maxParallelExecutions
        }
      };
      
      // Exécuter la coordination
      const result = await this.handler.execute(
        query,
        components,
        options.userContext,
        executionOptions
      );
      
      // Préparer le résultat
      const queryResult: QueryResult = {
        success: result.success,
        traceId: result.traceId,
        result: result.result,
        duration: result.duration,
        componentResults: {}
      };
      
      // Ajouter les résultats des composants si demandé
      if (options.includeDebugInfo && result.componentResults) {
        queryResult.debugInfo = {
          executionPath: result.executionPath,
          failedComponents: result.failedComponents,
          metrics: result.metrics
        };
        
        // Convertir la Map en objet pour la sortie JSON
        result.componentResults.forEach((value, key) => {
          queryResult.componentResults[key] = value;
        });
      }
      
      this.logger.info(`[${traceId}] Requête traitée avec succès en ${result.duration}ms`);
      return queryResult;
      
    } catch (error) {
      const duration = Date.now() - startTime;
      this.logger.error(`[${traceId}] Erreur lors du traitement: ${error.message}`);
      
      return {
        success: false,
        traceId,
        result: null,
        error: error.message,
        duration
      };
    }
  }

  /**
   * Enregistre un composant externe
   * @param componentConfig Configuration du composant
   * @returns ID du composant enregistré
   */
  public registerComponent(componentConfig: any): string {
    return this.registry.register(componentConfig);
  }

  /**
   * Désactive un composant
   * @param componentId ID du composant
   * @returns true si succès
   */
  public disableComponent(componentId: string): boolean {
    return this.registry.disable(componentId);
  }

  /**
   * Active un composant
   * @param componentId ID du composant
   * @returns true si succès
   */
  public enableComponent(componentId: string): boolean {
    return this.registry.enable(componentId);
  }

  /**
   * Génère un rapport sur l'état du système
   * @returns Rapport
   */
  public generateSystemReport(): any {
    const registryReport = this.registry.generateReport();
    const validationResult = this.registry.validateDependencies();
    
    const report = {
      timestamp: new Date().toISOString(),
      componentRegistry: {
        report: registryReport,
        validationIssues: {
          missingDependencies: validationResult.missingDependencies,
          circularDependencies: validationResult.circularDependencies
        }
      },
      configuration: this.defaultConfig,
      status: {
        isReady: validationResult.missingDependencies.length === 0 && 
                validationResult.circularDependencies.length === 0
      }
    };
    
    return report;
  }

  /**
   * Change le niveau de journalisation
   * @param level Nouveau niveau
   */
  public setLogLevel(level: LogLevel): void {
    this.logger.setLevel(level);
    this.logger.info(`Niveau de journalisation changé à ${LogLevel[level]}`);
  }
} 