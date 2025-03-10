import { Injectable, Inject, OnModuleInit, Optional } from '@nestjs/common';
import { LOGGER_TOKEN, ILogger } from './logger-tokens';
import { CircuitBreaker, CircuitBreakerConfig, CircuitState } from '../../legacy/utils/circuit-breaker';
import { EventBusService, RagKagEventType } from '../core/event-bus.service';

/**
 * Interface pour les options de retry
 */
export interface RetryOptions {
  /**
   * Nombre maximum de tentatives
   */
  maxRetries: number;
  /**
   * Fonction qui détermine si une erreur doit déclencher un retry
   */
  retryCondition?: (error: any) => boolean;
  /**
   * Fonction appelée avant chaque tentative de retry
   */
  onRetry?: (error: any, attempt: number) => void;
  /**
   * Facteur d'augmentation exponentielle du délai entre les tentatives
   */
  backoffFactor?: number;
}

/**
 * Configuration par défaut des circuit breakers par service
 */
const DEFAULT_CIRCUIT_BREAKER_CONFIGS: Record<string, CircuitBreakerConfig> = {
  // API Externes
  'google-ai': {
    failureThreshold: 3,         // 3 échecs consécutifs ouvrent le circuit
    resetTimeout: 30000,         // 30 secondes avant de passer à l'état semi-ouvert
    successThreshold: 2,         // 2 succès consécutifs ferment le circuit
    timeout: 10000,              // 10 secondes de timeout pour les requêtes
    monitorInterval: 60000,      // Monitoring toutes les 60 secondes
    name: 'google-ai'
  },
  'qwen-ai': {
    failureThreshold: 3,
    resetTimeout: 30000,
    successThreshold: 2,
    timeout: 12000,              // 12 secondes de timeout (API potentiellement plus lente)
    monitorInterval: 60000,
    name: 'qwen-ai'
  },
  'deepseek-ai': {
    failureThreshold: 3,
    resetTimeout: 30000,
    successThreshold: 2,
    timeout: 15000,              // 15 secondes de timeout
    monitorInterval: 60000,
    name: 'deepseek-ai'
  },
  // Modèles locaux
  'house-model': {
    failureThreshold: 5,         // Plus tolérant pour les modèles locaux
    resetTimeout: 15000,         // 15 secondes avant de réessayer
    successThreshold: 1,         // 1 succès suffit pour réactiver
    timeout: 20000,              // 20 secondes de timeout pour l'inférence locale
    monitorInterval: 60000,
    name: 'house-model'
  },
  // API Python pour modèles
  'python-api': {
    failureThreshold: 4,         // 4 échecs consécutifs ouvrent le circuit
    resetTimeout: 20000,         // 20 secondes avant de réessayer
    successThreshold: 1,         // 1 succès suffit pour réactiver
    timeout: 30000,              // 30 secondes de timeout pour les opérations potentiellement longues
    monitorInterval: 60000,
    name: 'python-api'
  }
};

// Implémentation simplifiée de l'interface Logger pour CircuitBreaker
class SimpleLogger {
  constructor(private readonly logger: ILogger) {}
  
  info(message: string, meta?: any) {
    this.logger.info(message, meta);
  }
  
  warn(message: string, meta?: any) {
    this.logger.warn(message, meta);
  }
  
  error(message: string, meta?: any) {
    this.logger.error(message, meta);
  }
  
  debug(message: string, meta?: any) {
    this.logger.debug(message, meta);
  }
}

/**
 * Service gérant la résilience de l'application
 * Fournit des circuit breakers pour les différents services
 * Intégré avec EventBus pour notifier des changements d'état
 */
@Injectable()
export class ResilienceService implements OnModuleInit {
  private circuitBreakers: Map<string, CircuitBreaker> = new Map();
  private simpleLogger: SimpleLogger;
  private circuitStates: Map<string, CircuitState> = new Map();

  constructor(
    @Inject(LOGGER_TOKEN) private readonly logger: ILogger,
    @Optional() private readonly eventBus?: EventBusService
  ) {
    this.simpleLogger = new SimpleLogger(logger);
  }

  /**
   * Initialise les circuit breakers lors du démarrage du module
   */
  async onModuleInit() {
    this.logger.info('Initialisation du service de résilience');
    
    // Créer les circuit breakers par défaut
    for (const [serviceName, config] of Object.entries(DEFAULT_CIRCUIT_BREAKER_CONFIGS)) {
      this.createCircuitBreaker(serviceName, config);
    }
    
    // Émettre un événement d'initialisation
    if (this.eventBus) {
      this.eventBus.emit({
        type: RagKagEventType.SYSTEM_INIT,
        source: 'ResilienceService',
        payload: {
          circuitBreakers: Object.keys(DEFAULT_CIRCUIT_BREAKER_CONFIGS)
        }
      });
    }
    
    // Démarrer le monitoring des circuit breakers
    this.startCircuitBreakerMonitoring();
  }

  /**
   * Crée un circuit breaker pour un service spécifique
   */
  private createCircuitBreaker(serviceName: string, config: CircuitBreakerConfig): CircuitBreaker {
    // TypeScript ne vérifie pas complètement la compatibilité de SimpleLogger avec Logger
    // mais CircuitBreaker n'utilise que les méthodes que nous avons implémentées
    const circuitBreaker = new CircuitBreaker(this.simpleLogger as any, config);
    this.circuitBreakers.set(serviceName, circuitBreaker);
    this.circuitStates.set(serviceName, CircuitState.CLOSED);
    
    this.logger.info(`Circuit breaker créé pour le service ${serviceName}`);
    
    return circuitBreaker;
  }

  /**
   * Obtient un circuit breaker pour un service spécifique
   * Si le circuit breaker n'existe pas, il est créé avec la configuration par défaut
   */
  getCircuitBreaker(serviceName: string): CircuitBreaker {
    if (!this.circuitBreakers.has(serviceName)) {
      const defaultConfig = DEFAULT_CIRCUIT_BREAKER_CONFIGS[serviceName] || {
        failureThreshold: 3,
        resetTimeout: 30000,
        successThreshold: 2,
        timeout: 10000,
        monitorInterval: 60000,
        name: serviceName
      };
      
      return this.createCircuitBreaker(serviceName, defaultConfig);
    }
    
    return this.circuitBreakers.get(serviceName)!;
  }

  /**
   * Exécute une opération avec protection par circuit breaker
   * @param serviceName Nom du service à protéger
   * @param operation Fonction à exécuter
   * @param fallback Fonction de secours optionnelle
   */
  async executeWithCircuitBreaker<T>(
    serviceName: string, 
    operation: () => Promise<T>,
    fallback?: (error: Error) => Promise<T>
  ): Promise<T> {
    const circuitBreaker = this.getCircuitBreaker(serviceName);
    
    try {
      const result = await circuitBreaker.execute(operation, fallback);
      
      // Vérifier si l'état a changé après l'exécution
      this.checkAndNotifyStateChange(serviceName);
      
      return result;
    } catch (error) {
      this.logger.error(`Erreur avec circuit breaker pour ${serviceName}`, { error });
      
      // Vérifier si l'état a changé après l'erreur
      this.checkAndNotifyStateChange(serviceName);
      
      throw error;
    }
  }
  
  /**
   * Récupère l'état de tous les circuit breakers
   */
  getAllCircuitBreakersStatus(): Record<string, { state: CircuitState, metrics: any }> {
    const status: Record<string, { state: CircuitState, metrics: any }> = {};
    
    for (const [serviceName, circuitBreaker] of this.circuitBreakers.entries()) {
      const metrics = circuitBreaker.getMetrics();
      status[serviceName] = {
        state: metrics.state,
        metrics: {
          failures: metrics.failures,
          successes: metrics.successes,
          totalFailures: metrics.totalFailures,
          totalSuccesses: metrics.totalSuccesses,
          lastFailureTime: metrics.lastFailureTime,
          openCircuitCount: metrics.openCircuitCount
        }
      };
      
      // Mettre à jour l'état stocké
      this.circuitStates.set(serviceName, metrics.state);
    }
    
    return status;
  }
  
  /**
   * Force la réinitialisation d'un circuit breaker spécifique
   */
  resetCircuitBreaker(serviceName: string): boolean {
    if (this.circuitBreakers.has(serviceName)) {
      const circuitBreaker = this.circuitBreakers.get(serviceName)!;
      const oldState = circuitBreaker.getMetrics().state;
      
      circuitBreaker.reset();
      this.logger.info(`Circuit breaker pour ${serviceName} réinitialisé manuellement`);
      
      // Émettre un événement de réinitialisation
      if (this.eventBus) {
        this.eventBus.emit({
          type: RagKagEventType.CIRCUIT_BREAKER_CLOSED,
          source: 'ResilienceService',
          payload: {
            serviceName,
            oldState,
            newState: CircuitState.CLOSED,
            reason: 'manual_reset',
            timestamp: Date.now()
          }
        });
      }
      
      // Mettre à jour l'état stocké
      this.circuitStates.set(serviceName, CircuitState.CLOSED);
      
      return true;
    }
    
    this.logger.warn(`Tentative de réinitialisation d'un circuit breaker inexistant: ${serviceName}`);
    return false;
  }
  
  /**
   * Vérifie si l'état d'un circuit breaker a changé et émet un événement si c'est le cas
   */
  private checkAndNotifyStateChange(serviceName: string): void {
    if (!this.eventBus || !this.circuitBreakers.has(serviceName)) return;
    
    const circuitBreaker = this.circuitBreakers.get(serviceName)!;
    const currentState = circuitBreaker.getMetrics().state;
    const previousState = this.circuitStates.get(serviceName) || CircuitState.CLOSED;
    
    if (currentState !== previousState) {
      // L'état a changé, émettre un événement
      const eventType = currentState === CircuitState.OPEN 
        ? RagKagEventType.CIRCUIT_BREAKER_OPENED 
        : (currentState === CircuitState.HALF_OPEN 
          ? RagKagEventType.CUSTOM 
          : RagKagEventType.CIRCUIT_BREAKER_CLOSED);
      
      this.eventBus.emit({
        type: eventType,
        source: 'ResilienceService',
        payload: {
          serviceName,
          oldState: previousState,
          newState: currentState,
          metrics: circuitBreaker.getMetrics(),
          timestamp: Date.now(),
          eventType: currentState === CircuitState.HALF_OPEN ? 'CIRCUIT_BREAKER_HALF_OPEN' : undefined
        }
      });
      
      this.logger.info(`État du circuit breaker ${serviceName} changé: ${previousState} -> ${currentState}`);
      
      // Mettre à jour l'état stocké
      this.circuitStates.set(serviceName, currentState);
    }
  }
  
  /**
   * Démarre le monitoring périodique des circuit breakers
   */
  private startCircuitBreakerMonitoring(): void {
    // Vérifier l'état des circuit breakers toutes les minutes
    setInterval(() => {
      this.logger.debug('Vérification périodique des circuit breakers');
      
      for (const [serviceName] of this.circuitBreakers.entries()) {
        this.checkAndNotifyStateChange(serviceName);
      }
      
      // Émettre un événement de métriques
      if (this.eventBus) {
        this.eventBus.emit({
          type: RagKagEventType.PERFORMANCE_METRIC,
          source: 'ResilienceService',
          payload: {
            circuitBreakers: this.getAllCircuitBreakersStatus(),
            timestamp: Date.now()
          }
        });
      }
    }, 60000); // Vérification toutes les minutes
  }

  /**
   * Exécute une opération avec retry en cas d'échec
   * @param fn Fonction à exécuter
   * @param options Options de retry
   * @returns Résultat de la fonction
   */
  public async executeWithRetry<T>(
    fn: () => Promise<T>,
    options: RetryOptions
  ): Promise<T> {
    let lastError: any;
    const { maxRetries, retryCondition, onRetry, backoffFactor = 1.5 } = options;

    for (let attempt = 1; attempt <= maxRetries + 1; attempt++) {
      try {
        return await fn();
      } catch (error) {
        lastError = error;
        
        // Dernière tentative échouée
        if (attempt > maxRetries) {
          break;
        }
        
        // Vérifier si on doit réessayer pour cette erreur
        if (retryCondition && !retryCondition(error)) {
          this.logger.info(`Retry abandonné: la condition d'erreur n'est pas satisfaite`, {
            error: error.message,
            attempt
          });
          break;
        }
        
        // Notification de retry
        if (onRetry) {
          onRetry(error, attempt);
        }
        
        // Log de la tentative
        this.logger.warn(`Tentative ${attempt}/${maxRetries} échouée, nouvelle tentative prévue`, {
          error: error.message,
          attempt,
          nextAttempt: attempt + 1
        });
        
        // Attente exponentielle
        const delayMs = 1000 * Math.pow(backoffFactor, attempt - 1);
        this.logger.debug(`Attente de ${delayMs}ms avant la prochaine tentative`, {
          attempt,
          delayMs
        });
        
        await new Promise(resolve => setTimeout(resolve, delayMs));
      }
    }
    
    // Émettre un événement d'échec si EventBus est disponible
    if (this.eventBus) {
      this.eventBus.emit({
        type: RagKagEventType.QUERY_ERROR,
        source: 'ResilienceService',
        payload: {
          error: lastError,
          operation: 'retry',
          maxRetries
        }
      });
    }
    
    this.logger.error(`Toutes les tentatives ont échoué (${maxRetries + 1} tentatives)`, {
      error: lastError.message,
      stack: lastError.stack
    });
    
    throw lastError;
  }
} 