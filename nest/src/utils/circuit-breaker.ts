import { Logger } from './logger';

/**
 * États possibles du disjoncteur
 */
export enum CircuitState {
  CLOSED = 'CLOSED',       // Circuit fermé - opérations permises
  OPEN = 'OPEN',           // Circuit ouvert - opérations bloquées
  HALF_OPEN = 'HALF_OPEN'  // Circuit semi-ouvert - tester la récupération
}

/**
 * Configuration du disjoncteur
 */
export interface CircuitBreakerConfig {
  failureThreshold: number;        // Nombre d'échecs avant ouverture du circuit
  resetTimeout: number;            // Délai avant passage à l'état HALF_OPEN (ms)
  successThreshold: number;        // Nombre de succès en HALF_OPEN avant fermeture
  timeout?: number;                // Timeout pour les opérations (ms)
  monitorInterval?: number;        // Intervalle de surveillance des métriques (ms)
  name?: string;                   // Nom du disjoncteur (pour les logs)
}

/**
 * Métriques du disjoncteur
 */
export interface CircuitBreakerMetrics {
  state: CircuitState;             // État actuel
  failures: number;                // Nombre d'échecs consécutifs
  successes: number;               // Nombre de succès consécutifs
  lastFailure?: Error;             // Dernière erreur rencontrée
  lastFailureTime?: number;        // Timestamp de la dernière erreur
  totalFailures: number;           // Nombre total d'échecs
  totalSuccesses: number;          // Nombre total de succès
  totalTimeouts: number;           // Nombre total de timeouts
  openCircuitCount: number;        // Nombre d'ouvertures du circuit
  lastStateChange: number;         // Timestamp du dernier changement d'état
}

/**
 * Erreur spécifique au disjoncteur
 */
export class CircuitBreakerError extends Error {
  constructor(
    message: string,
    public readonly state: CircuitState,
    public readonly underlyingError?: Error
  ) {
    super(message);
    this.name = 'CircuitBreakerError';
  }
}

/**
 * Implémentation du pattern Circuit Breaker
 * Permet d'isoler un composant défaillant et d'éviter la cascade d'échecs
 */
export class CircuitBreaker {
  private state: CircuitState = CircuitState.CLOSED;
  private failures: number = 0;
  private successes: number = 0;
  private totalFailures: number = 0;
  private totalSuccesses: number = 0;
  private totalTimeouts: number = 0;
  private openCircuitCount: number = 0;
  private lastFailure?: Error;
  private lastFailureTime?: number;
  private lastStateChange: number = Date.now();
  private resetTimer?: NodeJS.Timeout;
  private monitorTimer?: NodeJS.Timeout;
  private readonly config: Required<CircuitBreakerConfig>;
  private logger: Logger;

  /**
   * Crée une instance de CircuitBreaker
   * @param logger Logger
   * @param config Configuration
   */
  constructor(logger: Logger, config: CircuitBreakerConfig) {
    this.logger = logger;
    
    // Valeurs par défaut pour la configuration
    this.config = {
      failureThreshold: 5,
      resetTimeout: 30000, // 30 secondes
      successThreshold: 2,
      timeout: 10000,      // 10 secondes
      monitorInterval: 60000, // 1 minute
      name: 'circuit-' + Math.floor(Math.random() * 10000),
      ...config
    };

    this.logger.debug(`CircuitBreaker '${this.config.name}' initialisé`, {
      config: this.config
    });

    // Démarrer la surveillance des métriques si demandé
    if (this.config.monitorInterval > 0) {
      this.startMetricsMonitoring();
    }
  }

  /**
   * Exécute une opération avec protection par disjoncteur
   * @param operation Fonction à exécuter
   * @param fallback Fonction de repli en cas d'échec (optionnelle)
   * @returns Résultat de l'opération ou du fallback
   * @throws CircuitBreakerError si le circuit est ouvert et pas de fallback
   */
  public async execute<T>(
    operation: () => Promise<T>,
    fallback?: (error: Error) => Promise<T>
  ): Promise<T> {
    try {
      // Vérifier si le circuit est ouvert
      if (this.state === CircuitState.OPEN) {
        return this.handleOpenCircuit(fallback);
      }

      // Exécuter l'opération avec timeout si configuré
      const result = await this.executeWithTimeout(operation);
      
      // Gérer le succès
      this.onSuccess();
      return result;
    } catch (error) {
      // Gérer l'échec
      this.onFailure(error as Error);
      
      // Utiliser le fallback si disponible
      if (fallback) {
        try {
          return await fallback(error as Error);
        } catch (fallbackError) {
          this.logger.error(`Fallback a également échoué pour '${this.config.name}'`, {
            originalError: error,
            fallbackError
          });
          throw fallbackError;
        }
      }
      
      throw error;
    }
  }

  /**
   * Réinitialise le disjoncteur à l'état CLOSED
   */
  public reset(): void {
    this.changeState(CircuitState.CLOSED);
    this.failures = 0;
    this.successes = 0;
    this.logger.info(`CircuitBreaker '${this.config.name}' réinitialisé manuellement`);
  }

  /**
   * Force l'ouverture du circuit
   */
  public forceOpen(): void {
    this.changeState(CircuitState.OPEN);
    this.logger.info(`CircuitBreaker '${this.config.name}' forcé à l'état OPEN`);
  }

  /**
   * Obtient les métriques actuelles du disjoncteur
   */
  public getMetrics(): CircuitBreakerMetrics {
    return {
      state: this.state,
      failures: this.failures,
      successes: this.successes,
      lastFailure: this.lastFailure,
      lastFailureTime: this.lastFailureTime,
      totalFailures: this.totalFailures,
      totalSuccesses: this.totalSuccesses,
      totalTimeouts: this.totalTimeouts,
      openCircuitCount: this.openCircuitCount,
      lastStateChange: this.lastStateChange
    };
  }

  /**
   * Démarrre la surveillance des métriques
   */
  private startMetricsMonitoring(): void {
    this.monitorTimer = setInterval(() => {
      this.logger.debug(`Métriques CircuitBreaker '${this.config.name}'`, this.getMetrics());
    }, this.config.monitorInterval);
  }

  /**
   * Exécute une opération avec timeout
   */
  private async executeWithTimeout<T>(operation: () => Promise<T>): Promise<T> {
    if (!this.config.timeout) {
      return operation();
    }

    return new Promise<T>((resolve, reject) => {
      let timeoutReached = false;
      let operationComplete = false;

      // Créer le timer de timeout
      const timer = setTimeout(() => {
        if (!operationComplete) {
          timeoutReached = true;
          this.totalTimeouts++;
          const error = new Error(`Opération timeout après ${this.config.timeout}ms`);
          this.logger.warn(`Timeout pour CircuitBreaker '${this.config.name}'`, { timeout: this.config.timeout });
          reject(error);
        }
      }, this.config.timeout);

      // Exécuter l'opération
      operation()
        .then(result => {
          if (!timeoutReached) {
            operationComplete = true;
            clearTimeout(timer);
            resolve(result);
          }
        })
        .catch(error => {
          if (!timeoutReached) {
            operationComplete = true;
            clearTimeout(timer);
            reject(error);
          }
        });
    });
  }

  /**
   * Gère le cas où le circuit est ouvert
   */
  private async handleOpenCircuit<T>(fallback?: (error: Error) => Promise<T>): Promise<T> {
    // Si en mode semi-ouvert, permettre l'exécution d'une requête test
    if (this.state === CircuitState.HALF_OPEN) {
      this.logger.debug(`Tentative de test en mode HALF_OPEN pour '${this.config.name}'`);
      return Promise.reject(new CircuitBreakerError(
        `Circuit '${this.config.name}' en cours de test (HALF_OPEN)`,
        this.state
      ));
    }

    const error = new CircuitBreakerError(
      `Circuit '${this.config.name}' ouvert - opération non exécutée`,
      this.state,
      this.lastFailure
    );

    if (fallback) {
      return fallback(error);
    }

    return Promise.reject(error);
  }

  /**
   * Gère un succès d'exécution
   */
  private onSuccess(): void {
    // Réinitialiser le compteur d'échecs
    this.failures = 0;
    this.totalSuccesses++;

    // Si en mode semi-ouvert, incrémenter le compteur de succès
    if (this.state === CircuitState.HALF_OPEN) {
      this.successes++;
      
      // Si le seuil de succès est atteint, fermer le circuit
      if (this.successes >= this.config.successThreshold) {
        this.logger.info(`CircuitBreaker '${this.config.name}' rétabli après ${this.successes} succès consécutifs`);
        this.changeState(CircuitState.CLOSED);
      }
    }
  }

  /**
   * Gère un échec d'exécution
   */
  private onFailure(error: Error): void {
    this.failures++;
    this.totalFailures++;
    this.lastFailure = error;
    this.lastFailureTime = Date.now();

    this.logger.warn(`Échec dans CircuitBreaker '${this.config.name}' (${this.failures}/${this.config.failureThreshold})`, {
      error: error.message
    });

    // Si le circuit est en mode semi-ouvert, ouvrir immédiatement
    if (this.state === CircuitState.HALF_OPEN) {
      this.logger.info(`Test échoué en mode HALF_OPEN pour '${this.config.name}', réouverture du circuit`);
      this.openCircuit();
      return;
    }

    // Si le seuil d'échecs est atteint, ouvrir le circuit
    if (this.state === CircuitState.CLOSED && this.failures >= this.config.failureThreshold) {
      this.openCircuit();
    }
  }

  /**
   * Ouvre le circuit
   */
  private openCircuit(): void {
    this.changeState(CircuitState.OPEN);
    this.openCircuitCount++;
    
    this.logger.warn(`CircuitBreaker '${this.config.name}' ouvert suite à ${this.failures} échecs consécutifs`);
    
    // Planifier la transition vers HALF_OPEN après le délai configuré
    this.resetTimer = setTimeout(() => {
      this.changeState(CircuitState.HALF_OPEN);
      this.logger.info(`CircuitBreaker '${this.config.name}' passé à l'état HALF_OPEN pour test`);
    }, this.config.resetTimeout);
  }

  /**
   * Change l'état du circuit
   */
  private changeState(newState: CircuitState): void {
    if (this.state === newState) return;
    
    const oldState = this.state;
    this.state = newState;
    this.lastStateChange = Date.now();
    
    // Réinitialiser les compteurs appropriés
    if (newState === CircuitState.CLOSED) {
      this.failures = 0;
      this.successes = 0;
      
      // Nettoyer le timer de reset si existant
      if (this.resetTimer) {
        clearTimeout(this.resetTimer);
        this.resetTimer = undefined;
      }
    } else if (newState === CircuitState.HALF_OPEN) {
      this.successes = 0;
    }
    
    this.logger.info(`CircuitBreaker '${this.config.name}' changé d'état: ${oldState} → ${newState}`);
  }

  /**
   * Nettoyage des ressources lors de la destruction
   */
  public destroy(): void {
    if (this.resetTimer) {
      clearTimeout(this.resetTimer);
    }
    
    if (this.monitorTimer) {
      clearInterval(this.monitorTimer);
    }
    
    this.logger.debug(`CircuitBreaker '${this.config.name}' détruit`);
  }
} 