import { Logger } from '../utils/logger';

/**
 * États possibles d'un disjoncteur
 */
export enum CircuitState {
  CLOSED = 'CLOSED',       // Circuit fermé - opérations normales
  OPEN = 'OPEN',           // Circuit ouvert - échecs rapides
  HALF_OPEN = 'HALF_OPEN'  // Circuit semi-ouvert - test de récupération
}

/**
 * Options pour configurer un disjoncteur
 */
export interface CircuitBreakerOptions {
  failureThreshold: number;       // Nombre d'échecs nécessaires pour ouvrir le circuit
  resetTimeout: number;           // Temps en ms avant de passer à l'état semi-ouvert
  successThreshold: number;       // Nombre de succès nécessaires pour fermer le circuit
  timeout: number;                // Délai en ms avant qu'une opération ne soit considérée comme échouée
  monitorInterval?: number;       // Intervalle de surveillance des statistiques en ms
  volumeThreshold?: number;       // Volume minimal d'appels avant de considérer les statistiques
  errorPercentageThreshold?: number; // Pourcentage d'erreurs nécessaire pour ouvrir le circuit
}

/**
 * Statistiques d'un disjoncteur
 */
export interface CircuitStats {
  state: CircuitState;            // État actuel du circuit
  failures: number;               // Nombre d'échecs depuis la dernière réinitialisation
  successes: number;              // Nombre de succès depuis la dernière réinitialisation
  rejected: number;               // Nombre d'appels rejetés (circuit ouvert)
  lastFailure?: Error;            // Dernière erreur rencontrée
  lastFailureTime?: number;       // Horodatage de la dernière erreur
  lastStateChange: number;        // Horodatage du dernier changement d'état
  totalCalls: number;             // Nombre total d'appels
  totalFailures: number;          // Nombre total d'échecs
  currentErrorPercentage: number; // Pourcentage d'erreurs actuel
}

/**
 * Événements émis par le disjoncteur
 */
export enum CircuitEvent {
  STATE_CHANGE = 'STATE_CHANGE',  // Changement d'état du circuit
  SUCCESS = 'SUCCESS',            // Exécution réussie
  FAILURE = 'FAILURE',            // Échec d'exécution
  REJECTED = 'REJECTED',          // Exécution rejetée (circuit ouvert)
  TIMEOUT = 'TIMEOUT',            // Dépassement du délai d'exécution
  FALLBACK = 'FALLBACK'           // Utilisation d'une fonction de repli
}

/**
 * Type d'écouteur d'événements
 */
type CircuitListener = (event: CircuitEvent, data?: any) => void;

/**
 * Exception spécifique levée lorsqu'un circuit ouvert rejette une opération
 */
export class CircuitOpenError extends Error {
  constructor(serviceName: string) {
    super(`Circuit ${serviceName} ouvert - opération rejetée`);
    this.name = 'CircuitOpenError';
  }
}

/**
 * Implémentation du modèle Circuit Breaker
 * Permet de protéger le système contre les cascades d'erreurs
 */
export class CircuitBreaker {
  private state: CircuitState = CircuitState.CLOSED;
  private failures: number = 0;
  private successes: number = 0;
  private rejected: number = 0;
  private lastFailure?: Error;
  private lastFailureTime?: number;
  private lastStateChange: number = Date.now();
  private stateTimer?: NodeJS.Timeout;
  private monitorTimer?: NodeJS.Timeout;
  private listeners: Map<CircuitEvent, CircuitListener[]> = new Map();
  private totalCalls: number = 0;
  private totalFailures: number = 0;
  private recentCalls: number = 0;
  private recentFailures: number = 0;
  private statsResetTime: number = Date.now();
  
  /**
   * Crée un nouveau disjoncteur
   * @param name Nom du service protégé
   * @param options Options de configuration
   * @param logger Instance du logger
   */
  constructor(
    private readonly name: string,
    private readonly options: CircuitBreakerOptions,
    private readonly logger: Logger
  ) {
    this.logger.info(`Circuit Breaker [${this.name}] initialisé`, {
      failureThreshold: this.options.failureThreshold,
      resetTimeout: this.options.resetTimeout,
      successThreshold: this.options.successThreshold
    });
    
    // Configurer la surveillance si demandée
    if (this.options.monitorInterval) {
      this.monitorTimer = setInterval(() => this.resetRecentStats(), this.options.monitorInterval);
    }
  }
  
  /**
   * Exécute une fonction protégée par le disjoncteur
   * @param fn Fonction à exécuter
   * @param fallback Fonction de repli en cas d'erreur (optionnelle)
   * @returns Résultat de l'exécution
   */
  public async execute<T>(fn: () => Promise<T>, fallback?: (error: Error) => Promise<T>): Promise<T> {
    this.totalCalls++;
    this.recentCalls++;
    
    // Vérifier si le circuit est ouvert
    if (this.state === CircuitState.OPEN) {
      this.rejected++;
      this.emit(CircuitEvent.REJECTED, { serviceName: this.name });
      
      this.logger.debug(`Circuit [${this.name}] ouvert - appel rejeté`);
      
      const circuitOpenError = new CircuitOpenError(this.name);
      
      // Utiliser la fonction de repli si disponible
      if (fallback) {
        this.emit(CircuitEvent.FALLBACK, { error: circuitOpenError });
        return await fallback(circuitOpenError);
      }
      
      throw circuitOpenError;
    }
    
    // Créer une promesse avec timeout
    return new Promise<T>((resolve, reject) => {
      let timeoutId: NodeJS.Timeout;
      
      // Configuration du timeout
      if (this.options.timeout > 0) {
        timeoutId = setTimeout(() => {
          const timeoutError = new Error(`Opération ${this.name} expirée après ${this.options.timeout}ms`);
          this.emit(CircuitEvent.TIMEOUT, { error: timeoutError });
          this.handleFailure(timeoutError);
          
          // Utiliser la fonction de repli si disponible
          if (fallback) {
            this.emit(CircuitEvent.FALLBACK, { error: timeoutError });
            resolve(fallback(timeoutError));
          } else {
            reject(timeoutError);
          }
        }, this.options.timeout);
      }
      
      // Exécuter la fonction
      fn()
        .then((result) => {
          if (timeoutId) clearTimeout(timeoutId);
          this.handleSuccess();
          resolve(result);
        })
        .catch((error) => {
          if (timeoutId) clearTimeout(timeoutId);
          this.handleFailure(error);
          
          // Utiliser la fonction de repli si disponible
          if (fallback) {
            this.emit(CircuitEvent.FALLBACK, { error });
            resolve(fallback(error));
          } else {
            reject(error);
          }
        });
    });
  }
  
  /**
   * Gère un succès d'exécution
   */
  private handleSuccess(): void {
    this.emit(CircuitEvent.SUCCESS);
    
    // En état semi-ouvert, augmenter le compteur de succès
    if (this.state === CircuitState.HALF_OPEN) {
      this.successes++;
      
      // Vérifier si le seuil de succès est atteint pour fermer le circuit
      if (this.successes >= this.options.successThreshold) {
        this.toClosedState();
      }
    } else {
      // Réinitialiser les échecs en cas de succès en état fermé
      this.failures = 0;
    }
    
    this.logger.debug(`Circuit [${this.name}] - opération réussie`, {
      state: this.state,
      successes: this.successes
    });
  }
  
  /**
   * Gère un échec d'exécution
   * @param error Erreur rencontrée
   */
  private handleFailure(error: Error): void {
    this.failures++;
    this.totalFailures++;
    this.recentFailures++;
    this.lastFailure = error;
    this.lastFailureTime = Date.now();
    
    this.emit(CircuitEvent.FAILURE, { error });
    
    const errorPercentage = (this.recentFailures / this.recentCalls) * 100;
    
    // Vérifier les conditions d'ouverture du circuit
    if (this.state === CircuitState.CLOSED) {
      const thresholdMet = this.failures >= this.options.failureThreshold;
      const volumeMet = !this.options.volumeThreshold || this.recentCalls >= this.options.volumeThreshold;
      const percentageMet = !this.options.errorPercentageThreshold || 
                            (errorPercentage >= this.options.errorPercentageThreshold);
      
      if (thresholdMet && volumeMet && percentageMet) {
        this.toOpenState();
      }
    } else if (this.state === CircuitState.HALF_OPEN) {
      // En état semi-ouvert, tout échec ouvre immédiatement le circuit
      this.toOpenState();
    }
    
    this.logger.debug(`Circuit [${this.name}] - opération échouée`, {
      state: this.state,
      failures: this.failures,
      error: error.message,
      errorPercentage
    });
  }
  
  /**
   * Passe à l'état ouvert
   */
  private toOpenState(): void {
    this.logger.info(`Circuit [${this.name}] - passage à l'état OUVERT`, {
      failures: this.failures,
      lastError: this.lastFailure?.message
    });
    
    if (this.state !== CircuitState.OPEN) {
      this.state = CircuitState.OPEN;
      this.lastStateChange = Date.now();
      this.emit(CircuitEvent.STATE_CHANGE, { oldState: CircuitState.CLOSED, newState: CircuitState.OPEN });
      
      // Planifier le passage à l'état semi-ouvert
      this.stateTimer = setTimeout(() => {
        this.toHalfOpenState();
      }, this.options.resetTimeout);
    }
  }
  
  /**
   * Passe à l'état semi-ouvert
   */
  private toHalfOpenState(): void {
    this.logger.info(`Circuit [${this.name}] - passage à l'état SEMI-OUVERT`, {
      resetTimeout: this.options.resetTimeout
    });
    
    this.state = CircuitState.HALF_OPEN;
    this.lastStateChange = Date.now();
    this.failures = 0;
    this.successes = 0;
    
    this.emit(CircuitEvent.STATE_CHANGE, {
      oldState: CircuitState.OPEN,
      newState: CircuitState.HALF_OPEN
    });
  }
  
  /**
   * Passe à l'état fermé
   */
  private toClosedState(): void {
    this.logger.info(`Circuit [${this.name}] - passage à l'état FERMÉ`, {
      successThreshold: this.options.successThreshold
    });
    
    this.state = CircuitState.CLOSED;
    this.lastStateChange = Date.now();
    this.failures = 0;
    this.successes = 0;
    
    this.emit(CircuitEvent.STATE_CHANGE, {
      oldState: CircuitState.HALF_OPEN,
      newState: CircuitState.CLOSED
    });
  }
  
  /**
   * Réinitialise les statistiques récentes
   */
  private resetRecentStats(): void {
    const now = Date.now();
    const errorPercentage = this.recentCalls > 0 
      ? (this.recentFailures / this.recentCalls) * 100 
      : 0;
    
    this.logger.debug(`Circuit [${this.name}] - statistiques de surveillance`, {
      state: this.state,
      calls: this.recentCalls,
      failures: this.recentFailures,
      errorPercentage: errorPercentage.toFixed(2) + '%',
      elapsedMs: now - this.statsResetTime
    });
    
    // Ouvrir le circuit si le pourcentage d'erreur est trop élevé
    if (this.state === CircuitState.CLOSED &&
        this.options.errorPercentageThreshold &&
        this.options.volumeThreshold &&
        this.recentCalls >= this.options.volumeThreshold &&
        errorPercentage >= this.options.errorPercentageThreshold) {
      
      this.logger.warn(`Circuit [${this.name}] - ouverture basée sur le pourcentage d'erreurs`, {
        errorPercentage: errorPercentage.toFixed(2) + '%',
        threshold: this.options.errorPercentageThreshold + '%'
      });
      
      this.toOpenState();
    }
    
    // Réinitialiser les compteurs
    this.recentCalls = 0;
    this.recentFailures = 0;
    this.statsResetTime = now;
  }
  
  /**
   * S'abonne à un événement du disjoncteur
   * @param event Type d'événement
   * @param listener Fonction à appeler
   * @returns Fonction pour se désabonner
   */
  public on(event: CircuitEvent, listener: CircuitListener): () => void {
    if (!this.listeners.has(event)) {
      this.listeners.set(event, []);
    }
    
    this.listeners.get(event)!.push(listener);
    
    // Retourner une fonction pour se désabonner
    return () => {
      const eventListeners = this.listeners.get(event);
      if (eventListeners) {
        const index = eventListeners.indexOf(listener);
        if (index !== -1) {
          eventListeners.splice(index, 1);
        }
      }
    };
  }
  
  /**
   * Émet un événement aux abonnés
   * @param event Type d'événement
   * @param data Données associées
   */
  private emit(event: CircuitEvent, data?: any): void {
    const eventListeners = this.listeners.get(event);
    if (eventListeners) {
      for (const listener of eventListeners) {
        try {
          listener(event, data);
        } catch (error) {
          this.logger.error(`Erreur dans un écouteur d'événement ${event}`, { error });
        }
      }
    }
  }
  
  /**
   * Récupère les statistiques actuelles du disjoncteur
   * @returns Statistiques du disjoncteur
   */
  public getStats(): CircuitStats {
    const errorPercentage = this.recentCalls > 0 
      ? (this.recentFailures / this.recentCalls) * 100 
      : 0;
    
    return {
      state: this.state,
      failures: this.failures,
      successes: this.successes,
      rejected: this.rejected,
      lastFailure: this.lastFailure,
      lastFailureTime: this.lastFailureTime,
      lastStateChange: this.lastStateChange,
      totalCalls: this.totalCalls,
      totalFailures: this.totalFailures,
      currentErrorPercentage: errorPercentage
    };
  }
  
  /**
   * Réinitialise manuellement le disjoncteur à l'état fermé
   */
  public reset(): void {
    // Nettoyer les timers existants
    if (this.stateTimer) {
      clearTimeout(this.stateTimer);
      this.stateTimer = undefined;
    }
    
    this.logger.info(`Circuit [${this.name}] réinitialisé manuellement`);
    
    // Réinitialiser les compteurs et l'état
    this.failures = 0;
    this.successes = 0;
    this.rejected = 0;
    this.recentCalls = 0;
    this.recentFailures = 0;
    
    // Forcer l'état fermé
    const oldState = this.state;
    this.state = CircuitState.CLOSED;
    this.lastStateChange = Date.now();
    
    if (oldState !== CircuitState.CLOSED) {
      this.emit(CircuitEvent.STATE_CHANGE, {
        oldState,
        newState: CircuitState.CLOSED
      });
    }
  }
  
  /**
   * Libère les ressources du disjoncteur
   */
  public dispose(): void {
    // Nettoyer les timers
    if (this.stateTimer) {
      clearTimeout(this.stateTimer);
      this.stateTimer = undefined;
    }
    
    if (this.monitorTimer) {
      clearInterval(this.monitorTimer);
      this.monitorTimer = undefined;
    }
    
    // Supprimer tous les écouteurs
    this.listeners.clear();
    
    this.logger.debug(`Circuit [${this.name}] libéré`);
  }
} 