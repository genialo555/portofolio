import { Injectable, Inject, OnModuleInit } from '@nestjs/common';
import { LOGGER_TOKEN, ILogger } from '../utils/logger-tokens';
import { v4 as uuidv4 } from 'uuid';

/**
 * Types d'événements dans le système RAG/KAG
 */
export enum RagKagEventType {
  // Événements du cycle de vie
  SYSTEM_INIT = 'SYSTEM_INIT',
  SYSTEM_READY = 'SYSTEM_READY',
  SYSTEM_SHUTDOWN = 'SYSTEM_SHUTDOWN',
  
  // Événements de traitement des requêtes
  QUERY_RECEIVED = 'QUERY_RECEIVED',
  QUERY_ROUTED = 'QUERY_ROUTED',
  QUERY_PROCESSED = 'QUERY_PROCESSED',
  QUERY_ERROR = 'QUERY_ERROR',
  
  // Événements RAG
  RAG_RETRIEVAL_STARTED = 'RAG_RETRIEVAL_STARTED',
  RAG_RETRIEVAL_COMPLETED = 'RAG_RETRIEVAL_COMPLETED',
  RAG_DOCUMENTS_RANKED = 'RAG_DOCUMENTS_RANKED',
  
  // Événements KAG
  KAG_GENERATION_STARTED = 'KAG_GENERATION_STARTED',
  KAG_GENERATION_COMPLETED = 'KAG_GENERATION_COMPLETED',
  KAG_KNOWLEDGE_VERIFIED = 'KAG_KNOWLEDGE_VERIFIED',
  
  // Événements des pools d'agents
  POOL_EXECUTION_STARTED = 'POOL_EXECUTION_STARTED',
  POOL_EXECUTION_COMPLETED = 'POOL_EXECUTION_COMPLETED',
  AGENT_EXECUTION_STARTED = 'AGENT_EXECUTION_STARTED',
  AGENT_EXECUTION_COMPLETED = 'AGENT_EXECUTION_COMPLETED',
  
  // Événements du système de débat
  DEBATE_STARTED = 'DEBATE_STARTED',
  DEBATE_COMPLETED = 'DEBATE_COMPLETED',
  CONTRADICTION_FOUND = 'CONTRADICTION_FOUND',
  CONSENSUS_REACHED = 'CONSENSUS_REACHED',
  
  // Événements des modèles
  MODEL_TRAINING_STARTED = 'MODEL_TRAINING_STARTED',
  MODEL_TRAINING_COMPLETED = 'MODEL_TRAINING_COMPLETED',
  MODEL_EVALUATION_STARTED = 'MODEL_EVALUATION_STARTED',
  MODEL_EVALUATION_COMPLETED = 'MODEL_EVALUATION_COMPLETED',
  MODEL_EVALUATION_INITIALIZED = 'MODEL_EVALUATION_INITIALIZED',
  MODEL_EVALUATION_ERROR = 'MODEL_EVALUATION_ERROR',
  
  // Événements de performance et erreurs
  PERFORMANCE_METRIC = 'PERFORMANCE_METRIC',
  CIRCUIT_BREAKER_OPENED = 'CIRCUIT_BREAKER_OPENED',
  CIRCUIT_BREAKER_CLOSED = 'CIRCUIT_BREAKER_CLOSED',
  ANOMALY_DETECTED = 'ANOMALY_DETECTED',
  
  // Événements de détection d'anomalies
  ANOMALY_DETECTION_STARTED = 'ANOMALY_DETECTION_STARTED',
  ANOMALY_DETECTION_COMPLETED = 'ANOMALY_DETECTION_COMPLETED',
  ANOMALY_ALERT = 'ANOMALY_ALERT',
  ANOMALY_CRITICAL = 'ANOMALY_CRITICAL',
  
  // Événements de synthèse
  SYNTHESIS_STARTED = 'SYNTHESIS_STARTED',
  SYNTHESIS_COMPLETED = 'SYNTHESIS_COMPLETED',
  SYNTHESIS_ERROR = 'SYNTHESIS_ERROR',
  
  // Événements du graphe de connaissances
  KNOWLEDGE_NODE_ADDED = 'KNOWLEDGE_NODE_ADDED',
  KNOWLEDGE_EDGE_ADDED = 'KNOWLEDGE_EDGE_ADDED',
  KNOWLEDGE_GRAPH_UPDATED = 'KNOWLEDGE_GRAPH_UPDATED',
  
  // Événement générique
  CUSTOM = 'CUSTOM'
}

/**
 * Interface pour un événement du système RAG/KAG
 */
export interface RagKagEvent {
  id: string;
  type: RagKagEventType;
  timestamp: number;
  source: string;
  target?: string;
  payload?: any;
  metadata?: Record<string, any>;
}

/**
 * Type pour une fonction écouteur d'événements
 */
export type EventListener = (event: RagKagEvent) => void | Promise<void>;

/**
 * Options pour les abonnements aux événements
 */
export interface SubscriptionOptions {
  once?: boolean;
  filter?: (event: RagKagEvent) => boolean;
  priority?: number;
  async?: boolean;
  timeout?: number;
}

/**
 * Information interne d'abonnement
 */
interface Subscription {
  id: string;
  eventType: RagKagEventType;
  listener: EventListener;
  options: Required<SubscriptionOptions>;
}

/**
 * Service de bus d'événements pour le système RAG/KAG
 * Permet la communication asynchrone entre les différents composants
 */
@Injectable()
export class EventBusService implements OnModuleInit {
  private subscriptions: Map<string, Subscription> = new Map();
  private eventTypeSubscriptions: Map<RagKagEventType, Set<string>> = new Map();
  private eventHistory: RagKagEvent[] = [];
  private maxHistorySize: number = 100;
  private enabled: boolean = true;

  constructor(@Inject(LOGGER_TOKEN) private readonly logger: ILogger) {}

  /**
   * Initialisation du module
   */
  onModuleInit() {
    this.logger.debug('EventBusService initialisé', { service: 'EventBusService' });
    this.emit({
      type: RagKagEventType.SYSTEM_INIT,
      source: 'EventBusService',
      payload: { timestamp: Date.now() }
    });
  }

  /**
   * S'abonne à un type d'événement
   * @param eventType Type d'événement à écouter
   * @param listener Fonction à exécuter lors de la réception de l'événement
   * @param options Options d'abonnement
   * @returns ID de l'abonnement (à utiliser pour se désabonner)
   */
  public subscribe(
    eventType: RagKagEventType,
    listener: EventListener,
    options: SubscriptionOptions = {}
  ): string {
    const subscriptionId = uuidv4();
    
    const subscription: Subscription = {
      id: subscriptionId,
      eventType,
      listener,
      options: {
        once: options.once ?? false,
        filter: options.filter ?? (() => true),
        priority: options.priority ?? 0,
        async: options.async ?? false,
        timeout: options.timeout ?? 5000
      }
    };
    
    this.subscriptions.set(subscriptionId, subscription);
    
    if (!this.eventTypeSubscriptions.has(eventType)) {
      this.eventTypeSubscriptions.set(eventType, new Set());
    }
    
    this.eventTypeSubscriptions.get(eventType)!.add(subscriptionId);
    
    this.logger.debug(
      `Abonnement #${subscriptionId} créé pour l'événement ${eventType}`,
      { service: 'EventBusService' }
    );
    
    return subscriptionId;
  }

  /**
   * S'abonne à un événement et se désabonne après la première exécution
   * @param eventType Type d'événement
   * @param listener Fonction écouteur
   * @param options Options d'abonnement
   * @returns ID de l'abonnement
   */
  public once(
    eventType: RagKagEventType,
    listener: EventListener,
    options: Omit<SubscriptionOptions, 'once'> = {}
  ): string {
    return this.subscribe(eventType, listener, { ...options, once: true });
  }

  /**
   * Se désabonne d'un événement
   * @param subscriptionId ID de l'abonnement
   * @returns true si désabonné avec succès
   */
  public unsubscribe(subscriptionId: string): boolean {
    const subscription = this.subscriptions.get(subscriptionId);
    
    if (!subscription) {
      this.logger.warn(`Tentative de désabonnement avec un ID invalide: ${subscriptionId}`, { service: 'EventBusService' });
      return false;
    }
    
    this.subscriptions.delete(subscriptionId);
    
    const eventTypeSet = this.eventTypeSubscriptions.get(subscription.eventType);
    if (eventTypeSet) {
      eventTypeSet.delete(subscriptionId);
      if (eventTypeSet.size === 0) {
        this.eventTypeSubscriptions.delete(subscription.eventType);
      }
    }
    
    this.logger.debug(`Abonnement #${subscriptionId} supprimé`, { service: 'EventBusService' });
    return true;
  }

  /**
   * Émet un événement dans le bus d'événements
   * @param event Événement à émettre (sans id ni timestamp, ils seront ajoutés)
   * @returns true si l'événement a été émis avec succès
   */
  public emit(event: Omit<RagKagEvent, 'id' | 'timestamp'>): boolean {
    if (!this.enabled) {
      return false;
    }
    
    const fullEvent: RagKagEvent = {
      id: uuidv4(),
      timestamp: Date.now(),
      ...event
    };
    
    this.addToHistory(fullEvent);
    
    const eventType = fullEvent.type;
    const subscriptionIds = this.eventTypeSubscriptions.get(eventType);
    
    if (!subscriptionIds || subscriptionIds.size === 0) {
      this.logger.debug(`Aucun abonné pour l'événement ${eventType}`, { service: 'EventBusService' });
      return true;
    }
    
    // Récupérer et trier les abonnements par priorité
    const subscriptions = Array.from(subscriptionIds)
      .map(id => this.subscriptions.get(id)!)
      .filter(sub => sub.options.filter(fullEvent))
      .sort((a, b) => b.options.priority - a.options.priority);
    
    // Exécuter les écouteurs
    for (const subscription of subscriptions) {
      try {
        if (subscription.options.async) {
          this.executeAsync(subscription, fullEvent);
        } else {
          subscription.listener(fullEvent);
          
          if (subscription.options.once) {
            this.unsubscribe(subscription.id);
          }
        }
      } catch (error) {
        this.logger.error(
          `Erreur lors de l'exécution de l'abonné #${subscription.id}: ${error.message}`,
          { service: 'EventBusService', error: error.stack }
        );
      }
    }
    
    return true;
  }

  /**
   * Exécute un écouteur de manière asynchrone
   * @param subscription Abonnement concerné
   * @param event Événement à traiter
   */
  private executeAsync(subscription: Subscription, event: RagKagEvent): void {
    const timeoutPromise = new Promise((_, reject) => {
      setTimeout(() => {
        reject(new Error(`Timeout dépassé pour l'abonné #${subscription.id}`));
      }, subscription.options.timeout);
    });
    
    Promise.race([
      Promise.resolve().then(() => subscription.listener(event)),
      timeoutPromise
    ])
      .catch(error => {
        this.logger.error(
          `Erreur asynchrone pour l'abonné #${subscription.id}: ${error.message}`,
          { service: 'EventBusService', error: error.stack }
        );
      })
      .finally(() => {
        if (subscription.options.once) {
          this.unsubscribe(subscription.id);
        }
      });
  }

  /**
   * Ajoute un événement à l'historique
   * @param event Événement à ajouter
   */
  private addToHistory(event: RagKagEvent): void {
    this.eventHistory.push(event);
    
    // Limiter la taille de l'historique
    if (this.eventHistory.length > this.maxHistorySize) {
      this.eventHistory = this.eventHistory.slice(-this.maxHistorySize);
    }
  }

  /**
   * Récupère l'historique des événements
   * @param limit Nombre maximum d'événements à récupérer
   * @param filter Fonction de filtrage optionnelle
   * @returns Événements de l'historique correspondant aux critères
   */
  public getHistory(
    limit?: number,
    filter?: (event: RagKagEvent) => boolean
  ): RagKagEvent[] {
    let events = this.eventHistory;
    
    if (filter) {
      events = events.filter(filter);
    }
    
    if (limit && limit > 0) {
      events = events.slice(-limit);
    }
    
    return [...events];
  }

  /**
   * Récupère les événements d'un type spécifique
   * @param eventType Type d'événement recherché
   * @param limit Nombre maximum d'événements à récupérer
   * @returns Événements correspondants au type
   */
  public getEventsByType(eventType: RagKagEventType, limit: number = 10): RagKagEvent[] {
    return this.getHistory(limit, event => event.type === eventType);
  }

  /**
   * Attend un événement spécifique
   * @param eventType Type d'événement à attendre
   * @param timeout Temps d'attente maximum en ms
   * @param filter Filtre optionnel
   * @returns Promise résolue avec l'événement ou rejetée si timeout
   */
  public waitForEvent(
    eventType: RagKagEventType,
    timeout: number = 10000,
    filter?: (event: RagKagEvent) => boolean
  ): Promise<RagKagEvent> {
    return new Promise((resolve, reject) => {
      const timeoutId = setTimeout(() => {
        this.unsubscribe(subscriptionId);
        reject(new Error(`Timeout atteint en attendant l'événement ${eventType}`));
      }, timeout);
      
      const subscriptionId = this.subscribe(
        eventType,
        (event) => {
          clearTimeout(timeoutId);
          resolve(event);
        },
        {
          once: true,
          filter: filter || (() => true)
        }
      );
    });
  }

  /**
   * Active ou désactive le bus d'événements
   * @param enabled État d'activation
   */
  public setEnabled(enabled: boolean): void {
    this.enabled = enabled;
    this.logger.debug(`EventBus ${enabled ? 'activé' : 'désactivé'}`, { service: 'EventBusService' });
  }

  /**
   * Supprime tous les abonnements
   */
  public clearSubscriptions(): void {
    this.subscriptions.clear();
    this.eventTypeSubscriptions.clear();
    this.logger.debug('Tous les abonnements ont été supprimés', { service: 'EventBusService' });
  }

  /**
   * Efface l'historique des événements
   */
  public clearHistory(): void {
    this.eventHistory = [];
    this.logger.debug('Historique des événements effacé', { service: 'EventBusService' });
  }

  /**
   * Définit la taille maximale de l'historique
   * @param size Nouvelle taille maximale
   */
  public setMaxHistorySize(size: number): void {
    if (size <= 0) {
      throw new Error('La taille de l\'historique doit être supérieure à 0');
    }
    
    this.maxHistorySize = size;
    
    // Ajuster l'historique si nécessaire
    if (this.eventHistory.length > size) {
      this.eventHistory = this.eventHistory.slice(-size);
    }
    
    this.logger.debug(`Taille maximale de l'historique définie à ${size}`, { service: 'EventBusService' });
  }
}

// Étendre l'interface ILogger pour inclure une méthode log
declare module '../utils/logger-tokens' {
  interface ILogger {
    log(message: string, context?: Record<string, any>): void;
  }
} 