import { v4 as uuidv4 } from 'uuid';
import { Logger } from '../utils/logger';

/**
 * Types d'événements du système
 */
export enum SystemEventType {
  // Événements du cycle de vie
  SYSTEM_INIT = 'SYSTEM_INIT',
  SYSTEM_READY = 'SYSTEM_READY',
  SYSTEM_SHUTDOWN = 'SYSTEM_SHUTDOWN',
  
  // Événements des composants
  COMPONENT_REGISTERED = 'COMPONENT_REGISTERED',
  COMPONENT_UNREGISTERED = 'COMPONENT_UNREGISTERED',
  COMPONENT_ENABLED = 'COMPONENT_ENABLED',
  COMPONENT_DISABLED = 'COMPONENT_DISABLED',
  COMPONENT_ERROR = 'COMPONENT_ERROR',
  
  // Événements de traitement
  QUERY_RECEIVED = 'QUERY_RECEIVED',
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
  
  // Événements de telemetrie
  PERFORMANCE_METRIC = 'PERFORMANCE_METRIC',
  RESOURCE_USAGE = 'RESOURCE_USAGE',
  
  // Événements de configuration
  CONFIG_CHANGED = 'CONFIG_CHANGED',
  
  // Événements génériques
  CUSTOM = 'CUSTOM'
}

/**
 * Interface de base pour tous les événements du système
 */
export interface SystemEvent {
  id: string;                    // ID unique de l'événement
  type: SystemEventType;         // Type d'événement
  timestamp: number;             // Horodatage de création
  source: string;                // Source de l'événement (composant émetteur)
  payload?: Record<string, any>; // Contenu de l'événement
  metadata?: Record<string, any>; // Métadonnées additionnelles
}

/**
 * Interface pour les écouteurs d'événements
 */
export interface EventListener<T extends SystemEvent = SystemEvent> {
  (event: T): void | Promise<void>;
}

/**
 * Options pour l'abonnement à des événements
 */
export interface SubscriptionOptions {
  once?: boolean;              // Si true, se désabonne après la première exécution
  filter?: (event: SystemEvent) => boolean; // Filtre supplémentaire pour les événements
  priority?: number;           // Priorité de l'écouteur (plus élevée = exécution plus tôt)
  async?: boolean;             // Si true, exécute l'écouteur de manière asynchrone
  timeout?: number;            // Timeout en ms pour l'exécution (async uniquement)
}

/**
 * Structure interne pour stocker les abonnements
 */
interface Subscription {
  id: string;
  eventType: SystemEventType;
  listener: EventListener;
  options: Required<SubscriptionOptions>;
}

/**
 * Bus d'événements central pour la communication inter-composants
 */
export class EventBus {
  private subscriptions: Map<string, Subscription> = new Map();
  private eventTypeSubscriptions: Map<SystemEventType, Set<string>> = new Map();
  private logger: Logger;
  private enabled: boolean = true;
  private eventHistory: SystemEvent[] = [];
  private maxHistorySize: number = 100;
  
  /**
   * Crée une nouvelle instance du bus d'événements
   * @param logger Instance du logger
   */
  constructor(logger: Logger) {
    this.logger = logger;
    
    // Initialiser les maps pour chaque type d'événement
    Object.values(SystemEventType).forEach(eventType => {
      this.eventTypeSubscriptions.set(eventType as SystemEventType, new Set());
    });
    
    this.logger.info('Bus d\'événements initialisé');
  }
  
  /**
   * S'abonne à un type d'événement spécifique
   * @param eventType Type d'événement
   * @param listener Fonction à appeler lors de l'émission
   * @param options Options d'abonnement
   * @returns ID de l'abonnement (pour se désabonner)
   */
  public subscribe<T extends SystemEvent>(
    eventType: SystemEventType,
    listener: EventListener<T>,
    options: SubscriptionOptions = {}
  ): string {
    const id = uuidv4();
    
    // Valeurs par défaut pour les options
    const completeOptions: Required<SubscriptionOptions> = {
      once: options.once || false,
      filter: options.filter || (() => true),
      priority: options.priority || 0,
      async: options.async || false,
      timeout: options.timeout || 5000
    };
    
    // Stocker l'abonnement
    this.subscriptions.set(id, {
      id,
      eventType,
      listener: listener as EventListener,
      options: completeOptions
    });
    
    // Ajouter à la map par type d'événement
    this.eventTypeSubscriptions.get(eventType)?.add(id);
    
    this.logger.debug(`Abonnement à l'événement ${eventType}`, {
      subscriptionId: id,
      options: completeOptions
    });
    
    return id;
  }
  
  /**
   * S'abonne à un événement et se désabonne après la première exécution
   * @param eventType Type d'événement
   * @param listener Fonction à appeler
   * @param options Options d'abonnement
   * @returns ID de l'abonnement
   */
  public once<T extends SystemEvent>(
    eventType: SystemEventType,
    listener: EventListener<T>,
    options: Omit<SubscriptionOptions, 'once'> = {}
  ): string {
    return this.subscribe(eventType, listener, { ...options, once: true });
  }
  
  /**
   * Se désabonne d'un événement
   * @param subscriptionId ID de l'abonnement
   * @returns true si l'abonnement a été trouvé et supprimé
   */
  public unsubscribe(subscriptionId: string): boolean {
    const subscription = this.subscriptions.get(subscriptionId);
    
    if (!subscription) {
      return false;
    }
    
    // Supprimer de la map principale
    this.subscriptions.delete(subscriptionId);
    
    // Supprimer de la map par type
    this.eventTypeSubscriptions.get(subscription.eventType)?.delete(subscriptionId);
    
    this.logger.debug(`Désabonnement de l'événement ${subscription.eventType}`, {
      subscriptionId
    });
    
    return true;
  }
  
  /**
   * Émet un événement
   * @param event Événement à émettre
   * @returns true si le bus est activé et que des écouteurs ont été notifiés
   */
  public emit<T extends SystemEvent>(event: Omit<T, 'id' | 'timestamp'>): boolean {
    if (!this.enabled) {
      this.logger.debug('Émission ignorée: bus d\'événements désactivé');
      return false;
    }
    
    // Compléter l'événement
    const completeEvent: SystemEvent = {
      ...event,
      id: uuidv4(),
      timestamp: Date.now()
    } as SystemEvent;
    
    // Ajouter à l'historique
    this.addToHistory(completeEvent);
    
    // Récupérer les abonnements pour ce type d'événement
    const subscriptionIds = this.eventTypeSubscriptions.get(event.type) || new Set<string>();
    
    if (subscriptionIds.size === 0) {
      this.logger.debug(`Aucun écouteur pour l'événement ${event.type}`);
      return false;
    }
    
    // Récupérer les abonnements réels
    const activeSubscriptions = Array.from(subscriptionIds)
      .map(id => this.subscriptions.get(id))
      .filter(sub => sub !== undefined) as Subscription[];
    
    // Trier par priorité (décroissante)
    activeSubscriptions.sort((a, b) => b.options.priority - a.options.priority);
    
    // Notifier les écouteurs
    activeSubscriptions.forEach(subscription => {
      // Vérifier le filtre
      if (!subscription.options.filter(completeEvent)) {
        return;
      }
      
      try {
        // Désabonner si 'once'
        if (subscription.options.once) {
          this.unsubscribe(subscription.id);
        }
        
        // Exécuter l'écouteur
        if (subscription.options.async) {
          // Exécution asynchrone avec timeout
          this.executeAsync(subscription, completeEvent);
        } else {
          // Exécution synchrone
          subscription.listener(completeEvent);
        }
      } catch (error) {
        this.logger.error(`Erreur lors de l'exécution de l'écouteur pour ${event.type}`, {
          subscriptionId: subscription.id,
          error
        });
      }
    });
    
    this.logger.debug(`Événement ${event.type} émis`, {
      eventId: completeEvent.id,
      listenersCount: activeSubscriptions.length
    });
    
    return true;
  }
  
  /**
   * Exécute un écouteur de manière asynchrone avec timeout
   * @param subscription Abonnement à exécuter
   * @param event Événement à traiter
   */
  private executeAsync(subscription: Subscription, event: SystemEvent): void {
    Promise.race([
      Promise.resolve().then(() => subscription.listener(event)),
      new Promise((_, reject) => 
        setTimeout(() => reject(new Error('Timeout')), subscription.options.timeout)
      )
    ]).catch(error => {
      this.logger.error(`Erreur asynchrone lors de l'exécution de l'écouteur pour ${event.type}`, {
        subscriptionId: subscription.id,
        error
      });
    });
  }
  
  /**
   * Ajoute un événement à l'historique
   * @param event Événement à ajouter
   */
  private addToHistory(event: SystemEvent): void {
    this.eventHistory.push(event);
    
    // Limiter la taille de l'historique
    if (this.eventHistory.length > this.maxHistorySize) {
      this.eventHistory.shift();
    }
  }
  
  /**
   * Récupère l'historique des événements
   * @param limit Nombre maximum d'événements à récupérer
   * @param filter Filtre optionnel
   * @returns Liste des événements
   */
  public getHistory(limit?: number, filter?: (event: SystemEvent) => boolean): SystemEvent[] {
    let events = [...this.eventHistory];
    
    // Appliquer le filtre si présent
    if (filter) {
      events = events.filter(filter);
    }
    
    // Limiter le nombre d'événements
    if (limit && limit > 0 && limit < events.length) {
      events = events.slice(events.length - limit);
    }
    
    return events;
  }
  
  /**
   * Récupère les derniers événements d'un type spécifique
   * @param eventType Type d'événement
   * @param limit Nombre maximum d'événements
   * @returns Liste des événements
   */
  public getEventsByType(eventType: SystemEventType, limit: number = 10): SystemEvent[] {
    return this.getHistory(limit, event => event.type === eventType);
  }
  
  /**
   * Attend qu'un événement spécifique soit émis
   * @param eventType Type d'événement à attendre
   * @param timeout Timeout en ms
   * @param filter Filtre optionnel
   * @returns Promesse résolue avec l'événement ou rejetée si timeout
   */
  public waitForEvent<T extends SystemEvent>(
    eventType: SystemEventType,
    timeout: number = 10000,
    filter?: (event: T) => boolean
  ): Promise<T> {
    return new Promise<T>((resolve, reject) => {
      // Gérer le timeout
      const timeoutId = setTimeout(() => {
        this.unsubscribe(subscriptionId);
        reject(new Error(`Timeout en attendant l'événement ${eventType}`));
      }, timeout);
      
      // S'abonner à l'événement
      const subscriptionId = this.once<T>(
        eventType,
        (event: T) => {
          clearTimeout(timeoutId);
          resolve(event);
        },
        {
          filter: filter as ((event: SystemEvent) => boolean) | undefined,
          priority: 100 // Priorité élevée
        }
      );
    });
  }
  
  /**
   * Active ou désactive le bus d'événements
   * @param enabled État souhaité
   */
  public setEnabled(enabled: boolean): void {
    this.enabled = enabled;
    this.logger.info(`Bus d'événements ${enabled ? 'activé' : 'désactivé'}`);
  }
  
  /**
   * Supprime tous les abonnements
   */
  public clearSubscriptions(): void {
    this.subscriptions.clear();
    
    // Réinitialiser les maps par type
    Object.values(SystemEventType).forEach(eventType => {
      this.eventTypeSubscriptions.set(eventType as SystemEventType, new Set());
    });
    
    this.logger.info('Tous les abonnements ont été supprimés');
  }
  
  /**
   * Supprime l'historique des événements
   */
  public clearHistory(): void {
    this.eventHistory = [];
    this.logger.info('Historique des événements effacé');
  }
  
  /**
   * Configure la taille maximale de l'historique
   * @param size Nouvelle taille maximale
   */
  public setMaxHistorySize(size: number): void {
    if (size < 0) {
      throw new Error('La taille maximale de l\'historique ne peut pas être négative');
    }
    
    this.maxHistorySize = size;
    
    // Réduire l'historique si nécessaire
    while (this.eventHistory.length > this.maxHistorySize) {
      this.eventHistory.shift();
    }
    
    this.logger.debug(`Taille maximale de l'historique définie à ${size} événements`);
  }
  
  /**
   * Crée un événement personnalisé
   * @param source Source de l'événement
   * @param payload Données de l'événement
   * @param metadata Métadonnées additionnelles
   * @returns Événement créé
   */
  public static createCustomEvent(
    source: string,
    payload: Record<string, any>,
    metadata?: Record<string, any>
  ): SystemEvent {
    return {
      id: uuidv4(),
      type: SystemEventType.CUSTOM,
      timestamp: Date.now(),
      source,
      payload,
      metadata
    };
  }
} 