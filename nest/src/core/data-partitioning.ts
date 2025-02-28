import { v4 as uuidv4 } from 'uuid';
import { Logger } from '../utils/logger';

/**
 * Stratégies de partitionnement des données
 */
export enum PartitioningStrategy {
  RANDOM = 'RANDOM',               // Partitionnement aléatoire
  HASH = 'HASH',                   // Partitionnement par hachage
  RANGE = 'RANGE',                 // Partitionnement par plage de valeurs
  SEMANTIC = 'SEMANTIC',           // Partitionnement sémantique
  TEMPORAL = 'TEMPORAL',           // Partitionnement temporel
  HYBRID = 'HYBRID'                // Combinaison de plusieurs stratégies
}

/**
 * Type de données à partitionner
 */
export enum DataType {
  DOCUMENT = 'DOCUMENT',           // Document textuel
  VECTOR = 'VECTOR',               // Vecteur d'embedding
  KNOWLEDGE_NODE = 'KNOWLEDGE_NODE', // Nœud de connaissance
  QUERY = 'QUERY',                 // Requête utilisateur
  METRIC = 'METRIC',               // Métrique de performance
  EVENT = 'EVENT',                 // Événement système
  CUSTOM = 'CUSTOM'                // Type personnalisé
}

/**
 * Configuration d'une partition
 */
export interface PartitionConfig {
  id: string;                      // Identifiant unique
  name: string;                    // Nom descriptif
  strategy: PartitioningStrategy;  // Stratégie de partitionnement
  dataType: DataType;              // Type de données
  keyField: string;                // Champ utilisé pour le partitionnement
  filter?: (data: any) => boolean; // Filtre optionnel
  capacity?: number;               // Capacité maximale
  metadata?: Record<string, any>;  // Métadonnées
}

/**
 * Statistiques d'une partition
 */
export interface PartitionStats {
  id: string;                      // ID de la partition
  itemCount: number;               // Nombre d'éléments
  sizeBytes: number;               // Taille approximative en octets
  lastAccess: number;              // Dernier accès
  accessCount: number;             // Nombre d'accès
  hitRatio: number;                // Ratio de succès
  creationTime: number;            // Date de création
  avgAccessTime: number;           // Temps d'accès moyen
}

/**
 * Résultat d'une opération sur une partition
 */
export interface PartitionOperationResult<T = any> {
  success: boolean;                // Succès de l'opération
  partitionId: string;             // ID de la partition concernée
  data?: T;                        // Données retournées
  error?: Error;                   // Erreur éventuelle
  metrics: {
    timeMs: number;                // Temps d'exécution
    itemsProcessed: number;        // Éléments traités
  };
}

/**
 * Options pour la recherche dans les partitions
 */
export interface PartitionSearchOptions {
  limit?: number;                  // Limite de résultats
  offset?: number;                 // Décalage
  sortBy?: string;                 // Champ de tri
  sortDirection?: 'asc' | 'desc';  // Direction de tri
  includeMetadata?: boolean;       // Inclure les métadonnées
  filter?: Record<string, any>;    // Filtre
  threshold?: number;              // Seuil de pertinence
}

/**
 * Classe de gestion de partitionnement intelligent
 * Permet de répartir efficacement les données pour optimiser l'accès
 */
export class DataPartitioner {
  private partitions: Map<string, PartitionConfig> = new Map();
  private partitionData: Map<string, Map<string, any>> = new Map();
  private partitionStats: Map<string, PartitionStats> = new Map();
  private partitionsByType: Map<DataType, Set<string>> = new Map();
  private partitionsByStrategy: Map<PartitioningStrategy, Set<string>> = new Map();
  private dataLocationCache: Map<string, string> = new Map();
  private logger: Logger;
  
  /**
   * Crée une instance du gestionnaire de partitionnement
   * @param logger Instance du logger
   */
  constructor(logger: Logger) {
    this.logger = logger;
    
    // Initialiser les maps par type et stratégie
    Object.values(DataType).forEach(type => {
      this.partitionsByType.set(type as DataType, new Set());
    });
    
    Object.values(PartitioningStrategy).forEach(strategy => {
      this.partitionsByStrategy.set(strategy as PartitioningStrategy, new Set());
    });
    
    this.logger.info('Système de partitionnement initialisé');
  }
  
  /**
   * Crée une nouvelle partition
   * @param config Configuration de la partition
   * @returns ID de la partition créée
   */
  public createPartition(config: Omit<PartitionConfig, 'id'>): string {
    const id = uuidv4();
    const completeConfig: PartitionConfig = {
      ...config,
      id
    };
    
    this.partitions.set(id, completeConfig);
    this.partitionData.set(id, new Map());
    this.partitionStats.set(id, {
      id,
      itemCount: 0,
      sizeBytes: 0,
      lastAccess: Date.now(),
      accessCount: 0,
      hitRatio: 0,
      creationTime: Date.now(),
      avgAccessTime: 0
    });
    
    // Ajouter aux index
    this.partitionsByType.get(config.dataType)?.add(id);
    this.partitionsByStrategy.get(config.strategy)?.add(id);
    
    this.logger.info(`Partition créée: ${config.name} (${id})`, {
      strategy: config.strategy,
      dataType: config.dataType
    });
    
    return id;
  }
  
  /**
   * Supprime une partition
   * @param partitionId ID de la partition à supprimer
   * @returns true si supprimée
   */
  public removePartition(partitionId: string): boolean {
    const partition = this.partitions.get(partitionId);
    
    if (!partition) {
      return false;
    }
    
    // Supprimer des index
    this.partitionsByType.get(partition.dataType)?.delete(partitionId);
    this.partitionsByStrategy.get(partition.strategy)?.delete(partitionId);
    
    // Supprimer du cache
    for (const [key, value] of this.dataLocationCache.entries()) {
      if (value === partitionId) {
        this.dataLocationCache.delete(key);
      }
    }
    
    // Supprimer les données et métadonnées
    this.partitions.delete(partitionId);
    this.partitionData.delete(partitionId);
    this.partitionStats.delete(partitionId);
    
    this.logger.info(`Partition supprimée: ${partition.name} (${partitionId})`);
    
    return true;
  }
  
  /**
   * Ajoute un élément à une partition
   * @param partitionId ID de la partition
   * @param data Données à ajouter
   * @param key Clé d'accès (optionnelle)
   * @returns Résultat de l'opération
   */
  public addToPartition(
    partitionId: string,
    data: any,
    key?: string
  ): PartitionOperationResult {
    const startTime = Date.now();
    
    try {
      const partition = this.partitions.get(partitionId);
      
      if (!partition) {
        throw new Error(`Partition non trouvée: ${partitionId}`);
      }
      
      const dataMap = this.partitionData.get(partitionId)!;
      const stats = this.partitionStats.get(partitionId)!;
      
      // Appliquer le filtre si défini
      if (partition.filter && !partition.filter(data)) {
        throw new Error(`Données rejetées par le filtre de la partition ${partition.name}`);
      }
      
      // Générer une clé si non fournie
      const dataKey = key || this.generateKey(partition, data);
      
      // Vérifier la capacité
      if (partition.capacity && dataMap.size >= partition.capacity) {
        throw new Error(`Capacité maximale atteinte pour la partition ${partition.name}`);
      }
      
      // Stocker les données
      dataMap.set(dataKey, data);
      
      // Mettre à jour le cache
      this.dataLocationCache.set(dataKey, partitionId);
      
      // Estimer la taille
      const dataSizeBytes = this.estimateDataSize(data);
      
      // Mettre à jour les statistiques
      stats.itemCount = dataMap.size;
      stats.sizeBytes += dataSizeBytes;
      stats.lastAccess = Date.now();
      stats.accessCount += 1;
      
      const endTime = Date.now();
      
      this.logger.debug(`Élément ajouté à la partition ${partition.name}`, {
        key: dataKey,
        partitionId,
        executionTime: endTime - startTime
      });
      
      return {
        success: true,
        partitionId,
        metrics: {
          timeMs: endTime - startTime,
          itemsProcessed: 1
        }
      };
    } catch (error) {
      const endTime = Date.now();
      
      this.logger.error(`Échec d'ajout à la partition ${partitionId}`, {
        error,
        executionTime: endTime - startTime
      });
      
      return {
        success: false,
        partitionId,
        error: error as Error,
        metrics: {
          timeMs: endTime - startTime,
          itemsProcessed: 0
        }
      };
    }
  }
  
  /**
   * Récupère un élément d'une partition
   * @param key Clé de l'élément
   * @returns Résultat de l'opération
   */
  public get(key: string): PartitionOperationResult {
    const startTime = Date.now();
    
    try {
      // Vérifier le cache
      const partitionId = this.dataLocationCache.get(key);
      
      if (!partitionId) {
        // Chercher dans toutes les partitions
        for (const [id, dataMap] of this.partitionData.entries()) {
          if (dataMap.has(key)) {
            // Mettre à jour le cache
            this.dataLocationCache.set(key, id);
            
            const data = dataMap.get(key);
            this.updateAccessStats(id);
            
            const endTime = Date.now();
            
            return {
              success: true,
              partitionId: id,
              data,
              metrics: {
                timeMs: endTime - startTime,
                itemsProcessed: 1
              }
            };
          }
        }
        
        throw new Error(`Élément non trouvé: ${key}`);
      }
      
      // Récupérer depuis la partition connue
      const dataMap = this.partitionData.get(partitionId);
      
      if (!dataMap) {
        throw new Error(`Partition non trouvée: ${partitionId}`);
      }
      
      const data = dataMap.get(key);
      
      if (data === undefined) {
        throw new Error(`Élément non trouvé dans la partition ${partitionId}: ${key}`);
      }
      
      this.updateAccessStats(partitionId);
      
      const endTime = Date.now();
      
      return {
        success: true,
        partitionId,
        data,
        metrics: {
          timeMs: endTime - startTime,
          itemsProcessed: 1
        }
      };
    } catch (error) {
      const endTime = Date.now();
      
      return {
        success: false,
        partitionId: '',
        error: error as Error,
        metrics: {
          timeMs: endTime - startTime,
          itemsProcessed: 0
        }
      };
    }
  }
  
  /**
   * Recherche dans les partitions d'un type spécifique
   * @param dataType Type de données
   * @param query Requête de recherche
   * @param options Options de recherche
   * @returns Résultat de l'opération
   */
  public search(
    dataType: DataType,
    query: any,
    options: PartitionSearchOptions = {}
  ): PartitionOperationResult<any[]> {
    const startTime = Date.now();
    const results: any[] = [];
    
    try {
      const partitionIds = this.partitionsByType.get(dataType) || new Set<string>();
      
      if (partitionIds.size === 0) {
        return {
          success: true,
          partitionId: 'multiple',
          data: [],
          metrics: {
            timeMs: Date.now() - startTime,
            itemsProcessed: 0
          }
        };
      }
      
      let itemsProcessed = 0;
      
      // Rechercher dans chaque partition
      for (const partitionId of partitionIds) {
        const partition = this.partitions.get(partitionId)!;
        const dataMap = this.partitionData.get(partitionId)!;
        
        // Appliquer les filtres
        for (const [key, value] of dataMap.entries()) {
          itemsProcessed++;
          
          // Filtre de base par type de données
          if (options.filter) {
            let match = true;
            
            // Vérifier tous les critères du filtre
            for (const [filterKey, filterValue] of Object.entries(options.filter)) {
              // Accéder aux propriétés imbriquées avec notation par points
              const actualValue = this.getNestedProperty(value, filterKey);
              
              if (actualValue !== filterValue) {
                match = false;
                break;
              }
            }
            
            if (!match) continue;
          }
          
          // Appliquer la fonction de filtrage de la partition
          if (partition.filter && !partition.filter(value)) {
            continue;
          }
          
          results.push(value);
        }
        
        // Mettre à jour les statistiques
        this.updateAccessStats(partitionId);
      }
      
      // Trier les résultats
      if (options.sortBy) {
        results.sort((a, b) => {
          const aValue = this.getNestedProperty(a, options.sortBy!);
          const bValue = this.getNestedProperty(b, options.sortBy!);
          
          if (options.sortDirection === 'desc') {
            return bValue > aValue ? 1 : bValue < aValue ? -1 : 0;
          } else {
            return aValue > bValue ? 1 : aValue < bValue ? -1 : 0;
          }
        });
      }
      
      // Appliquer la pagination
      let paginatedResults = results;
      
      if (options.offset !== undefined || options.limit !== undefined) {
        const offset = options.offset || 0;
        const limit = options.limit || results.length;
        
        paginatedResults = results.slice(offset, offset + limit);
      }
      
      const endTime = Date.now();
      
      this.logger.debug(`Recherche effectuée sur les partitions de type ${dataType}`, {
        resultCount: paginatedResults.length,
        totalProcessed: itemsProcessed,
        executionTime: endTime - startTime
      });
      
      return {
        success: true,
        partitionId: 'multiple',
        data: paginatedResults,
        metrics: {
          timeMs: endTime - startTime,
          itemsProcessed
        }
      };
    } catch (error) {
      const endTime = Date.now();
      
      this.logger.error(`Erreur lors de la recherche dans les partitions de type ${dataType}`, {
        error,
        executionTime: endTime - startTime
      });
      
      return {
        success: false,
        partitionId: 'multiple',
        error: error as Error,
        metrics: {
          timeMs: endTime - startTime,
          itemsProcessed: 0
        }
      };
    }
  }
  
  /**
   * Supprime un élément d'une partition
   * @param key Clé de l'élément
   * @returns Résultat de l'opération
   */
  public remove(key: string): PartitionOperationResult {
    const startTime = Date.now();
    
    try {
      // Vérifier le cache
      const partitionId = this.dataLocationCache.get(key);
      
      if (!partitionId) {
        // Chercher dans toutes les partitions
        for (const [id, dataMap] of this.partitionData.entries()) {
          if (dataMap.has(key)) {
            const data = dataMap.get(key);
            
            // Supprimer les données
            dataMap.delete(key);
            this.dataLocationCache.delete(key);
            
            // Mettre à jour les statistiques
            const stats = this.partitionStats.get(id)!;
            stats.itemCount = dataMap.size;
            stats.sizeBytes -= this.estimateDataSize(data);
            
            const endTime = Date.now();
            
            this.logger.debug(`Élément supprimé de la partition ${id}`, {
              key,
              executionTime: endTime - startTime
            });
            
            return {
              success: true,
              partitionId: id,
              data,
              metrics: {
                timeMs: endTime - startTime,
                itemsProcessed: 1
              }
            };
          }
        }
        
        throw new Error(`Élément non trouvé: ${key}`);
      }
      
      // Supprimer depuis la partition connue
      const dataMap = this.partitionData.get(partitionId);
      
      if (!dataMap) {
        throw new Error(`Partition non trouvée: ${partitionId}`);
      }
      
      const data = dataMap.get(key);
      
      if (data === undefined) {
        throw new Error(`Élément non trouvé dans la partition ${partitionId}: ${key}`);
      }
      
      // Supprimer les données
      dataMap.delete(key);
      this.dataLocationCache.delete(key);
      
      // Mettre à jour les statistiques
      const stats = this.partitionStats.get(partitionId)!;
      stats.itemCount = dataMap.size;
      stats.sizeBytes -= this.estimateDataSize(data);
      
      const endTime = Date.now();
      
      this.logger.debug(`Élément supprimé de la partition ${partitionId}`, {
        key,
        executionTime: endTime - startTime
      });
      
      return {
        success: true,
        partitionId,
        data,
        metrics: {
          timeMs: endTime - startTime,
          itemsProcessed: 1
        }
      };
    } catch (error) {
      const endTime = Date.now();
      
      return {
        success: false,
        partitionId: '',
        error: error as Error,
        metrics: {
          timeMs: endTime - startTime,
          itemsProcessed: 0
        }
      };
    }
  }
  
  /**
   * Récupère les statistiques d'une partition
   * @param partitionId ID de la partition
   * @returns Statistiques ou null si non trouvée
   */
  public getPartitionStats(partitionId: string): PartitionStats | null {
    const stats = this.partitionStats.get(partitionId);
    return stats ? { ...stats } : null;
  }
  
  /**
   * Récupère les statistiques pour toutes les partitions
   * @returns Map des statistiques par ID de partition
   */
  public getAllPartitionStats(): Record<string, PartitionStats> {
    const result: Record<string, PartitionStats> = {};
    
    for (const [id, stats] of this.partitionStats.entries()) {
      result[id] = { ...stats };
    }
    
    return result;
  }
  
  /**
   * Réorganise les données en fonction des statistiques d'accès
   * @returns Nombre de partitions réorganisées
   */
  public rebalance(): number {
    const startTime = Date.now();
    let reorganizedCount = 0;
    
    this.logger.info('Début de la réorganisation des partitions');
    
    // Identifier les partitions à réorganiser
    const allStats = Array.from(this.partitionStats.entries())
      .map(([id, stats]) => ({ id, stats }))
      .sort((a, b) => a.stats.hitRatio - b.stats.hitRatio);
    
    // Réorganiser les partitions avec le plus faible taux de succès
    for (const { id, stats } of allStats.slice(0, Math.ceil(allStats.length * 0.2))) {
      const partition = this.partitions.get(id);
      
      if (!partition) continue;
      
      // Réduire la capacité des partitions peu utilisées
      if (partition.capacity && stats.itemCount < partition.capacity * 0.5) {
        this.logger.debug(`Réduction de la capacité de la partition ${partition.name}`, {
          oldCapacity: partition.capacity,
          newCapacity: Math.floor(partition.capacity * 0.8)
        });
        
        partition.capacity = Math.floor(partition.capacity * 0.8);
        reorganizedCount++;
      }
    }
    
    const endTime = Date.now();
    
    this.logger.info(`Réorganisation terminée: ${reorganizedCount} partitions modifiées`, {
      executionTime: endTime - startTime
    });
    
    return reorganizedCount;
  }
  
  /**
   * Génère une clé pour les données en fonction de la stratégie de partitionnement
   * @param partition Configuration de la partition
   * @param data Données à stocker
   * @returns Clé générée
   */
  private generateKey(partition: PartitionConfig, data: any): string {
    // Obtenir la valeur du champ de clé
    const keyValue = this.getNestedProperty(data, partition.keyField);
    
    if (keyValue === undefined) {
      throw new Error(`Champ de clé ${partition.keyField} non trouvé dans les données`);
    }
    
    switch (partition.strategy) {
      case PartitioningStrategy.HASH:
        // Hachage simple
        return `${partition.id}:${this.simpleHash(keyValue.toString())}`;
        
      case PartitioningStrategy.RANGE:
        // Plage numérique
        if (typeof keyValue === 'number') {
          return `${partition.id}:range:${keyValue}`;
        }
        throw new Error('La stratégie RANGE nécessite une valeur numérique');
        
      case PartitioningStrategy.TEMPORAL:
        // Date/heure
        if (keyValue instanceof Date || typeof keyValue === 'number') {
          const timestamp = keyValue instanceof Date ? keyValue.getTime() : keyValue;
          return `${partition.id}:time:${timestamp}`;
        }
        throw new Error('La stratégie TEMPORAL nécessite une date ou un timestamp');
        
      case PartitioningStrategy.SEMANTIC:
        // Préfixe sémantique + valeur
        return `${partition.id}:sem:${keyValue.toString()}`;
        
      case PartitioningStrategy.RANDOM:
      default:
        // Clé aléatoire
        return `${partition.id}:${uuidv4()}`;
    }
  }
  
  /**
   * Met à jour les statistiques d'accès à une partition
   * @param partitionId ID de la partition
   */
  private updateAccessStats(partitionId: string): void {
    const stats = this.partitionStats.get(partitionId);
    
    if (stats) {
      const now = Date.now();
      const accessTime = now - stats.lastAccess;
      
      stats.accessCount += 1;
      stats.lastAccess = now;
      
      // Moyenne mobile des temps d'accès
      stats.avgAccessTime = stats.avgAccessTime === 0
        ? accessTime
        : stats.avgAccessTime * 0.9 + accessTime * 0.1;
        
      // Calculer le ratio de succès (cache hit ratio)
      const totalLookups = this.dataLocationCache.size;
      const partitionLookups = Array.from(this.dataLocationCache.values())
        .filter(id => id === partitionId).length;
        
      stats.hitRatio = totalLookups > 0 ? partitionLookups / totalLookups : 0;
    }
  }
  
  /**
   * Estime la taille en octets des données
   * @param data Données à estimer
   * @returns Taille estimée en octets
   */
  private estimateDataSize(data: any): number {
    if (data === null || data === undefined) {
      return 0;
    }
    
    if (typeof data === 'string') {
      return data.length * 2; // UTF-16
    }
    
    if (typeof data === 'number') {
      return 8; // Double
    }
    
    if (typeof data === 'boolean') {
      return 1;
    }
    
    if (data instanceof Date) {
      return 8;
    }
    
    if (Array.isArray(data)) {
      return data.reduce((size, item) => size + this.estimateDataSize(item), 0);
    }
    
    if (typeof data === 'object') {
      let size = 0;
      for (const key in data) {
        if (Object.prototype.hasOwnProperty.call(data, key)) {
          size += key.length * 2; // Clé
          size += this.estimateDataSize(data[key]); // Valeur
        }
      }
      return size;
    }
    
    return 8; // Par défaut
  }
  
  /**
   * Accède à une propriété imbriquée en utilisant la notation par points
   * @param obj Objet
   * @param path Chemin d'accès (e.g. "user.address.city")
   * @returns Valeur ou undefined si non trouvée
   */
  private getNestedProperty(obj: any, path: string): any {
    return path.split('.').reduce((prev, curr) => {
      return prev ? prev[curr] : undefined;
    }, obj);
  }
  
  /**
   * Fonction de hachage simple
   * @param str Chaîne à hacher
   * @returns Valeur de hachage
   */
  private simpleHash(str: string): string {
    let hash = 0;
    
    for (let i = 0; i < str.length; i++) {
      const char = str.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash; // Conversion en 32 bits
    }
    
    return Math.abs(hash).toString(16);
  }
  
  /**
   * Récupère les configurations de toutes les partitions
   * @returns Configurations des partitions
   */
  public getAllPartitions(): Record<string, PartitionConfig> {
    const result: Record<string, PartitionConfig> = {};
    
    for (const [id, config] of this.partitions.entries()) {
      // Omettre la fonction de filtre pour la sérialisation
      const { filter, ...rest } = config;
      result[id] = rest as PartitionConfig;
    }
    
    return result;
  }
  
  /**
   * Récupère les statistiques globales du système de partitionnement
   * @returns Statistiques globales
   */
  public getGlobalStats(): Record<string, any> {
    const totalItems = Array.from(this.partitionStats.values())
      .reduce((sum, stats) => sum + stats.itemCount, 0);
      
    const totalSize = Array.from(this.partitionStats.values())
      .reduce((sum, stats) => sum + stats.sizeBytes, 0);
      
    return {
      partitionCount: this.partitions.size,
      totalItems,
      totalSizeBytes: totalSize,
      cacheSize: this.dataLocationCache.size,
      partitionsByType: Object.fromEntries(
        Array.from(this.partitionsByType.entries())
          .map(([type, ids]) => [type, ids.size])
      ),
      partitionsByStrategy: Object.fromEntries(
        Array.from(this.partitionsByStrategy.entries())
          .map(([strategy, ids]) => [strategy, ids.size])
      )
    };
  }
  
  /**
   * Nettoie la mémoire et les ressources
   */
  public clear(): void {
    this.partitions.clear();
    this.partitionData.clear();
    this.partitionStats.clear();
    this.dataLocationCache.clear();
    
    Object.values(DataType).forEach(type => {
      this.partitionsByType.set(type as DataType, new Set());
    });
    
    Object.values(PartitioningStrategy).forEach(strategy => {
      this.partitionsByStrategy.set(strategy as PartitioningStrategy, new Set());
    });
    
    this.logger.info('Système de partitionnement réinitialisé');
  }
} 