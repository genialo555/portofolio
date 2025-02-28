import { v4 as uuidv4 } from 'uuid';
import { Logger } from '../utils/logger';

/**
 * Types de relations entre les nœuds du graphe
 */
export enum RelationType {
  IS_A = 'IS_A',                   // Relation de type 'est un'
  PART_OF = 'PART_OF',             // Relation de partie à tout
  RELATED_TO = 'RELATED_TO',       // Relation générique
  CAUSES = 'CAUSES',               // Relation causale
  SIMILAR_TO = 'SIMILAR_TO',       // Similarité conceptuelle
  DIFFERENT_FROM = 'DIFFERENT_FROM', // Différence conceptuelle
  DEPENDS_ON = 'DEPENDS_ON',       // Dépendance
  LEADS_TO = 'LEADS_TO',           // Succession logique
  INSTANCE_OF = 'INSTANCE_OF',     // Instance d'un concept
  PRECEDES = 'PRECEDES',           // Précédence temporelle
  CUSTOM = 'CUSTOM'                // Relation personnalisée
}

/**
 * Source d'une connaissance
 */
export enum KnowledgeSource {
  RAG = 'RAG',                     // Retrieval Augmented Generation
  KAG = 'KAG',                     // Knowledge Augmented Generation
  USER_INPUT = 'USER_INPUT',       // Entrée utilisateur
  INFERENCE = 'INFERENCE',         // Inférence logique
  EXTERNAL_API = 'EXTERNAL_API',   // API externe
  SYSTEM = 'SYSTEM'                // Généré par le système
}

/**
 * Structure d'un nœud de connaissance
 */
export interface KnowledgeNode {
  id: string;                      // Identifiant unique
  label: string;                   // Libellé du nœud
  type: string;                    // Type de nœud
  content: string;                 // Contenu textuel
  metadata?: Record<string, any>;  // Métadonnées
  confidence: number;              // Score de confiance (0-1)
  source: KnowledgeSource;         // Source de la connaissance
  timestamp: number;               // Horodatage de création
  lastUpdated?: number;            // Dernier mise à jour
  version?: number;                // Version du nœud
}

/**
 * Structure d'une relation entre nœuds
 */
export interface KnowledgeEdge {
  id: string;                      // Identifiant unique
  sourceId: string;                // ID du nœud source
  targetId: string;                // ID du nœud cible
  type: RelationType;              // Type de relation
  label?: string;                  // Libellé de la relation
  weight: number;                  // Poids de la relation (0-1)
  metadata?: Record<string, any>;  // Métadonnées
  confidence: number;              // Score de confiance (0-1)
  bidirectional: boolean;          // Si la relation est bidirectionnelle
  timestamp: number;               // Horodatage de création
}

/**
 * Options de recherche dans le graphe
 */
export interface GraphSearchOptions {
  maxDepth?: number;               // Profondeur maximale de recherche
  minConfidence?: number;          // Confiance minimale des nœuds
  relationTypes?: RelationType[];  // Types de relations à suivre
  nodeTypes?: string[];            // Types de nœuds à inclure
  maxResults?: number;             // Nombre maximum de résultats
  sortByRelevance?: boolean;       // Trier par pertinence
  includeMetadata?: boolean;       // Inclure les métadonnées
  includeRelations?: boolean;      // Inclure les relations
}

/**
 * Résultat d'une recherche dans le graphe
 */
export interface GraphSearchResult {
  nodes: KnowledgeNode[];          // Nœuds trouvés
  edges?: KnowledgeEdge[];         // Relations entre ces nœuds
  relevanceScores?: Map<string, number>; // Scores de pertinence par nœud
  queryNodes?: string[];           // Nœuds correspondant à la requête
  metainfo?: {
    totalNodesExplored: number;    // Nombre total de nœuds explorés
    searchDepth: number;           // Profondeur de recherche atteinte
    executionTimeMs: number;       // Temps d'exécution en ms
  };
}

/**
 * Graphe de connaissances
 * Structure de données centrale pour stocker et interroger les connaissances
 */
export class KnowledgeGraph {
  private nodes: Map<string, KnowledgeNode> = new Map();
  private edges: Map<string, KnowledgeEdge> = new Map();
  private nodeEdges: Map<string, Set<string>> = new Map();
  private logger: Logger;
  private embeddings: Map<string, number[]> = new Map(); // Simuler les embeddings
  
  /**
   * Crée une instance de graphe de connaissances
   * @param logger Instance du logger
   */
  constructor(logger: Logger) {
    this.logger = logger;
    this.logger.info('Graphe de connaissances initialisé');
  }
  
  /**
   * Ajoute un nœud de connaissance au graphe
   * @param node Nœud à ajouter
   * @returns ID du nœud ajouté
   */
  public addNode(node: Omit<KnowledgeNode, 'id' | 'timestamp'>): string {
    const id = uuidv4();
    const timestamp = Date.now();
    
    const completeNode: KnowledgeNode = {
      ...node,
      id,
      timestamp
    };
    
    this.nodes.set(id, completeNode);
    this.nodeEdges.set(id, new Set());
    
    this.logger.debug(`Nœud ajouté au graphe: ${node.label} (${id})`, {
      nodeType: node.type,
      nodeSource: node.source
    });
    
    return id;
  }
  
  /**
   * Ajoute une relation entre deux nœuds
   * @param edge Relation à ajouter
   * @returns ID de la relation ajoutée
   */
  public addEdge(edge: Omit<KnowledgeEdge, 'id' | 'timestamp'>): string {
    // Vérifier que les nœuds existent
    if (!this.nodes.has(edge.sourceId) || !this.nodes.has(edge.targetId)) {
      throw new Error(`Impossible d'ajouter la relation: un des nœuds n'existe pas`);
    }
    
    const id = uuidv4();
    const timestamp = Date.now();
    
    const completeEdge: KnowledgeEdge = {
      ...edge,
      id,
      timestamp
    };
    
    this.edges.set(id, completeEdge);
    
    // Ajouter la référence de l'arête aux nœuds
    this.nodeEdges.get(edge.sourceId)!.add(id);
    this.nodeEdges.get(edge.targetId)!.add(id);
    
    this.logger.debug(`Relation ajoutée: ${this.nodes.get(edge.sourceId)!.label} → ${this.nodes.get(edge.targetId)!.label}`, {
      relationType: edge.type,
      bidirectional: edge.bidirectional
    });
    
    return id;
  }
  
  /**
   * Ajoute un fait sous forme de triplet sujet-prédicat-objet
   * @param subject Sujet (nœud existant ou label pour nouveau nœud)
   * @param predicate Type de relation
   * @param object Objet (nœud existant ou label pour nouveau nœud)
   * @param confidence Score de confiance (0-1)
   * @param options Options supplémentaires
   * @returns IDs des nœuds et de la relation créés
   */
  public addFact(
    subject: string | KnowledgeNode,
    predicate: RelationType | string,
    object: string | KnowledgeNode,
    confidence: number = 0.8,
    options: {
      bidirectional?: boolean;
      weight?: number;
      source?: KnowledgeSource;
      metadata?: Record<string, any>;
    } = {}
  ): { subjectId: string; objectId: string; edgeId: string } {
    // Déterminer le type de relation
    const relationType = Object.values(RelationType).includes(predicate as RelationType)
      ? predicate as RelationType
      : RelationType.CUSTOM;
      
    // Options par défaut
    const {
      bidirectional = false,
      weight = 1.0,
      source = KnowledgeSource.SYSTEM,
      metadata = {}
    } = options;
    
    // Gérer le sujet (existant ou nouveau)
    let subjectId: string;
    if (typeof subject === 'string') {
      // Chercher un nœud existant avec ce label
      const existingSubject = Array.from(this.nodes.values())
        .find(n => n.label.toLowerCase() === subject.toLowerCase());
        
      if (existingSubject) {
        subjectId = existingSubject.id;
      } else {
        // Créer un nouveau nœud
        subjectId = this.addNode({
          label: subject,
          type: 'concept',
          content: subject,
          confidence,
          source
        });
      }
    } else {
      subjectId = subject.id;
    }
    
    // Gérer l'objet (existant ou nouveau)
    let objectId: string;
    if (typeof object === 'string') {
      // Chercher un nœud existant avec ce label
      const existingObject = Array.from(this.nodes.values())
        .find(n => n.label.toLowerCase() === object.toLowerCase());
        
      if (existingObject) {
        objectId = existingObject.id;
      } else {
        // Créer un nouveau nœud
        objectId = this.addNode({
          label: object,
          type: 'concept',
          content: object,
          confidence,
          source
        });
      }
    } else {
      objectId = object.id;
    }
    
    // Créer la relation
    let relationLabel = typeof predicate === 'string' ? predicate : undefined;
    if (relationType === RelationType.CUSTOM && !relationLabel) {
      relationLabel = 'custom_relation';
    }
    
    const edgeId = this.addEdge({
      sourceId: subjectId,
      targetId: objectId,
      type: relationType,
      label: relationLabel,
      weight,
      confidence,
      bidirectional,
      metadata
    });
    
    return { subjectId, objectId, edgeId };
  }
  
  /**
   * Recherche les nœuds correspondant à une requête textuelle
   * @param query Texte de la requête
   * @param options Options de recherche
   * @returns Résultat de la recherche
   */
  public search(query: string, options: GraphSearchOptions = {}): GraphSearchResult {
    const startTime = Date.now();
    const {
      maxDepth = 2,
      minConfidence = 0.5,
      relationTypes,
      nodeTypes,
      maxResults = 10,
      sortByRelevance = true,
      includeMetadata = false,
      includeRelations = true
    } = options;
    
    // Dans une implémentation réelle, utiliser un embedding pour la recherche sémantique
    // Ici on simule par une recherche textuelle simple
    const queryTokens = query.toLowerCase().split(/\s+/);
    
    // Trouver les nœuds correspondant directement à la requête
    const directMatches = Array.from(this.nodes.values())
      .filter(node => 
        node.confidence >= minConfidence && 
        (!nodeTypes || nodeTypes.includes(node.type)) &&
        (
          queryTokens.some(token => 
            node.label.toLowerCase().includes(token) || 
            node.content.toLowerCase().includes(token)
          )
        )
      );
      
    // IDs des nœuds de requête directs
    const directMatchIds = directMatches.map(n => n.id);
    
    // Explorer le graphe à partir des correspondances directes
    const exploredNodes = new Set<string>(directMatchIds);
    const resultNodes = new Map<string, KnowledgeNode>();
    const resultEdges = new Set<string>();
    let totalExplored = directMatchIds.length;
    
    // Ajouter les correspondances directes au résultat
    directMatches.forEach(node => {
      if (!includeMetadata) {
        // eslint-disable-next-line @typescript-eslint/no-unused-vars
        const { metadata, ...rest } = node;
        resultNodes.set(node.id, rest as KnowledgeNode);
      } else {
        resultNodes.set(node.id, node);
      }
    });
    
    // Explorer le graphe en largeur d'abord (BFS)
    let currentDepth = 0;
    let currentFrontier = directMatchIds;
    
    while (currentDepth < maxDepth && currentFrontier.length > 0) {
      currentDepth++;
      const nextFrontier: string[] = [];
      
      for (const nodeId of currentFrontier) {
        const nodeEdgeIds = this.nodeEdges.get(nodeId) || new Set<string>();
        
        for (const edgeId of nodeEdgeIds) {
          const edge = this.edges.get(edgeId)!;
          
          // Filtrer par type de relation si spécifié
          if (relationTypes && !relationTypes.includes(edge.type)) {
            continue;
          }
          
          // Déterminer l'autre extrémité de l'arête
          const otherNodeId = edge.sourceId === nodeId ? edge.targetId : edge.sourceId;
          
          if (!exploredNodes.has(otherNodeId)) {
            totalExplored++;
            exploredNodes.add(otherNodeId);
            
            const otherNode = this.nodes.get(otherNodeId)!;
            
            // Filtrer par confiance et type si spécifié
            if (
              otherNode.confidence >= minConfidence && 
              (!nodeTypes || nodeTypes.includes(otherNode.type))
            ) {
              if (!includeMetadata) {
                // eslint-disable-next-line @typescript-eslint/no-unused-vars
                const { metadata, ...rest } = otherNode;
                resultNodes.set(otherNodeId, rest as KnowledgeNode);
              } else {
                resultNodes.set(otherNodeId, otherNode);
              }
              
              resultEdges.add(edgeId);
              nextFrontier.push(otherNodeId);
            }
          } else if (includeRelations && !resultEdges.has(edgeId)) {
            // L'arête connecte deux nœuds déjà explorés
            resultEdges.add(edgeId);
          }
        }
      }
      
      currentFrontier = nextFrontier;
    }
    
    // Convertir en tableaux pour le résultat
    let resultNodesList = Array.from(resultNodes.values());
    const resultEdgesList = includeRelations
      ? Array.from(resultEdges).map(id => this.edges.get(id)!)
      : undefined;
    
    // Calculer les scores de pertinence
    const relevanceScores = new Map<string, number>();
    if (sortByRelevance) {
      resultNodesList.forEach(node => {
        // Score de base selon la correspondance directe
        let score = directMatchIds.includes(node.id) ? 0.8 : 0.4;
        
        // Augmenter le score selon la confiance
        score += node.confidence * 0.2;
        
        // Dans une implémentation réelle, utiliser la similarité des embeddings
        
        relevanceScores.set(node.id, score);
      });
      
      // Trier par pertinence
      resultNodesList.sort((a, b) => 
        (relevanceScores.get(b.id) || 0) - (relevanceScores.get(a.id) || 0)
      );
    }
    
    // Limiter le nombre de résultats
    if (maxResults > 0 && resultNodesList.length > maxResults) {
      resultNodesList = resultNodesList.slice(0, maxResults);
    }
    
    const endTime = Date.now();
    
    // Construire le résultat
    const result: GraphSearchResult = {
      nodes: resultNodesList,
      edges: resultEdgesList,
      queryNodes: directMatchIds,
      relevanceScores: sortByRelevance ? relevanceScores : undefined,
      metainfo: {
        totalNodesExplored: totalExplored,
        searchDepth: currentDepth,
        executionTimeMs: endTime - startTime
      }
    };
    
    this.logger.debug(`Recherche dans le graphe pour "${query}"`, {
      nodesFound: resultNodesList.length,
      edgesFound: resultEdgesList?.length || 0,
      searchDepthReached: currentDepth,
      executionTime: endTime - startTime
    });
    
    return result;
  }
  
  /**
   * Recherche les plus courts chemins entre deux nœuds
   * @param startNodeId ID du nœud de départ
   * @param endNodeId ID du nœud d'arrivée
   * @param options Options de recherche
   * @returns Chemins trouvés (séquences d'arêtes)
   */
  public findPaths(
    startNodeId: string,
    endNodeId: string,
    options: {
      maxDepth?: number;
      relationTypes?: RelationType[];
      maxPaths?: number;
    } = {}
  ): Array<{ path: KnowledgeEdge[]; length: number }> {
    const {
      maxDepth = 4,
      relationTypes,
      maxPaths = 3
    } = options;
    
    if (!this.nodes.has(startNodeId) || !this.nodes.has(endNodeId)) {
      this.logger.warn('findPaths: Un des nœuds n\'existe pas', {
        startNodeId,
        endNodeId,
        startExists: this.nodes.has(startNodeId),
        endExists: this.nodes.has(endNodeId)
      });
      return [];
    }
    
    // Algorithme de recherche en largeur (BFS)
    const visited = new Set<string>([startNodeId]);
    const queue: Array<{
      nodeId: string;
      path: KnowledgeEdge[];
      depth: number;
    }> = [{ nodeId: startNodeId, path: [], depth: 0 }];
    
    const results: Array<{ path: KnowledgeEdge[]; length: number }> = [];
    
    while (queue.length > 0 && results.length < maxPaths) {
      const { nodeId, path, depth } = queue.shift()!;
      
      if (depth >= maxDepth) {
        continue;
      }
      
      const edgeIds = this.nodeEdges.get(nodeId) || new Set<string>();
      
      for (const edgeId of edgeIds) {
        const edge = this.edges.get(edgeId)!;
        
        // Filtrer par type de relation si spécifié
        if (relationTypes && !relationTypes.includes(edge.type)) {
          continue;
        }
        
        const nextNodeId = edge.sourceId === nodeId ? edge.targetId : edge.sourceId;
        
        // Vérifier si l'arête connecte au nœud cible
        if (nextNodeId === endNodeId) {
          const completePath = [...path, edge];
          results.push({
            path: completePath,
            length: completePath.length
          });
          
          if (results.length >= maxPaths) {
            break;
          }
          
          continue;
        }
        
        // Ne pas revisiter les nœuds
        if (!visited.has(nextNodeId)) {
          visited.add(nextNodeId);
          queue.push({
            nodeId: nextNodeId,
            path: [...path, edge],
            depth: depth + 1
          });
        }
      }
    }
    
    // Trier par longueur de chemin
    results.sort((a, b) => a.length - b.length);
    
    this.logger.debug(`Recherche de chemins entre ${startNodeId} et ${endNodeId}`, {
      pathsFound: results.length,
      shortestPathLength: results[0]?.length
    });
    
    return results;
  }

  /**
   * Génère un sous-graphe à partir d'un ensemble de nœuds
   * @param nodeIds IDs des nœuds à inclure
   * @param includeConnections Inclure les connexions entre les nœuds
   * @returns Sous-graphe (nœuds et arêtes)
   */
  public generateSubgraph(
    nodeIds: string[],
    includeConnections: boolean = true
  ): { nodes: KnowledgeNode[]; edges: KnowledgeEdge[] } {
    const subgraphNodes = new Map<string, KnowledgeNode>();
    const subgraphEdges = new Set<string>();
    
    // Collecter les nœuds
    for (const nodeId of nodeIds) {
      if (this.nodes.has(nodeId)) {
        subgraphNodes.set(nodeId, this.nodes.get(nodeId)!);
      }
    }
    
    // Collecter les arêtes si demandé
    if (includeConnections) {
      for (const nodeId of nodeIds) {
        const edgeIds = this.nodeEdges.get(nodeId) || new Set<string>();
        
        for (const edgeId of edgeIds) {
          const edge = this.edges.get(edgeId)!;
          
          // N'inclure que les arêtes entre les nœuds du sous-graphe
          if (nodeIds.includes(edge.sourceId) && nodeIds.includes(edge.targetId)) {
            subgraphEdges.add(edgeId);
          }
        }
      }
    }
    
    return {
      nodes: Array.from(subgraphNodes.values()),
      edges: Array.from(subgraphEdges).map(id => this.edges.get(id)!)
    };
  }
  
  /**
   * Calcule les statistiques sur le graphe
   * @returns Statistiques du graphe
   */
  public getStatistics(): Record<string, any> {
    const nodeTypes = new Map<string, number>();
    const edgeTypes = new Map<RelationType, number>();
    const nodeSources = new Map<KnowledgeSource, number>();
    
    // Compter les types de nœuds
    for (const node of this.nodes.values()) {
      nodeTypes.set(node.type, (nodeTypes.get(node.type) || 0) + 1);
      nodeSources.set(node.source, (nodeSources.get(node.source) || 0) + 1);
    }
    
    // Compter les types d'arêtes
    for (const edge of this.edges.values()) {
      edgeTypes.set(edge.type, (edgeTypes.get(edge.type) || 0) + 1);
    }
    
    // Calculer la densité du graphe
    const nodeCount = this.nodes.size;
    const edgeCount = this.edges.size;
    const maxPossibleEdges = nodeCount * (nodeCount - 1) / 2;
    const density = nodeCount > 1 ? edgeCount / maxPossibleEdges : 0;
    
    return {
      nodeCount,
      edgeCount,
      nodeTypes: Object.fromEntries(nodeTypes),
      edgeTypes: Object.fromEntries(edgeTypes),
      nodeSources: Object.fromEntries(nodeSources),
      density,
      averageDegree: nodeCount > 0 ? (edgeCount * 2) / nodeCount : 0,
      createdAt: new Date().toISOString()
    };
  }
  
  /**
   * Remet à zéro le graphe
   */
  public clear(): void {
    this.nodes.clear();
    this.edges.clear();
    this.nodeEdges.clear();
    this.embeddings.clear();
    this.logger.info('Graphe de connaissances réinitialisé');
  }
  
  /**
   * Exporte le graphe sous forme sérialisable
   * @returns Représentation JSON du graphe
   */
  public export(): Record<string, any> {
    return {
      nodes: Array.from(this.nodes.values()),
      edges: Array.from(this.edges.values()),
      stats: this.getStatistics()
    };
  }
  
  /**
   * Importe un graphe depuis une représentation sérialisée
   * @param data Représentation JSON du graphe
   */
  public import(data: { nodes: KnowledgeNode[]; edges: KnowledgeEdge[] }): void {
    this.clear();
    
    // Importer les nœuds
    for (const node of data.nodes) {
      this.nodes.set(node.id, node);
      this.nodeEdges.set(node.id, new Set());
    }
    
    // Importer les arêtes
    for (const edge of data.edges) {
      this.edges.set(edge.id, edge);
      
      // Mettre à jour les références des nœuds
      if (this.nodeEdges.has(edge.sourceId)) {
        this.nodeEdges.get(edge.sourceId)!.add(edge.id);
      }
      
      if (this.nodeEdges.has(edge.targetId)) {
        this.nodeEdges.get(edge.targetId)!.add(edge.id);
      }
    }
    
    this.logger.info(`Graphe importé: ${data.nodes.length} nœuds, ${data.edges.length} arêtes`);
  }
} 