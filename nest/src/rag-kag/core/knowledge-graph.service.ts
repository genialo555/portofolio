import { Injectable, Inject } from '@nestjs/common';
import { LOGGER_TOKEN, ILogger } from '../utils/logger-tokens';
import { v4 as uuidv4 } from 'uuid';
import { EventBusService, RagKagEventType } from './event-bus.service';

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
  SYSTEM = 'SYSTEM',               // Généré par le système
  AGENT = 'AGENT'                  // Agent spécifique
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
 * Options pour les recherches dans le graphe
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
  filter?: (node: KnowledgeNode) => boolean; // Filtre personnalisé sur les nœuds
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
 * Service de graphe de connaissances pour le système RAG/KAG
 * Stocke et organise les connaissances sous forme de graphe
 */
@Injectable()
export class KnowledgeGraphService {
  private nodes: Map<string, KnowledgeNode> = new Map();
  private edges: Map<string, KnowledgeEdge> = new Map();
  private nodeEdges: Map<string, Set<string>> = new Map();
  private embeddings: Map<string, number[]> = new Map(); // Simuler les embeddings pour le prototype
  
  constructor(
    @Inject(LOGGER_TOKEN) private readonly logger: ILogger,
    private readonly eventBus: EventBusService
  ) {}

  /**
   * Ajoute un nœud au graphe de connaissances
   * @param node Données du nœud à ajouter
   * @returns ID du nœud créé
   */
  public addNode(node: Omit<KnowledgeNode, 'id' | 'timestamp'>): string {
    const nodeId = uuidv4();
    const timestamp = Date.now();
    
    const newNode: KnowledgeNode = {
      id: nodeId,
      timestamp,
      ...node
    };
    
    this.nodes.set(nodeId, newNode);
    this.nodeEdges.set(nodeId, new Set());
    
    // Simuler un embedding pour le contenu du nœud (version prototype)
    this.embeddings.set(nodeId, this.simulateEmbedding(node.content));
    
    this.logger.debug(`Nœud de connaissance ajouté: ${nodeId} (${node.label})`, { id: nodeId, type: node.type });
    
    // Émettre un événement de création de nœud
    this.eventBus.emit({
      type: RagKagEventType.KNOWLEDGE_NODE_ADDED,
      source: 'KnowledgeGraphService',
      payload: {
        nodeId,
        nodeType: node.type,
        source: node.source
      }
    });
    
    return nodeId;
  }

  /**
   * Ajoute une relation entre deux nœuds
   * @param edge Données de la relation à ajouter
   * @returns ID de la relation créée
   */
  public addEdge(edge: Omit<KnowledgeEdge, 'id' | 'timestamp'>): string {
    // Vérifier que les nœuds existent
    if (!this.nodes.has(edge.sourceId)) {
      throw new Error(`Nœud source non trouvé: ${edge.sourceId}`);
    }
    
    if (!this.nodes.has(edge.targetId)) {
      throw new Error(`Nœud cible non trouvé: ${edge.targetId}`);
    }
    
    const edgeId = uuidv4();
    const timestamp = Date.now();
    
    const newEdge: KnowledgeEdge = {
      id: edgeId,
      timestamp,
      ...edge
    };
    
    this.edges.set(edgeId, newEdge);
    
    // Mettre à jour les mappings nœud-relation
    this.nodeEdges.get(edge.sourceId)!.add(edgeId);
    this.nodeEdges.get(edge.targetId)!.add(edgeId);
    
    this.logger.debug(
      `Relation ajoutée: ${edge.sourceId} -[${edge.type}]-> ${edge.targetId}`,
      { id: edgeId, type: edge.type }
    );
    
    // Émettre un événement de création de relation
    this.eventBus.emit({
      type: RagKagEventType.KNOWLEDGE_EDGE_ADDED,
      source: 'KnowledgeGraphService',
      payload: {
        edgeId,
        sourceId: edge.sourceId,
        targetId: edge.targetId,
        relationType: edge.type
      }
    });
    
    return edgeId;
  }

  /**
   * Ajoute un fait au graphe de connaissances
   * @param subject Sujet (nœud ou label)
   * @param predicate Type de relation
   * @param object Objet (nœud ou label)
   * @param confidence Niveau de confiance
   * @param options Options additionnelles
   * @returns IDs des nœuds et de la relation créés
   */
  public addFact(
    subject: string | Partial<KnowledgeNode>,
    predicate: RelationType | string,
    object: string | Partial<KnowledgeNode>,
    confidence: number = 0.8,
    options: {
      bidirectional?: boolean;
      weight?: number;
      source?: KnowledgeSource;
      metadata?: Record<string, any>;
    } = {}
  ): { subjectId: string; objectId: string; edgeId: string } {
    // Valeurs par défaut
    const defaultOptions = {
      bidirectional: false,
      weight: 0.5,
      source: KnowledgeSource.SYSTEM,
      metadata: {}
    };
    
    const mergedOptions = { ...defaultOptions, ...options };
    
    // Traiter le sujet
    let subjectId: string;
    if (typeof subject === 'string') {
      // Rechercher un nœud existant avec ce label
      const existingSubject = Array.from(this.nodes.values())
        .find(node => node.label.toLowerCase() === subject.toLowerCase());
      
      if (existingSubject) {
        subjectId = existingSubject.id;
      } else {
        // Créer un nouveau nœud
        subjectId = this.addNode({
          label: subject,
          type: 'CONCEPT',
          content: subject,
          confidence,
          source: mergedOptions.source,
          metadata: { ...mergedOptions.metadata, autoCreated: true }
        });
      }
    } else {
      // Utiliser les propriétés fournies pour créer un nœud
      subjectId = this.addNode({
        label: subject.label || 'Concept sans nom',
        type: subject.type || 'CONCEPT',
        content: subject.content || subject.label || '',
        confidence: subject.confidence || confidence,
        source: subject.source || mergedOptions.source,
        metadata: { ...mergedOptions.metadata, ...(subject.metadata || {}) }
      });
    }
    
    // Traiter l'objet
    let objectId: string;
    if (typeof object === 'string') {
      // Rechercher un nœud existant avec ce label
      const existingObject = Array.from(this.nodes.values())
        .find(node => node.label.toLowerCase() === object.toLowerCase());
      
      if (existingObject) {
        objectId = existingObject.id;
      } else {
        // Créer un nouveau nœud
        objectId = this.addNode({
          label: object,
          type: 'CONCEPT',
          content: object,
          confidence,
          source: mergedOptions.source,
          metadata: { ...mergedOptions.metadata, autoCreated: true }
        });
      }
    } else {
      // Utiliser les propriétés fournies pour créer un nœud
      objectId = this.addNode({
        label: object.label || 'Concept sans nom',
        type: object.type || 'CONCEPT',
        content: object.content || object.label || '',
        confidence: object.confidence || confidence,
        source: object.source || mergedOptions.source,
        metadata: { ...mergedOptions.metadata, ...(object.metadata || {}) }
      });
    }
    
    // Traiter le prédicat
    const relationType = typeof predicate === 'string' 
      ? RelationType.CUSTOM 
      : predicate;
    
    // Créer la relation
    const edgeId = this.addEdge({
      sourceId: subjectId,
      targetId: objectId,
      type: relationType,
      label: typeof predicate === 'string' ? predicate : undefined,
      weight: mergedOptions.weight,
      confidence,
      bidirectional: mergedOptions.bidirectional,
      metadata: mergedOptions.metadata
    });
    
    return { subjectId, objectId, edgeId };
  }

  /**
   * Recherche dans le graphe de connaissances à partir d'une requête textuelle
   * @param query Requête de recherche
   * @param options Options de recherche
   * @returns Résultats de la recherche
   */
  public search(query: string, options: GraphSearchOptions = {}): GraphSearchResult {
    this.logger.debug('Recherche dans le graphe de connaissances', { query, options });
    
    const {
      maxDepth = 2,
      minConfidence = 0.5,
      relationTypes,
      nodeTypes,
      maxResults = 10,
      sortByRelevance = false,
      includeMetadata = false,
      includeRelations = false,
      filter = () => true
    } = options;
    
    const startTime = Date.now();
    
    // Simuler un embedding pour la requête
    const queryEmbedding = this.simulateEmbedding(query);
    
    // Trouver les nœuds les plus pertinents
    const relevanceScores = new Map<string, number>();
    
    // Calculer les scores de similarité pour tous les nœuds
    for (const [nodeId, nodeEmbedding] of this.embeddings.entries()) {
      const similarity = this.cosineSimilarity(queryEmbedding, nodeEmbedding);
      relevanceScores.set(nodeId, similarity);
    }
    
    // Filtrer par confiance et type
    const filteredNodeIds = Array.from(this.nodes.entries())
      .filter(([nodeId, node]) => {
        const score = relevanceScores.get(nodeId) || 0;
        const passesConfidence = node.confidence >= minConfidence;
        const passesType = nodeTypes.length === 0 || nodeTypes.includes(node.type);
        const passesRelevance = score > 0.2; // Seuil de pertinence minimal
        
        return passesConfidence && passesType && passesRelevance && filter(node);
      })
      .map(([nodeId]) => nodeId);
    
    // Trier par pertinence si demandé
    if (sortByRelevance) {
      filteredNodeIds.sort((a, b) => 
        (relevanceScores.get(b) || 0) - (relevanceScores.get(a) || 0)
      );
    }
    
    // Limiter le nombre de résultats
    const limitedNodeIds = filteredNodeIds.slice(0, maxResults);
    
    // Exploration du graphe à partir des nœuds pertinents
    const visitedNodeIds = new Set<string>(limitedNodeIds);
    const nodesToExplore = [...limitedNodeIds];
    const exploredEdgeIds = new Set<string>();
    let totalNodesExplored = limitedNodeIds.length;
    let currentDepth = 0;
    
    // Explorer le graphe jusqu'à la profondeur maximale
    while (nodesToExplore.length > 0 && currentDepth < maxDepth) {
      const currentNodeId = nodesToExplore.shift()!;
      const nodeEdgeIds = this.nodeEdges.get(currentNodeId) || new Set();
      
      for (const edgeId of nodeEdgeIds) {
        const edge = this.edges.get(edgeId)!;
        
        // Vérifier que le type de relation est inclus
        if (!relationTypes.includes(edge.type)) {
          continue;
        }
        
        exploredEdgeIds.add(edgeId);
        
        // Trouver l'autre nœud de la relation
        const otherNodeId = edge.sourceId === currentNodeId ? edge.targetId : edge.sourceId;
        
        // Si le nœud n'a pas encore été visité, l'ajouter
        if (!visitedNodeIds.has(otherNodeId)) {
          const otherNode = this.nodes.get(otherNodeId)!;
          
          // Vérifier la confiance et le type
          const passesConfidence = otherNode.confidence >= minConfidence;
          const passesType = nodeTypes.length === 0 || nodeTypes.includes(otherNode.type);
          
          if (passesConfidence && passesType) {
            visitedNodeIds.add(otherNodeId);
            nodesToExplore.push(otherNodeId);
            totalNodesExplored++;
          }
        }
      }
      
      // Si la file est vide mais qu'il reste des niveaux à explorer
      if (nodesToExplore.length === 0 && currentDepth < maxDepth - 1) {
        currentDepth++;
      }
    }
    
    // Construire le résultat
    const resultNodes = Array.from(visitedNodeIds)
      .map(nodeId => this.nodes.get(nodeId)!)
      .filter(node => !!node);
    
    let resultEdges: KnowledgeEdge[] = [];
    if (includeRelations) {
      resultEdges = Array.from(exploredEdgeIds)
        .map(edgeId => this.edges.get(edgeId)!)
        .filter(edge => !!edge);
    }
    
    const executionTime = Date.now() - startTime;
    
    this.logger.debug(
      `Recherche effectuée: "${query}" (${resultNodes.length} nœuds, ${resultEdges.length} relations)`,
      { executionTime, depth: currentDepth }
    );
    
    return {
      nodes: resultNodes,
      edges: includeRelations ? resultEdges : undefined,
      relevanceScores: relevanceScores,
      queryNodes: limitedNodeIds,
      metainfo: {
        totalNodesExplored,
        searchDepth: currentDepth,
        executionTimeMs: executionTime
      }
    };
  }

  /**
   * Trouve les chemins entre deux nœuds du graphe
   * @param startNodeId ID du nœud de départ
   * @param endNodeId ID du nœud d'arrivée
   * @param options Options de recherche
   * @returns Chemins trouvés avec leur longueur
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
    // Vérifier que les nœuds existent
    if (!this.nodes.has(startNodeId)) {
      throw new Error(`Nœud de départ non trouvé: ${startNodeId}`);
    }
    
    if (!this.nodes.has(endNodeId)) {
      throw new Error(`Nœud d'arrivée non trouvé: ${endNodeId}`);
    }
    
    // Options par défaut
    const defaultOptions = {
      maxDepth: 4,
      relationTypes: Object.values(RelationType),
      maxPaths: 5
    };
    
    const mergedOptions = { ...defaultOptions, ...options };
    
    // Tableaux pour stocker les résultats
    const paths: Array<{ path: KnowledgeEdge[]; length: number }> = [];
    
    // Exploration par BFS modifiée pour trouver tous les chemins
    const queue: Array<{ nodeId: string; path: KnowledgeEdge[] }> = [
      { nodeId: startNodeId, path: [] }
    ];
    
    const visited = new Map<string, boolean>();
    visited.set(startNodeId, true);
    
    while (queue.length > 0 && paths.length < mergedOptions.maxPaths) {
      const { nodeId, path } = queue.shift()!;
      
      // Si on a atteint le nœud cible
      if (nodeId === endNodeId && path.length > 0) {
        paths.push({ path, length: path.length });
        continue;
      }
      
      // Si on a atteint la profondeur maximale
      if (path.length >= mergedOptions.maxDepth) {
        continue;
      }
      
      // Explorer les voisins
      const edgeIds = this.nodeEdges.get(nodeId) || new Set();
      
      for (const edgeId of edgeIds) {
        const edge = this.edges.get(edgeId)!;
        
        // Vérifier le type de relation
        if (!mergedOptions.relationTypes.includes(edge.type)) {
          continue;
        }
        
        // Trouver le nœud voisin
        const nextNodeId = edge.sourceId === nodeId ? edge.targetId : edge.sourceId;
        
        // Vérifier la direction de la relation
        if (edge.sourceId !== nodeId && !edge.bidirectional) {
          continue; // Relation unidirectionnelle dans le mauvais sens
        }
        
        // Éviter les cycles
        if (path.some(p => p.sourceId === nextNodeId || p.targetId === nextNodeId)) {
          continue;
        }
        
        // Créer un nouveau chemin
        const newPath = [...path, edge];
        
        // Ajouter à la file
        queue.push({ nodeId: nextNodeId, path: newPath });
      }
    }
    
    // Trier les chemins par longueur
    paths.sort((a, b) => a.length - b.length);
    
    this.logger.debug(
      `Chemins trouvés entre ${startNodeId} et ${endNodeId}: ${paths.length}`,
      { pathLengths: paths.map(p => p.length) }
    );
    
    return paths;
  }

  /**
   * Crée un sous-graphe à partir d'une liste de nœuds
   * @param nodeIds IDs des nœuds à inclure
   * @param includeConnections Inclure les connexions entre les nœuds
   * @returns Sous-graphe avec nœuds et relations
   */
  public generateSubgraph(
    nodeIds: string[],
    includeConnections: boolean = true
  ): { nodes: KnowledgeNode[]; edges: KnowledgeEdge[] } {
    const nodes = nodeIds
      .filter(id => this.nodes.has(id))
      .map(id => this.nodes.get(id)!);
    
    let edges: KnowledgeEdge[] = [];
    
    if (includeConnections) {
      // Trouver toutes les relations entre les nœuds spécifiés
      const nodeIdSet = new Set(nodeIds);
      
      for (const nodeId of nodeIds) {
        const edgeIds = this.nodeEdges.get(nodeId) || new Set();
        
        for (const edgeId of edgeIds) {
          const edge = this.edges.get(edgeId)!;
          
          // Ajouter la relation seulement si les deux nœuds sont dans l'ensemble
          if (nodeIdSet.has(edge.sourceId) && nodeIdSet.has(edge.targetId)) {
            edges.push(edge);
          }
        }
      }
      
      // Éliminer les doublons
      edges = Array.from(new Map(edges.map(edge => [edge.id, edge])).values());
    }
    
    this.logger.debug(
      `Sous-graphe généré avec ${nodes.length} nœuds et ${edges.length} relations`,
      { nodeCount: nodes.length, edgeCount: edges.length }
    );
    
    return { nodes, edges };
  }

  /**
   * Récupère des statistiques sur le graphe de connaissances
   * @returns Statistiques diverses
   */
  public getStatistics(): Record<string, any> {
    const nodeCount = this.nodes.size;
    const edgeCount = this.edges.size;
    
    // Compter les nœuds par type
    const nodeTypeCounts: Record<string, number> = {};
    for (const node of this.nodes.values()) {
      nodeTypeCounts[node.type] = (nodeTypeCounts[node.type] || 0) + 1;
    }
    
    // Compter les relations par type
    const edgeTypeCounts: Record<string, number> = {};
    for (const edge of this.edges.values()) {
      edgeTypeCounts[edge.type] = (edgeTypeCounts[edge.type] || 0) + 1;
    }
    
    // Calculer la densité du graphe
    const maxPossibleEdges = nodeCount * (nodeCount - 1) / 2;
    const density = maxPossibleEdges > 0 ? edgeCount / maxPossibleEdges : 0;
    
    // Calculer le degré moyen
    let totalDegree = 0;
    for (const nodeEdgeSet of this.nodeEdges.values()) {
      totalDegree += nodeEdgeSet.size;
    }
    const averageDegree = nodeCount > 0 ? totalDegree / nodeCount : 0;
    
    return {
      nodeCount,
      edgeCount,
      nodeTypeCounts,
      edgeTypeCounts,
      density,
      averageDegree,
      timestamp: Date.now()
    };
  }

  /**
   * Vide le graphe de connaissances
   */
  public clear(): void {
    this.nodes.clear();
    this.edges.clear();
    this.nodeEdges.clear();
    this.embeddings.clear();
    
    this.logger.debug('Graphe de connaissances vidé');
    
    // Émettre un événement de mise à jour du graphe
    this.eventBus.emit({
      type: RagKagEventType.KNOWLEDGE_GRAPH_UPDATED,
      source: 'KnowledgeGraphService',
      payload: { action: 'clear', timestamp: Date.now() }
    });
  }

  /**
   * Exporte le graphe complet
   * @returns Nœuds et relations du graphe
   */
  public export(): { nodes: KnowledgeNode[]; edges: KnowledgeEdge[] } {
    return {
      nodes: Array.from(this.nodes.values()),
      edges: Array.from(this.edges.values())
    };
  }

  /**
   * Importe un graphe
   * @param data Nœuds et relations à importer
   */
  public import(data: { nodes: KnowledgeNode[]; edges: KnowledgeEdge[] }): void {
    // Vider le graphe actuel
    this.clear();
    
    // Importer les nœuds
    for (const node of data.nodes) {
      this.nodes.set(node.id, node);
      this.nodeEdges.set(node.id, new Set());
      this.embeddings.set(node.id, this.simulateEmbedding(node.content));
    }
    
    // Importer les relations
    for (const edge of data.edges) {
      this.edges.set(edge.id, edge);
      
      // Mettre à jour les mappings nœud-relation
      if (this.nodeEdges.has(edge.sourceId)) {
        this.nodeEdges.get(edge.sourceId)!.add(edge.id);
      }
      
      if (this.nodeEdges.has(edge.targetId)) {
        this.nodeEdges.get(edge.targetId)!.add(edge.id);
      }
    }
    
    this.logger.debug(
      `Graphe importé: ${data.nodes.length} nœuds, ${data.edges.length} relations`,
      { nodeCount: data.nodes.length, edgeCount: data.edges.length }
    );
    
    // Émettre un événement de mise à jour du graphe
    this.eventBus.emit({
      type: RagKagEventType.KNOWLEDGE_GRAPH_UPDATED,
      source: 'KnowledgeGraphService',
      payload: {
        action: 'import',
        nodeCount: data.nodes.length,
        edgeCount: data.edges.length,
        timestamp: Date.now()
      }
    });
  }

  /**
   * Simuler un embedding pour un texte (version prototype)
   * @param text Texte à encoder
   * @returns Vecteur d'embedding simulé
   */
  private simulateEmbedding(text: string): number[] {
    // Simplification pour le prototype: génère un vecteur basé sur le hash du texte
    // Dans une implémentation réelle, on utiliserait un vrai modèle d'embedding
    const hash = this.simpleHash(text);
    const vector = new Array(128).fill(0);
    
    // Remplir le vecteur avec des valeurs dérivées du hash
    for (let i = 0; i < 128; i++) {
      vector[i] = Math.sin(hash * (i + 1)) * 0.5 + 0.5;
    }
    
    return vector;
  }

  /**
   * Calcule une valeur de hash simple pour un texte
   * @param text Texte à hasher
   * @returns Valeur de hash
   */
  private simpleHash(text: string): number {
    let hash = 0;
    if (text.length === 0) return hash;
    
    for (let i = 0; i < text.length; i++) {
      const char = text.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash; // Convertir en entier 32 bits
    }
    
    return hash;
  }

  /**
   * Calcule la similarité cosinus entre deux vecteurs
   * @param a Premier vecteur
   * @param b Second vecteur
   * @returns Score de similarité (0-1)
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
    
    if (normA === 0 || normB === 0) {
      return 0;
    }
    
    return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
  }
} 