import { Injectable, Inject, Optional } from '@nestjs/common';
import { LOGGER_TOKEN, ILogger } from '../utils/logger-tokens';
import { UserQuery, TargetPools } from '../types';
import { QueryAnalyzerService, QueryAnalysisResult } from '../core/query-analyzer.service';
import { EventBusService, RagKagEventType } from '../core/event-bus.service';
import { KnowledgeGraphService, KnowledgeSource, RelationType } from '../core/knowledge-graph.service';

/**
 * Service de routage pour déterminer quels pools d'agents doivent être utilisés
 * Utilise l'analyse avancée du QueryAnalyzerService
 * Intègre le KnowledgeGraph pour l'apprentissage des routages
 */
@Injectable()
export class RouterService {
  private readonly logger: ILogger;
  private readonly routingHistory: Map<string, { 
    count: number, 
    successRate: number,
    pools: TargetPools 
  }> = new Map();
  
  constructor(
    @Inject(LOGGER_TOKEN) logger: ILogger,
    private readonly queryAnalyzer: QueryAnalyzerService,
    private readonly eventBus: EventBusService,
    @Optional() private readonly knowledgeGraph?: KnowledgeGraphService
  ) {
    this.logger = logger;
    
    // Charger l'historique de routage depuis le graphe de connaissances
    if (this.knowledgeGraph) {
      this.loadRoutingHistoryFromGraph();
    }
  }

  /**
   * Détermine les pools d'agents à utiliser pour une requête
   * @param query Requête utilisateur
   * @returns Configuration des pools à utiliser
   */
  async determineTargetPools(query: UserQuery): Promise<TargetPools> {
    // Utiliser le QueryAnalyzerService avancé
    const analysisResult = await this.queryAnalyzer.analyzeQuery(query);
    
    // Vérifier si nous avons un routage similaire dans l'historique
    const similarQuery = this.findSimilarQuery(query, analysisResult);
    let targetPools: TargetPools;
    
    if (similarQuery && similarQuery.successRate > 0.7) {
      // Utiliser le routage qui a bien fonctionné dans le passé
      targetPools = similarQuery.pools;
      this.logger.debug('Utilisation d\'un routage historique avec succès', {
        similarityScore: similarQuery.similarityScore,
        successRate: similarQuery.successRate
      });
    } else {
      // Utiliser l'analyse standard
      targetPools = this.mapAnalysisToTargetPools(analysisResult, query);
    }
    
    this.logger.debug(`Pools cibles déterminés:`, {
      commercial: targetPools.commercial,
      marketing: targetPools.marketing,
      sectoriel: targetPools.sectoriel,
      educational: targetPools.educational,
      primaryFocus: targetPools.primaryFocus
    });
    
    // Émettre un événement de routage
    this.eventBus.emit({
      type: RagKagEventType.QUERY_ROUTED,
      source: 'RouterService',
      payload: {
        query: typeof query === 'string' ? query : query.content,
        targetPools,
        domains: analysisResult.domains
      }
    });
    
    // Stocker le routage dans le graphe de connaissances
    if (this.knowledgeGraph) {
      this.storeRoutingDecision(query, targetPools, analysisResult);
    }
    
    return targetPools;
  }

  /**
   * Mappe le résultat d'analyse aux pools cibles
   * @param analysis Résultat d'analyse de requête
   * @param query Requête originale
   * @returns Configuration des pools cibles
   */
  private mapAnalysisToTargetPools(analysis: QueryAnalysisResult, query: UserQuery): TargetPools {
    const targetPools: TargetPools = {
            commercial: false,
            marketing: false,
      sectoriel: false,
      educational: false
    };
    
    // Activer les pools en fonction des domaines détectés
    for (const domain of analysis.domains) {
      if (domain === 'commercial') {
        targetPools.commercial = true;
      } else if (domain === 'marketing') {
        targetPools.marketing = true;
      } else if (domain === 'sectoriel') {
        targetPools.sectoriel = true;
      } else if (domain === 'educational') {
        targetPools.educational = true;
      }
    }

    // Si aucun domaine n'a été détecté ou si la confiance est faible, utiliser une heuristique
    if (analysis.domains.length === 0 || analysis.confidenceScore < 0.5) {
      targetPools.commercial = this.hasCommercialTerms(analysis.keywords);
      targetPools.marketing = this.hasMarketingTerms(analysis.keywords);
      targetPools.sectoriel = this.hasSectorialTerms(analysis.keywords);
      targetPools.educational = this.hasEducationalTerms(analysis.keywords);
    }
    
    // Vérifier les préférences utilisateur si disponibles
    if (query && typeof query !== 'string' && query.preferences) {
      // Si l'utilisateur a demandé une adaptation éducative
      if (query.preferences.educationalLevel) {
        targetPools.educational = true;
      }
    }
    
    // Si aucun pool n'est activé, activer le commercial par défaut
    if (!targetPools.commercial && !targetPools.marketing && !targetPools.sectoriel && !targetPools.educational) {
      targetPools.commercial = true;
    }
    
    // Déterminer le focus principal
    targetPools.primaryFocus = this.determinePrimaryFocus(targetPools, analysis);
    
    return targetPools;
  }

  /**
   * Détermine le pool principal à utiliser
   * @param targetPools Pools activés
   * @param analysis Résultat d'analyse
   * @returns Type de pool principal
   */
  private determinePrimaryFocus(targetPools: TargetPools, analysis: QueryAnalysisResult): 'COMMERCIAL' | 'MARKETING' | 'SECTORIEL' | 'EDUCATIONAL' | undefined {
    // Si un seul pool est activé, c'est le focus principal
    if (targetPools.commercial && !targetPools.marketing && !targetPools.sectoriel && !targetPools.educational) {
      return 'COMMERCIAL';
    }
    if (!targetPools.commercial && targetPools.marketing && !targetPools.sectoriel && !targetPools.educational) {
      return 'MARKETING';
    }
    if (!targetPools.commercial && !targetPools.marketing && targetPools.sectoriel && !targetPools.educational) {
      return 'SECTORIEL';
    }
    if (!targetPools.commercial && !targetPools.marketing && !targetPools.sectoriel && targetPools.educational) {
      return 'EDUCATIONAL';
    }
    
    // Sinon, sélectionner en fonction du domaine le plus pertinent
    if (analysis.domains.length > 0) {
      const primaryDomain = analysis.domains[0];
      
      if (primaryDomain === 'commercial' && targetPools.commercial) {
        return 'COMMERCIAL';
      } else if (primaryDomain === 'marketing' && targetPools.marketing) {
        return 'MARKETING';
      } else if (primaryDomain === 'sectoriel' && targetPools.sectoriel) {
        return 'SECTORIEL';
      } else if (primaryDomain === 'educational' && targetPools.educational) {
        return 'EDUCATIONAL';
      }
    }
    
    // Si aucun domaine clair, utiliser le type de requête pour guider
    if (analysis.queryType === 'informational' && targetPools.educational) {
      return 'EDUCATIONAL';
    } else if (analysis.queryType === 'analytical' && targetPools.sectoriel) {
      return 'SECTORIEL';
    } else if (analysis.queryType === 'actionable' && targetPools.commercial) {
      return 'COMMERCIAL';
    } else if (analysis.queryType === 'comparative' && targetPools.marketing) {
      return 'MARKETING';
    }
    
    // Par défaut, pas de focus principal
    return undefined;
  }

  /**
   * Vérifie si les mots-clés contiennent des termes commerciaux
   */
  private hasCommercialTerms(keywords: string[]): boolean {
    const commercialTerms = ['vente', 'client', 'offre', 'produit', 'service', 'prix', 'commande'];
    return this.countTermsInKeywords(keywords, commercialTerms) > 0;
  }

  /**
   * Vérifie si les mots-clés contiennent des termes marketing
   */
  private hasMarketingTerms(keywords: string[]): boolean {
    const marketingTerms = ['campagne', 'publicité', 'marque', 'segment', 'cible', 'promotion'];
    return this.countTermsInKeywords(keywords, marketingTerms) > 0;
  }

  /**
   * Vérifie si les mots-clés contiennent des termes sectoriels
   */
  private hasSectorialTerms(keywords: string[]): boolean {
    const sectorialTerms = ['secteur', 'industrie', 'marché', 'concurrence', 'tendance'];
    return this.countTermsInKeywords(keywords, sectorialTerms) > 0;
  }

  /**
   * Vérifie si les mots-clés contiennent des termes éducatifs
   */
  private hasEducationalTerms(keywords: string[]): boolean {
    const educationalTerms = ['apprendre', 'enseigner', 'formation', 'éducation', 'cours', 'étudiant'];
    return this.countTermsInKeywords(keywords, educationalTerms) > 0;
  }

  /**
   * Compte combien de termes d'une liste apparaissent dans les mots-clés
   */
  private countTermsInKeywords(keywords: string[], terms: string[]): number {
    let count = 0;
    for (const keyword of keywords) {
      if (terms.includes(keyword.toLowerCase())) {
        count++;
      }
    }
    return count;
  }
  
  /**
   * Charge l'historique de routage depuis le graphe de connaissances
   */
  private async loadRoutingHistoryFromGraph(): Promise<void> {
    if (!this.knowledgeGraph) return;
    
    try {
      const routingNodes = await this.knowledgeGraph.search('routing decision', {
        nodeTypes: ['ROUTING_DECISION'],
        maxResults: 1000,
        sortByRelevance: false
      });
      
      for (const node of routingNodes.nodes) {
        const queryHash = node.metadata.queryHash;
        const pools = node.metadata.pools;
        const successRate = node.metadata.successRate || 0.5;
        const count = node.metadata.count || 1;
        
        this.routingHistory.set(queryHash, {
          count,
          successRate,
          pools
        });
      }
      
      this.logger.info(`Historique de routage chargé: ${this.routingHistory.size} entrées`);
    } catch (error) {
      this.logger.error('Erreur lors du chargement de l\'historique de routage', { error });
    }
  }
  
  /**
   * Stocke une décision de routage dans le graphe de connaissances
   */
  private async storeRoutingDecision(
    query: UserQuery, 
    targetPools: TargetPools, 
    analysis: QueryAnalysisResult
  ): Promise<void> {
    if (!this.knowledgeGraph) return;
    
    try {
      const queryContent = typeof query === 'string' ? query : query.content;
      const queryHash = this.hashString(queryContent);
      const keywords = analysis.keywords.join(',');
      
      // Vérifier si cette décision existe déjà
      const existingNodes = await this.knowledgeGraph.search(queryHash, {
        nodeTypes: ['ROUTING_DECISION'],
        maxResults: 1
      });
      
      if (existingNodes.nodes.length > 0) {
        // Mettre à jour le nœud existant
        const existingNode = existingNodes.nodes[0];
        const currentCount = existingNode.metadata.count || 1;
        const currentSuccessRate = existingNode.metadata.successRate || 0.5;
        
        // Mettre à jour les métadonnées
        // Note: Dans une implémentation réelle, nous aurions besoin d'une API pour mettre à jour les nœuds
        this.logger.debug('Mise à jour d\'une décision de routage existante', {
          queryHash,
          count: currentCount + 1
        });
      } else {
        // Créer un nouveau nœud pour cette décision
        const nodeId = this.knowledgeGraph.addNode({
          label: `Routing: ${queryContent.substring(0, 30)}...`,
          type: 'ROUTING_DECISION',
          content: queryContent,
          confidence: 0.8,
          source: KnowledgeSource.SYSTEM,
          metadata: {
            queryHash,
            query: queryContent,
            pools: targetPools,
            domains: analysis.domains,
            keywords,
            successRate: 0.5,
            count: 1,
            timestamp: Date.now()
          }
        });
        
        // Créer des relations avec les domaines
        for (const domain of analysis.domains) {
          this.knowledgeGraph.addFact(
            nodeId,
            'RELATED_TO',
            {
              label: `Domain: ${domain}`,
              type: 'DOMAIN',
              content: `Domain information: ${domain}`,
              confidence: 0.9,
              source: KnowledgeSource.SYSTEM
            },
            0.8,
            { bidirectional: true, weight: 0.7 }
          );
        }
        
        this.logger.debug('Nouvelle décision de routage stockée', { queryHash });
      }
    } catch (error) {
      this.logger.error('Erreur lors du stockage de la décision de routage', { error });
    }
  }
  
  /**
   * Met à jour le taux de succès d'une décision de routage
   */
  public async updateRoutingSuccess(
    query: UserQuery, 
    success: boolean
  ): Promise<void> {
    if (!this.knowledgeGraph) return;
    
    try {
      const queryContent = typeof query === 'string' ? query : query.content;
      const queryHash = this.hashString(queryContent);
      
      // Rechercher la décision de routage
      const routingNodes = await this.knowledgeGraph.search(queryHash, {
        nodeTypes: ['ROUTING_DECISION'],
        maxResults: 1
      });
      
      if (routingNodes.nodes.length > 0) {
        const node = routingNodes.nodes[0];
        const currentCount = node.metadata.count || 1;
        const currentSuccessRate = node.metadata.successRate || 0.5;
        
        // Calculer le nouveau taux de succès
        const newSuccessRate = (currentSuccessRate * currentCount + (success ? 1 : 0)) / (currentCount + 1);
        
        // Mettre à jour les métadonnées
        // Note: Dans une implémentation réelle, nous aurions besoin d'une API pour mettre à jour les nœuds
        this.logger.debug('Mise à jour du taux de succès d\'une décision de routage', {
          queryHash,
          oldSuccessRate: currentSuccessRate,
          newSuccessRate,
          count: currentCount + 1
        });
        
        // Mettre à jour l'historique local
        this.routingHistory.set(queryHash, {
          count: currentCount + 1,
          successRate: newSuccessRate,
          pools: node.metadata.pools
        });
      }
    } catch (error) {
      this.logger.error('Erreur lors de la mise à jour du taux de succès', { error });
    }
  }
  
  /**
   * Trouve une requête similaire dans l'historique
   */
  private findSimilarQuery(
    query: UserQuery, 
    analysis: QueryAnalysisResult
  ): { pools: TargetPools; successRate: number; similarityScore: number } | null {
    if (this.routingHistory.size === 0) return null;
    
    const queryContent = typeof query === 'string' ? query : query.content;
    const queryKeywords = analysis.keywords;
    
    let bestMatch = null;
    let highestScore = 0;
    
    // Parcourir l'historique pour trouver la meilleure correspondance
    for (const [hash, entry] of this.routingHistory.entries()) {
      // Dans une implémentation réelle, nous utiliserions une similarité sémantique
      // Pour cet exemple, nous utilisons une similarité basée sur les mots-clés
      const similarityScore = this.calculateSimilarity(queryContent, hash, queryKeywords);
      
      if (similarityScore > highestScore && similarityScore > 0.7) {
        highestScore = similarityScore;
        bestMatch = {
          pools: entry.pools,
          successRate: entry.successRate,
          similarityScore
        };
      }
    }
    
    return bestMatch;
  }
  
  /**
   * Calcule la similarité entre deux requêtes
   * Implémentation simplifiée pour l'exemple
   */
  private calculateSimilarity(query1: string, hash: string, keywords: string[]): number {
    // Dans une implémentation réelle, nous utiliserions une similarité sémantique
    // Pour cet exemple, nous retournons une valeur aléatoire entre 0.5 et 0.9
    return 0.5 + Math.random() * 0.4;
  }
  
  /**
   * Génère un hash simple pour une chaîne
   */
  private hashString(str: string): string {
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
      const char = str.charCodeAt(i);
      hash = (hash << 5) - hash + char;
      hash = hash & hash; // Convertir en entier 32 bits
    }
    return hash.toString(16);
  }
} 