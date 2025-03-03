import { Injectable, Inject } from '@nestjs/common';
import { LOGGER_TOKEN, ILogger } from '../utils/logger-tokens';
import { EventBusService, RagKagEventType } from './event-bus.service';
import { KnowledgeGraphService, KnowledgeSource } from './knowledge-graph.service';
import { UserQuery, ComponentType } from '../../types';

/**
 * Résultat de l'analyse d'une requête utilisateur
 */
export interface QueryAnalysisResult {
  queryType: string;            // Type de requête (information, action, etc.)
  domains: string[];            // Domaines associés à la requête
  keywords: string[];           // Mots-clés extraits
  entities: string[];           // Entités identifiées
  sentiment: number;            // Score de sentiment (-1 à 1)
  complexity: number;           // Score de complexité (0 à 1)
  languageDetected: string;     // Langue détectée
  confidenceScore: number;      // Score de confiance global
  suggestedComponents: ComponentType[]; // Composants recommandés
}

/**
 * Service d'analyse de requête
 * Ce service analyse les requêtes utilisateur pour identifier les caractéristiques
 * et orienter le traitement par les composants suivants
 */
@Injectable()
export class QueryAnalyzerService {
  // Configuration par défaut
  private config = {
    minComplexityThreshold: 0.3,
    keywordMinLength: 4,
    confidenceThreshold: 0.7,
    languageDetectionEnabled: true,
    defaultLanguage: 'fr',
    domainDetectionSensitivity: 0.5
  };

  // Dictionnaires de termes par domaine
  private domainTerms: Record<string, string[]> = {
    commercial: ['vente', 'client', 'offre', 'produit', 'service', 'prix', 'commande', 'achat', 'contrat'],
    marketing: ['campagne', 'publicité', 'marque', 'segment', 'cible', 'promotion', 'communication', 'audience'],
    sectoriel: ['secteur', 'industrie', 'marché', 'concurrence', 'tendance', 'réglementation', 'économie'],
    educational: ['apprendre', 'enseigner', 'formation', 'éducation', 'cours', 'étudiant', 'professeur', 'pédagogie']
  };

  // Termes associés aux types de requêtes
  private queryTypeTerms: Record<string, string[]> = {
    informational: ['qu\'est-ce que', 'comment', 'pourquoi', 'quand', 'où', 'qui', 'définir', 'expliquer'],
    actionable: ['faire', 'créer', 'mettre en œuvre', 'planifier', 'exécuter', 'comment faire', 'étapes'],
    comparative: ['différence', 'comparer', 'versus', 'vs', 'avantages', 'inconvénients', 'meilleur'],
    analytical: ['analyser', 'évaluer', 'impact', 'causes', 'effets', 'conséquences', 'tendances']
  };
  
  constructor(
    @Inject(LOGGER_TOKEN) private readonly logger: ILogger,
    private readonly eventBus: EventBusService,
    private readonly knowledgeGraph: KnowledgeGraphService
  ) {}

  /**
   * Analyse une requête utilisateur
   * @param query Requête à analyser
   * @returns Résultat de l'analyse
   */
  async analyzeQuery(query: UserQuery): Promise<QueryAnalysisResult> {
    const startTime = Date.now();
    let text: string;
    
    if (typeof query === 'string') {
      text = query;
    } else if ('text' in query) {
      text = query.text;
    } else {
      // Fallback pour autres formats potentiels
      text = (query as any).content || '';
    }
    
    this.logger.debug(`Analyse de la requête: "${text.substring(0, 50)}${text.length > 50 ? '...' : ''}"`, 
      { queryLength: text.length });

    // Émettre un événement de début d'analyse
    this.eventBus.emit({
      type: RagKagEventType.QUERY_RECEIVED,
      source: 'QueryAnalyzerService',
      payload: { query: text }
    });

    // Effectuer l'analyse
    const languageDetected = this.detectLanguage(text);
    const domains = this.detectDomains(text);
    const keywords = this.extractKeywords(text);
    const entities = this.extractEntities(text);
    const sentiment = this.analyzeSentiment(text);
    const complexity = this.computeComplexity(text);
    const queryType = this.determineQueryType(text);
    const suggestedComponents = this.suggestComponents(queryType, complexity, domains);
    
    // Construction du résultat
    const result: QueryAnalysisResult = {
      queryType,
      domains,
      keywords,
      entities,
      sentiment,
      complexity,
      languageDetected,
      confidenceScore: this.calculateConfidenceScore(domains, keywords, complexity),
      suggestedComponents
    };

    // Enrichir le graphe de connaissances avec les informations de la requête
    this.enrichKnowledgeGraph(text, result);
    
    // Émettre un événement de fin d'analyse
    const processingTime = Date.now() - startTime;
    this.eventBus.emit({
      type: RagKagEventType.QUERY_PROCESSED,
      source: 'QueryAnalyzerService',
      payload: { 
        query: text,
        analysis: result,
        processingTime
      }
    });
    
    this.logger.debug(`Analyse de requête terminée en ${processingTime}ms`, {
      queryType: result.queryType,
      domains: result.domains,
      keywords: result.keywords.length,
      complexity: result.complexity,
      confidence: result.confidenceScore
    });
    
    return result;
  }

  /**
   * Détecte les domaines associés à la requête
   * @param text Texte de la requête
   * @returns Liste des domaines identifiés
   */
  private detectDomains(text: string): string[] {
    const textLower = text.toLowerCase();
    const domainScores: Record<string, number> = {};
    
    // Calculer un score pour chaque domaine
    for (const [domain, terms] of Object.entries(this.domainTerms)) {
      domainScores[domain] = 0;
      
      for (const term of terms) {
        if (textLower.includes(term)) {
          domainScores[domain] += 1;
        }
      }
      
      // Normaliser le score par rapport au nombre de termes
      domainScores[domain] = domainScores[domain] / terms.length;
    }
    
    // Filtrer les domaines avec un score supérieur au seuil
    const detectedDomains = Object.entries(domainScores)
      .filter(([_, score]) => score >= this.config.domainDetectionSensitivity * 0.5)
      .sort((a, b) => b[1] - a[1]) // Trier par score décroissant
      .map(([domain]) => domain);
    
    return detectedDomains;
  }

  /**
   * Extrait les mots-clés de la requête
   * @param text Texte de la requête
   * @returns Liste des mots-clés
   */
  private extractKeywords(text: string): string[] {
    // Mots vides à ignorer
    const stopWords = ['le', 'la', 'les', 'un', 'une', 'des', 'et', 'ou', 'de', 'du', 'à', 'en', 'pour', 'avec', 'par'];
    
    // Nettoyer et tokeniser le texte
    const cleanText = text.toLowerCase()
      .replace(/[.,!?;:()"']/g, ' ')
      .replace(/\s+/g, ' ')
      .trim();
    
    const words = cleanText.split(' ');
    
    // Filtrer les mots vides et les mots trop courts
    const keywords = words.filter(word => 
      word.length >= this.config.keywordMinLength && 
      !stopWords.includes(word)
    );
    
    // Éliminer les doublons
    return [...new Set(keywords)];
  }

  /**
   * Extrait les entités nommées de la requête
   * @param text Texte de la requête
   * @returns Liste des entités identifiées
   */
  private extractEntities(text: string): string[] {
    // Implémentation simplifée - dans un cas réel, on utiliserait un modèle NER
    const entities: string[] = [];
    
    // Détecter les entités potentielles par les majuscules
    const potentialEntities = text.match(/[A-Z][a-zÀ-ÿ]+(?:\s+[A-Z][a-zÀ-ÿ]+)*/g) || [];
    
    // Filtrer et ajouter à la liste
    for (const entity of potentialEntities) {
      if (entity.length > 1 && !entities.includes(entity)) {
        entities.push(entity);
      }
    }
    
    return entities;
  }

  /**
   * Analyse le sentiment exprimé dans la requête
   * @param text Texte de la requête
   * @returns Score de sentiment (-1 à 1)
   */
  private analyzeSentiment(text: string): number {
    // Implémentation simplifiée - dans un cas réel, on utiliserait un modèle
    const textLower = text.toLowerCase();
    
    // Mots positifs
    const positiveTerms = ['bon', 'bien', 'excellent', 'super', 'génial', 'aime', 'positif', 'avantage'];
    // Mots négatifs
    const negativeTerms = ['mauvais', 'mal', 'problème', 'difficulté', 'négatif', 'inconvénient', 'erreur'];
    
    let positiveScore = 0;
    let negativeScore = 0;
    
    for (const term of positiveTerms) {
      if (textLower.includes(term)) {
        positiveScore += 1;
      }
    }
    
    for (const term of negativeTerms) {
      if (textLower.includes(term)) {
        negativeScore += 1;
      }
    }
    
    // Normaliser les scores
    const normalizedPositive = positiveScore / positiveTerms.length;
    const normalizedNegative = negativeScore / negativeTerms.length;
    
    // Calculer le score final entre -1 et 1
    return normalizedPositive - normalizedNegative;
  }

  /**
   * Calcule la complexité de la requête
   * @param text Texte de la requête
   * @returns Score de complexité (0 à 1)
   */
  private computeComplexity(text: string): number {
    // Facteurs de complexité
    const words = text.split(/\s+/).length;
    const sentences = text.split(/[.!?]+/).filter(s => s.trim().length > 0).length;
    const avgWordLength = text.replace(/[.,!?;:()"'\s]/g, '').length / words;
    const complexWords = text.split(/\s+/).filter(word => word.length > 6).length / words;
    
    // Mots indiquant des relations complexes
    const complexityIndicators = ['pourquoi', 'comment', 'relation', 'impact', 'effet', 'cause', 'analyser'];
    let indicatorCount = 0;
    
    for (const indicator of complexityIndicators) {
      if (text.toLowerCase().includes(indicator)) {
        indicatorCount += 1;
      }
    }
    
    // Pondération des facteurs
    const lengthFactor = Math.min(words / 50, 1) * 0.3; // Longueur
    const sentenceFactor = Math.min(sentences / 5, 1) * 0.2; // Nombre de phrases
    const wordLengthFactor = Math.min((avgWordLength - 3) / 3, 1) * 0.15; // Longueur moyenne des mots
    const complexWordsFactor = complexWords * 0.15; // Proportion de mots complexes
    const indicatorFactor = (indicatorCount / complexityIndicators.length) * 0.2; // Indicateurs de complexité
    
    // Score final (0 à 1)
    let complexityScore = lengthFactor + sentenceFactor + wordLengthFactor + complexWordsFactor + indicatorFactor;
    
    // Limiter le score entre 0 et 1
    return Math.max(0, Math.min(1, complexityScore));
  }

  /**
   * Détecte la langue de la requête
   * @param text Texte de la requête
   * @returns Code de langue détecté
   */
  private detectLanguage(text: string): string {
    if (!this.config.languageDetectionEnabled) {
      return this.config.defaultLanguage;
    }
    
    // Implémentation simplifiée - dans un cas réel, on utiliserait un détecteur de langue
    const frenchIndicators = ['le', 'la', 'les', 'un', 'une', 'des', 'et', 'ou', 'de', 'du', 'à', 'en', 'pour', 'avec'];
    const englishIndicators = ['the', 'a', 'an', 'and', 'or', 'of', 'to', 'in', 'for', 'with', 'by', 'at', 'from'];
    
    let frenchScore = 0;
    let englishScore = 0;
    
    const words = text.toLowerCase().split(/\s+/);
    
    for (const word of words) {
      if (frenchIndicators.includes(word)) {
        frenchScore += 1;
      }
      if (englishIndicators.includes(word)) {
        englishScore += 1;
      }
    }
    
    if (frenchScore > englishScore) {
      return 'fr';
    } else if (englishScore > frenchScore) {
      return 'en';
    } else {
      // Si égalité ou pas de détection claire, utiliser la langue par défaut
      return this.config.defaultLanguage;
    }
  }

  /**
   * Détermine le type de requête
   * @param text Texte de la requête
   * @returns Type de requête identifié
   */
  private determineQueryType(text: string): string {
    const textLower = text.toLowerCase();
    const typeScores: Record<string, number> = {};
    
    // Calculer un score pour chaque type de requête
    for (const [type, terms] of Object.entries(this.queryTypeTerms)) {
      typeScores[type] = 0;
      
      for (const term of terms) {
        if (textLower.includes(term)) {
          typeScores[type] += 1;
        }
      }
      
      // Normaliser le score
      typeScores[type] = typeScores[type] / terms.length;
    }
    
    // Trouver le type avec le score le plus élevé
    let maxScore = 0;
    let detectedType = 'informational'; // Type par défaut
    
    for (const [type, score] of Object.entries(typeScores)) {
      if (score > maxScore) {
        maxScore = score;
        detectedType = type;
      }
    }
    
    return detectedType;
  }

  /**
   * Suggère les composants à utiliser pour traiter la requête
   * @param queryType Type de requête
   * @param complexity Complexité de la requête
   * @param domains Domaines identifiés
   * @returns Liste des composants suggérés
   */
  private suggestComponents(queryType: string, complexity: number, domains: string[]): ComponentType[] {
    const components: ComponentType[] = [];
    
    // Toujours inclure l'analyseur de requête
    components.push(ComponentType.QUERY_ANALYZER);
    
    // Pour les requêtes informationnelles, privilégier RAG
    if (queryType === 'informational') {
      components.push(ComponentType.KNOWLEDGE_RETRIEVER);
      components.push(ComponentType.RAG_ENGINE);
      
      // Si la complexité est élevée, ajouter aussi KAG
      if (complexity > this.config.minComplexityThreshold) {
        components.push(ComponentType.KAG_ENGINE);
        components.push(ComponentType.DEBATE_PROTOCOL);
      }
    }
    // Pour les requêtes d'analyse, privilégier KAG
    else if (queryType === 'analytical') {
      components.push(ComponentType.KAG_ENGINE);
      
      // Si la complexité est élevée, ajouter aussi RAG pour le contexte
      if (complexity > this.config.minComplexityThreshold) {
        components.push(ComponentType.KNOWLEDGE_RETRIEVER);
        components.push(ComponentType.RAG_ENGINE);
        components.push(ComponentType.DEBATE_PROTOCOL);
      }
    }
    // Pour les requêtes d'action ou comparatives, utiliser les deux
    else {
      components.push(ComponentType.KNOWLEDGE_RETRIEVER);
      components.push(ComponentType.RAG_ENGINE);
      components.push(ComponentType.KAG_ENGINE);
      components.push(ComponentType.DEBATE_PROTOCOL);
    }
    
    // Toujours inclure la synthèse
    components.push(ComponentType.SYNTHESIS);
    
    return components;
  }

  /**
   * Calcule le score de confiance global pour l'analyse
   * @param domains Domaines identifiés
   * @param keywords Mots-clés extraits
   * @param complexity Complexité calculée
   * @returns Score de confiance (0 à 1)
   */
  private calculateConfidenceScore(domains: string[], keywords: string[], complexity: number): number {
    // Facteurs affectant la confiance
    const hasDomains = domains.length > 0;
    const hasKeywords = keywords.length >= 3;
    const isComplexityManageable = complexity <= 0.8; // Trop complexe = moins de confiance
    
    // Pondération des facteurs
    const domainFactor = hasDomains ? 0.4 : 0.1;
    const keywordFactor = hasKeywords ? 0.3 : 0.1;
    const complexityFactor = isComplexityManageable ? 0.3 : 0.1;
    
    // Score final (0 à 1)
    return domainFactor + keywordFactor + complexityFactor;
  }

  /**
   * Enrichit le graphe de connaissances avec les informations de la requête
   * @param text Texte de la requête
   * @param analysis Résultat de l'analyse
   */
  private enrichKnowledgeGraph(text: string, analysis: QueryAnalysisResult): void {
    try {
      // Créer un nœud pour la requête
      const queryNodeId = this.knowledgeGraph.addNode({
        label: `Query: ${text.substring(0, 30)}${text.length > 30 ? '...' : ''}`,
        type: 'QUERY',
        content: text,
        confidence: analysis.confidenceScore,
        source: KnowledgeSource.USER_INPUT
      });
      
      // Ajouter les mots-clés comme nœuds
      for (const keyword of analysis.keywords) {
        // Créer ou récupérer un nœud pour le mot-clé
        const { subjectId } = this.knowledgeGraph.addFact(
          queryNodeId,
          'HAS_KEYWORD',
          {
            label: keyword,
            type: 'KEYWORD',
            content: keyword,
            confidence: 0.8,
            source: KnowledgeSource.INFERENCE
          },
          0.8,
          {
            bidirectional: false,
            weight: 0.7
          }
        );
      }
      
      // Ajouter les domaines
      for (const domain of analysis.domains) {
        this.knowledgeGraph.addFact(
          queryNodeId,
          'BELONGS_TO_DOMAIN',
          {
            label: domain,
            type: 'DOMAIN',
            content: `Domaine: ${domain}`,
            confidence: 0.9,
            source: KnowledgeSource.INFERENCE
          },
          0.9,
          {
            bidirectional: false,
            weight: 0.8
          }
        );
      }
      
      // Ajouter les entités
      for (const entity of analysis.entities) {
        this.knowledgeGraph.addFact(
          queryNodeId,
          'MENTIONS_ENTITY',
          {
            label: entity,
            type: 'ENTITY',
            content: entity,
            confidence: 0.7,
            source: KnowledgeSource.INFERENCE
          },
          0.7,
          {
            bidirectional: false,
            weight: 0.6
          }
        );
      }
      
      this.logger.debug(`Graphe de connaissances enrichi avec la requête et ${analysis.keywords.length} mots-clés, ${analysis.domains.length} domaines, ${analysis.entities.length} entités`, {
        queryNodeId
      });
    } catch (error) {
      this.logger.error(`Erreur lors de l'enrichissement du graphe de connaissances: ${error.message}`, { error });
    }
  }

  /**
   * Extrait les connaissances clés du texte
   * @param content Contenu de l'analyse
   * @returns Liste des connaissances clés
   */
  private extractKeyInsights(content: string): string[] {
    // Approche simplifiée - dans un cas réel, on utiliserait NLP
    const paragraphs = content.split(/\n\n|\r\n\r\n/);
    const insights: string[] = [];
    
    for (const paragraph of paragraphs) {
      if (paragraph.length > 50 && paragraph.length < 200) {
        insights.push(paragraph.trim());
      }
      
      if (insights.length >= 3) break; // Limiter à 3 insights
    }
    
    return insights;
  }
} 