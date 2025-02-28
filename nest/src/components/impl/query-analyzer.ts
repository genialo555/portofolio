import { v4 as uuidv4 } from 'uuid';
import { ComponentType, ComponentRegistration } from '../registry';
import { ComponentPriority, ComponentStatus, CoordinationContext } from '../../handlers/coordination-handler';
import { Logger, LogLevel } from '../../utils/logger';

/**
 * Résultat de l'analyse de requête
 */
export interface QueryAnalysisResult {
  queryType: string;            // Type de requête (information, action, etc.)
  domains: string[];            // Domaines associés à la requête
  keywords: string[];           // Mots-clés extraits
  entities: string[];           // Entités nommées détectées
  sentiment: number;            // Score de sentiment (-1 à 1)
  complexity: number;           // Score de complexité (0 à 1)
  languageDetected: string;     // Langue détectée
  confidenceScore: number;      // Score de confiance global
  requiredComponents: string[]; // Composants recommandés pour traiter la requête
}

/**
 * Composant d'analyse de requête avancé
 * Ce composant analyse la requête utilisateur pour déterminer ses caractéristiques
 * et orienter le traitement par les composants suivants.
 */
export class QueryAnalyzerComponent {
  private logger: Logger;
  private config: {
    minComplexityThreshold: number;
    keywordMinLength: number;
    confidenceThreshold: number;
    languageDetectionEnabled: boolean;
    defaultLanguage: string;
    domainDetectionSensitivity: number;
  };

  /**
   * Crée un composant d'analyse de requête
   * @param logger Instance du logger
   */
  constructor(logger: Logger) {
    this.logger = logger;
    
    // Configuration par défaut
    this.config = {
      minComplexityThreshold: 0.3,
      keywordMinLength: 4,
      confidenceThreshold: 0.7,
      languageDetectionEnabled: true,
      defaultLanguage: 'fr',
      domainDetectionSensitivity: 0.6
    };
  }

  /**
   * Crée l'enregistrement du composant pour le registre
   * @returns Enregistrement du composant
   */
  public createRegistration(): ComponentRegistration {
    return {
      id: `query-analyzer-${uuidv4().substring(0, 8)}`,
      type: ComponentType.QUERY_ANALYZER,
      name: "Analyseur de Requête Avancé",
      description: "Analyse les requêtes utilisateur pour déterminer leurs caractéristiques et orienter le traitement",
      version: "1.0.0",
      priority: ComponentPriority.CRITICAL, // Priorité critique car première étape
      executeFunction: this.execute.bind(this),
      isEnabled: true
    };
  }

  /**
   * Analyse une requête
   * @param context Contexte de coordination
   * @returns Résultat de l'analyse
   */
  private async execute(context: CoordinationContext): Promise<QueryAnalysisResult> {
    this.logger.debug(`[${context.traceId}] Démarrage analyse de requête: "${context.query.substring(0, 50)}..."`);
    
    const startTime = Date.now();
    const query = context.query;
    
    try {
      // Extraire les mots-clés (implémentation simplifiée)
      const keywords = this.extractKeywords(query);
      
      // Déterminer les domaines concernés
      const domains = this.detectDomains(query, keywords);
      
      // Estimer la complexité
      const complexity = this.estimateComplexity(query, keywords);
      
      // Détecter le sentiment
      const sentiment = this.detectSentiment(query);
      
      // Déterminer la langue
      const languageDetected = this.detectLanguage(query);
      
      // Identifier les entités
      const entities = this.extractEntities(query);
      
      // Déterminer les composants requis
      const requiredComponents = this.determineRequiredComponents(domains, complexity, entities);
      
      // Calculer le score de confiance global
      const confidenceScore = this.calculateConfidenceScore(keywords, domains, complexity);
      
      // Déterminer le type de requête
      const queryType = this.determineQueryType(query, domains, entities);
      
      const result: QueryAnalysisResult = {
        queryType,
        domains,
        keywords,
        entities,
        sentiment,
        complexity,
        languageDetected,
        confidenceScore,
        requiredComponents
      };
      
      const duration = Date.now() - startTime;
      this.logger.debug(`[${context.traceId}] Analyse de requête terminée en ${duration}ms`);
      
      return result;
      
    } catch (error) {
      this.logger.error(`[${context.traceId}] Erreur lors de l'analyse de requête: ${error.message}`);
      throw error;
    }
  }

  /**
   * Extrait les mots-clés d'une requête
   * @param query Requête à analyser
   * @returns Liste des mots-clés
   */
  private extractKeywords(query: string): string[] {
    // Implémentation simplifiée
    const stopWords = ['le', 'la', 'les', 'un', 'une', 'des', 'et', 'ou', 'pour', 'par', 'dans', 'sur', 'avec'];
    
    return query
      .toLowerCase()
      .replace(/[.,?!;:'"()\[\]{}]/g, ' ')
      .split(/\s+/)
      .filter(word => 
        word.length >= this.config.keywordMinLength && 
        !stopWords.includes(word)
      );
  }

  /**
   * Détecte les domaines concernés par la requête
   * @param query Requête à analyser
   * @param keywords Mots-clés extraits
   * @returns Liste des domaines
   */
  private detectDomains(query: string, keywords: string[]): string[] {
    // Définition simplifiée des domaines et de leurs termes associés
    const domainKeywords: Record<string, string[]> = {
      'commercial': ['vente', 'client', 'achat', 'prix', 'produit', 'offre', 'commercial'],
      'marketing': ['marketing', 'publicité', 'campagne', 'stratégie', 'marque', 'cible', 'promotion'],
      'finance': ['finance', 'budget', 'coût', 'investissement', 'profit', 'rentabilité'],
      'technique': ['technique', 'logiciel', 'application', 'développement', 'système', 'technologie'],
      'ressources_humaines': ['recrutement', 'employé', 'poste', 'carrière', 'compétence', 'formation']
    };
    
    const domains: string[] = [];
    
    // Calculer les scores pour chaque domaine
    Object.entries(domainKeywords).forEach(([domain, domainTerms]) => {
      let score = 0;
      
      // Vérifier la présence des termes du domaine dans les mots-clés
      keywords.forEach(keyword => {
        if (domainTerms.some(term => keyword.includes(term) || term.includes(keyword))) {
          score += 1;
        }
      });
      
      // Normaliser le score par rapport au nombre de mots-clés
      const normalizedScore = keywords.length > 0 ? score / keywords.length : 0;
      
      // Ajouter le domaine si le score dépasse le seuil de sensibilité
      if (normalizedScore >= this.config.domainDetectionSensitivity) {
        domains.push(domain);
      }
    });
    
    // Si aucun domaine n'est détecté, utiliser 'general'
    if (domains.length === 0) {
      domains.push('general');
    }
    
    return domains;
  }

  /**
   * Estime la complexité de la requête
   * @param query Requête à analyser
   * @param keywords Mots-clés extraits
   * @returns Score de complexité (0-1)
   */
  private estimateComplexity(query: string, keywords: string[]): number {
    // Facteurs de complexité (implémentation simplifiée)
    const factors = {
      length: Math.min(1, query.length / 200),  // Longueur max considérée: 200 caractères
      keywords: Math.min(1, keywords.length / 10),  // Nombre max de mots-clés: 10
      questionCount: (query.match(/\?/g) || []).length / 3,  // Nombre de questions (max 3)
      complexity: ((query.match(/comment|pourquoi|expliquer|analyser|comparer/gi) || []).length) / 2  // Termes complexes (max 2)
    };
    
    // Pondération des facteurs
    const weights = {
      length: 0.2,
      keywords: 0.3,
      questionCount: 0.2,
      complexity: 0.3
    };
    
    // Calcul du score pondéré
    let score = 0;
    for (const [factor, value] of Object.entries(factors)) {
      score += value * weights[factor as keyof typeof weights];
    }
    
    // Limiter entre 0 et 1
    return Math.max(0, Math.min(1, score));
  }

  /**
   * Détecte le sentiment exprimé dans la requête
   * @param query Requête à analyser
   * @returns Score de sentiment (-1 à 1)
   */
  private detectSentiment(query: string): number {
    // Implémentation simplifiée du sentiment
    const positiveTerms = ['bon', 'bien', 'excellent', 'super', 'génial', 'content', 'satisfait', 'aime', 'positif'];
    const negativeTerms = ['mauvais', 'problème', 'erreur', 'bug', 'difficile', 'compliqué', 'déçu', 'négatif', 'impossible'];
    
    let score = 0;
    const lowercaseQuery = query.toLowerCase();
    
    // Détecter termes positifs
    positiveTerms.forEach(term => {
      if (lowercaseQuery.includes(term)) {
        score += 0.2;  // +0.2 par terme positif
      }
    });
    
    // Détecter termes négatifs
    negativeTerms.forEach(term => {
      if (lowercaseQuery.includes(term)) {
        score -= 0.2;  // -0.2 par terme négatif
      }
    });
    
    // Limiter entre -1 et 1
    return Math.max(-1, Math.min(1, score));
  }

  /**
   * Détecte la langue de la requête
   * @param query Requête à analyser
   * @returns Code ISO de la langue
   */
  private detectLanguage(query: string): string {
    // Implémentation simplifiée (à remplacer par une vraie librairie de détection)
    if (!this.config.languageDetectionEnabled) {
      return this.config.defaultLanguage;
    }
    
    // Heuristique très simplifiée basée sur quelques mots spécifiques
    const languagePatterns: Record<string, RegExp> = {
      'fr': /^(le|la|les|un|une|des|et|ou|pour|comment|pourquoi|quand|où)\s|est-ce que/i,
      'en': /^(the|a|an|and|or|for|how|why|when|where)\s|is it/i,
      'es': /^(el|la|los|las|un|una|unos|unas|y|o|para|cómo|por qué|cuándo|dónde)\s|es/i
    };
    
    for (const [lang, pattern] of Object.entries(languagePatterns)) {
      if (pattern.test(query)) {
        return lang;
      }
    }
    
    // Par défaut
    return this.config.defaultLanguage;
  }

  /**
   * Extrait les entités nommées de la requête
   * @param query Requête à analyser
   * @returns Liste des entités détectées
   */
  private extractEntities(query: string): string[] {
    // Implémentation simplifiée (à remplacer par un vrai NER)
    const entities: string[] = [];
    
    // Recherche simplifiée de patterns pouvant indiquer des entités
    
    // Majuscules suivies de minuscules (potentiellement un nom propre)
    const capitalizedWords = query.match(/[A-Z][a-z]+/g) || [];
    capitalizedWords.forEach(word => {
      if (!entities.includes(word)) {
        entities.push(word);
      }
    });
    
    // Recherche de motifs numériques (prix, dates, etc.)
    const numericPatterns = query.match(/\d+([.,]\d+)?(\s*[€$%])?/g) || [];
    numericPatterns.forEach(pattern => {
      if (!entities.includes(pattern)) {
        entities.push(pattern);
      }
    });
    
    return entities;
  }

  /**
   * Détermine les composants requis pour traiter cette requête
   * @param domains Domaines détectés
   * @param complexity Complexité estimée
   * @param entities Entités détectées
   * @returns Liste des types de composants recommandés
   */
  private determineRequiredComponents(domains: string[], complexity: number, entities: string[]): string[] {
    const requiredComponents: string[] = [];
    
    // Toujours ajouter le sélecteur de pools
    requiredComponents.push(ComponentType.POOL_SELECTOR);
    
    // Ajouter les agents spécifiques selon les domaines
    if (domains.includes('commercial')) {
      requiredComponents.push(ComponentType.COMMERCIAL_AGENT);
    }
    
    if (domains.includes('marketing')) {
      requiredComponents.push(ComponentType.MARKETING_AGENT);
    }
    
    if (domains.some(d => !['commercial', 'marketing', 'general'].includes(d))) {
      requiredComponents.push(ComponentType.SECTORAL_AGENT);
    }
    
    // Pour les requêtes complexes, utiliser des moteurs avancés
    if (complexity > 0.6) {
      requiredComponents.push(ComponentType.DEBATE_ENGINE);
      requiredComponents.push(ComponentType.SYNTHESIS_ENGINE);
    }
    
    // Si complexité moyenne ou haute, ajouter le détecteur d'anomalies
    if (complexity > 0.4) {
      requiredComponents.push(ComponentType.ANOMALY_DETECTOR);
    }
    
    // Si entités détectées, utiliser le moteur RAG
    if (entities.length > 0) {
      requiredComponents.push(ComponentType.RAG_ENGINE);
    }
    
    // Si complexité élevée, ajouter KAG
    if (complexity > 0.7) {
      requiredComponents.push(ComponentType.KAG_ENGINE);
    }
    
    // Toujours ajouter le formateur de sortie
    requiredComponents.push(ComponentType.OUTPUT_FORMATTER);
    
    return requiredComponents;
  }

  /**
   * Calcule le score de confiance global
   * @param keywords Mots-clés extraits
   * @param domains Domaines détectés
   * @param complexity Complexité estimée
   * @returns Score de confiance (0-1)
   */
  private calculateConfidenceScore(keywords: string[], domains: string[], complexity: number): number {
    // Facteurs affectant la confiance
    let confidenceScore = 0.5; // Base de départ
    
    // Plus de mots-clés = plus de confiance
    confidenceScore += Math.min(0.2, keywords.length * 0.02);
    
    // Domaines spécifiques = plus de confiance
    if (domains.length > 0 && !domains.includes('general')) {
      confidenceScore += 0.1;
    }
    
    // Complexité modérée = plus de confiance 
    // (trop simple ou trop complexe réduit la confiance)
    const complexityFactor = -0.2 * Math.pow(complexity - 0.5, 2) + 0.2;
    confidenceScore += complexityFactor;
    
    // Limiter entre 0 et 1
    return Math.max(0, Math.min(1, confidenceScore));
  }

  /**
   * Détermine le type de requête
   * @param query Requête à analyser
   * @param domains Domaines détectés
   * @param entities Entités détectées
   * @returns Type de requête
   */
  private determineQueryType(query: string, domains: string[], entities: string[]): string {
    // Vérifier s'il s'agit d'une question
    if (query.includes('?') || /^(comment|pourquoi|quand|où|qui|quoi|combien)/i.test(query)) {
      return 'question';
    }
    
    // Vérifier s'il s'agit d'une commande
    if (/^(faire|trouver|chercher|analyser|comparer|créer|générer)/i.test(query)) {
      return 'command';
    }
    
    // Vérifier s'il s'agit d'une recherche
    if (entities.length > 0 || query.length < 20) {
      return 'search';
    }
    
    // Par défaut, considérer comme une demande d'information
    return 'information';
  }
} 