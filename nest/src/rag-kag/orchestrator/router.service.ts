import { Injectable, Inject } from '@nestjs/common';
import { LOGGER_TOKEN, ILogger } from '../utils/logger-tokens';
import { UserQuery, TargetPools } from '../types';

/**
 * Service de routage des requêtes vers les pools appropriés
 */
@Injectable()
export class RouterService {
  private readonly logger: ILogger;
  
  constructor(
    @Inject(LOGGER_TOKEN) logger: ILogger
  ) {
    this.logger = logger;
    this.logger.info('Service de routage initialisé');
  }
  
  /**
   * Détermine quels pools d'agents cibler en fonction de la requête
   * @param query Requête utilisateur
   * @returns Configuration des pools à activer
   */
  async determineTargetPools(query: UserQuery): Promise<TargetPools> {
    this.logger.debug('Analyse de la requête pour déterminer les pools', { queryId: query.sessionId });
    
    // Dans une implémentation réelle, on utiliserait des modèles d'analyse 
    // plus sophistiqués avec NLP pour déterminer les pools pertinents
    
    // Recherche simpliste de mots-clés
    const text = query.text.toLowerCase();
    
    // Analyse de domaine basique
    const hasCommercialTerms = this.hasCommercialTerms(text);
    const hasMarketingTerms = this.hasMarketingTerms(text);
    const hasSectorialTerms = this.hasSectorialTerms(text);
    
    // Prendre en compte les indices de domaine fournis par l'utilisateur
    if (query.domainHints && query.domainHints.length > 0) {
      for (const hint of query.domainHints) {
        const lowerHint = hint.toLowerCase();
        
        if (this.commercialDomains.includes(lowerHint)) {
          this.logger.debug('Indice explicite de domaine commercial détecté');
          return {
            commercial: true,
            marketing: false,
            sectoriel: false,
            primaryFocus: 'COMMERCIAL'
          };
        }
        
        if (this.marketingDomains.includes(lowerHint)) {
          this.logger.debug('Indice explicite de domaine marketing détecté');
          return {
            commercial: false,
            marketing: true,
            sectoriel: false,
            primaryFocus: 'MARKETING'
          };
        }
        
        if (this.sectorialDomains.includes(lowerHint)) {
          this.logger.debug('Indice explicite de domaine sectoriel détecté');
          return {
            commercial: false,
            marketing: false,
            sectoriel: true,
            primaryFocus: 'SECTORIEL'
          };
        }
      }
    }
    
    // Stratégie de priorité basique
    let primaryFocus = undefined;
    
    if (hasCommercialTerms && !hasMarketingTerms && !hasSectorialTerms) {
      primaryFocus = 'COMMERCIAL';
    } else if (!hasCommercialTerms && hasMarketingTerms && !hasSectorialTerms) {
      primaryFocus = 'MARKETING';
    } else if (!hasCommercialTerms && !hasMarketingTerms && hasSectorialTerms) {
      primaryFocus = 'SECTORIEL';
    } else if (hasCommercialTerms && hasMarketingTerms && !hasSectorialTerms) {
      // Si à la fois commercial et marketing, choisir le plus probable
      primaryFocus = this.countTerms(text, this.commercialTerms) > this.countTerms(text, this.marketingTerms)
        ? 'COMMERCIAL' : 'MARKETING';
    }
    
    const result: TargetPools = {
      commercial: hasCommercialTerms,
      marketing: hasMarketingTerms,
      sectoriel: hasSectorialTerms,
      primaryFocus
    };
    
    this.logger.debug('Pools ciblés identifiés', { result });
    return result;
  }
  
  // Termes de domaine - dans une implémentation réelle, ces listes seraient bien plus étendues
  private commercialTerms = ['vente', 'client', 'offre', 'produit', 'service', 'prix', 'commande'];
  private marketingTerms = ['campagne', 'publicité', 'marque', 'segment', 'cible', 'promotion'];
  private sectorialTerms = ['secteur', 'industrie', 'marché', 'concurrence', 'tendance'];
  
  // Domaines explicites
  private commercialDomains = ['commercial', 'vente', 'crm'];
  private marketingDomains = ['marketing', 'publicité', 'communication'];
  private sectorialDomains = ['sectoriel', 'industriel', 'marché'];
  
  /**
   * Vérifie si le texte contient des termes commerciaux
   */
  private hasCommercialTerms(text: string): boolean {
    return this.commercialTerms.some(term => text.includes(term));
  }
  
  /**
   * Vérifie si le texte contient des termes marketing
   */
  private hasMarketingTerms(text: string): boolean {
    return this.marketingTerms.some(term => text.includes(term));
  }
  
  /**
   * Vérifie si le texte contient des termes sectoriels
   */
  private hasSectorialTerms(text: string): boolean {
    return this.sectorialTerms.some(term => text.includes(term));
  }
  
  /**
   * Compte le nombre de termes présents dans le texte
   */
  private countTerms(text: string, terms: string[]): number {
    return terms.filter(term => text.includes(term)).length;
  }
} 