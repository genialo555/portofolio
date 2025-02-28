import { Injectable, Inject } from '@nestjs/common';
import { LOGGER_TOKEN, ILogger } from '../utils/logger-tokens';
import { PromptsService, PromptTemplateType } from '../prompts/prompts.service';
import { ApiProviderFactory } from '../apis/api-provider-factory.service';
import { 
  UserQuery, 
  DebateResult, 
  FinalResponse, 
  FinalResponseOptions, 
  ExpertiseLevel,
  ConfidenceLevel
} from '../types';

/**
 * Options pour la synthèse
 */
export interface SynthesisOptions {
  expertiseLevel: ExpertiseLevel; 
  includeSuggestions: boolean;
  maxLength?: number;
}

/**
 * Service responsable de la génération des réponses finales
 */
@Injectable()
export class SynthesisService {
  private readonly logger: ILogger;
  private readonly defaultOptions: SynthesisOptions = {
    expertiseLevel: 'INTERMEDIATE',
    includeSuggestions: true,
    maxLength: 1500
  };

  constructor(
    @Inject(LOGGER_TOKEN) logger: ILogger,
    private readonly promptsService: PromptsService,
    private readonly apiProviderFactory: ApiProviderFactory
  ) {
    this.logger = logger;
    this.logger.info('Service de synthèse initialisé');
  }

  /**
   * Génère une réponse finale à partir des résultats du débat
   * @param query Requête utilisateur
   * @param debateResult Résultat du débat
   * @param options Options de synthèse
   * @returns Réponse finale
   */
  async generateFinalResponse(
    query: UserQuery,
    debateResult: DebateResult,
    options?: Partial<SynthesisOptions>
  ): Promise<FinalResponse> {
    const startTime = Date.now();
    const mergedOptions = { ...this.defaultOptions, ...options };
    
    this.logger.debug('Génération de la réponse finale', {
      queryId: query.sessionId,
      hasConsensus: debateResult.hasConsensus,
      expertiseLevel: mergedOptions.expertiseLevel
    });

    try {
      // 1. Déterminer l'approche de génération
      const hasConsensus = debateResult.hasConsensus;
      
      // 2. Obtenir le prompt de synthèse
      const synthesisPrompt = this.promptsService.getPromptTemplate(PromptTemplateType.SYNTHESIS);
      
      // 3. Préparer les données pour le prompt
      const promptData = {
        query: query.text,
        debateResult: debateResult.content,
        expertiseLevel: mergedOptions.expertiseLevel,
        themes: debateResult.identifiedThemes.join(', ')
      };
      
      // 4. Construire le prompt complet
      const fullPrompt = this.promptsService.fillTemplate(synthesisPrompt, promptData);
      
      // 5. Générer la réponse via une API
      const synthesisResponse = await this.apiProviderFactory.generateResponse(
        'google', // Préférer Google pour la synthèse finale
        fullPrompt,
        {
          temperature: hasConsensus ? 0.3 : 0.5, // Plus créatif si pas de consensus
          top_p: 0.92,
          top_k: 40,
          max_tokens: Math.min(2000, mergedOptions.maxLength * 1.5) // Marge pour post-traitement
        }
      );
      
      // 6. Traiter le contenu généré
      let finalContent = '';
      
      if (hasConsensus) {
        // Si consensus, formater directement le contenu
        finalContent = this.formatContentForExpertiseLevel(
          synthesisResponse.text,
          mergedOptions.expertiseLevel
        );
      } else {
        // Si pas de consensus, créer un contenu multi-perspectives
        finalContent = this.createMultiPerspectiveContent(
          synthesisResponse.text,
          debateResult,
          mergedOptions.expertiseLevel
        );
      }
      
      // 7. Préparer les suggestions de suivi si demandé
      let suggestions: string[] = [];
      
      if (mergedOptions.includeSuggestions) {
        suggestions = await this.generateFollowUpSuggestions(
          query,
          debateResult.identifiedThemes
        );
      }
      
      // 8. Calculer le niveau de confiance agrégé
      const confidenceLevel = this.deriveOverallConfidenceLevel(debateResult);
      
      // 9. Retourner la réponse finale
      const processingTime = Date.now() - startTime;
      const finalResponse: FinalResponse = {
        content: finalContent,
        metaData: {
          sourceTypes: this.countDistinctSourceTypes(debateResult),
          confidenceLevel,
          processingTime,
          usedAgentCount: this.countDistinctAgents(debateResult),
          expertiseLevel: mergedOptions.expertiseLevel,
          topicsIdentified: debateResult.identifiedThemes
        },
        suggestedFollowUp: suggestions
      };
      
      this.logger.info('Réponse finale générée avec succès', {
        processingTime,
        contentLength: finalContent.length,
        suggestionsCount: suggestions.length
      });
      
      return finalResponse;
      
    } catch (error) {
      this.logger.error('Erreur lors de la génération de la réponse finale', {
        error: error.message
      });
      
      // Retourner une réponse minimale en cas d'erreur
      return {
        content: `Nous rencontrons des difficultés pour générer une réponse complète: ${error.message}`,
        metaData: {
          sourceTypes: [],
          confidenceLevel: 'LOW' as ConfidenceLevel,
          processingTime: Date.now() - startTime,
          usedAgentCount: 0,
          expertiseLevel: mergedOptions.expertiseLevel,
          topicsIdentified: []
        },
        error: error.message
      };
    }
  }

  /**
   * Compte les agents distincts utilisés
   * @param debateResult Résultat du débat
   * @returns Nombre d'agents
   */
  private countDistinctAgents(debateResult: DebateResult): number {
    // Dans une implémentation réelle, on extrairait cette information des métadonnées
    // Simulé pour le prototype
    const poolsCount = debateResult.sourceMetrics?.poolsUsed?.length || 0;
    const estimatedAgentsPerPool = 2; // Estimation
    
    return poolsCount * estimatedAgentsPerPool;
  }
  
  /**
   * Compte les types de sources distinctes
   * @param debateResult Résultat du débat
   * @returns Types de sources
   */
  private countDistinctSourceTypes(debateResult: DebateResult): string[] {
    const sources = [];
    
    // Ajouter KAG si utilisé
    if (debateResult.sourceMetrics?.kagConfidence > 0) {
      sources.push('KAG');
    }
    
    // Ajouter RAG si utilisé
    if (debateResult.sourceMetrics?.ragConfidence > 0) {
      sources.push('RAG');
    }
    
    // Ajouter les pools utilisés
    if (debateResult.sourceMetrics?.poolsUsed?.length > 0) {
      // Convertir les enum en strings si nécessaire
      const poolNames = debateResult.sourceMetrics.poolsUsed.map(pool => {
        if (typeof pool === 'string') return pool;
        return pool; // Déjà une string ou un PoolType qui sera converti
      });
      
      sources.push(...poolNames);
    }
    
    return sources;
  }
  
  /**
   * Extrait les thèmes d'une requête
   * @param query Requête utilisateur
   * @returns Thèmes identifiés
   */
  private extractThemesFromQuery(query: UserQuery): string[] {
    // Dans une implémentation réelle, utiliser NLP pour extraire les thèmes
    // Simulé pour le prototype
    return query.domainHints || ['Commerce', 'Marketing'];
  }
  
  /**
   * Formate le contenu selon le niveau d'expertise
   * @param content Contenu brut
   * @param level Niveau d'expertise cible
   * @returns Contenu formaté
   */
  private formatContentForExpertiseLevel(content: string, level: ExpertiseLevel): string {
    // Dans une implémentation réelle, utiliser des transformations NLP
    
    switch (level) {
      case 'BEGINNER':
        return this.simplifyContent(content);
      case 'ADVANCED':
        return this.enrichContent(content);
      case 'INTERMEDIATE':
      default:
        return content; // Niveau par défaut
    }
  }
  
  /**
   * Simplifie le contenu pour un niveau débutant
   * @param content Contenu à simplifier
   * @returns Contenu simplifié
   */
  private simplifyContent(content: string): string {
    // Dans une implémentation réelle, utiliser des techniques de simplification
    return content; // Simulé
  }
  
  /**
   * Enrichit le contenu pour un niveau avancé
   * @param content Contenu à enrichir
   * @returns Contenu enrichi
   */
  private enrichContent(content: string): string {
    // Dans une implémentation réelle, enrichir avec des termes techniques
    return content; // Simulé
  }
  
  /**
   * Crée un contenu présentant plusieurs perspectives
   * @param content Contenu de base
   * @param debateResult Résultat du débat
   * @param level Niveau d'expertise
   * @returns Contenu multi-perspectives
   */
  private createMultiPerspectiveContent(
    content: string,
    debateResult: DebateResult,
    level: ExpertiseLevel
  ): string {
    // Dans une implémentation réelle, structurer le contenu avec plusieurs viewpoints
    
    // Version simplifiée
    const formattedContent = this.formatContentForExpertiseLevel(content, level);
    
    const intro = "Cette question présente plusieurs perspectives valides. Voici une synthèse des différents points de vue:\n\n";
    
    return intro + formattedContent;
  }
  
  /**
   * Génère des suggestions de suivi
   * @param query Requête initiale
   * @param themes Thèmes identifiés
   * @returns Liste de suggestions
   */
  private async generateFollowUpSuggestions(
    query: UserQuery,
    themes: string[]
  ): Promise<string[]> {
    try {
      // Obtenir le prompt pour les suggestions
      const suggestionsPrompt = this.promptsService.getPromptTemplate(
        PromptTemplateType.FOLLOWUP_SUGGESTIONS
      );
      
      // Préparer les données pour le prompt
      const promptData = {
        query: query.text,
        themes: themes.join(', ')
      };
      
      // Construire le prompt complet
      const fullPrompt = this.promptsService.fillTemplate(suggestionsPrompt, promptData);
      
      // Générer les suggestions via une API
      const suggestionsResponse = await this.apiProviderFactory.generateResponse(
        'qwen', // Qwen pour les suggestions
        fullPrompt,
        {
          temperature: 0.7, // Créatif pour les suggestions
          top_p: 0.9,
          max_tokens: 300
        }
      );
      
      // Transformer le texte en liste de suggestions
      const suggestionsText = suggestionsResponse.text;
      
      // Version simplifiée: on suppose que chaque ligne est une suggestion
      const suggestions = suggestionsText
        .split('\n')
        .map(line => line.trim())
        .filter(line => line.length > 10 && line.includes('?'))
        .slice(0, 3); // Limiter à 3 suggestions
      
      return suggestions.length > 0 
        ? suggestions 
        : ['Comment pourrions-nous approfondir cette analyse?', 
           'Souhaitez-vous des précisions sur un aspect particulier?'];
      
    } catch (error) {
      this.logger.warn('Erreur lors de la génération des suggestions', {
        error: error.message
      });
      
      // Fallback sur des suggestions génériques
      return [
        'Avez-vous d\'autres questions sur ce sujet?',
        'Souhaitez-vous des précisions sur un aspect particulier?'
      ];
    }
  }
  
  /**
   * Dérive un niveau de confiance global
   * @param debateResult Résultat du débat
   * @returns Niveau de confiance global
   */
  private deriveOverallConfidenceLevel(debateResult: DebateResult): ConfidenceLevel {
    const { kagConfidence = 0, ragConfidence = 0 } = debateResult.sourceMetrics || {};
    
    // Calculer un score moyen pondéré
    const hasConsensus = debateResult.hasConsensus;
    const consensusBonus = hasConsensus ? 0.1 : 0;
    
    const avgConfidence = (kagConfidence + ragConfidence) / 2 + consensusBonus;
    
    // Mapper à un niveau de confiance
    if (avgConfidence >= 0.8) return 'HIGH';
    if (avgConfidence >= 0.5) return 'MEDIUM';
    return 'LOW';
  }
} 