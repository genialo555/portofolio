import { Injectable, Inject } from '@nestjs/common';
import { LOGGER_TOKEN, ILogger } from '../utils/logger-tokens';

/**
 * Types de templates de prompts disponibles
 */
export enum PromptTemplateType {
  // Templates pour le processus de débat
  KAG_ANALYSIS = 'kag_analysis',
  RAG_ANALYSIS = 'rag_analysis',
  KAG_RAG_DEBATE = 'kag_rag_debate',
  DEBATE = 'debate',
  
  // Templates pour les pools d'agents
  COMMERCIAL_AGENT = 'commercial_agent',
  MARKETING_AGENT = 'marketing_agent',
  SECTORIAL_AGENT = 'sectorial_agent',
  
  // Templates pour la synthèse et suivi
  SYNTHESIS = 'synthesis',
  FOLLOWUP_SUGGESTIONS = 'followup_suggestions',
}

/**
 * Interface définissant un template de prompt
 */
export interface PromptTemplate {
  type: PromptTemplateType;
  template: string;
}

/**
 * Service de gestion des templates de prompts
 */
@Injectable()
export class PromptsService {
  private readonly logger: ILogger;
  private readonly templates: Map<PromptTemplateType, string> = new Map();

  constructor(
    @Inject(LOGGER_TOKEN) logger: ILogger
  ) {
    this.logger = logger;
    this.logger.info('Service de prompts initialisé');
    this.loadPromptTemplates();
  }

  /**
   * Charge les templates de prompts
   * Dans une implémentation réelle, ces templates seraient chargés depuis des fichiers
   */
  private loadPromptTemplates(): void {
    this.logger.debug('Chargement des templates de prompts');

    // Template pour l'analyse KAG
    this.templates.set(
      PromptTemplateType.KAG_ANALYSIS,
      `Analysez la requête utilisateur suivante du point de vue commercial: "{{query}}". 
      Concentrez-vous sur les enjeux business, les opportunités de croissance et les stratégies commerciales.`
    );

    // Template pour l'analyse RAG
    this.templates.set(
      PromptTemplateType.RAG_ANALYSIS,
      `Analysez la requête utilisateur suivante en utilisant les connaissances pertinentes: "{{query}}". 
      Fournissez des informations factuelles et des réponses basées sur des sources fiables.`
    );

    // Template pour le débat KAG/RAG
    this.templates.set(
      PromptTemplateType.KAG_RAG_DEBATE,
      `Examinez les deux analyses suivantes concernant la requête: "{{query}}".
      
      Analyse KAG:
      {{kagAnalysis}}
      
      Analyse RAG:
      {{ragAnalysis}}
      
      Comparez ces analyses, identifiez les points de convergence et de divergence, et formulez une synthèse qui intègre les perspectives les plus pertinentes.
      Indiquez clairement si un consensus a été atteint, et sur quels points il existe encore des divergences.`
    );

    // Template pour le débat standard
    this.templates.set(
      PromptTemplateType.DEBATE,
      `Examinez les deux analyses suivantes concernant la requête: "{{query}}".
      
      Analyse KAG:
      {{kagAnalysis}}
      
      Analyse RAG:
      {{ragAnalysis}}
      
      Comparez ces analyses, identifiez les points de convergence et de divergence, et formulez une synthèse qui intègre les perspectives les plus pertinentes.`
    );

    // Template pour l'agent commercial
    this.templates.set(
      PromptTemplateType.COMMERCIAL_AGENT,
      `En tant qu'expert commercial, analysez la requête suivante: "{{query}}".
      
      Concentrez-vous sur:
      - Les opportunités de vente et de revenus
      - Les stratégies commerciales pertinentes
      - L'analyse de la concurrence et du positionnement
      - Les modèles de tarification et propositions de valeur
      
      Fournissez une analyse concise et orientée résultats.`
    );

    // Template pour l'agent marketing
    this.templates.set(
      PromptTemplateType.MARKETING_AGENT,
      `En tant qu'expert marketing, analysez la requête suivante: "{{query}}".
      
      Concentrez-vous sur:
      - Le positionnement de marque et la communication
      - Les segments de clientèle et personas
      - Les canaux marketing à privilégier
      - Les messages clés et la proposition de valeur
      
      Fournissez une analyse concise et orientée résultats.`
    );

    // Template pour l'agent sectoriel
    this.templates.set(
      PromptTemplateType.SECTORIAL_AGENT,
      `En tant qu'expert sectoriel, analysez la requête suivante: "{{query}}".
      
      Concentrez-vous sur:
      - Les tendances et évolutions du secteur
      - La réglementation et conformité spécifiques
      - Les meilleures pratiques observées chez les leaders
      - Les enjeux d'innovation et de transformation
      
      Fournissez une analyse concise et orientée résultats.`
    );

    // Template pour la synthèse
    this.templates.set(
      PromptTemplateType.SYNTHESIS,
      `Rédigez une réponse finale à la requête: "{{query}}" basée sur le résultat du débat suivant:
      
      {{debateResult}}
      
      Niveau d'expertise cible: {{expertiseLevel}}
      Adaptez le niveau de détail et le vocabulaire en conséquence.`
    );

    // Template pour les suggestions de suivi
    this.templates.set(
      PromptTemplateType.FOLLOWUP_SUGGESTIONS,
      `Générez 3 questions de suivi pertinentes pour approfondir la discussion sur: "{{query}}"
      
      Thèmes identifiés: {{themes}}
      
      Les questions doivent être ouvertes et encourager l'exploration de nouveaux aspects du sujet.`
    );

    this.logger.info('Templates de prompts chargés avec succès', {
      count: this.templates.size
    });
  }

  /**
   * Récupère un template de prompt par son type
   * @param type Type de template
   * @returns Template de prompt
   */
  getPromptTemplate(type: PromptTemplateType): string {
    const template = this.templates.get(type);
    
    if (!template) {
      this.logger.warn(`Template de prompt non trouvé: ${type}`);
      throw new Error(`Template de prompt non trouvé: ${type}`);
    }
    
    this.logger.debug(`Template récupéré: ${type.substring(0, 15)}...`);
    return template;
  }

  /**
   * Remplit un template avec les valeurs données
   * @param template Template à remplir
   * @param values Valeurs à injecter
   * @returns Template complété
   */
  fillTemplate(template: string, values: Record<string, any>): string {
    let filledTemplate = template;
    
    for (const [key, value] of Object.entries(values)) {
      const placeholder = `{{${key}}}`;
      filledTemplate = filledTemplate.replace(
        new RegExp(placeholder, 'g'), 
        String(value)
      );
    }
    
    return filledTemplate;
  }
} 