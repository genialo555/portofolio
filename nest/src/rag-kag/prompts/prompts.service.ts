import { Injectable, Inject } from '@nestjs/common';
import { LOGGER_TOKEN, ILogger } from '../utils/logger-tokens';
import { generateCommercialPrompt } from '../../legacy/prompts/base-prompts/commercial';
import { generateMarketingPrompt } from '../../legacy/prompts/base-prompts/marketing';
import { generateSectorielPrompt } from '../../legacy/prompts/base-prompts/sectoriel';
import { generateEducationalPrompt } from '../../legacy/prompts/base-prompts/educational';
import { AgentType, ApiProvider } from '../../legacy/types/agent.types';
import { AgentConfig } from '../../legacy/config/poolConfig';

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
  EDUCATIONAL_AGENT = 'educational_agent',
  
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
  
  // Configuration mock pour les agents legacy
  private readonly mockAgentConfig: AgentConfig = {
    id: 'mock-agent-id', 
    name: 'MockAgent',
    type: AgentType.COMMERCIAL,
    api: ApiProvider.HOUSE_MODEL,
    parameters: {
      temperature: 0.7,
      top_p: 0.9,
      top_k: 40,
      max_tokens: 1000,
      context_window: 8000,
      presence_penalty: 0.1,
      frequency_penalty: 0.1
    },
    description: 'Agent de mock pour la compatibilité avec l\'ancienne architecture'
  };

  constructor(
    @Inject(LOGGER_TOKEN) logger: ILogger
  ) {
    this.logger = logger;
    this.logger.info('Service de prompts initialisé');
    this.loadPromptTemplates();
  }

  /**
   * Charge les templates de prompts
   * Intègre les templates legacy lorsque c'est pertinent
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

    // Pour l'agent commercial, utiliser un template dynamique basé sur le template legacy
    this.templates.set(
      PromptTemplateType.COMMERCIAL_AGENT,
      `Template de base pour l'agent commercial. Sera remplacé par un template legacy.`
    );

    // Pour l'agent marketing, utiliser un template dynamique basé sur le template legacy
    this.templates.set(
      PromptTemplateType.MARKETING_AGENT,
      `Template de base pour l'agent marketing. Sera remplacé par un template legacy.`
    );

    // Pour l'agent sectoriel, utiliser un template dynamique basé sur le template legacy
    this.templates.set(
      PromptTemplateType.SECTORIAL_AGENT,
      `Template de base pour l'agent sectoriel. Sera remplacé par un template legacy.`
    );
    
    // Pour l'agent éducatif, utiliser un template dynamique basé sur le template legacy
    this.templates.set(
      PromptTemplateType.EDUCATIONAL_AGENT,
      `Template de base pour l'agent éducatif. Sera remplacé par un template legacy.`
    );

    // Template pour la synthèse
    this.templates.set(
      PromptTemplateType.SYNTHESIS,
      `Basé sur les résultats du débat et les points d'accord identifiés, synthétisez une réponse complète et pertinente à la requête: "{{query}}".
      
      Points d'accord:
      {{agreements}}
      
      Adaptez le niveau technique de la réponse pour un niveau d'expertise: {{expertiseLevel}}`
    );

    // Template pour les suggestions de suivi
    this.templates.set(
      PromptTemplateType.FOLLOWUP_SUGGESTIONS,
      `Basé sur la requête "{{query}}" et la réponse fournie, suggérez 3 questions de suivi pertinentes que l'utilisateur pourrait poser pour approfondir sa compréhension.`
    );

    this.logger.info('Templates de prompts chargés avec succès', { count: this.templates.size });
  }

  /**
   * Récupère un template de prompt
   */
  getPromptTemplate(type: PromptTemplateType): string {
    if (!this.templates.has(type)) {
      this.logger.warn(`Template de prompt non trouvé: ${type}`);
      return '';
    }
    return this.templates.get(type) || '';
  }
  
  /**
   * Récupère un template spécifique pour un agent, en utilisant les templates legacy
   * @param type Type d'agent
   * @param agentId ID de l'agent (optionnel)
   * @param query Requête de l'utilisateur
   */
  getAgentPrompt(type: PromptTemplateType, query: string, agentId: string = 'default'): string {
    // Configuration de l'agent adaptée pour les fonctions legacy
    const adaptedConfig = { 
      ...this.mockAgentConfig,
      id: agentId
    };
    
    switch (type) {
      case PromptTemplateType.COMMERCIAL_AGENT:
        return generateCommercialPrompt(adaptedConfig, query);
      case PromptTemplateType.MARKETING_AGENT:
        return generateMarketingPrompt(adaptedConfig, query);
      case PromptTemplateType.SECTORIAL_AGENT:
        return generateSectorielPrompt(adaptedConfig, query);
      case PromptTemplateType.EDUCATIONAL_AGENT:
        return generateEducationalPrompt(adaptedConfig, query);
      default:
        // Fallback au template standard
        const template = this.getPromptTemplate(type);
        return this.fillTemplate(template, { query });
    }
  }

  /**
   * Remplit un template avec des valeurs
   */
  fillTemplate(template: string, values: Record<string, any>): string {
    let filledTemplate = template;
    for (const [key, value] of Object.entries(values)) {
      const placeholder = `{{${key}}}`;
      filledTemplate = filledTemplate.replace(new RegExp(placeholder, 'g'), String(value));
    }
    return filledTemplate;
  }
} 