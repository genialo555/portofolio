import { Injectable, Inject } from '@nestjs/common';
import { LOGGER_TOKEN, ILogger } from '../utils/logger-tokens';
import { KagEngineService } from './kag-engine.service';
import { RagEngineService } from './rag-engine.service';
import { PromptsService, PromptTemplateType } from '../prompts/prompts.service';
import { ApiProviderFactory } from '../apis/api-provider-factory.service';
import { EventBusService, RagKagEventType } from '../core/event-bus.service';
import { KnowledgeGraphService, KnowledgeSource, RelationType } from '../core/knowledge-graph.service';
import { generateKagRagDebatePrompt } from '../../legacy/prompts/debate-prompts/kag-rag-debate';
import { 
  UserQuery, 
  DebateResult, 
  PoolOutputs, 
  KagAnalysis,
  RagAnalysis,
  DebateInput
} from '../types';
import { DebateInput as LegacyDebateInput } from '../../legacy/types/prompt.types';

/**
 * Service responsable de la conduite du débat entre RAG et KAG
 */
@Injectable()
export class DebateService {
  constructor(
    @Inject(LOGGER_TOKEN) private readonly logger: ILogger,
    private readonly kagEngine: KagEngineService,
    private readonly ragEngine: RagEngineService,
    private readonly promptsService: PromptsService,
    private readonly apiProviderFactory: ApiProviderFactory,
    private readonly eventBus: EventBusService,
    private readonly knowledgeGraph: KnowledgeGraphService
  ) {}

  /**
   * Génère un débat entre les approches RAG et KAG pour une requête
   * @param query Requête utilisateur
   * @param options Options additionnelles
   * @returns Résultat du débat structuré
   */
  async generateDebate(query: UserQuery, options: { 
    includePoolOutputs?: boolean;
    prioritizeSpeed?: boolean;
    useLegacyPrompt?: boolean;
  } = {}): Promise<DebateResult> {
    const startTime = Date.now();
    const queryText = typeof query === 'string' ? query : query.text || query.content || '';
    
    this.logger.info(`Démarrage du débat pour la requête: "${queryText.substring(0, 50)}${queryText.length > 50 ? '...' : ''}"`, {
      includePoolOutputs: options.includePoolOutputs,
      prioritizeSpeed: options.prioritizeSpeed,
      useLegacyPrompt: options.useLegacyPrompt
    });
    
    // Émettre un événement de début de débat
    this.eventBus.emit({
      type: RagKagEventType.DEBATE_STARTED,
      source: 'DebateService',
      payload: { 
        query: queryText,
        options 
      }
    });
    
    try {
      // 1. Génération de l'analyse KAG
      this.logger.debug('Génération de l\'analyse KAG');
      const kagAnalysis = await this.kagEngine.generateAnalysis(query);
      
      // 2. Génération de l'analyse RAG
      this.logger.debug('Génération de l\'analyse RAG');
      const ragAnalysis = await this.ragEngine.generateAnalysis(query);

      // 3. Création de l'input pour le débat
      const debateInput: DebateInput = {
        query,
        kagAnalysis,
        ragAnalysis,
        poolOutputs: options.includePoolOutputs ? {
          commercial: [],
          marketing: [],
          sectoriel: [],
          educational: []
        } : undefined
      };
      
      // 4. Exécution du débat
      const debateResult = await this.conductDebate(debateInput, options);
      
      // 5. Stocker les résultats dans le graphe de connaissances
      this.storeDebateInGraph(queryText, debateResult, ragAnalysis, kagAnalysis);
      
      // 6. Émettre un événement de fin de débat
      this.eventBus.emit({
        type: RagKagEventType.DEBATE_COMPLETED,
        source: 'DebateService',
        payload: {
          query: queryText,
          duration: Date.now() - startTime,
          consensusLevel: debateResult.hasConsensus ? 'HIGH' : 'LOW',
          sourceMetrics: debateResult.sourceMetrics
        }
      });
      
      return debateResult;
    } catch (error) {
      this.logger.error(`Erreur pendant le débat: ${error.message}`, {
        error,
        query: queryText
      });
      
      this.eventBus.emit({
        type: RagKagEventType.QUERY_ERROR,
        source: 'DebateService',
        payload: {
          error,
          query: queryText
        }
      });
      
      // Retourner un résultat d'erreur
      return {
        content: "Une erreur est survenue pendant le débat d'analyse.",
        hasConsensus: false,
        identifiedThemes: [],
        processingTime: Date.now() - startTime,
        sourceMetrics: {
          poolsUsed: [],
        },
        error: error.message
      };
    }
  }
  
  /**
   * Exécute le débat entre les analyses RAG et KAG
   * @param input Entrée du débat
   * @param options Options supplémentaires
   * @returns Résultat du débat
   */
  private async conductDebate(
    input: DebateInput, 
    options: { useLegacyPrompt?: boolean } = {}
  ): Promise<DebateResult> {
    const startDebateTime = Date.now();
    
    // Extraction des analyses
    const { query, ragAnalysis, kagAnalysis } = input;
    const queryText = typeof query === 'string' ? query : query.text;
    
    this.logger.debug('Préparation du débat entre analyses RAG et KAG', {
      ragConfidence: ragAnalysis.confidenceScore,
      kagConfidence: kagAnalysis.confidenceScore,
      useLegacyPrompt: options.useLegacyPrompt
    });
    
    // Choix du prompt: legacy ou standard
    let debatePrompt: string;
    
    if (options.useLegacyPrompt) {
      // Préparer le prompt legacy avancé
      const legacyDebateInput: LegacyDebateInput = {
        kagAnalysis: kagAnalysis.content,
        ragAnalysis: ragAnalysis.content,
        query: queryText,
        poolOutputs: input.poolOutputs || {}
      };
      
      debatePrompt = generateKagRagDebatePrompt(legacyDebateInput);
      this.logger.debug('Utilisation du prompt de débat legacy avancé');
    } else {
      // Utiliser le prompt standard
      debatePrompt = this.promptsService.getPromptTemplate(PromptTemplateType.KAG_RAG_DEBATE);
      debatePrompt = this.promptsService.fillTemplate(debatePrompt, {
        query: queryText,
        kagAnalysis: kagAnalysis.content,
        ragAnalysis: ragAnalysis.content
      });
      this.logger.debug('Utilisation du prompt de débat standard');
    }
    
    // Exécuter le débat avec le modèle approprié
    const provider = this.apiProviderFactory.recommendProvider(undefined, {
      prioritizeReliability: true
    });
    
    const startGeneration = Date.now();
    const response = await this.apiProviderFactory.generateResponse(provider, debatePrompt, {
      temperature: 0.5,
      max_tokens: 2000
    });
    
    this.logger.debug('Génération du débat terminée', {
      generationTime: Date.now() - startGeneration,
      responseLength: response.text.length
    });
    
    // Analyser la réponse pour extraire les informations pertinentes
    const consensus = this.extractConsensusLevel(response.text);
    const hasConsensus = consensus > 0.5;
    const identifiedThemes = this.extractThemes(response.text);
    const majorContradictions = this.detectMajorContradictions(response.text);
    
    // Construire le résultat du débat
    const result: DebateResult = {
      content: response.text,
      hasConsensus: hasConsensus,
      identifiedThemes: identifiedThemes,
      processingTime: Date.now() - startDebateTime,
      sourceMetrics: {
        kagConfidence: kagAnalysis.confidenceScore,
        ragConfidence: ragAnalysis.confidenceScore,
        poolsUsed: [],
        sourcesUsed: []
      },
      consensusLevel: consensus,
      debateTimestamp: new Date()
    };
    
    // Émettre un événement approprié
    if (hasConsensus) {
      this.eventBus.emit({
        type: RagKagEventType.CONSENSUS_REACHED,
        source: 'DebateService',
        payload: {
          consensusLevel: consensus,
          identifiedThemes
        }
      });
    } else if (majorContradictions) {
      this.eventBus.emit({
        type: RagKagEventType.CONTRADICTION_FOUND,
        source: 'DebateService',
        payload: {
          contradictions: this.extractContradictions(response.text)
        }
      });
    }
    
    return result;
  }
  
  /**
   * Extrait le niveau de consensus du texte du débat
   * @param text Texte du débat
   * @returns Niveau de consensus (0-10)
   */
  private extractConsensusLevel(text: string): number {
    try {
      const consensusMatch = text.match(/Niveau de consensus\s*:\s*(\d+)/i);
      if (consensusMatch && consensusMatch[1]) {
        const level = parseInt(consensusMatch[1], 10);
        return isNaN(level) ? 5 : Math.min(Math.max(level, 0), 10);
      }
      return 5; // Valeur par défaut
    } catch (error) {
      this.logger.error(`Erreur lors de l'extraction du niveau de consensus: ${error.message}`);
      return 5;
    }
  }
  
  /**
   * Extrait les thèmes identifiés du texte du débat
   * @param text Texte du débat
   * @returns Liste des thèmes
   */
  private extractThemes(text: string): string[] {
    try {
      const themesMatch = text.match(/Thèmes identifiés\s*:\s*\[(.*?)\]/is);
      if (themesMatch && themesMatch[1]) {
        return themesMatch[1]
          .split(',')
          .map(theme => theme.trim())
          .filter(theme => theme.length > 0);
      }
      return [];
    } catch (error) {
      this.logger.error(`Erreur lors de l'extraction des thèmes: ${error.message}`);
      return [];
    }
  }
  
  /**
   * Détecte s'il y a des contradictions majeures dans le débat
   * @param text Texte du débat
   * @returns Vrai si des contradictions majeures sont détectées
   */
  private detectMajorContradictions(text: string): boolean {
    const contradictionSection = text.match(/Contradictions\s*:\s*\[(.*?)\]/is);
    if (!contradictionSection || !contradictionSection[1]) {
      return false;
    }
    
    const contradictions = contradictionSection[1].trim();
    
    // Si la section des contradictions contient des termes significatifs
    const significantTerms = ['majeur', 'important', 'fondamental', 'significatif', 'critique'];
    return significantTerms.some(term => contradictions.toLowerCase().includes(term)) && 
           contradictions.length > 50;
  }
  
  /**
   * Extrait les contradictions du texte du débat
   * @param text Texte du débat
   * @returns Liste des contradictions
   */
  private extractContradictions(text: string): string[] {
    try {
      const contradictionsMatch = text.match(/Contradictions\s*:\s*\[(.*?)\]/is);
      if (contradictionsMatch && contradictionsMatch[1]) {
        return contradictionsMatch[1]
          .split(',')
          .map(contradiction => contradiction.trim())
          .filter(contradiction => contradiction.length > 0);
      }
      return [];
    } catch (error) {
      this.logger.error(`Erreur lors de l'extraction des contradictions: ${error.message}`);
      return [];
    }
  }
  
  /**
   * Stocke le résultat du débat dans le graphe de connaissances
   * @param query Requête originale
   * @param debate Résultat du débat
   * @param ragAnalysis Analyse RAG
   * @param kagAnalysis Analyse KAG
   */
  private storeDebateInGraph(
    query: string, 
    debate: DebateResult,
    ragAnalysis: RagAnalysis,
    kagAnalysis: KagAnalysis
  ): void {
    try {
      // Créer un nœud pour le débat
      const debateNodeId = this.knowledgeGraph.addNode({
        label: `Debate: ${query.substring(0, 30)}${query.length > 30 ? '...' : ''}`,
        type: 'DEBATE_RESULT',
        content: debate.content,
        confidence: (debate.consensusLevel || 5) / 10,
        source: KnowledgeSource.INFERENCE
      });
      
      // Rechercher les nœuds des analyses
      const searchResults = this.knowledgeGraph.search(query, {
        nodeTypes: ['RAG_ANALYSIS', 'KAG_ANALYSIS', 'QUERY'],
        maxResults: 10,
        maxDepth: 0
      });
      
      // Lier le débat aux analyses et à la requête
      const linkToNode = (nodeType: string, relationType: string) => {
        const node = searchResults.nodes.find(n => n.type === nodeType);
        if (node) {
          this.knowledgeGraph.addFact(
            debateNodeId,
            relationType,
            node.id,
            0.9,
            { bidirectional: true, weight: 0.8 }
          );
        }
      };
      
      linkToNode('RAG_ANALYSIS', 'USES_RAG_ANALYSIS');
      linkToNode('KAG_ANALYSIS', 'USES_KAG_ANALYSIS');
      linkToNode('QUERY', 'ANSWERS_QUERY');
      
      // Ajouter les thèmes comme nœuds
      for (const theme of debate.identifiedThemes) {
        this.knowledgeGraph.addFact(
          debateNodeId,
          'HAS_THEME',
          {
            label: theme,
            type: 'THEME',
            content: theme,
            confidence: 0.8,
            source: KnowledgeSource.INFERENCE
          },
          0.8,
          { bidirectional: false, weight: 0.7 }
        );
      }
      
      this.logger.debug(`Débat stocké dans le graphe de connaissances`, {
        debateId: debateNodeId,
        consensus: debate.consensusLevel,
        themes: debate.identifiedThemes.length
      });
    } catch (error) {
      this.logger.error(`Erreur lors du stockage du débat dans le graphe: ${error.message}`, {
        error: error.stack
      });
    }
  }
} 