import { Injectable, Inject } from '@nestjs/common';
import { LOGGER_TOKEN, ILogger } from '../utils/logger-tokens';
import { RouterService } from './router.service';
import { OutputCollectorService } from './output-collector.service';
import { PoolManagerService } from '../pools/pool-manager.service';
import { DebateService } from '../debate/debate.service';
import { SynthesisService } from '../synthesis/synthesis.service';
import { 
  UserQuery, 
  FinalResponse, 
  PoolOutputs, 
  TargetPools,
  DebateResult,
  ExpertiseLevel,
  KagAnalysis, 
  RagAnalysis
} from '../types';

interface ProcessingOptions {
  includeSuggestions?: boolean;
  maxLength?: number;
}

/**
 * Service d'orchestration principal du système RAG/KAG
 */
@Injectable()
export class OrchestratorService {
  private readonly logger: ILogger;

  constructor(
    @Inject(LOGGER_TOKEN) logger: ILogger,
    private readonly router: RouterService,
    private readonly poolManager: PoolManagerService,
    private readonly debateService: DebateService,
    private readonly outputCollector: OutputCollectorService,
    private readonly synthesisService: SynthesisService
  ) {
    this.logger = logger;
    this.logger.info('Service d\'orchestration initialisé');
  }

  /**
   * Processus principal de traitement des requêtes
   * @param query Requête utilisateur
   * @param expertiseLevel Niveau d'expertise du destinataire
   * @param options Options de traitement
   * @returns Réponse finale formatée
   */
  async processQuery(
    query: UserQuery, 
    expertiseLevel: ExpertiseLevel = 'INTERMEDIATE',
    options: ProcessingOptions = {}
  ): Promise<FinalResponse> {
    const startTime = Date.now();
    
    try {
      this.logger.info('Début du traitement de la requête', { query: query.text });

      // 1. Déterminer les pools cibles basés sur la requête
      const targetPools: TargetPools = await this.router.determineTargetPools(query);
      this.logger.info('Pools ciblés identifiés', { targetPools });

      // 2. Exécuter les agents dans chaque pool en parallèle
      const poolOutputs: PoolOutputs = await this.poolManager.executeAgents(targetPools, query);
      this.logger.info('Outputs des pools récupérés');
      
      // 3. Collecter et formater les sorties pour le moteur de débat
      const processedOutputs = await this.outputCollector.collectAndProcess(poolOutputs);
      
      // 4. Traitement RAG/KAG en parallèle
      const [kagAnalysis, ragAnalysis] = await Promise.all([
        this.debateService.generateKagAnalysis(query),
        this.debateService.generateRagAnalysis(query)
      ]);
      
      // 5. Exécuter le protocole de débat
      const debateResult = await this.debateService.facilitateDebate({
        query,
        kagAnalysis,
        ragAnalysis,
        poolOutputs: await processedOutputs,
      });
      
      // 6. Générer la réponse finale
      const response = await this.synthesisService.generateFinalResponse(
        query,
        debateResult,
        {
          expertiseLevel,
          includeSuggestions: options.includeSuggestions !== false,
          maxLength: options.maxLength
        }
      );
      
      const processingTime = Date.now() - startTime;
      
      // Ajouter le temps de traitement à la réponse
      response.metaData.processingTime = processingTime;
      
      this.logger.info('Traitement terminé avec succès', { 
        processingTime, 
        confidence: response.metaData.confidenceLevel 
      });
      
      return response;
      
    } catch (error) {
      const processingTime = Date.now() - startTime;
      this.logger.error('Erreur lors du traitement de la requête', { 
        error: error.message, 
        processingTime 
      });
      
      // En cas d'erreur, retourner une réponse d'erreur formattée
      return {
        content: `Une erreur est survenue lors du traitement de votre requête: ${error.message}`,
        metaData: {
          sourceTypes: [],
          confidenceLevel: 'LOW',
          processingTime,
          usedAgentCount: 0,
          expertiseLevel,
          topicsIdentified: [],
        }
      };
    }
  }
} 