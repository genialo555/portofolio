/**
 * Orchestrateur principal du système mixte RAG/KAG
 * Gère le flux complet de traitement des requêtes utilisateur
 */

import { router } from './router';
import { poolManager } from './poolManager';
import { outputCollector } from './outputCollector';
import { kagEngine } from '../debate/kagEngine';
import { ragEngine } from '../debate/ragEngine';
import { debateProtocol } from '../debate/debateProtocol';
import { anomalyDetector } from '../utils/anomalyDetector';
import { merger } from '../synthesis/merger';
import { contradictionResolver } from '../synthesis/contradictionResolver';
import { responseFormatter } from '../synthesis/responseFormatter';
import { 
  UserQuery, 
  PoolOutputs, 
  TargetPools,
  KagAnalysis,
  RagAnalysis,
  DebateResult,
  AnomalyReport,
  MergedInsights,
  ResolvedInsights,
  FinalResponse,
  ExpertiseLevel
} from '../types';
import { Logger, LogLevel } from '../utils/logger';

// Créer une instance du logger
const logger = new Logger({ level: LogLevel.INFO });

/**
 * Processus principal du système
 * @param query - Requête de l'utilisateur
 * @param expertiseLevel - Niveau d'expertise du destinataire (débutant, intermédiaire, avancé)
 * @returns Réponse finale synthétisée
 */
export async function processQuery(
  query: UserQuery, 
  expertiseLevel: ExpertiseLevel = 'INTERMEDIATE'
): Promise<FinalResponse> {
  try {
    logger.info('Début du traitement de la requête', { query });

    // 1. Déterminer les pools cibles basés sur la requête
    const targetPools: TargetPools = router.determineTargetPools(query);
    logger.info('Pools ciblés identifiés', { targetPools });

    // 2. Exécuter les agents dans chaque pool en parallèle
    const poolOutputs: PoolOutputs = await poolManager.executeAgents(targetPools, query);
    logger.info('Outputs des pools récupérés');

    // 3. Vérifier les anomalies dans les outputs des pools
    const anomalyReport: AnomalyReport = await anomalyDetector.detectAnomalies(poolOutputs);
    logger.info('Détection d\'anomalies terminée', { 
      anomalyCount: anomalyReport.highPriorityAnomalies.length + 
                   anomalyReport.mediumPriorityAnomalies.length
    });

    // 4. Filtrer ou corriger les outputs basés sur le rapport d'anomalies
    const filteredOutputs: PoolOutputs = poolManager.filterAnomalies(poolOutputs, anomalyReport);
    logger.info('Outputs filtrés des anomalies majeures');

    // 5. Collecter et structurer les outputs filtrés
    const structuredOutputs: PoolOutputs = outputCollector.collect(filteredOutputs);
    logger.info('Outputs structurés prêts pour analyse');

    // 6. Analyse KAG (Knowledge-Augmented Generation)
    const kagAnalysis: KagAnalysis = await kagEngine.analyze(structuredOutputs);
    logger.info('Analyse KAG terminée');

    // 7. Analyse RAG (Retrieval-Augmented Generation)
    const ragAnalysis: RagAnalysis = await ragEngine.enrich(structuredOutputs);
    logger.info('Analyse RAG terminée');

    // 8. Débat entre KAG et RAG
    const debateResult: DebateResult = await debateProtocol.debate(kagAnalysis, ragAnalysis);
    logger.info('Débat KAG vs RAG terminé');

    // 9. Fusion des insights du débat
    const mergedInsights: MergedInsights = merger.merge(debateResult);
    logger.info('Fusion des insights terminée');

    // 10. Résolution des contradictions potentielles restantes
    const resolvedInsights: ResolvedInsights = contradictionResolver.resolve(mergedInsights);
    logger.info('Résolution des contradictions terminée');

    // 11. Formater la réponse finale selon le niveau d'expertise
    const finalResponse: FinalResponse = responseFormatter.format(resolvedInsights, expertiseLevel);
    logger.info('Réponse finale formatée', { expertiseLevel });

    return finalResponse;
  } catch (error) {
    logger.error('Erreur lors du traitement de la requête', { error });
    throw error;
  }
}

/**
 * Processus simplifié pour les requêtes urgentes ou simples
 * Contourne le débat KAG/RAG pour une réponse plus rapide
 */
export async function processSimpleQuery(
  query: UserQuery, 
  expertiseLevel: ExpertiseLevel = 'INTERMEDIATE'
): Promise<FinalResponse> {
  try {
    logger.info('Début du traitement simplifié', { query });

    // Version simplifiée avec moins d'étapes
    const targetPools: TargetPools = router.determineTargetPools(query);
    const poolOutputs: PoolOutputs = await poolManager.executeAgents(targetPools, query);
    const structuredOutputs: PoolOutputs = outputCollector.collect(poolOutputs);
    
    // Contourner le débat, aller directement à la synthèse
    const simplifiedInsights = {
      outputs: structuredOutputs,
      confidence: 'MEDIUM',
      source: 'SIMPLIFIED_PROCESS'
    };
    
    // Formater directement
    const finalResponse: FinalResponse = responseFormatter.formatSimplified(
      simplifiedInsights, 
      expertiseLevel
    );
    
    logger.info('Réponse simplifiée formatée');
    return finalResponse;
  } catch (error) {
    logger.error('Erreur lors du traitement simplifié', { error });
    throw error;
  }
}

/**
 * API publique de l'orchestrateur
 */
export const orchestrator = {
  processQuery,
  processSimpleQuery
}; 