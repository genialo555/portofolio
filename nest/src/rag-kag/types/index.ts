/**
 * Réexportation des interfaces et types pour le système RAG/KAG
 */

// Types essentiels - réimporté explicitement pour résoudre les erreurs de compilation
import { UserQuery as ImportedUserQuery } from '../../types';
export * from '../../types';

/**
 * Niveaux de confiance dans les réponses (plus spécifique que le type global)
 */
export type ConfidenceLevel = 'HIGH' | 'MEDIUM' | 'LOW';

/**
 * Types d'utilisateurs du système
 */
export enum UserType {
  EXPERT = 'EXPERT',
  STANDARD = 'STANDARD',
  GUEST = 'GUEST',
}

/**
 * Niveaux d'expertise pour l'adaptation des réponses
 */
export type ExpertiseLevel = 'BEGINNER' | 'INTERMEDIATE' | 'ADVANCED';

/**
 * Types de pools d'agents disponibles
 */
export enum PoolType {
  COMMERCIAL = 'COMMERCIAL',
  MARKETING = 'MARKETING',
  SECTORIEL = 'SECTORIEL',
}

/**
 * Types d'API disponibles pour les LLMs
 */
export enum ApiType {
  GOOGLE_AI = 'GOOGLE_AI',
  QWEN_AI = 'QWEN_AI',
  DEEPSEEK_AI = 'DEEPSEEK_AI',
}

/**
 * Interface pour une requête utilisateur (redéfinie pour spécificités RAG/KAG)
 */
export interface UserQuery extends ImportedUserQuery {
  userType?: UserType;
  domainHints?: string[];
}

/**
 * Résultat de l'exécution d'un agent
 */
export interface AgentResult {
  agentId: string;
  agentName: string;
  poolType: PoolType;
  content: string;
  timestamp: number;
  processingTime: number;
  error?: string;
  metadata?: Record<string, any>;
}

/**
 * Configuration pour l'exécution d'un pool d'agents
 */
export interface PoolExecutionConfig {
  minAgentsToExecute?: number;
  maxAgentsToExecute?: number;
  timeoutMs?: number;
  retryCount?: number;
}

/**
 * Résultat de l'exécution d'un pool d'agents
 */
export interface PoolResult {
  poolType: PoolType;
  results: AgentResult[];
  successRate: number;
  processingTime: number;
  timestamp: number;
}

/**
 * Enrichissement de l'interface PoolOutputs
 */
export interface PoolOutputs {
  commercial: any[];
  marketing: any[];
  sectoriel: any[];
  errors?: string[];
  timestamp?: Date;
  query?: UserQuery;
}

/**
 * Analyse KAG (Knowledge Augmented Generation)
 */
export interface KagAnalysis {
  content: string;
  agentResults?: AgentResult[];
  processingTime: number;
  poolsUsed?: PoolType[];
  confidence: ConfidenceLevel;
  confidenceScore?: number;
  sourceType?: 'KAG';
  factualityScore?: number;
  relevanceScore?: number;
  keyInsights?: string[];
  knowledgeDomains?: string[];
  timestamp?: Date;
  error?: string;
}

/**
 * Analyse RAG (Retrieval Augmented Generation)
 */
export interface RagAnalysis {
  content: string;
  sources?: string[];
  processingTime: number;
  confidence: ConfidenceLevel;
  confidenceScore?: number;
  sourceType?: 'RAG';
  factualityScore?: number;
  relevanceScore?: number;
  retrievedDocuments?: any[];
  sourcesUsed?: string[];
  timestamp?: Date;
  error?: string;
}

/**
 * Entrée pour le processus de débat
 */
export interface DebateInput {
  query: UserQuery;
  kagAnalysis: KagAnalysis;
  ragAnalysis: RagAnalysis;
  poolOutputs?: PoolOutputs;
}

/**
 * Résultat du processus de débat entre KAG et RAG
 * Cette interface est spécifique à notre implémentation
 */
export interface DebateResult {
  content: string;
  hasConsensus: boolean;
  identifiedThemes: string[];
  processingTime: number;
  sourceMetrics: {
    kagConfidence?: number;
    ragConfidence?: number;
    poolsUsed: PoolType[] | string[];
    sourcesUsed?: string[];
  };
  consensusLevel?: number;
  debateTimestamp?: Date;
  error?: string;
  output?: any; // Pour compatibilité avec les modules existants
}

/**
 * Métriques d'utilisation de l'API
 */
export interface ApiUsageMetrics {
  promptTokens: number;
  completionTokens: number;
  totalTokens: number;
  cost?: number;
  processingTime: number;
}

/**
 * Options pour la génération de la réponse finale
 */
export interface FinalResponseOptions {
  format?: 'TEXT' | 'MARKDOWN' | 'HTML';
  expertiseLevel: ExpertiseLevel;
  maxLength?: number;
  includeSuggestions?: boolean;
}

/**
 * Structure de la réponse finale
 */
export interface FinalResponse {
  content: string;
  metaData: {
    sourceTypes: string[];
    confidenceLevel: ConfidenceLevel;
    processingTime: number;
    usedAgentCount: number;
    expertiseLevel: ExpertiseLevel;
    topicsIdentified: string[];
  };
  suggestedFollowUp?: string[];
  error?: string;
}

/**
 * Structure pour les métriques d'exécution
 */
export interface ExecutionMetrics {
  totalProcessingTime: number;
  kagProcessingTime: number;
  ragProcessingTime: number;
  debateProcessingTime: number;
  synthesisProcessingTime: number;
  agentsUsed: number;
  tokensUsed: number;
  estimatedCost: number;
}

/**
 * Options pour la gestion de la requête
 */
export interface QueryOptions {
  timeout?: number;
  prioritizeSpeed?: boolean;
  expertiseLevel?: ExpertiseLevel;
  detailedMetrics?: boolean;
}

/**
 * Résultat complet du traitement d'une requête
 */
export interface QueryResult {
  query: UserQuery;
  response: FinalResponse;
  metrics?: ExecutionMetrics;
  executionPath?: string[];
  successful: boolean;
  errorMessage?: string;
} 