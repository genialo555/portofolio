/**
 * Définitions des types pour le système mixte RAG/KAG
 */

// Types pour les requêtes et réponses
export interface UserQuery {
  text: string;
  contextInfo?: any;
  domainHints?: string[];
  timestamp?: Date | number;
  sessionId?: string;
  userId?: string;
  metadata?: Record<string, any>;
  userType?: any; // UserType est défini dans l'autre fichier de types
}

export type ExpertiseLevel = 'BEGINNER' | 'INTERMEDIATE' | 'ADVANCED';

export interface FinalResponse {
  content: string;
  metaData: {
    sourceTypes: ('KAG' | 'RAG' | 'POOL' | 'DEBATE')[];
    confidenceLevel: ConfidenceLevel;
    processingTime: number;
    usedAgentCount: number;
    expertiseLevel: ExpertiseLevel;
    topicsIdentified: string[];
  };
  suggestedFollowUp?: string[];
  error?: string;
}

// Types pour les pools d'agents
export type PoolType = 'COMMERCIAL' | 'MARKETING' | 'SECTORIEL';

export interface TargetPools {
  commercial: boolean;
  marketing: boolean;
  sectoriel: boolean;
  primaryFocus?: PoolType;
}

export interface AgentConfig {
  id: string;
  name: string;
  poolType: PoolType;
  api: 'qwen' | 'google' | 'deepseek';
  parameters: {
    temperature: number;
    top_p: number;
    top_k: number;
    max_tokens: number;
    context_window: number;
    repetition_penalty?: number;
  };
  promptTemplate: string;
}

export interface AgentOutput {
  agentId: string;
  poolType: PoolType;
  content: string;
  confidence: number;
  processingTime: number;
  metadata: {
    temperature: number;
    api: string;
    tokensUsed: number;
    perspectiveType?: string;
  };
}

export interface PoolOutputs {
  commercial?: AgentOutput[];
  marketing?: AgentOutput[];
  sectoriel?: AgentOutput[];
  timestamp?: Date;
  query?: UserQuery;
}

// Types pour le détecteur d'anomalies
export type AnomalyType = 
  | 'LOGICAL_INCONSISTENCY' 
  | 'FACTUAL_ERROR' 
  | 'METHODOLOGICAL_FLAW'
  | 'STATISTICAL_ERROR'
  | 'UNJUSTIFIED_GENERALIZATION'
  | 'COGNITIVE_BIAS'
  | 'CITATION_ISSUE'
  | 'CONCEPTUAL_CONFUSION';

export type AnomalySeverity = 'HIGH' | 'MEDIUM' | 'LOW';

export interface Anomaly {
  type: AnomalyType;
  description: string;
  location: {
    agentId: string;
    poolType: PoolType;
    contentFragment: string;
  };
  severity: AnomalySeverity;
  suggestedResolution?: string;
}

export interface AnomalyReport {
  overallReliability: number; // 0-1
  highPriorityAnomalies: Anomaly[];
  mediumPriorityAnomalies: Anomaly[];
  minorIssues: Anomaly[];
  lowPriorityAnomalies?: Anomaly[];
  report?: string;
  systemicPatterns?: string[];
}

// Types pour le système de débat KAG vs RAG
export type ConfidenceLevel = 'VERY_HIGH' | 'HIGH' | 'MEDIUM' | 'LOW' | 'VERY_LOW';

export interface KnowledgeClaim {
  claim: string;
  source: 'KAG' | 'RAG' | 'POOL';
  confidence: ConfidenceLevel;
  supportingEvidence?: string;
}

export interface KagAnalysis {
  keyClaims: KnowledgeClaim[];
  verificationResults: {
    verifiedClaims: KnowledgeClaim[];
    unverifiedClaims: KnowledgeClaim[];
    contradictoryClaims: KnowledgeClaim[][];
  };
  theoreticalFrameworks: {
    name: string;
    relevance: number; // 0-1
    applicability: string;
  }[];
  knowledgeGaps: string[];
  overallEvaluation: string;
}

export interface RagAnalysis {
  enrichedClaims: {
    originalClaim: KnowledgeClaim;
    retrievedEvidence: string[];
    updatedClaim?: KnowledgeClaim;
  }[];
  additionalContext: {
    topic: string;
    retrievedInformation: string;
    source?: string;
    relevance: number; // 0-1
  }[];
  caseStudies: {
    title: string;
    description: string;
    relevance: number; // 0-1
    source?: string;
  }[];
  statistics: {
    description: string;
    value: string;
    source?: string;
    date?: string;
  }[];
}

export interface DebatePoint {
  topic: string;
  kagPerspective: {
    position: string;
    strength: number; // 0-1
    reasoning: string;
  };
  ragPerspective: {
    position: string;
    strength: number; // 0-1
    reasoning: string;
    evidence?: string;
  };
  synthesis?: string;
  irreconcilable: boolean;
}

export interface DebateResult {
  agreementPoints: KnowledgeClaim[];
  debatePoints: DebatePoint[];
  highConfidenceInsights: KnowledgeClaim[];
  openQuestions: string[];
  metaAnalysis: string;
  output?: any; // Pour compatibilité avec les modules existants
}

// Types pour le module de synthèse
export interface MergedInsights {
  coreClaims: KnowledgeClaim[];
  supportingEvidence: {
    claim: string;
    evidence: string[];
  }[];
  practicalApplications: string[];
  theoreticalFoundations: string[];
  confidence: ConfidenceLevel;
}

export interface ResolvedInsights extends MergedInsights {
  resolvedContradictions: {
    originalClaims: string[];
    resolution: string;
    confidence: ConfidenceLevel;
  }[];
  uncertaintyAreas: string[];
}

// Types pour l'API
export interface ProcessQueryRequest {
  query: {
    text: string;
    contextInfo?: any;
  };
  expertiseLevel?: ExpertiseLevel;
  useSimplifiedProcess?: boolean;
}

export interface ProcessQueryResponse {
  response: FinalResponse;
  status: 'SUCCESS' | 'PARTIAL_SUCCESS' | 'FAILURE';
  errorMessage?: string;
}

// Types utilitaires
export interface LogEntry {
  timestamp: Date;
  level: 'INFO' | 'WARN' | 'ERROR' | 'DEBUG';
  message: string;
  data?: any;
}

export interface PerformanceMetrics {
  totalTime?: number;
  componentTimes?: Record<string, number>;
  tokenCounts?: Record<string, number>;
  modelCalls?: number;
  dbQueries?: number;
  costEstimate?: number;
  endpoint?: string;
  responseTime?: number;
  tokensUsed?: number;
  cost?: number;
  success?: boolean;
}

/**
 * Type pour le statut des composants dans le système
 */
export enum ComponentStatus {
  READY = 'READY',
  INITIALIZING = 'INITIALIZING',
  PROCESSING = 'PROCESSING',
  COMPLETED = 'COMPLETED',
  ERROR = 'ERROR',
  IDLE = 'IDLE'
}

/**
 * Type pour le niveau de log
 */
export enum LogLevel {
  DEBUG = 'DEBUG',
  INFO = 'INFO',
  WARN = 'WARN',
  ERROR = 'ERROR'
}

/**
 * Interface de base pour une entité du système
 */
export interface BaseEntity {
  id: string;
  name: string;
  type: string;
  status?: ComponentStatus;
  metadata?: Record<string, any>;
}

/**
 * Interface de base pour un événement système
 */
export interface SystemEvent {
  id: string;
  timestamp: number;
  type: string;
  source: string;
  target?: string;
  data?: any;
  payload?: any; // Pour compatibilité avec event-bus.ts
}

/**
 * Interface pour le contexte d'exécution
 */
export interface ExecutionContext {
  sessionId: string;
  query: UserQuery;
  results: Record<string, ComponentResult>;
  startTime: number;
  metadata?: Record<string, any>;
}

/**
 * Types d'options pour les exécutions
 */
export interface ExecutionOptions {
  timeout?: number;
  retry?: number;
  priority?: ComponentPriority;
  mode?: ExecutionMode;
}

/**
 * Modes d'exécution pour les composants
 */
export enum ExecutionMode {
  SEQUENTIAL = 'SEQUENTIAL',
  PARALLEL = 'PARALLEL',
  ADAPTIVE = 'ADAPTIVE'
}

/**
 * Niveaux de priorité pour les composants
 */
export enum ComponentPriority {
  HIGH = 'HIGH',
  MEDIUM = 'MEDIUM',
  LOW = 'LOW'
}

/**
 * Interface pour le résultat d'un composant
 */
export interface ComponentResult {
  id: string;
  componentId: string;
  status: ComponentStatus;
  data?: any;
  error?: string;
  processingTime: number;
  startTime: number;
  endTime: number;
}

/**
 * Types pour les composants
 */
export enum ComponentType {
  QUERY_ANALYZER = 'QUERY_ANALYZER',
  KNOWLEDGE_RETRIEVER = 'KNOWLEDGE_RETRIEVER',
  RAG_ENGINE = 'RAG_ENGINE',
  KAG_ENGINE = 'KAG_ENGINE',
  DEBATE_PROTOCOL = 'DEBATE_PROTOCOL',
  SYNTHESIS = 'SYNTHESIS'
}

/**
 * Types de sources de connaissance
 */
export enum KnowledgeSourceType {
  DATABASE = 'DATABASE',
  API = 'API',
  FILE_SYSTEM = 'FILE_SYSTEM',
  MODEL = 'MODEL'
}

/**
 * Résultat de l'analyse de requête
 */
export interface QueryAnalysisResult {
  queryType: string;
  entities: string[];
  intent: string;
  complexity: number;
  suggestedComponents: ComponentType[];
  domainHints: string[];
}

/**
 * Résultat de récupération de connaissance
 */
export interface KnowledgeRetrievalResult {
  sourceType: KnowledgeSourceType;
  documents: any[];
  relevanceScores: number[];
  sourceMappings: Record<string, string>;
}

/**
 * Résultat d'analyse RAG
 */
export interface RagAnalysisResult {
  contextualKnowledge: string;
  confidenceScore: number;
  references: string[];
  reasoning: string;
}

/**
 * Résultat d'analyse KAG
 */
export interface KagAnalysisResult {
  modelKnowledge: string;
  confidenceScore: number;
  uncertaintyAreas: string[];
  reasoning: string;
}

/**
 * Résultat de synthèse
 */
export interface SynthesisResult {
  finalContent: string;
  confidence: number;
  suggestedQuestions: string[];
  metadata: Record<string, any>;
}

// Exports
export * from './agent.types';
export * from './prompt.types';
// export * from './response.types'; 