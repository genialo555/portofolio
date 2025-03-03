/**
 * Interface pour les entrées de débat entre KAG et RAG
 */
export interface DebateInput {
  kagAnalysis: string;
  ragAnalysis: string;
  query: string;
  poolOutputs: any;
}

/**
 * Interface pour les résultats du débat
 */
export interface DebateResult {
  agreements: string[];
  contradictions: string[];
  synthesis: string;
  confidenceLevel: number;
  unresolvedIssues?: string[];
}

/**
 * Types pour la gestion des prompts dans l'ancienne architecture
 */

/**
 * Type de prompt
 */
export enum PromptType {
  COMMERCIAL = 'COMMERCIAL',
  MARKETING = 'MARKETING',
  SECTORIEL = 'SECTORIEL',
  RAG = 'RAG',
  KAG = 'KAG',
  DEBATE = 'DEBATE',
  SYNTHESIS = 'SYNTHESIS',
  COORDINATION = 'COORDINATION',
  ORCHESTRATOR = 'ORCHESTRATOR',
  ANOMALY = 'ANOMALY',
  AGENT = 'AGENT'
}

/**
 * Prompt avec son type
 */
export interface Prompt {
  type: PromptType;
  content: string;
  metadata?: { 
    [key: string]: any; 
    description?: string; 
    version?: string; 
  };
  variables?: Record<string, string | number | boolean>;
}

/**
 * Template de prompt avec ses variables
 */
export interface PromptTemplate {
  type: PromptType;
  template: string;
  description: string;
  variables: string[];
}

/**
 * Interface pour les paramètres de prompt
 */
export interface PromptParameters {
  temperature: number;
  top_p: number;
  top_k: number;
  max_tokens: number;
  stop_sequences?: string[];
  presence_penalty?: number;
  frequency_penalty?: number;
  [key: string]: any;
}

/**
 * Interface générique pour les prompts
 */
export interface Prompt {
  type: PromptType;
  content: string;
  parameters: PromptParameters;
  metadata?: {
    description?: string;
    version?: string;
    [key: string]: any;
  };
}

/**
 * Interface pour les prompts d'anomalie
 */
export interface AnomalyPrompt extends Prompt {
  type: PromptType.ANOMALY;
  thresholds: {
    contradiction: number;
    factualError: number;
    hallucination: number;
    [key: string]: number;
  };
} 