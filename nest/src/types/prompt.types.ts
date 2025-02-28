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
 * Types de prompts disponibles dans le système
 */
export enum PromptType {
  AGENT = 'AGENT',
  ORCHESTRATOR = 'ORCHESTRATOR',
  KAG = 'KAG',
  RAG = 'RAG',
  DEBATE = 'DEBATE',
  SYNTHESIS = 'SYNTHESIS',
  ANOMALY = 'ANOMALY',
  COORDINATION = 'COORDINATION'
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