/**
 * Énumération des types d'agents disponibles
 */
export enum AgentType {
  COMMERCIAL = 'COMMERCIAL',
  MARKETING = 'MARKETING',
  SECTORIEL = 'SECTORIEL',
  EDUCATIONAL = 'EDUCATIONAL'
}

/**
 * Énumération des fournisseurs d'API disponibles
 */
export enum ApiProvider {
  QWEN = 'QWEN',
  GOOGLE = 'GOOGLE',
  DEEPSEEK = 'DEEPSEEK',
  HOUSE_MODEL = 'HOUSE_MODEL'
}

/**
 * Interface pour les réponses d'agent
 */
export interface AgentResponse {
  agentId: string;
  content: string;
  confidence: number;
  metadata: {
    processingTime: number;
    tokenCount: number;
    [key: string]: any;
  };
}

/**
 * Interface pour les requêtes utilisateur
 */
export interface UserQuery {
  id: string;
  content: string;
  context?: {
    previousQueries?: string[];
    relevantData?: any;
    [key: string]: any;
  };
  preferences?: {
    maxResponseTime?: number;
    preferredAgentTypes?: AgentType[];
    [key: string]: any;
  };
} 