import { AgentType, ApiProvider } from '../types/agent.types';

export interface AgentConfig {
  id: string;
  name: string;
  type: AgentType;
  api: ApiProvider;
  parameters: {
    temperature: number;
    top_p: number;
    top_k: number;
    max_tokens: number;
    context_window: number;
    stop_sequences?: string[];
    presence_penalty?: number;
    frequency_penalty?: number;
  };
  description: string;
}

// Agents du pool Commercial
export const commercialAgents: AgentConfig[] = [
  {
    id: 'commercial-agent-1',
    name: 'Agent Commercial Analytique',
    type: AgentType.COMMERCIAL,
    api: ApiProvider.QWEN,
    parameters: {
      temperature: 0.2,
      top_p: 0.85,
      top_k: 40,
      max_tokens: 1000,
      context_window: 8000,
      presence_penalty: 0.1,
      frequency_penalty: 0.1
    },
    description: 'Agent focalisé sur l\'analyse de données de vente et comportements d\'achat'
  },
  {
    id: 'commercial-agent-2',
    name: 'Agent Commercial Relationnel',
    type: AgentType.COMMERCIAL,
    api: ApiProvider.GOOGLE,
    parameters: {
      temperature: 0.5,
      top_p: 0.9,
      top_k: 50,
      max_tokens: 1200,
      context_window: 8000,
      presence_penalty: 0.3,
      frequency_penalty: 0.2
    },
    description: 'Agent spécialisé en relation client et fidélisation'
  },
  {
    id: 'commercial-agent-3',
    name: 'Agent Commercial Stratégique',
    type: AgentType.COMMERCIAL,
    api: ApiProvider.DEEPSEEK,
    parameters: {
      temperature: 0.7,
      top_p: 0.92,
      top_k: 60,
      max_tokens: 1500,
      context_window: 10000,
      presence_penalty: 0.2,
      frequency_penalty: 0.3
    },
    description: 'Agent focalisé sur les stratégies de vente à long terme'
  },
  {
    id: 'commercial-agent-4',
    name: 'Agent Commercial Innovant',
    type: AgentType.COMMERCIAL,
    api: ApiProvider.QWEN,
    parameters: {
      temperature: 0.9,
      top_p: 0.95,
      top_k: 80,
      max_tokens: 1800,
      context_window: 12000,
      presence_penalty: 0.4,
      frequency_penalty: 0.4
    },
    description: 'Agent spécialisé en approches commerciales disruptives et innovantes'
  }
];

// Agents du pool Marketing
export const marketingAgents: AgentConfig[] = [
  {
    id: 'marketing-agent-1',
    name: 'Agent Marketing Analytique',
    type: AgentType.MARKETING,
    api: ApiProvider.GOOGLE,
    parameters: {
      temperature: 0.3,
      top_p: 0.88,
      top_k: 45,
      max_tokens: 1200,
      context_window: 9000,
      presence_penalty: 0.2,
      frequency_penalty: 0.2
    },
    description: 'Agent focalisé sur l\'analyse de données marketing et métriques'
  },
  {
    id: 'marketing-agent-2',
    name: 'Agent Marketing Contenu',
    type: AgentType.MARKETING,
    api: ApiProvider.DEEPSEEK,
    parameters: {
      temperature: 0.6,
      top_p: 0.92,
      top_k: 55,
      max_tokens: 1500,
      context_window: 10000,
      presence_penalty: 0.3,
      frequency_penalty: 0.3
    },
    description: 'Agent spécialisé en stratégies de contenu et narration'
  },
  {
    id: 'marketing-agent-3',
    name: 'Agent Marketing Digital',
    type: AgentType.MARKETING,
    api: ApiProvider.QWEN,
    parameters: {
      temperature: 0.7,
      top_p: 0.9,
      top_k: 60,
      max_tokens: 1300,
      context_window: 10000,
      presence_penalty: 0.25,
      frequency_penalty: 0.35
    },
    description: 'Agent focalisé sur les stratégies marketing digitales et réseaux sociaux'
  },
  {
    id: 'marketing-agent-4',
    name: 'Agent Marketing Créatif',
    type: AgentType.MARKETING,
    api: ApiProvider.GOOGLE,
    parameters: {
      temperature: 0.95,
      top_p: 0.98,
      top_k: 100,
      max_tokens: 2000,
      context_window: 12000,
      presence_penalty: 0.5,
      frequency_penalty: 0.3
    },
    description: 'Agent spécialisé en idées créatives et campagnes disruptives'
  }
];

// Agents du pool Sectoriel
export const sectorielAgents: AgentConfig[] = [
  {
    id: 'sectoriel-agent-1',
    name: 'Agent Sectoriel B2B',
    type: AgentType.SECTORIEL,
    api: ApiProvider.DEEPSEEK,
    parameters: {
      temperature: 0.3,
      top_p: 0.85,
      top_k: 40,
      max_tokens: 1500,
      context_window: 10000,
      presence_penalty: 0.2,
      frequency_penalty: 0.2
    },
    description: 'Agent spécialisé dans les problématiques B2B et ventes complexes'
  },
  {
    id: 'sectoriel-agent-2',
    name: 'Agent Sectoriel B2C',
    type: AgentType.SECTORIEL,
    api: ApiProvider.QWEN,
    parameters: {
      temperature: 0.5,
      top_p: 0.9,
      top_k: 50,
      max_tokens: 1300,
      context_window: 9000,
      presence_penalty: 0.3,
      frequency_penalty: 0.3
    },
    description: 'Agent focalisé sur les marchés grand public et comportement consommateur'
  },
  {
    id: 'sectoriel-agent-3',
    name: 'Agent Sectoriel Tech',
    type: AgentType.SECTORIEL,
    api: ApiProvider.GOOGLE,
    parameters: {
      temperature: 0.6,
      top_p: 0.92,
      top_k: 60,
      max_tokens: 1600,
      context_window: 11000,
      presence_penalty: 0.25,
      frequency_penalty: 0.25
    },
    description: 'Agent spécialisé dans les secteurs technologiques et innovation'
  },
  {
    id: 'sectoriel-agent-4',
    name: 'Agent Sectoriel Émergent',
    type: AgentType.SECTORIEL,
    api: ApiProvider.DEEPSEEK,
    parameters: {
      temperature: 0.8,
      top_p: 0.95,
      top_k: 75,
      max_tokens: 1800,
      context_window: 12000,
      presence_penalty: 0.4,
      frequency_penalty: 0.3
    },
    description: 'Agent focalisé sur les marchés émergents et tendances futures'
  }
];

// Export de toutes les configurations d'agents
export const allAgentConfigs: AgentConfig[] = [
  ...commercialAgents,
  ...marketingAgents,
  ...sectorielAgents
];

// Fonctions utilitaires pour récupérer des configurations
export function getAgentConfig(agentId: string): AgentConfig | undefined {
  return allAgentConfigs.find(agent => agent.id === agentId);
}

export function getAgentsByType(type: AgentType): AgentConfig[] {
  return allAgentConfigs.filter(agent => agent.type === type);
} 