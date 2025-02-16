export type AIModel = "gemini" | "qwen-max" | "qwen-plus" | "qwen-turbo" | "deepseek" | "deepseek-reasoner";
export type DebateRole = "pour" | "contre" | "synthese";

export interface Message {
  id: string;
  content: string;
  role: DebateRole;
  timestamp: Date;
  media?: {
    type: "image" | "video";
    url: string;
    alt?: string;
  };
}

export interface Conversation {
  id: string;
  title: string;
  messages: Message[];
  lastUpdated: Date;
  models: {
    pour: AIModel;
    contre: AIModel;
    synthese: AIModel;
  };
}

export interface ModelConfig {
  id: AIModel;
  name: string;
  description: string;
  modelId: string;
  maxLength: number;
  temperature: number;
  recommendedRoles?: DebateRole[];
}

export interface CrossModelConfig {
  pour: AIModel;
  contre: AIModel;
  synthese: AIModel;
} 