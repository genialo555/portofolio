# Architecture du Système Mixte RAG/KAG en TypeScript

```
/agent-mixte-rag-kag/
│
├── /src/                          # Code source principal
│   ├── /config/                   # Configuration du système
│   │   ├── apiConfig.ts           # Configuration des APIs (Qwen, Google, DeepSeek)
│   │   ├── poolConfig.ts          # Configuration des pools d'agents
│   │   └── index.ts               # Point d'entrée des configurations
│   │
│   ├── /types/                    # Définitions des types TypeScript
│   │   ├── agent.types.ts         # Types pour les agents
│   │   ├── prompt.types.ts        # Types pour les prompts
│   │   ├── response.types.ts      # Types pour les réponses
│   │   └── index.ts               # Export des types
│   │
│   ├── /apis/                     # Intégrations avec les APIs LLM
│   │   ├── qwen.api.ts            # API Qwen
│   │   ├── google.api.ts          # API Google
│   │   ├── deepseek.api.ts        # API DeepSeek
│   │   └── index.ts               # Façade d'API unifiée
│   │
│   ├── /pools/                    # Implémentation des pools d'agents
│   │   ├── /commercial/           # Pool d'agents commerciaux
│   │   │   ├── agent1.ts          # Agent commercial 1
│   │   │   ├── agent2.ts          # Agent commercial 2
│   │   │   ├── agent3.ts          # Agent commercial 3
│   │   │   ├── agent4.ts          # Agent commercial 4
│   │   │   └── index.ts           # Gestionnaire du pool commercial
│   │   │
│   │   ├── /marketing/            # Pool d'agents marketing
│   │   │   ├── agent1.ts          # Agent marketing 1
│   │   │   ├── agent2.ts          # Agent marketing 2
│   │   │   ├── agent3.ts          # Agent marketing 3
│   │   │   ├── agent4.ts          # Agent marketing 4
│   │   │   └── index.ts           # Gestionnaire du pool marketing
│   │   │
│   │   ├── /sectoriel/            # Pool d'agents sectoriels
│   │   │   ├── agent1.ts          # Agent sectoriel 1
│   │   │   ├── agent2.ts          # Agent sectoriel 2
│   │   │   ├── agent3.ts          # Agent sectoriel 3
│   │   │   ├── agent4.ts          # Agent sectoriel 4
│   │   │   └── index.ts           # Gestionnaire du pool sectoriel
│   │   │
│   │   └── index.ts               # Façade unifiée des pools
│   │
│   ├── /prompts/                  # Définitions des prompts
│   │   ├── /base-prompts/         # Prompts de base
│   │   │   ├── commercial.ts      # Prompts de base commerciaux
│   │   │   ├── marketing.ts       # Prompts de base marketing
│   │   │   └── sectoriel.ts       # Prompts de base sectoriels
│   │   │
│   │   ├── /meta-prompts/         # Méta-prompts
│   │   │   ├── orchestrator.ts    # Prompts pour l'orchestrateur
│   │   │   ├── anomaly.ts         # Prompts pour la détection d'anomalies
│   │   │   └── synthesis.ts       # Prompts pour la synthèse
│   │   │
│   │   ├── /debate-prompts/       # Prompts pour le débat
│   │   │   ├── kag.ts             # Prompts pour le modèle KAG
│   │   │   ├── rag.ts             # Prompts pour le modèle RAG
│   │   │   └── debate.ts          # Prompts pour le processus de débat
│   │   │
│   │   └── index.ts               # Export des prompts
│   │
│   ├── /orchestrator/             # Système d'orchestration
│   │   ├── router.ts              # Routeur des requêtes
│   │   ├── poolManager.ts         # Gestionnaire des pools
│   │   ├── outputCollector.ts     # Collecteur d'outputs
│   │   └── index.ts               # Point d'entrée de l'orchestrateur
│   │
│   ├── /debate/                   # Système de débat KAG vs RAG
│   │   ├── kagEngine.ts           # Moteur KAG
│   │   ├── ragEngine.ts           # Moteur RAG
│   │   ├── debateProtocol.ts      # Protocole de débat
│   │   └── index.ts               # Point d'entrée du système de débat
│   │
│   ├── /synthesis/                # Module de synthèse
│   │   ├── merger.ts              # Fusion des outputs
│   │   ├── contradictionResolver.ts # Résolution des contradictions
│   │   ├── responseFormatter.ts   # Formattage des réponses
│   │   └── index.ts               # Point d'entrée du module de synthèse
│   │
│   ├── /utils/                    # Utilitaires
│   │   ├── logger.ts              # Système de logging
│   │   ├── errorHandler.ts        # Gestion des erreurs
│   │   └── helpers.ts             # Fonctions helper
│   │
│   └── index.ts                   # Point d'entrée principal
│
├── /tests/                        # Tests
│   ├── /unit/                     # Tests unitaires
│   ├── /integration/              # Tests d'intégration
│   └── /e2e/                      # Tests end-to-end
│
├── /scripts/                      # Scripts utilitaires
│   ├── setup.ts                   # Script de configuration
│   └── deploy.ts                  # Script de déploiement
│
├── .env                           # Variables d'environnement
├── .gitignore                     # Fichiers ignorés par git
├── package.json                   # Dépendances et scripts
├── tsconfig.json                  # Configuration TypeScript
└── README.md                      # Documentation
```

## Structure des Fichiers Clés

### 1. Configuration des Agents

**`poolConfig.ts`**
```typescript
export interface AgentConfig {
  id: string;
  name: string;
  api: 'qwen' | 'google' | 'deepseek';
  parameters: {
    temperature: number;
    top_p: number;
    top_k: number;
    max_tokens: number;
    context_window: number;
    // Autres paramètres spécifiques
  };
  promptTemplate: string;
}

export const commercialAgents: AgentConfig[] = [
  // 4 configurations d'agents commerciaux
];

export const marketingAgents: AgentConfig[] = [
  // 4 configurations d'agents marketing
];

export const sectorielAgents: AgentConfig[] = [
  // 4 configurations d'agents sectoriels
];
```

### 2. Orchestrateur

**`orchestrator/index.ts`**
```typescript
import { router } from './router';
import { poolManager } from './poolManager';
import { outputCollector } from './outputCollector';
import { UserQuery } from '../types';

export async function processQuery(query: UserQuery) {
  // 1. Router la requête vers les pools appropriés
  const targetPools = router.determineTargetPools(query);
  
  // 2. Exécuter les agents dans chaque pool en parallèle
  const poolOutputs = await poolManager.executeAgents(targetPools, query);
  
  // 3. Collecter et structurer les outputs
  const structuredOutputs = outputCollector.collect(poolOutputs);
  
  // 4. Retourner les outputs pour débat et synthèse
  return structuredOutputs;
}
```

### 3. Système de Débat

**`debate/index.ts`**
```typescript
import { kagEngine } from './kagEngine';
import { ragEngine } from './ragEngine';
import { debateProtocol } from './debateProtocol';
import { PoolOutputs } from '../types';

export async function conductDebate(poolOutputs: PoolOutputs) {
  // 1. Analyse KAG des outputs
  const kagAnalysis = await kagEngine.analyze(poolOutputs);
  
  // 2. Enrichissement RAG des outputs
  const ragAnalysis = await ragEngine.enrich(poolOutputs);
  
  // 3. Confrontation KAG vs RAG
  const debateResult = await debateProtocol.debate(kagAnalysis, ragAnalysis);
  
  // 4. Retourner le résultat du débat
  return debateResult;
}
```

### 4. Module de Synthèse

**`synthesis/index.ts`**
```typescript
import { merger } from './merger';
import { contradictionResolver } from './contradictionResolver';
import { responseFormatter } from './responseFormatter';
import { DebateResult } from '../types';

export async function synthesize(debateResult: DebateResult) {
  // 1. Fusionner les insights du débat
  const mergedInsights = merger.merge(debateResult);
  
  // 2. Résoudre les contradictions potentielles
  const resolvedInsights = contradictionResolver.resolve(mergedInsights);
  
  // 3. Formater la réponse finale
  const finalResponse = responseFormatter.format(resolvedInsights);
  
  return finalResponse;
}
``` 