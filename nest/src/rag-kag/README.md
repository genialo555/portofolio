# Module RAG/KAG NestJS

Ce module implémente un système de génération de réponses combinant les approches RAG (Retrieval Augmented Generation) et KAG (Knowledge Augmented Generation), structuré selon les principes NestJS.

## Architecture

Le module est organisé selon les principes de conception NestJS avec une structure modulaire :

```
src/rag-kag/
├── agents/                # Gestion des agents
│   ├── agent-factory.service.ts
│   └── agents.module.ts
├── apis/                  # Intégration avec les APIs LLM
│   ├── api-provider-factory.service.ts
│   ├── apis.module.ts
│   ├── deepseek-ai.service.ts
│   ├── google-ai.service.ts
│   ├── house-model.service.ts
│   ├── model-evaluation.service.ts
│   ├── model-training.service.ts
│   ├── model-utils.service.ts
│   ├── qwen-ai.service.ts
│   └── tokenizer.service.ts
├── controllers/           # Endpoints API
│   └── rag-kag.controller.ts
├── core/                  # Services fondamentaux
│   ├── core.module.ts
│   ├── event-bus.service.ts
│   ├── knowledge-graph.service.ts
│   ├── metrics.service.ts
│   └── query-analyzer.service.ts
├── debate/                # Système de débat KAG/RAG
│   ├── debate.module.ts
│   ├── debate.service.ts
│   ├── kag-engine.service.ts
│   └── rag-engine.service.ts
├── orchestrator/          # Orchestration des requêtes
│   ├── orchestrator.module.ts
│   ├── orchestrator.service.ts
│   ├── output-collector.service.ts
│   └── router.service.ts
├── pools/                 # Pools d'agents spécialisés
│   ├── commercial-pool.service.ts
│   ├── educational-pool.service.ts
│   ├── marketing-pool.service.ts
│   ├── pool-manager.service.ts
│   ├── pools.module.ts
│   └── sectorial-pool.service.ts
├── prompts/               # Gestion des templates de prompts
│   ├── prompts.module.ts
│   └── prompts.service.ts
├── synthesis/             # Synthèse des réponses
│   ├── synthesis.module.ts
│   └── synthesis.service.ts
├── testing/               # Outils de test automatisés
│   ├── auto-test.service.ts
│   └── fixtures/
├── types/                 # Définitions de types
│   ├── index.ts
│   └── types.module.ts
├── utils/                 # Utilitaires
│   ├── logger-tokens.ts
│   ├── logger.module.ts
│   └── resilience.service.ts
└── rag-kag.module.ts      # Module principal
```

## Flux de traitement des requêtes

Le traitement d'une requête suit le flux suivant :

1. **Réception** - `rag-kag.controller.ts`
2. **Orchestration** - `orchestrator.service.ts`
   - Analyse de la requête - `query-analyzer.service.ts`
   - Routage vers les pools appropriés - `router.service.ts`
3. **Exécution des agents** - `pool-manager.service.ts`
   - Pools spécialisés (commercial, marketing, sectoriel, éducatif)
   - Création et exécution des agents - `agent-factory.service.ts`
4. **Débat** - `debate.service.ts`
   - Analyse KAG - `kag-engine.service.ts`
   - Analyse RAG - `rag-engine.service.ts`
5. **Synthèse** - `synthesis.service.ts`
6. **Retour de la réponse**

## Intégration avec l'architecture legacy

Le module intègre des fonctionnalités avancées de l'architecture legacy via des adaptateurs spécifiques :

### 1. Orchestrateur hybride

L'`OrchestratorService` peut utiliser le système de coordination avancé de l'ancienne architecture :

```typescript
// Dans OrchestratorService
async processQuery(query: UserQuery, expertiseLevel: ExpertiseLevel, options: ProcessingOptions): Promise<FinalResponse> {
  if (options.useAdvancedCoordination) {
    return this.processWithAdvancedCoordination(query, expertiseLevel, options);
  }
  
  // Implémentation standard...
}
```

### 2. Système de débat amélioré

Le `DebateService` peut utiliser les prompts dialectiques sophistiqués de l'ancienne architecture :

```typescript
// Dans DebateService
async generateDebate(query: UserQuery, options: { useLegacyPrompt?: boolean }): Promise<DebateResult> {
  // ...
  if (options.useLegacyPrompt) {
    debatePrompt = generateKagRagDebatePrompt(legacyDebateInput);
  } else {
    debatePrompt = this.promptsService.getPromptTemplate(PromptTemplateType.KAG_RAG_DEBATE);
    // ...
  }
  // ...
}
```

### 3. Détection d'anomalies avancée

Le service de détection d'anomalies peut utiliser les analyseurs spécialisés de l'ancienne architecture :

```typescript
// Dans AnomalyDetectionService
async detectAnomalies(poolOutputs: PoolOutputs, options: AnomalyDetectionOptions): Promise<AnomalyReport> {
  // ...
  if (options.useAdvancedDetection) {
    report = await this.enrichWithAdvancedDetection(report, poolOutputs);
  }
  // ...
}
```

## Services principaux

### 1. EventBusService

Le bus d'événements sert de colonne vertébrale pour la communication entre services :

```typescript
// Exemples d'utilisation
// Émission d'événement
this.eventBus.emit({
  type: RagKagEventType.QUERY_RECEIVED,
  source: 'OrchestratorService',
  payload: { query: queryText }
});

// Abonnement à un événement
this.eventBus.subscribe(
  RagKagEventType.ANOMALY_DETECTED,
  (event) => this.handleAnomalyDetected(event),
  { priority: 10 }
);
```

### 2. KnowledgeGraphService

Le graphe de connaissances stocke et organise les informations :

```typescript
// Exemple d'utilisation
const nodeId = this.knowledgeGraph.addNode({
  label: 'MarketAnalysis',
  type: 'INSIGHT',
  content: 'Le marché IoT connaît une croissance de 15% annuelle',
  confidence: 0.85,
  source: KnowledgeSource.RAG
});

this.knowledgeGraph.addEdge({
  sourceId: queryNodeId,
  targetId: nodeId,
  type: RelationType.GENERATED,
  weight: 0.9,
  bidirectional: false
});
```

### 3. ResilienceService

Gère la résilience des appels aux services externes :

```typescript
// Exemple d'utilisation
return this.resilienceService.executeWithCircuitBreaker(
  'google-ai',
  async () => await this.googleApiClient.generateText(prompt),
  async (error) => this.fallbackGenerateText(prompt, error)
);
```

## Utilisation avancée

### 1. Modèles locaux (House Model)

Le service intègre un système de modèles locaux avec formation continue :

```typescript
// Exécution d'un modèle local
const result = await this.houseModelService.generateResponse(prompt, {
  modelName: 'phi-3-mini',
  maxLength: 500
});

// Formation d'un modèle local
await this.modelTrainingService.finetuneDistilledModel('phi-3-mini');
```

### 2. Tests automatisés

Le module inclut un système de tests automatisés via `AutoTestService` :

```typescript
// Lancement des tests automatisés
await this.autoTestService.runTests();

// Test d'un composant spécifique
await this.autoTestService.testComponent('debate');
```

### 3. Métriques de performance

Le système collecte et analyse des métriques de performance :

```typescript
// Enregistrement de métriques
this.metricsService.recordMetric({
  type: MetricType.LATENCY,
  value: processingTime,
  tags: { component: 'debate', provider: 'google-ai' },
  source: 'DebateService'
});

// Récupération de métriques
const latencyMetrics = this.metricsService.getMetrics({
  types: [MetricType.LATENCY],
  sources: ['DebateService'],
  startTime: Date.now() - 24 * 60 * 60 * 1000 // Dernier jour
});
```

## Configuration

Le module est configurable via des variables d'environnement et des fichiers de configuration :

```typescript
// Exemple de configuration
export const ragKagConfig = {
  apis: {
    googleAi: {
      apiKey: process.env.GOOGLE_AI_API_KEY,
      model: 'gemini-pro',
      defaultTemperature: 0.3
    },
    // ...
  },
  houseModels: {
    enabled: true,
    modelsDir: './models',
    defaultModel: 'phi-3-mini'
  },
  // ...
};
```

## Contribution au module

Pour contribuer au développement du module :

1. Respecter l'architecture modulaire NestJS
2. Utiliser l'injection de dépendances pour tous les services
3. Suivre les conventions de nommage établies
4. Documenter les interfaces publiques
5. Écrire des tests unitaires pour les nouvelles fonctionnalités
6. Utiliser l'EventBusService pour la communication entre composants
7. Consulter `ARCHITECTURE.md` pour comprendre l'intégration avec les composants legacy

## Documentation complémentaire

Pour plus d'informations sur l'architecture globale, se référer à :

- `ARCHITECTURE.md` - Vue d'ensemble de l'architecture hybride
- `src/legacy/README.md` - Documentation des composants legacy
- `test/README.md` - Guide pour les tests automatisés 