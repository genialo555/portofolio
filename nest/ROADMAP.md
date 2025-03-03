# Feuille de route pour le système RAG/KAG

## Architecture globale

Le système RAG/KAG est construit autour des composants principaux suivants :

1. **Core Components** 
   - EventBusService - Bus d'événements pour la communication asynchrone
   - KnowledgeGraphService - Graphe de connaissances pour stocker et récupérer des informations

2. **Flow Principal**
   - RouterService - Dirige les requêtes vers les pools appropriés
   - PoolManagerService - Gère l'exécution des agents dans différents pools
   - DebateService - Organise le débat entre RAG et KAG
   - SynthesisService - Produit les réponses finales

3. **Moteurs de Génération**
   - RagEngineService - Génération augmentée par récupération
   - KagEngineService - Génération augmentée par connaissances 
   - HouseModelService - Génération par modèles locaux

4. **Services Support**
   - ResilienceService - Gestion des circuit breakers
   - AnomalyDetectionService - Détection d'anomalies dans les réponses
   - ApiProviderFactory - Abstraction des fournisseurs d'API

5. **Gestion des modèles**
   - ModelTrainingService - Formation des modèles distillés
   - ModelEvaluationService - Évaluation des performances des modèles
   - ModelUtilsService - Utilitaires pour les modèles 

## État d'intégration actuel

| Composant | EventBus | KnowledgeGraph | Circuit Breakers | Tests | Statut |
|-----------|:--------:|:--------------:|:----------------:|:-----:|:------:|
| EventBusService | ✅ | N/A | N/A | ❌ | Complet |
| KnowledgeGraphService | ✅ | N/A | N/A | ❌ | Complet |
| RouterService | ✅ | ✅ | N/A | ❌ | Complet |
| PoolManagerService | ✅ | ❌ | ❌ | ❌ | Partiel |
| DebateService | ✅ | ✅ | N/A | ❌ | Complet |
| SynthesisService | ✅ | ✅ | N/A | ❌ | Complet |
| RagEngineService | ✅ | ✅ | ❌ | ❌ | Partiel |
| KagEngineService | ✅ | ✅ | ❌ | ❌ | Partiel |
| HouseModelService | ✅ | ✅ | N/A | ❌ | Complet |
| ResilienceService | ✅ | ❌ | ✅ | ❌ | Partiel |
| AnomalyDetectionService | ✅ | ✅ | N/A | ❌ | Complet |
| ApiProviderFactory | ✅ | ✅ | ✅ | ❌ | Complet |
| ModelTrainingService | ✅ | ✅ | N/A | ❌ | Complet |
| ModelEvaluationService | ✅ | ✅ | N/A | ❌ | Complet |
| ModelUtilsService | N/A | N/A | N/A | ❌ | Complet |

## Priorités d'intégration

1. **Priorité Haute** - Intégrations cruciales pour le fonctionnement cohérent
   - **ModelTrainingService** avec EventBus et KnowledgeGraph
   - **ModelEvaluationService** avec EventBus et KnowledgeGraph
   - **HouseModelService** pour stocker les exemples d'apprentissage dans KnowledgeGraph

2. **Priorité Moyenne** - Améliorations importantes
   - **PoolManagerService** avec EventBus et notification des déploiements
   - **ResilienceService** avec EventBus pour alerter des ouvertures/fermetures
   - **ApiProviderFactory** avec KnowledgeGraph pour historique de performance

3. **Priorité Basse** - Finitions et améliorations
   - **Suivi centralisé des métriques** via un nouveau service dédié
   - **Tests unitaires** pour tous les composants
   - **Documentation générée** à partir du code source

## Timeline d'implémentation

### Phase 1 - Intégrations essentielles (Priorité Haute)
- ✅ Intégrer AnomalyDetectionService avec KnowledgeGraph
- ✅ Intégrer SynthesisService avec EventBus et KnowledgeGraph
- ✅ Intégrer ModelTrainingService avec EventBus et KnowledgeGraph
- ✅ Intégrer ModelEvaluationService avec EventBus et KnowledgeGraph
- ✅ Améliorer HouseModelService pour utiliser KnowledgeGraph pour les exemples

### Phase 2 - Amélioration de la résilience (Priorité Moyenne)
- ✅ Intégrer PoolManagerService avec EventBus
- ✅ Enrichir RouterService avec KnowledgeGraph pour apprentissage des routages
- ✅ Intégrer ResilienceService avec EventBus pour les notifications
- ✅ Améliorer ApiProviderFactory pour métriques via KnowledgeGraph

### Phase 3 - Monitoring et qualité (Priorité Basse)
- ✅ Créer MetricsService pour collecter et stocker les métriques
- ✅ Implémenter un mécanisme de tests automatiques
- ✅ Ajouter un système d'alertes basé sur des seuils de performance
- ⬜ Créer une interface de visualisation des données du graphe

## Défis techniques identifiés

1. **Dépendances circulaires**
   - Plusieurs services ont des références mutuelles nécessitant forwardRef()
   - Surveiller d'éventuels problèmes de cycle de vie d'initialisation

2. **Performance du graphe de connaissances**
   - Surveiller la croissance du graphe et implémenter une stratégie de nettoyage
   - Optimiser les requêtes fréquentes

3. **Débordement de mémoire avec les exemples d'apprentissage**
   - Implémenter un mécanisme de persistence pour éviter le stockage en mémoire

4. **Gestion cohérente des erreurs**
   - Standardiser le format des erreurs émises via EventBus

## Progression actuelle

### Terminé
- Intégration de AnomalyDetectionService avec KnowledgeGraph
- Intégration de SynthesisService avec EventBus et KnowledgeGraph
- Intégration de ModelTrainingService avec EventBus et KnowledgeGraph
- Intégration de ModelEvaluationService avec EventBus et KnowledgeGraph
- Amélioration du HouseModelService pour utiliser KnowledgeGraph pour les exemples d'apprentissage
- Intégration de PoolManagerService avec EventBus
- Enrichissement de RouterService avec KnowledgeGraph pour l'apprentissage des routages
- Intégration de ResilienceService avec EventBus pour les notifications
- Amélioration de ApiProviderFactory pour les métriques via KnowledgeGraph
- Création de MetricsService pour la collecte et le stockage centralisés des métriques

### Phase 2 - Amélioration de la résilience
✅ Tous les composants de la Phase 2 ont été intégrés avec succès!

### Prochaines tâches (Phase 3)
- ✅ Créer MetricsService pour collecter et stocker les métriques
- ✅ Implémenter un mécanisme de tests automatiques
- ✅ Ajouter un système d'alertes basé sur des seuils de performance
- ⬜ Créer une interface de visualisation des données du graphe 

## Audit d'architecture

Un audit complet de l'architecture et de la qualité du code est en cours, en suivant l'arborescence complète du projet :

```
.
├── [ ] ROADMAP.md
├── [ ] agent_preprompts.md
├── [ ] architecture_systeme.md
├── [ ] .cursor
│   └── [ ] rules
├── [ ] debate_synthesis_prompts.md
├── [ ] dist
│   ├── [ ] app.module.d.ts
│   ├── [ ] app.module.js
│   ├── [ ] app.module.js.map
│   ├── [ ] components
│   ├── [ ] config
│   ├── [ ] core
│   ├── [ ] debate
│   ├── [ ] examples
│   ├── [ ] handlers
│   ├── [ ] index.d.ts
│   ├── [ ] index.js
│   ├── [ ] index.js.map
│   ├── [ ] main.d.ts
│   ├── [ ] main.js
│   ├── [ ] main.js.map
│   ├── [ ] orchestrator
│   ├── [ ] prompts
│   ├── [ ] rag-kag
│   │   ├── [ ] agents
│   │   │   ├── [ ] agent-factory.service.ts
│   │   │   └── [ ] agents.module.ts
│   │   ├── [ ] apis
│   │   │   ├── [ ] api-provider-factory.service.ts
│   │   │   ├── [ ] apis.module.ts
│   │   │   ├── [ ] deepseek-ai.service.ts
│   │   │   ├── [ ] google-ai.service.ts
│   │   │   ├── [ ] house-model.service.ts
│   │   │   ├── [ ] model-evaluation.service.ts
│   │   │   ├── [ ] model-training.service.ts
│   │   │   ├── [ ] model-utils.service.ts
│   │   │   ├── [ ] qwen-ai.service.ts
│   │   │   └── [ ] tokenizer.service.ts
│   │   ├── [ ] controllers
│   │   │   └── [ ] rag-kag.controller.ts
│   │   ├── [ ] core
│   │   │   ├── [ ] core.module.ts
│   │   │   ├── [ ] event-bus.service.ts
│   │   │   ├── [ ] knowledge-graph.service.ts
│   │   │   ├── [ ] metrics.service.ts
│   │   │   └── [ ] query-analyzer.service.ts
│   │   ├── [ ] debate
│   │   │   ├── [ ] debate.module.ts
│   │   │   ├── [ ] debate.service.ts
│   │   │   ├── [ ] kag-engine.service.ts
│   │   │   └── [ ] rag-engine.service.ts
│   │   ├── [ ] orchestrator
│   │   │   ├── [ ] orchestrator.module.ts
│   │   │   ├── [ ] orchestrator.service.ts
│   │   │   ├── [ ] output-collector.service.ts
│   │   │   └── [ ] router.service.ts
│   │   ├── [ ] pools
│   │   │   ├── [ ] commercial-pool.service.ts
│   │   │   ├── [ ] educational-pool.service.ts
│   │   │   ├── [ ] marketing-pool.service.ts
│   │   │   ├── [ ] pool-manager.service.ts
│   │   │   ├── [ ] pools.module.ts
│   │   │   └── [ ] sectorial-pool.service.ts
│   │   ├── [ ] prompts
│   │   │   ├── [ ] prompts.module.ts
│   │   │   └── [ ] prompts.service.ts
│   │   ├── [ ] rag-kag.module.ts
│   │   ├── [ ] synthesis
│   │   │   ├── [ ] synthesis.module.ts
│   │   │   └── [ ] synthesis.service.ts
│   │   ├── [ ] testing
│   │   │   ├── [ ] auto-test.service.ts
│   │   │   └── [ ] fixtures
│   │   ├── [ ] types
│   │   │   ├── [ ] index.ts
│   │   │   └── [ ] types.module.ts
│   │   └── [ ] utils
│   │       ├── [ ] logger.module.ts
│   │       ├── [ ] logger-tokens.ts
│   │       └── [ ] resilience.service.ts
│   ├── [ ] types
│   └── [ ] utils
├── [ ] .eslintrc.js
├── [ ] evaluations
├── [ ] .gitignore
├── [ ] models
│   └── [ ] vocab
│       ├── [ ] deepseek-r1_vocab.json
│       ├── [ ] llama-3-8b_vocab.json
│       ├── [ ] mistral-7b-fr_vocab.json
│       └── [ ] phi-3-mini_vocab.json
├── [ ] nest-cli.json
├── [ ] optimisations_prompts.md
├── [ ] package.json
├── [ ] .prettierrc
├── [ ] README.md
├── src
│   ├── [ ] app.module.ts
│   ├── [ ] components
│   │   ├── [ ] impl
│   │   │   ├── [ ] kag-engine.ts
│   │   │   ├── [ ] query-analyzer.ts
│   │   │   └── [ ] rag-engine.ts
│   │   └── [ ] registry.ts
│   ├── [ ] config
│   │   └── [ ] poolConfig.ts
│   ├── [ ] coordination-architecture.md
│   ├── [ ] core
│   │   ├── [ ] circuit-breaker.ts
│   │   ├── [ ] coordination-system.ts
│   │   ├── [ ] data-partitioning.ts
│   │   ├── [ ] event-bus.ts
│   │   └── [ ] knowledge-graph.ts
│   ├── [ ] debate
│   │   ├── [ ] debateProtocol.ts
│   │   ├── [ ] kagEngine.ts
│   │   └── [ ] ragEngine.ts
│   ├── [ ] examples
│   │   └── [ ] components-usage.ts
│   ├── [ ] handlers
│   │   └── [ ] coordination-handler.ts
│   ├── [ ] index.ts
│   ├── [ ] main.ts
│   ├── [ ] orchestrator
│   │   ├── [ ] index.ts
│   │   ├── [ ] outputCollector.ts
│   │   ├── [ ] poolManager.ts
│   │   └── [ ] router.ts
│   ├── [ ] prompts
│   │   ├── [ ] base-prompts
│   │   │   ├── [ ] commercial.ts
│   │   │   ├── [ ] educational.ts
│   │   │   ├── [ ] marketing.ts
│   │   │   └── [ ] sectoriel.ts
│   │   ├── [ ] debate-prompts
│   │   │   └── [ ] kag-rag-debate.ts
│   │   └── [ ] meta-prompts
│   │       ├── [ ] anomaly.ts
│   │       ├── [ ] coordination.ts
│   │       ├── [ ] handler.ts
│   │       ├── [ ] orchestrator.ts
│   │       └── [ ] synthesis.ts
│   ├── [ ] types
│   │   └── [ ] index.ts
│   ├── [ ] utils
│   │   ├── [ ] anomaly-detector.module.ts
│   │   ├── [ ] anomalyDetector.ts
│   │   ├── [ ] circuit-breaker.ts
│   │   └── [ ] logger.ts
│   ├── [ ] synthesis
│   │   ├── [ ] contradictionResolver.ts
│   │   ├── [ ] merger.ts
│   │   └── [ ] responseFormatter.ts
│   ├── [ ] app.module.ts
│   └── [ ] main.ts
├── [ ] tsconfig.build.json
├── [ ] tsconfig.json
├── [ ] tsconfig.json.backup
├── [ ] yarn.lock
└── test
    ├── [ ] app.e2e-spec.ts
    └── [ ] jest-e2e.json