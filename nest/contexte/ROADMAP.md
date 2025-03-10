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

## Défis critiques à résoudre

Cette section identifie les problèmes critiques de l'architecture actuelle et propose un plan d'action pour les résoudre.

### Problèmes techniques prioritaires

| Problème | Impact | Complexité | Solution proposée | Échéance | Statut |
|----------|:------:|:----------:|-------------------|:--------:|:------:|
| **Boucle de rétroaction d'erreurs** | Élevé | Moyenne | Implémenter un système de vérification externe pour les connaissances avant stockage dans le graphe | T1 | À faire |
| **Latence excessive** | Élevé | Élevée | Créer un système de décision pour n'activer le débat RAG/KAG que pour les requêtes complexes | T1 | **Implémenté** |
| **Dépendances circulaires** | Moyen | Moyenne | Refactoriser les services avec pattern médiateur et interfaces claires | T1 | À faire |
| **Croissance non contrôlée du graphe** | Moyen | Moyenne | Implémenter un système de nettoyage et consolidation périodique du graphe | T2 | À faire |
| **Consommation mémoire des modèles** | Élevé | Élevée | Développer un gestionnaire de ressources avec déchargement dynamique des modèles | T2 | À faire |
| **Propagation d'hallucinations** | Élevé | Élevée | Ajouter un système de vérification factuelle externe pour les connaissances critiques | T2 | À faire |
| **Coût computationnel** | Moyen | Moyenne | Optimiser la sélection des agents et implémenter un système de cache sémantique | T3 | À faire |
| **Dette technique hybride** | Moyen | Élevée | Planifier la migration complète vers NestJS avec timeline stricte | T3 | À faire |

### Plan d'action détaillé

#### Phase 1 (T1) : Stabilisation et optimisation critique

1. **Système de vérification des connaissances**
   ```typescript
   // Exemple d'implémentation
   class KnowledgeVerifier {
     async verify(claim: string, confidence: number): Promise<VerificationResult> {
       // Vérification par sources multiples
       // Détection de contradictions avec connaissances existantes
       // Validation par règles logiques
     }
   }
   ```

2. **Optimisation de la latence** ✅
   - ✅ Implémenter un classificateur rapide de complexité des requêtes
   - ✅ Créer un pipeline adaptatif qui active seulement les composants nécessaires
   - ✅ Développer un système de cache intelligent pour les requêtes similaires

   **Implementation**: Le `ComplexityAnalyzerService` utilise Phi-3-mini et un algorithme K-means pour classifier rapidement les requêtes en trois niveaux de complexité (simple, standard, complexe), déterminant ainsi le pipeline approprié:
   - Requêtes simples: Traitement direct par un modèle local sans débat
   - Requêtes standard: Utilisation de RAG ou KAG (mais pas les deux)
   - Requêtes complexes: Pipeline complet avec débat RAG/KAG

3. **Résolution des dépendances circulaires**
   - Introduire un pattern médiateur central
   - Définir des interfaces claires pour chaque service
   - Utiliser des événements plutôt que des appels directs entre services

#### Phase 2 (T2) : Gestion des ressources et fiabilité

1. **Système de nettoyage du graphe de connaissances**
   - Algorithme de détection des nœuds obsolètes ou redondants
   - Consolidation périodique des connaissances similaires
   - Stratégie de rétention basée sur l'utilité et la fraîcheur

2. **Gestionnaire de ressources pour modèles**
   ```typescript
   class ModelResourceManager {
     private activeModels: Map<string, { model: any, lastUsed: number }> = new Map();
     
     async getModel(modelName: string): Promise<any> {
       // Logique de chargement/déchargement dynamique
       // Priorisation basée sur l'usage récent et la mémoire disponible
     }
   }
   ```

3. **Système de vérification factuelle**
   - Intégration avec des sources externes fiables
   - Mécanisme de consensus entre sources multiples
   - Marquage explicite du niveau de confiance des connaissances

#### Phase 3 (T3) : Optimisation et modernisation

1. **Optimisation computationnelle**
   - Profilage détaillé de la consommation de ressources
   - Parallélisation intelligente des tâches indépendantes
   - Implémentation d'un cache sémantique à plusieurs niveaux

2. **Plan de migration NestJS**
   - Cartographie complète des dépendances legacy
   - Réécriture progressive par domaine fonctionnel
   - Tests A/B systématiques entre anciennes et nouvelles implémentations

3. **Métriques et monitoring avancés**
   - Dashboard temps réel de performance
   - Alertes précoces sur anomalies de comportement
   - Traçabilité complète des décisions du système

### Métriques de succès

| Métrique | Valeur actuelle | Objectif T1 | Objectif T2 | Objectif T3 |
|----------|:--------------:|:-----------:|:-----------:|:-----------:|
| Temps de réponse moyen | ~3000ms | <2000ms | <1000ms | <500ms |
| Utilisation mémoire | ~4GB | <3GB | <2GB | <1.5GB |
| Taux d'hallucinations | ~5% | <3% | <1% | <0.5% |
| Dépendances circulaires | 12 | <8 | <4 | 0 |
| Code legacy utilisé | 60% | <50% | <30% | <10% |
| Coût par requête | ~$0.05 | <$0.04 | <$0.03 | <$0.02 |

### Risques et mitigations

| Risque | Probabilité | Impact | Stratégie de mitigation |
|--------|:-----------:|:------:|-------------------------|
| Complexité croissante pendant la transition | Élevée | Élevé | Freezer les fonctionnalités pendant la refactorisation |
| Régression de performance | Moyenne | Élevé | Tests de performance automatisés pour chaque PR |
| Perte de connaissances lors du nettoyage du graphe | Moyenne | Moyen | Système de sauvegarde et restauration granulaire |
| Échec de la migration complète | Élevée | Moyen | Définir des jalons intermédiaires fonctionnels |
| Dépassement des ressources matérielles | Moyenne | Élevé | Monitoring proactif et scaling horizontal |

Cette roadmap sera révisée trimestriellement pour ajuster les priorités en fonction des progrès réalisés et des nouveaux défis identifiés.

## Phase 4 (T4) - Migration vers l'appel direct des modèles Python 🆕
- Créer une API Python (Flask) pour exposer les modèles
  - Endpoints pour chaque fonctionnalité de modèle requise
  - Validation des entrées et gestion des erreurs
  - Tests unitaires pour l'API
- Mettre à jour les services NestJS pour appeler l'API Python
  - Remplacer les appels à TensorFlow.js par des requêtes HTTP
  - Définir des interfaces TypeScript pour les entrées/sorties de l'API
  - Gérer les erreurs et la validation des réponses
- Adapter les tests NestJS pour couvrir les appels à l'API Python
  - Tests unitaires pour les services modifiés
  - Tests d'intégration couvrant le flux complet NestJS -> API Python -> modèles
- Mettre à jour la configuration de déploiement
  - Déployer l'API Python aux côtés de l'application NestJS
  - Configurer la communication entre les deux (URL, ports, etc.)
  - Adapter les scripts de build et de démarrage
- Tester rigoureusement le nouveau workflow
  - Tests manuels couvrant divers scénarios
  - Surveiller les performances et la stabilité
  - Comparer les résultats avec l'implémentation TensorFlow.js
- Nettoyer le code legacy lié à TensorFlow.js
  - Supprimer les dépendances inutiles
  - Refactoriser pour améliorer la lisibilité et la maintenabilité
- Documenter le nouveau workflow
  - Mettre à jour la documentation d'architecture
  - Écrire des guides pour le développement et le déploiement
  - Ajouter des exemples de code illustrant les appels à l'API Python

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