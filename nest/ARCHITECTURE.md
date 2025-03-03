# Architecture du Système RAG/KAG Hybride

## Vue d'ensemble

Le système RAG/KAG est une architecture hybride combinant des éléments de l'ancienne architecture (legacy) avec une nouvelle implémentation NestJS. Cette documentation détaille l'intégration des deux architectures, les problématiques technologiques et les recommandations d'utilisation.

```
┌─────────────────────────────────────────────────────────────┐
│                  Application Client                          │
└───────────────────────────┬─────────────────────────────────┘
                           │
┌───────────────────────────▼─────────────────────────────────┐
│                        API Gateway                           │
└───────────────────────────┬─────────────────────────────────┘
                           │
┌───────────────────────────▼─────────────────────────────────┐
│                   Orchestrateur Hybride                      │
│                                                              │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐    │
│  │ Routeur     │────▶│ Pool Manager│────▶│ Output      │    │
│  │ Standard    │     │             │     │ Collector   │    │
│  └─────────────┘     └─────────────┘     └─────────────┘    │
│         │                                                    │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐    │
│  │ Legacy      │     │ Advanced    │     │ Legacy      │    │
│  │ Router      │────▶│ Coordination│─────▶ Adapter     │    │
│  └─────────────┘     └─────────────┘     └─────────────┘    │
└───────────────────────────┬─────────────────────────────────┘
                           │
           ┌───────────────┼───────────────┐
           │               │               │
┌──────────▼───────┐┌──────▼───────┐┌──────▼──────────┐
│                  ││               ││                 │
│  Agent Pools     ││ Debate System ││  Synthesis      │
│  ┌────────────┐  ││ ┌───────────┐││                 │
│  │Commercial  │  ││ │KAG Engine │││                 │
│  └────────────┘  ││ └───────────┘││                 │
│  ┌────────────┐  ││ ┌───────────┐││                 │
│  │Marketing   │──┼┼─▶│Debate     │──┼┼─▶             │
│  └────────────┘  ││ │Protocol   │││                 │
│  ┌────────────┐  ││ └───────────┘││                 │
│  │Sectorial   │  ││ ┌───────────┐││                 │
│  └────────────┘  ││ │RAG Engine │││                 │
│  ┌────────────┐  ││ └───────────┘││                 │
│  │Educational │  ││               ││                 │
│  └────────────┘  ││               ││                 │
└──────────────────┘└───────────────┘└─────────────────┘
```

## Composants principaux

### 1. Architecture NestJS (Nouvelle)
- `src/rag-kag/` - Implémentation NestJS moderne
- Basé sur l'injection de dépendances
- Services modulaires et testables
- Utilise le système d'événements

### 2. Architecture Legacy (Ancienne)
- `src/legacy/` - Composants de l'ancienne architecture
- Système de prompts sophistiqué
- Logique de coordination avancée
- Détection d'anomalies spécialisée

### 3. Points d'intégration
- Orchestrateur hybride
- Adaptateurs pour le système de détection d'anomalies
- Intégration des prompts legacy
- Système de coordination avancée

## Fonctionnalités intégrées

### 1. Détection d'anomalies avancée

L'intégration permet d'utiliser les détecteurs d'anomalies spécialisés de l'ancienne architecture avec le service moderne.

```typescript
// Utilisation de la détection d'anomalies avancée
const report = await anomalyDetectionService.detectAnomalies(poolOutputs, {
  useAdvancedDetection: true // Activer les détecteurs legacy
});
```

Les détecteurs spécialisés disponibles sont :
- Détection de biais cognitifs
- Analyse des erreurs méthodologiques
- Vérification des problèmes statistiques
- Validation des citations

### 2. Prompts de débat avancés

Le système intègre les prompts dialectiques sophistiqués de l'ancienne architecture.

```typescript
// Utilisation des prompts de débat avancés
const debate = await debateService.generateDebate(query, {
  useLegacyPrompt: true // Activer les prompts dialectiques avancés
});
```

### 3. Coordination avancée

Le système intègre le CoordinationHandler legacy qui offre des stratégies d'exécution sophistiquées.

```typescript
// Utilisation de la coordination avancée
const response = await orchestratorService.processQuery(query, 'INTERMEDIATE', {
  useAdvancedCoordination: true, // Activer la coordination legacy
  executionMode: 'adaptive' // Mode d'exécution : 'sequential', 'parallel', 'adaptive'
});
```

### Modes d'exécution disponibles :

- **Sequential** : Exécution en série des composants, économe en ressources mais plus lente.
- **Parallel** : Exécution en parallèle, plus rapide mais consomme plus de ressources.
- **Adaptive** : Mode intelligent qui ajuste la stratégie en fonction de la complexité de la requête et de la charge du système.

## Problématiques technologiques à implémenter

### 1. Circuit Breakers pour APIs externes

**Problème** : Les appels aux APIs externes (Google AI, Qwen, Deepseek) peuvent échouer ou présenter des latences élevées.

**Solution à implémenter** :
- Intégrer pleinement le circuit breaker legacy à tous les points d'appel API
- Définir des stratégies de fallback pour chaque service
- Configurer des seuils de timeout adaptés à chaque fournisseur

```typescript
// Exemple d'implémentation recommandée
async callExternalApi(provider: string, request: any): Promise<any> {
  return this.resilienceService.executeWithCircuitBreaker(
    provider,
    async () => await this.apiProviderFactory.generateResponse(provider, request),
    async (error) => this.handleFailover(error, provider, request)
  );
}
```

### 2. Gestion de la mémoire pour les modèles locaux

**Problème** : Les modèles locaux peuvent occuper beaucoup de mémoire GPU/RAM, particulièrement lors d'exécutions en parallèle.

**Solution à implémenter** :
- Système de pool de threads/processus pour les modèles
- Limitation du nombre d'instances parallèles
- Déchargement dynamique des modèles inutilisés

```typescript
// Exemple d'architecture proposée
class ModelPoolManager {
  private modelPools: Map<string, ModelWorkerPool>;
  
  async getWorker(modelName: string): Promise<ModelWorker> {
    // Logique d'allocation de worker
  }
  
  releaseWorker(worker: ModelWorker): void {
    // Logique de libération
  }
}
```

### 3. Synchronisation des interfaces de type

**Problème** : Les types entre les architectures legacy et moderne peuvent diverger, causant des erreurs subtiles.

**Solution à implémenter** :
- Créer un système de validation de type runtime
- Mapper automatiquement les types entre les deux architectures
- Générer des rapports de compatibilité

```typescript
// Exemple d'adaptateur de type
function adaptLegacyToModern<T, U>(legacyData: T, schema: Schema<U>): U {
  const validated = schema.validate(legacyData);
  if (!validated.success) {
    throw new TypeError(`Invalid data structure: ${validated.error}`);
  }
  return validated.data;
}
```

### 4. Mécanisme de mise en cache intelligent

**Problème** : Les requêtes similaires génèrent des appels redondants aux modèles d'IA.

**Solution à implémenter** :
- Cache sémantique (pas seulement basé sur une correspondance exacte)
- Stratégies d'invalidation basées sur le temps et la pertinence
- Cache hiérarchique (mémoire, disque, distribué)

```typescript
// Exemple de mise en œuvre
class SemanticCache {
  async get(query: string, threshold: number = 0.92): Promise<CachedResult | null> {
    // Recherche sémantique
  }
  
  async set(query: string, result: any): Promise<void> {
    // Stockage avec embedding
  }
}
```

## Recommandations d'utilisation

### 1. Choix de la stratégie de coordination

| Contexte | Stratégie recommandée |
|----------|----------------------|
| Requêtes simples/courtes | `sequential` |
| Requêtes complexes nécessitant plusieurs perspectives | `parallel` |
| Production avec charge variable | `adaptive` |
| Phase de développement/test | `sequential` |

### 2. Migration progressive

- Commencer par activer les fonctionnalités legacy uniquement pour certains types de requêtes
- Établir des métriques comparatives entre les deux approches
- Migrer progressivement vers les implémentations modernes après validation

### 3. Monitoring et performances

- Implémenter des métriques distinctes pour les composants legacy et modernes
- Suivre les temps de réponse, la consommation mémoire et l'utilisation CPU
- Mettre en place des alertes spécifiques pour les échecs de composants legacy

## Exemples de cas d'utilisation

### 1. Requête commerciale complexe

```typescript
// Requête complexe d'analyse de marché
const marketAnalysisQuery = "Analysez l'impact des stratégies omnicanal sur la fidélisation client dans le secteur du luxe";

const response = await orchestratorService.processQuery(marketAnalysisQuery, 'ADVANCED', {
  useAdvancedCoordination: true,
  executionMode: 'parallel',
  prioritizeSpeed: false // Privilégier la qualité sur la vitesse
});
```

### 2. Requête éducative avec détection d'anomalies

```typescript
// Requête éducative nécessitant une vérification factuelle rigoureuse
const educationalQuery = "Expliquez les conséquences du réchauffement climatique sur les écosystèmes marins";

// Utilisation de la coordination standard avec détection d'anomalies avancée
const poolOutputs = await poolManagerService.executeAgents(targetPools, adaptedQuery);
const anomalyReport = await anomalyDetectionService.detectAnomalies(poolOutputs, {
  useAdvancedDetection: true,
  detectionLevel: AnomalyDetectionLevel.ALL,
  throwOnHigh: true // Arrêter le traitement en cas d'anomalie critique
});
```

### 3. Traitement batch optimisé

```typescript
// Traitement d'un lot de requêtes similaires
const batchQueries = ["Avantages du cloud computing", "Inconvénients du cloud computing", ...];

// Utiliser la coordination avancée en mode séquentiel pour économiser les ressources
for (const query of batchQueries) {
  await orchestratorService.processQuery(query, 'INTERMEDIATE', {
    useAdvancedCoordination: true,
    executionMode: 'sequential'
  });
}
```

## Plan de migration à long terme

### Phase 1: Stabilisation (actuelle)
- Intégration des composants legacy critiques
- Documentation des interfaces
- Définition des métriques de comparaison

### Phase 2: Évaluation (1-2 mois)
- Analyse comparative des performances
- Identification des goulots d'étranglement
- Sélection des composants à moderniser en priorité

### Phase 3: Modernisation (2-6 mois)
- Réimplémentation des composants legacy prioritaires
- Tests A/B entre versions legacy et moderne
- Migration progressive par fonctionnalité

### Phase 4: Consolidation (6+ mois)
- Suppression graduelle des dépendances legacy
- Unification de l'architecture
- Documentation finale et formation

## Conclusion

L'architecture hybride RAG/KAG offre le meilleur des deux mondes : les fonctionnalités avancées de l'architecture legacy et la modularité de l'architecture NestJS moderne. Cette approche permet une migration progressive tout en maintenant la qualité des réponses et en améliorant la maintenabilité du code. 