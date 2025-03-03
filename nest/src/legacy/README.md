# Code Legacy

Ce dossier contient des éléments de l'ancienne architecture qui ont été préservés mais isolés de la nouvelle structure.

## Structure du dossier

- `config/` : Configurations des pools d'agents et autres paramètres système
- `prompts/` : Templates de prompts sophistiqués organisés par type
  - `base-prompts/` : Prompts de base pour les différents types d'agents
  - `debate-prompts/` : Prompts spécialisés pour le débat KAG/RAG
  - `meta-prompts/` : Prompts de coordination et d'orchestration
- `types/` : Définitions de types pour l'architecture legacy
- `utils/` : Utilitaires et services de l'ancienne architecture
  - `circuit-breaker.ts` : Implémentation sophistiquée de circuit breaker
  - `anomalyDetector.ts` : Détection spécialisée d'anomalies

## Fonctionnalités préservées

### Système de coordination avancé

Le système de coordination avancé (`meta-prompts/handler.ts`) offre des stratégies d'exécution sophistiquées :

- **Mode séquentiel** : Exécution en série, optimisée pour la consommation de ressources
- **Mode parallèle** : Exécution simultanée, optimisée pour la vitesse
- **Mode adaptatif** : Ajustement dynamique basé sur la complexité et la charge

```typescript
// Utilisation via l'OrchestratorService
const response = await orchestratorService.processQuery(query, 'INTERMEDIATE', {
  useAdvancedCoordination: true,
  executionMode: 'adaptive' // 'sequential', 'parallel', 'adaptive'
});
```

### Détection d'anomalies spécialisée

Le détecteur d'anomalies (`anomalyDetector.ts`) comprend des analyseurs spécialisés :

- **Biais cognitifs** : Détecte les biais d'autorité, de confirmation, etc.
- **Failles méthodologiques** : Identifie les problèmes de raisonnement
- **Erreurs statistiques** : Vérifie l'usage correct des statistiques
- **Problèmes de citation** : Analyse les références et attributions

```typescript
// Utilisation via AnomalyDetectionService
const report = await anomalyDetectionService.detectAnomalies(poolOutputs, {
  useAdvancedDetection: true
});
```

### Prompts dialectiques sophistiqués

Les prompts de débat (`debate-prompts/kag-rag-debate.ts`) implémentent un système dialectique structuré :

- Format dialectique thèse-antithèse-synthèse
- Pondération des arguments par niveau de confiance
- Identification des contradictions et réconciliations
- Synthèse des positions convergentes

```typescript
// Utilisation via DebateService
const debate = await debateService.generateDebate(query, {
  useLegacyPrompt: true
});
```

## Correspondances avec la nouvelle architecture

| Composant Legacy | Composant NestJS | Description |
|------------------|------------------|-------------|
| `utils/circuit-breaker.ts` | `rag-kag/utils/resilience.service.ts` | Système de gestion de la résilience |
| `utils/anomalyDetector.ts` | `utils/anomaly-detection.service.ts` | Détection d'anomalies |
| `prompts/base-prompts/*` | `rag-kag/prompts/prompts.service.ts` | Gestion des templates de prompts |
| `config/poolConfig.ts` | `rag-kag/agents/agent-factory.service.ts` | Configuration des agents |
| `prompts/meta-prompts/handler.ts` | `rag-kag/orchestrator/orchestrator.service.ts` | Coordination d'exécution |
| `prompts/debate-prompts/*` | `rag-kag/debate/debate.service.ts` | Système de débat dialectique |

## Intégration et utilisation

Les composants legacy sont pleinement intégrés à l'architecture NestJS via des adaptateurs et peuvent être activés via des options dans les services correspondants. Pour plus d'informations sur l'utilisation, consultez :

- La documentation complète dans `ARCHITECTURE.md`
- Les exemples d'utilisation dans `src/rag-kag/*/examples`
- Les tests d'intégration dans `test/integration`

## Plan de migration long terme

1. Phase de stabilisation (actuelle)
   - Intégration des composants legacy critiques ✅
   - Documentation complète des interfaces ✅
   - Métriques de comparaison

2. Phase d'évaluation (1-2 mois)
   - Tests de performance comparatifs
   - Identification des priorités de modernisation

3. Phase de modernisation (2-6 mois)
   - Réimplémentation progressive
   - Tests A/B des nouvelles implémentations

4. Phase de consolidation (6+ mois)
   - Suppression graduelle des dépendances legacy
   - Migration complète vers l'architecture NestJS 