# Système de Coordination RAG/KAG

Un système de coordination robuste conçu pour orchestrer l'interaction entre les composants d'une architecture mixte RAG (Retrieval Augmented Generation) et KAG (Knowledge Augmented Generation).

## Vue d'ensemble

Le système de coordination constitue le cœur du système RAG/KAG, organisant le flux de traitement entre les différents composants spécialisés comme les agents commerciaux, les agents marketing, les agents sectoriels, les moteurs de débat KAG/RAG, et les détecteurs d'anomalies.

Cette architecture modulaire permet:
- Une grande adaptabilité aux différents types de requêtes
- Une exécution parallèle ou séquentielle optimisée selon le contexte
- Une détection fiable des anomalies dans les réponses
- Une gestion efficace des dépendances entre composants
- Une journalisation complète pour le suivi et le débogage

## Architecture

L'architecture du système est composée de plusieurs couches:

### 1. Cœur du Système

- **CoordinationSystem**: Interface principale pour les applications externes
- **ComponentRegistry**: Gestion centralisée des composants disponibles
- **CoordinationHandler**: Exécution des workflows de traitement

### 2. Composants

Différents types de composants peuvent être enregistrés:

- Analyseurs de requêtes
- Sélecteurs de pools d'agents
- Agents spécialisés (commercial, marketing, sectoriel)
- Moteurs RAG/KAG
- Détecteurs d'anomalies
- Formateurs de sortie

### 3. Infrastructure

- **Logger**: Système de journalisation unifié
- **Métriques**: Collecte et analyse des performances
- **Circuit Breakers**: Protection contre les cascades d'échecs

## Flux d'Exécution

1. **Réception de la requête**: Analyse initiale et planification
2. **Sélection des composants**: Détermination des composants nécessaires
3. **Exécution orchestrée**: Activation des composants dans l'ordre optimal
4. **Intégration des résultats**: Fusion des sorties des différents composants
5. **Détection d'anomalies**: Vérification de la cohérence des résultats
6. **Finalisation**: Formatage de la réponse finale

## Modes d'Exécution

Le système supporte trois modes d'exécution:

- **SEQUENTIAL**: Exécution séquentielle pour les requêtes simples ou les systèmes chargés
- **PARALLEL**: Exécution parallèle pour les requêtes complexes nécessitant plusieurs composants
- **ADAPTIVE**: Sélection automatique du mode optimal selon la complexité et la charge

## Installation

```bash
npm install
```

## Utilisation

```typescript
import { CoordinationSystem } from './core/coordination-system';
import { ExecutionMode } from './handlers/coordination-handler';

// Créer une instance du système
const system = new CoordinationSystem();

// Traiter une requête
const result = await system.processQuery(
  "Quelle est la meilleure stratégie marketing pour une entreprise de luxe?",
  { executionMode: ExecutionMode.ADAPTIVE }
);

console.log(result);
```

## Exemple de Résultat

```json
{
  "success": true,
  "traceId": "f8a7b6c5-d4e3-2c1b-a0f9-e8d7c6b5a4f3",
  "result": {
    "query": "Quelle est la meilleure stratégie marketing pour une entreprise de luxe?",
    "processedBy": [
      { "id": "query-analyzer", "name": "Analyseur de Requête" },
      { "id": "pool-selector", "name": "Sélecteur de Pools" },
      { "id": "marketing-agent", "name": "Agent Marketing" },
      { "id": "output-formatter", "name": "Formateur de Sortie" }
    ],
    "data": {
      "marketing-agent": {
        "recommendation": "Pour une entreprise de luxe, la stratégie marketing optimale repose sur l'exclusivité, l'expérience client personnalisée et la narration de marque authentique...",
        "confidence": 0.87
      }
    },
    "metadata": {
      "executionTime": 1250,
      "traceId": "f8a7b6c5-d4e3-2c1b-a0f9-e8d7c6b5a4f3"
    }
  },
  "duration": 1250,
  "debugInfo": {
    "executionPath": ["query-analyzer", "pool-selector", "marketing-agent", "output-formatter"],
    "metrics": {
      "totalDuration": 1250,
      "bottlenecks": ["marketing-agent"],
      "optimizationSuggestions": ["Considérer le mode parallèle pour réduire la latence totale"]
    }
  }
}
```

## Personnalisation

Le système est conçu pour être hautement personnalisable:

```typescript
// Enregistrer un nouveau composant
system.registerComponent({
  type: ComponentType.MARKETING_AGENT,
  name: "Agent Marketing Spécialisé Luxe",
  description: "Agent spécialisé dans le marketing de produits de luxe",
  version: "1.0.0",
  priority: ComponentPriority.HIGH,
  executeFunction: async (context) => {
    // Logique d'exécution personnalisée
    return {
      recommendation: "Stratégie de marketing pour le secteur du luxe...",
      confidence: 0.9
    };
  },
  isEnabled: true
});
```

## Développement

```bash
# Compilation TypeScript
npm run build

# Exécution
npm start

# Développement avec rechargement automatique
npm run dev

# Tests
npm test

# Lint et formatage
npm run lint
npm run format
```

## Licence

MIT 