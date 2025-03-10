# Système RAG/KAG avec Vérification de Connaissances

Un système avancé combinant Retrieval Augmented Generation (RAG) et Knowledge Augmented Generation (KAG) avec un mécanisme sophistiqué de vérification et validation des connaissances.

## Caractéristiques principales

- **Architecture hybride NestJS/Legacy**: Structure modulaire combinant le meilleur des deux mondes
- **Multi-agent spécialisé**: Pools d'agents dédiés (Commercial, Marketing, Sectoriel, Éducatif)
- **Système de débat dialectique**: Confrontation RAG-KAG pour produire des réponses optimales
- **Vérification avancée des connaissances**: Mécanisme multi-niveau pour garantir la fiabilité
- **Circuit breakers et résilience**: Protection contre les défaillances des services externes
- **Adaptation intelligente**: Orchestration dynamique selon la complexité des requêtes

## Architecture du système

```
┌─────────────────────────────────────────────────────────────┐
│                  NestJS Application                          │
└───────────────────────────┬─────────────────────────────────┘
                           │
┌───────────────────────────▼─────────────────────────────────┐
│                        Orchestrateur                         │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐    │
│  │ Router      │────▶│ Pool Manager│────▶│ Output      │    │
│  │             │     │             │     │ Collector   │    │
│  └─────────────┘     └─────────────┘     └─────────────┘    │
└───────────────────────────┬─────────────────────────────────┘
                           │
           ┌───────────────┼───────────────┐
           │               │               │
┌──────────▼───────┐┌──────▼───────┐┌──────▼──────────┐
│                  ││               ││                │
│  Agent Pools     ││ Debate System ││  Synthesis     │
│  ┌────────────┐  ││ ┌───────────┐││                │
│  │Commercial  │  ││ │KAG Engine │││                │
│  └────────────┘  ││ └───────────┘││                │
│  ┌────────────┐  ││ ┌───────────┐││                │
│  │Marketing   │──┼┼─▶│Debate     │──┼┼─▶           │
│  └────────────┘  ││ │Protocol   │││                │
│  ┌────────────┐  ││ └───────────┘││                │
│  │Sectorial   │  ││ ┌───────────┐││                │
│  └────────────┘  ││ │RAG Engine │││                │
│  ┌────────────┐  ││ └───────────┘││                │
│  │Educational │  ││               ││                │
│  └────────────┘  ││               ││                │
└──────────────────┘└───────────────┘└────────────────┘
```

### Nouveau système de vérification des connaissances

Le système utilise une approche multi-niveaux pour garantir la fiabilité des connaissances ajoutées au graphe:

```
┌──────────────────────┐     ┌──────────────────────┐     ┌──────────────────────┐
│                      │     │                      │     │                      │
│  Vérification API    │────▶│  Débat interne       │────▶│ Vérification graphe  │
│  (Consensus externe) │     │   (Pilpoul)          │     │  (Cohérence interne) │
│                      │     │                      │     │                      │
└──────────────────────┘     └──────────────────────┘     └──────────────────────┘
           │                           │                           │
           └───────────────┬───────────┴───────────────┬───────────┘
                           ▼                           ▼
           ┌──────────────────────────┐     ┌──────────────────────┐
           │                          │     │                      │
           │   Connaissance vérifiée  │     │     Quarantaine      │
           │    (Ajoutée au graphe)   │     │ (Attente vérification│
           │                          │     │     ultérieure)      │
           └──────────────────────────┘     └──────────────────────┘
```

## Installation

```bash
# Installation des dépendances
$ yarn install

# Développement
$ yarn start:dev

# Production
$ yarn start:prod
```

## Dépendances principales

- NestJS - Framework backend
- TensorFlow.js - Modèles locaux 
- ONNX Runtime - Exécution de modèles optimisés
- UUID - Génération d'identifiants uniques

## Structure des dossiers

```
src/
├── rag-kag/              # Implémentation NestJS
│   ├── agents/           # Gestion des agents
│   ├── apis/             # Intégration APIs (Google, Qwen, Deepseek)
│   ├── common/           # Services partagés
│   ├── controllers/      # Controllers REST API
│   ├── core/             # Services fondamentaux
│   │   ├── event-bus.service.ts
│   │   ├── knowledge-graph.service.ts
│   │   └── knowledge-verifier.service.ts  # Nouveau système de vérification
│   ├── debate/           # Moteur de débat
│   ├── orchestrator/     # Coordination des flux
│   ├── pools/            # Pools d'agents spécialisés
│   ├── synthesis/        # Génération de réponses finales
│   ├── testing/          # Outils de test automatisés
│   ├── types/            # Définitions de types
│   └── utils/            # Utilitaires divers
└── legacy/               # Composants de l'architecture legacy
    ├── config/           # Configurations
    ├── prompts/          # Templates de prompts
    ├── types/            # Types legacy
    └── utils/            # Utilitaires legacy
```

## Utilisation du système de vérification des connaissances

### Ajouter une connaissance avec vérification

```typescript
// Exemple d'ajout d'une connaissance avec vérification automatique
const nodeId = await knowledgeGraph.addNodeWithVerification({
  label: 'Fait historique',
  type: 'HISTORY',
  content: 'La Tour Eiffel a été construite en 1889',
  confidence: 0.8,
  source: KnowledgeSource.USER_INPUT
});

// Vérifier si l'ajout a réussi (null si placé en quarantaine)
if (nodeId) {
  console.log(`Connaissance vérifiée et ajoutée: ${nodeId}`);
} else {
  console.log('Connaissance mise en quarantaine pour vérification ultérieure');
}
```

### Vérification adaptative d'une affirmation

```typescript
// Vérification adaptative avec cache et contrainte de temps
const result = await knowledgeVerifier.verifyClaimAdaptive({
  claim: 'Paris est la capitale de la France',
  domain: 'Géographie',
  source: KnowledgeSource.USER_INPUT,
  confidence: 0.9
}, {
  useCache: true,
  timeConstraint: 5000, // 5 secondes max
  cacheSimilarityThreshold: 0.8
});

console.log(`Vérification: ${result.isVerified ? 'VALIDÉE' : 'REJETÉE'}`);
console.log(`Score: ${result.confidenceScore.toFixed(2)}`);
console.log(`Méthodes utilisées: ${result.methods.join(', ')}`);
```

## Configurations disponibles

### Niveaux de vérification

- **STRICT**: Vérification rigoureuse pour les informations critiques
  - Plusieurs APIs consultées
  - Débat interne complet (5 perspectives)
  - Vérification contre le graphe existant
  - Seuil de confiance élevé (0.85)

- **STANDARD**: Vérification équilibrée pour usage courant
  - 2 APIs consultées
  - Débat interne (3 perspectives)
  - Seuil de confiance moyen (0.7)

- **RELAXED**: Vérification légère pour informations non critiques
  - Modèle local uniquement
  - Débat interne simplifié (2 perspectives)
  - Seuil de confiance bas (0.6)

## Performances et optimisations

Le système de vérification adaptatif offre:

- **~80% moins d'appels API** pour les affirmations simples
- **~60% de réduction de latence** grâce au cache intelligent
- **Prioritisation automatique** des méthodes selon le contexte

## Roadmap

### Phase 1 (T1) - Haute priorité ✅
- Boucle de rétroaction d'erreurs - **COMPLÉTÉ**
- Latence excessive - **COMPLÉTÉ**
- Dépendances circulaires

### Phase 2 (T2) - Priorité moyenne 🔄
- Croissance non contrôlée du graphe
- Consommation mémoire des modèles
- Propagation d'hallucinations

### Phase 3 (T3) - Priorité basse ⏱️
- Coût computationnel
- Migration complète vers NestJS

## Licence

Ce projet est soumis à la licence MIT.
