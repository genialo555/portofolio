# Système de Coordination RAG/KAG

Un système avancé conçu pour orchestrer l'interaction entre les composants d'une architecture hybride RAG (Retrieval Augmented Generation) et KAG (Knowledge Augmented Generation), avec une optimisation spécifique pour les architectures Apple Silicon.

## Vue d'ensemble

Le système de coordination constitue le cœur de l'architecture RAG/KAG, organisant le flux de traitement entre différents composants spécialisés et optimisant l'allocation des ressources système. Cette plateforme complète comprend :

- **Backend NestJS** : Orchestre le flux de traitement des requêtes et les pools d'agents spécialisés
- **Services ML Python** : Gère l'inférence ML, les modèles et l'orchestration des ressources 
- **Frontend Next.js** : Fournit une interface utilisateur moderne et réactive

Cette architecture modulaire multi-niveaux permet :
- Une adaptation dynamique aux différents types de requêtes (informatives, analytiques, créatives)
- Une exécution optimisée selon les ressources disponibles et les priorités des requêtes
- Une gestion efficace de l'orchestration hybride RAG/KAG avec plusieurs stratégies d'exécution
- Une optimisation spécifique pour les architectures Apple Silicon (M-series)
- Une surveillance complète des performances et des ressources système

## Prérequis

- **Python** : 3.8+ (Python 3.10 recommandé)
- **Node.js** : 16.x+
- **Yarn** : 1.22+
- Pour **Apple Silicon** (M1/M2/M3) :
  - XCode Command Line Tools
  - Compilateur C++ avec support Metal

## Architecture Globale

L'architecture du système est organisée en trois composants principaux :

### 1. Backend NestJS (nest/)

Centre de coordination et d'orchestration principal avec :

- **Module RAG/KAG** : Implémentation NestJS du système RAG/KAG
  - **Orchestrateur** : Gestion du flux de traitement des requêtes
  - **Pools d'Agents** : Agents spécialisés par domaine (commercial, marketing, sectoriel)
  - **Système de Débat** : Organisation des débats entre RAG et KAG
  - **Synthèse** : Production des réponses finales

- **Services Fondamentaux**
  - **EventBusService** : Bus d'événements pour la communication asynchrone
  - **KnowledgeGraphService** : Gestion du graphe de connaissances
  - **ApiProviderFactory** : Abstraction pour les fournisseurs d'API LLM

### 2. Services ML Python (py-ml/)

Services d'inférence ML et d'orchestration des ressources :

- **Orchestration**
  - **ResourceOrchestrator** : Allocation des ressources système (CPU, RAM, GPU)
  - **ModelLifecycleManager** : Gestion du cycle de vie des modèles
  - **RequestCoordinator** : Files d'attente et prioritisation des requêtes
  - **HybridOrchestrator** : Coordination RAG/KAG avec stratégies adaptatives

- **Composants RAG/KAG**
  - **Retrievers** : Récupération de documents pertinents (RAG)
  - **KnowledgeBase** : Base de connaissances structurée (KAG)
  - **HybridGenerator** : Génération de réponses hybrides
  - **KnowledgeFusion** : Fusion des résultats RAG et KAG

- **Gestion des Modèles**
  - **ModelManager** : Chargement/déchargement des modèles
  - **ModelLoader** : Optimisation des modèles pour différentes architectures
  - **TeacherModel** : Modèle principal d'évaluation et de synthèse

### 3. Frontend Next.js (next/)

Interface utilisateur moderne et réactive :

- **Dashboard** : Vue d'ensemble et monitoring
- **Interface de Requêtes** : Soumission et suivi des requêtes
- **Visualisation** : Représentation des résultats et métadonnées
- **Administration** : Gestion des modèles et configuration

## Flux d'Exécution

Le traitement d'une requête suit le flux suivant :

1. **Réception et analyse** : Analyse initiale et détermination du type de requête
2. **Orchestration** : Sélection de la stratégie d'exécution optimale
   - Routage vers les pools d'agents appropriés
   - Allocation des ressources nécessaires
3. **Exécution** : Traitement par les composants RAG et/ou KAG selon la stratégie
   - RAG : Récupération et génération basée sur les documents
   - KAG : Génération basée sur les connaissances structurées
4. **Débat et synthèse** : Organisation d'un débat entre les approches et synthèse
5. **Finalisation** : Formatage de la réponse et enrichissement avec métadonnées

## Stratégies d'Exécution

Le système supporte six stratégies d'exécution via le `HybridOrchestrator` :

- **RAG_ONLY** : Utilise uniquement la Retrieval Augmented Generation
- **KAG_ONLY** : Utilise uniquement la Knowledge Augmented Generation
- **SEQUENTIAL** : Interroge RAG puis KAG séquentiellement
- **PARALLEL** : Interroge RAG et KAG en parallèle
- **ADAPTIVE** : Choisit dynamiquement la meilleure stratégie selon le type de requête
- **FUSION** : Fusionne les résultats de RAG et KAG pour une réponse intégrée

## Optimisations Spécifiques

### Apple Silicon (M-series)

Le système intègre des optimisations spécifiques pour les puces Apple :

- **Utilisation optimisée de MPS** (Metal Performance Shaders)
  - Détection automatique des capacités matérielles
  - Configuration optimale de `n_gpu_layers` pour Metal
  - Gestion adaptative de la mémoire GPU/CPU
- **Support spécifique pour llama-cpp-python avec Metal**
  - Compilation optimisée avec `-DLLAMA_METAL=on`
  - Déchargement KQV pour économiser la mémoire
  - Adaptation dynamique du batch size basée sur la mémoire disponible
- **Support pour les modèles quantifiés GGUF**
  - Chargement optimisé des modèles Q4_K_M, Q5_K_M et Q8_0
  - Inferfaces standardisées pour l'intégration dans l'API

### Gestion des Ressources

- Surveillance continue de l'utilisation CPU, RAM et GPU
- Déchargement automatique des modèles peu utilisés
- Prioritisation des requêtes critiques
- Paramètres adaptatifs basés sur la disponibilité des ressources :
  - `n_threads` optimisé selon le type de CPU
  - `n_batch` configuré selon la mémoire disponible
  - `n_ctx` ajusté selon les besoins de la requête

## Installation

### Installation Standard

```bash
# Installation des dépendances globales
yarn

# Installation des dépendances Python (ML Service)
cd py-ml
pip install -r requirements.txt
pip install -e .

# Retour à la racine et installation des dépendances NestJS
cd ../nest
yarn

# Installation des dépendances Next.js
cd ../next
yarn
```

### Optimisation pour Apple Silicon

Pour les systèmes Apple Silicon (M1/M2/M3), compiler llama-cpp-python avec le support Metal :

```bash
# Dans le répertoire py-ml
export CMAKE_ARGS="-DLLAMA_METAL=on" 
pip install llama-cpp-python --no-binary llama-cpp-python
```

## Démarrage

```bash
# Démarrer le service ML (depuis la racine)
cd py-ml
python -m src.ml_service.api.main

# Dans un nouveau terminal, démarrer le backend NestJS
cd nest
yarn start:dev

# Dans un nouveau terminal, démarrer le frontend Next.js
cd next
yarn dev
```

## Configuration Docker

Le système peut être déployé via Docker pour une installation simplifiée :

```bash
# Démarrer l'ensemble du système avec Docker Compose
docker-compose up -d

# Démarrer seulement le service ML
docker-compose up ml-api -d

# Arrêter tous les services
docker-compose down
```

Configuration Docker pour le service ML :

```yaml
services:
  ml-api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./model_cache:/app/model_cache
      - ./data:/app/data
    environment:
      - PORT=8000
      - MODEL_CACHE_DIR=/app/model_cache
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      - redis
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
  { 
    executionMode: ExecutionMode.ADAPTIVE,
    strategy: "FUSION"
  }
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
      { "id": "rag-engine", "name": "Moteur RAG" },
      { "id": "kag-engine", "name": "Moteur KAG" },
      { "id": "hybrid-orchestrator", "name": "Orchestrateur Hybride" },
      { "id": "output-formatter", "name": "Formateur de Sortie" }
    ],
    "data": {
      "marketing-agent": {
        "recommendation": "Pour une entreprise de luxe, la stratégie marketing optimale repose sur l'exclusivité, l'expérience client personnalisée et la narration de marque authentique...",
        "confidence": 0.87
      },
      "rag-engine": {
        "documents": ["doc1", "doc2", "doc3"],
        "relevance": 0.92
      },
      "kag-engine": {
        "concepts": ["luxury branding", "exclusivity", "customer experience"],
        "confidence": 0.89
      }
    },
    "executionStrategy": "FUSION",
    "executionTime": 1250,
    "resourceUsage": {
      "cpu": "42%",
      "memory": "1.2GB",
      "gpu": "36%"
    }
  }
}
```

## Intégration de Nouveaux Modèles

Le système permet d'intégrer facilement de nouveaux modèles GGUF :

1. **Téléchargement du modèle**
   ```bash
   mkdir -p models
   curl -L https://huggingface.co/TheBloke/[ModelName]-GGUF/resolve/main/[model-file].gguf -o models/[model-file].gguf
   ```

2. **Test du modèle**
   ```bash
   python src/ml_service/scripts/test_gguf_model.py
   ```

3. **Enregistrement dans ModelManager**
   - Ajouter le modèle à la liste des modèles supportés
   - Créer le wrapper d'API pour le modèle
   - Intégrer dans les routes API existantes

4. **Quantifications supportées**
   - Q4_K_M : Bon compromis taille/qualité pour appareils limités
   - Q5_K_M : Performance améliorée
   - Q8_0 : Haute qualité

## Personnalisation

Le système est conçu pour être hautement personnalisable :

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

## Feuille de Route

Le projet suit une feuille de route structurée en 5 phases :

1. **Phase 1 : Préparation et configuration**
   - Configuration des environnements de développement
   - Structuration des projets Frontend, Backend et ML
   - Mise en place des outils et contrats d'API

2. **Phase 2 : Développement de base**
   - Implmentation des composants UI et API REST
   - Développement des services fondamentaux
   - Optimisation des pipelines de traitement

3. **Phase 3 : Intégration et optimisation**
   - Intégration Frontend-Backend via API REST
   - Optimisation des communications entre NestJS et Python
   - Implémentation des mécanismes de mise en cache

4. **Phase 4 : Tests et qualité**
   - Développement des tests unitaires et d'intégration
   - Configuration des linters et hooks de pré-commit
   - Audits de sécurité et de performance

5. **Phase 5 : Finalisation et déploiement**
   - Documentation technique et guides d'utilisation
   - Configuration des pipelines CI/CD
   - Préparation des environnements de production

## Documentation

Pour plus de détails sur chaque composant :

- **Backend NestJS** : Voir `nest/README.md`
- **Services ML** : Voir `py-ml/doc/README.md`
- **Frontend** : Voir `next/README.md`

Documentation détaillée pour des aspects spécifiques:
- **Intégration GGUF** : `py-ml/doc/gguf_integration.md`
- **Nouveaux modèles** : `py-ml/doc/integration_new_gguf_model.md`
- **Plan d'implémentation** : `py-ml/doc/PLAN_IMPLEMENTATION_DETAILLE.md`

## Développement

```bash
# Compilation TypeScript
yarn build

# Exécution
yarn start

# Développement avec rechargement automatique
yarn dev

# Tests
yarn test

# Lint et formatage
yarn lint
yarn format
```

## Licence

MIT 