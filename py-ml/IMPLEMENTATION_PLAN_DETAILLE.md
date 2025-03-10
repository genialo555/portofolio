# Plan d'implémentation détaillé pour l'intégration des modèles ML

## Analyse de l'existant

Après analyse des fichiers et de la structure du projet, nous avons identifié:

- Une architecture ML Python existante avec plusieurs modèles:
  - `TeacherModel`: Modèle d'analyse et de synthèse de texte
  - `ImageTeacherModel`: Modèle de génération d'images
  - Une structure de base pour les modèles ML avec `BaseModel`

- Une application NestJS existante qui nécessite l'accès à ces modèles via une API

## Étape 1: Définir les contrats d'API (2 jours)

### 1.1 Identifier les modèles à exposer

| Modèle | Fonctionnalités | Endpoints à créer |
|--------|----------------|-------------------|
| TeacherModel | - Évaluation de réponses<br>- Synthèse de débats | - `/api/v1/models/teacher/evaluate`<br>- `/api/v1/models/teacher/synthesize` |
| ImageTeacherModel | - Génération d'images | - `/api/v1/models/image/generate` |

### 1.2 Définir les schemas de données

#### Schema pour TeacherModel - Endpoint d'évaluation
```typescript
// Request
interface EvaluateRequestDto {
  response: string;
  context?: {
    topic?: string;
    domain?: string;
  };
  model?: string; // Optionnel, par défaut 'qwen25'
}

// Response
interface EvaluateResponseDto {
  analysis: {
    initial: string;
    counterarguments: string;
    synthesis: string;
    recommendations: string;
  };
  metadata: {
    model_used: string;
    processing_time: number;
    version: string;
  }
}
```

#### Schema pour TeacherModel - Endpoint de synthèse
```typescript
// Request
interface SynthesizeRequestDto {
  perspective_a: string;
  perspective_b: string;
  debate_history: string[];
  context?: {
    topic?: string;
    depth?: 'shallow' | 'medium' | 'deep';
  };
}

// Response
interface SynthesizeResponseDto {
  synthesis: string;
  metadata: {
    model_used: string;
    processing_time: number;
    version: string;
  }
}
```

#### Schema pour ImageTeacherModel
```typescript
// Request
interface GenerateImageRequestDto {
  prompt: string;
  negative_prompt?: string;
  config?: {
    width?: number;
    height?: number;
    num_inference_steps?: number;
    guidance_scale?: number;
  };
}

// Response
interface GenerateImageResponseDto {
  image_url: string;
  quality_score: number;
  metadata: {
    model_used: string;
    processing_time: number;
    version: string;
  }
}
```

## Étape 2: Implémenter l'API FastAPI (3 jours)

### 2.1 Structure des fichiers à créer/modifier

```
src/ml_service/api/
├── routes/
│   ├── __init__.py (à modifier)
│   ├── models.py (à créer)
│   ├── health.py (existant)
├── middleware.py (existant)
├── main.py (existant)
├── models/
│   ├── __init__.py (à créer)
│   ├── dto.py (à créer - Data Transfer Objects)
│   ├── responses.py (à créer - Réponses standardisées)
├── core/
│   ├── __init__.py (à créer)
│   ├── manager.py (à créer - Gestionnaire de modèles)
│   ├── cache.py (à créer - Service de cache)
```

### 2.2 Implémentation des routes pour les modèles

Dans `src/ml_service/api/routes/models.py`:
- Définir le router FastAPI
- Implémenter les endpoints pour TeacherModel
- Implémenter les endpoints pour ImageTeacherModel
- Documenter avec OpenAPI

### 2.3 Implémenter le gestionnaire de modèles

Dans `src/ml_service/api/core/manager.py`:
- Créer une classe ModelManager qui instancie et gère les modèles ML
- Implémenter le chargement paresseux des modèles
- Implémenter des méthodes pour accéder à chaque type de modèle

### 2.4 Implémenter le service de cache

Dans `src/ml_service/api/core/cache.py`:
- Créer un cache LRU pour les réponses des modèles
- Configurer les TTL (time-to-live) selon le type de modèle
- Implémenter l'invalidation de cache

## Étape 3: Optimisation des performances (2 jours)

### 3.1 Mise en cache

- Implémenter un cache Redis pour les résultats d'inférence
- Configurer des clés de cache basées sur les paramètres d'entrée
- Configurer des expiration différentes selon les modèles

### 3.2 Optimisation du chargement des modèles

- Implémenter le chargement paresseux des modèles
- Utiliser la quantification pour réduire l'empreinte mémoire
- Configurer un pool de modèles pour gérer plusieurs requêtes

### 3.3 Gestion des ressources

- Implémenter un système de file d'attente pour les requêtes lourdes
- Configurer les timeouts pour éviter les blocages
- Ajouter un monitoring des ressources (CPU/GPU/Mémoire)

## Étape 4: Interfaces TypeScript (1 jour)

### 4.1 Création d'un package partagé

Créer un package de types partagés dans:
```
src/shared/types/
├── models/
│   ├── teacher.ts
│   ├── image.ts
│   ├── common.ts
```

### 4.2 DTOs NestJS

Créer les DTOs côté NestJS qui correspondent aux schemas d'API:
```
src/rag-kag/apis/python-models/dto/
├── teacher.dto.ts
├── image.dto.ts
```

## Étape 5: Service de consommation NestJS (2 jours)

### 5.1 Service PythonApiService

Dans `src/rag-kag/apis/python-models/python-api.service.ts`:
- Implémenter des méthodes pour chaque endpoint
- Ajouter les retry et circuit breakers
- Gestion des erreurs et fallbacks

## Étape 6: Tests et validation (2 jours)

### 6.1 Tests unitaires (1 jour)

- Tests pour les routes FastAPI
- Tests pour le gestionnaire de modèles
- Tests pour le service de cache

### 6.2 Tests d'intégration (1 jour)

- Tests de bout en bout avec NestJS et l'API Python
- Tests de charge pour valider les performances

## Étape 7: Documentation et déploiement (1 jour)

### 7.1 Documentation OpenAPI

- Générer et vérifier la documentation OpenAPI
- Créer des exemples de requêtes pour chaque endpoint

### 7.2 Dockerisation

- Créer un Dockerfile pour l'API Python
- Configurer les variables d'environnement
- Script de démarrage avec préchargement des modèles

## Calendrier d'exécution

| Jour | Activités |
|------|-----------|
| 1-2 | Définition des contrats d'API et schemas |
| 3-5 | Implémentation de l'API FastAPI |
| 6-7 | Optimisation des performances |
| 8 | Interfaces TypeScript |
| 9-10 | Service NestJS de consommation |
| 11-12 | Tests et validation |
| 13 | Documentation et déploiement |

## Prochaines étapes immédiates

1. Créer la structure de fichiers pour l'API FastAPI
2. Implémenter les routes pour TeacherModel
3. Implémenter le gestionnaire de modèles
4. Configurer la documentation OpenAPI
5. Créer les DTOs TypeScript pour NestJS 