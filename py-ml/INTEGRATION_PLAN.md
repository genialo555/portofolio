# Plan d'intégration des modèles ML

## Contexte

L'application backend NestJS existe déjà. L'objectif est maintenant d'exposer les modèles Python via une API et de les intégrer correctement dans l'architecture existante.

## Structure actuelle

```
.
├── src/
│   ├── ml_service/         # Services ML en Python (existant)
│   │   ├── api/            # API à développer pour exposer les modèles
│   │   ├── training/
│   │   ├── pilpoul/
│   │   ├── business/
│   │   ├── scripts/
│   │   ├── agents/
│   │   ├── models/
│   │   ├── utils/
│   │   ├── monitoring/
│   │   ├── data/
│   │   ├── pipelines/
│   │   ├── config.py
│   │   └── __init__.py
├── rag-kag/                # Application NestJS (existante)
│   ├── agents/
│   ├── apis/
│   │   ├── python-models/  # À développer pour consommer l'API Python
│   │   └── in-house-models/
│   ├── controllers/
│   ├── core/
│   └── ...
```

## Étapes d'intégration

### 1. Développement de l'API Python (1 semaine)

- [ ] **Définir les contrats d'API**
  - [ ] Identifier tous les modèles qui doivent être exposés
  - [ ] Définir les endpoints, paramètres et formats de réponse
  - [ ] Documenter l'API avec OpenAPI/Swagger

- [ ] **Implémenter l'API Python avec FastAPI**
  - [ ] Créer le squelette de l'API dans `src/ml_service/api/`
  - [ ] Implémenter les endpoints pour chaque modèle
  - [ ] Ajouter la validation des entrées
  - [ ] Implémenter la gestion des erreurs et exceptions
  - [ ] Ajouter le versionnage de l'API

- [ ] **Optimisation des performances**
  - [ ] Mettre en place un mécanisme de cache
  - [ ] Optimiser le chargement des modèles
  - [ ] Configurer l'utilisation des ressources (GPU/CPU)

### 2. Consommation de l'API depuis NestJS (3 jours)

- [ ] **Définir les interfaces TypeScript**
  - [ ] Créer des interfaces pour les entrées/sorties de chaque endpoint
  - [ ] Implémenter les DTOs pour la validation

- [ ] **Implémenter le service de consommation**
  - [ ] Développer `PythonApiService` pour appeler l'API Python
  - [ ] Implémenter la gestion des erreurs et retries
  - [ ] Ajouter des mécanismes de fallback
  - [ ] Configurer les circuit breakers

### 3. Tests et validation (2 jours)

- [ ] **Tests unitaires**
  - [ ] Tester les endpoints Python en isolation
  - [ ] Tester le service NestJS avec des mocks

- [ ] **Tests d'intégration**
  - [ ] Tester l'interaction entre NestJS et l'API Python
  - [ ] Valider les résultats avec des données réelles

### 4. Monitoring et observabilité (2 jours)

- [ ] **Logging structuré**
  - [ ] Configurer le logging dans l'API Python
  - [ ] Harmoniser avec le logging NestJS

- [ ] **Métriques et alertes**
  - [ ] Ajouter des métriques de performance (latence, charge, etc.)
  - [ ] Configurer des alertes pour détecter les problèmes

### 5. Documentation et déploiement (1 jour)

- [ ] **Documentation**
  - [ ] Mettre à jour la documentation technique
  - [ ] Créer des exemples d'utilisation

- [ ] **Configuration pour le déploiement**
  - [ ] Dockeriser l'API Python
  - [ ] Préparer les scripts de déploiement

## Risques et challenges

1. **Performances** : Latence entre NestJS et l'API Python. Solution : Mise en cache et optimisation.
2. **Fiabilité** : Gestion des erreurs et des cas où l'API Python est indisponible. Solution : Circuit breakers et stratégies de fallback.
3. **Synchronisation des types** : Maintenir la cohérence entre les interfaces TypeScript et les modèles Python. Solution : Génération automatique des types depuis l'API.

## Prochaines étapes immédiates

1. Finaliser la spécification des endpoints de l'API Python
2. Développer un prototype de l'API avec un modèle simple
3. Tester l'intégration avec NestJS sur ce premier modèle
4. Étendre progressivement à tous les modèles 