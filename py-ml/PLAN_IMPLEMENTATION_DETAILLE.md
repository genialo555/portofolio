# Plan d'implémentation détaillé pour l'intégration Python-NestJS

## 1. Vue d'ensemble

Ce plan détaille les étapes nécessaires pour implémenter et intégrer une API Python exposant des modèles ML (Machine Learning) avec une application NestJS existante. L'architecture s'appuie sur le modèle RAG-KAG (Retrieval Augmented Generation - Knowledge Augmented Generation).

## 2. Architecture technique

```
+----------------+      REST API      +----------------+      Python      +----------------+
|                |  <-------------->  |                |  <------------>  |                |
|  NestJS API    |                    |  Python API    |                  |  Modèles ML    |
|  (TypeScript)  |                    |  (FastAPI)     |                  |  (PyTorch)     |
|                |                    |                |                  |                |
+----------------+                    +----------------+                  +----------------+
```

## 3. Composants à développer

### 3.1 API Python (FastAPI)

#### 3.1.1 Structure de dossiers
```
src/ml_service/
├── api/
│   ├── main.py                    # Point d'entrée de l'API FastAPI
│   ├── middleware.py              # Middlewares pour logging, CORS, etc.
│   ├── models/                    # Modèles de données pour l'API
│   │   ├── __init__.py
│   │   ├── dto.py                 # Data Transfer Objects
│   │   └── responses.py           # Formats de réponse standards
│   ├── routes/                    # Endpoints de l'API
│   │   ├── __init__.py
│   │   ├── models.py              # Endpoints pour les modèles ML
│   │   └── health.py              # Endpoints de vérification de santé
│   ├── core/                      # Logique métier principale
│   │   ├── __init__.py
│   │   ├── manager.py             # Gestionnaire de modèles ML
│   │   ├── cache.py               # Service de mise en cache
│   │   ├── scheduler.py           # Planificateur de tâches
│   │   └── monitor.py             # Monitoring des performances
│   └── utils/                     # Utilitaires
│       ├── __init__.py
│       ├── logger.py              # Configuration de logging
│       └── exceptions.py          # Gestion des exceptions
├── models/                        # Définition des modèles ML
│   ├── __init__.py
│   ├── teacher_model.py           # Modèle d'évaluation et de synthèse
│   ├── image_teacher.py           # Modèle de génération d'images
│   └── base_model.py              # Classe de base pour les modèles
├── rag/                          # Composants RAG
│   ├── __init__.py
│   ├── vectorizer.py             # Vectorisation des documents
│   ├── retriever.py              # Récupération des documents pertinents
│   └── document_store.py         # Stockage des documents vectorisés
├── kag/                          # Composants KAG
│   ├── __init__.py
│   ├── knowledge_graph.py        # Gestionnaire du graphe de connaissances
│   └── sync.py                   # Synchronisation avec NestJS
├── monitoring/                   # Outils de surveillance
│   ├── __init__.py
│   ├── metrics.py                # Collection de métriques
│   └── prometheus.py             # Exporter Prometheus
└── config.py                     # Configuration globale
```

#### 3.1.2 Endpoints API à implémenter
1. **Gestion des modèles** (`/api/v1/models`)
   - `GET /list` - Liste tous les modèles disponibles
   - `POST /teacher/evaluate` - Évalue une réponse avec le TeacherModel
   - `POST /teacher/synthesize` - Synthétise un débat avec le TeacherModel
   - `POST /image/generate` - Génère une image avec l'ImageTeacherModel

2. **Vérification de santé** (`/api/v1/health`)
   - `GET /` - Vérification basique de l'état de l'API
   - `GET /stats` - Statistiques détaillées sur l'API et le système

3. **RAG (Retrieval Augmented Generation)** (`/api/v1/rag`)
   - `POST /documents/store` - Stocke et vectorise des documents
   - `POST /retrieve` - Récupère les documents pertinents pour une requête
   - `POST /generate` - Génère une réponse basée sur les documents récupérés

4. **KAG (Knowledge Augmented Generation)** (`/api/v1/kag`)
   - `POST /generate` - Génère une réponse basée sur le graphe de connaissances
   - `POST /verify` - Vérifie une réponse générée contre le graphe de connaissances
   - `POST /sync` - Synchronise le graphe de connaissances avec NestJS

5. **Hybrid RAG-KAG** (`/api/v1/hybrid`)
   - `POST /generate` - Génère une réponse en combinant RAG et KAG

6. **Monitoring** (`/api/v1/monitoring`)
   - `GET /metrics` - Exposer les métriques Prometheus
   - `GET /rag/metrics` - Métriques spécifiques à RAG
   - `GET /kag/metrics` - Métriques spécifiques à KAG

#### 3.1.3 Gestionnaire de modèles

La classe `ModelManager` sera le cœur de l'API, responsable de:
- Charger et gérer les modèles ML
- Gérer le cycle de vie des modèles (chargement, déchargement)
- Optimiser l'utilisation des ressources (GPU, mémoire)
- Fournir une interface unifiée pour tous les modèles

```python
class ModelManager:
    def __init__(self, model_cache_dir='./models'):
        self.model_cache_dir = model_cache_dir
        self.models = {}  # Modèles chargés en mémoire
        self.supported_models = ['teacher', 'image_teacher']
        # Initialisation des configurations...
        
    def load_model(self, model_name):
        # Logique pour charger un modèle
        
    def unload_model(self, model_name):
        # Libérer la mémoire
        
    def get_metrics(self, model_name):
        # Récupérer les métriques d'un modèle
        
    # Méthodes spécifiques par type de modèle
    def evaluate_response(self, response, model='teacher'):
        # Évaluer une réponse avec TeacherModel
        
    def synthesize_debate(self, perspective_a, perspective_b, history, model='teacher'):
        # Synthétiser un débat avec TeacherModel
        
    def generate_image(self, prompt, config=None, model='image_teacher'):
        # Générer une image avec ImageTeacherModel
        
    # Méthodes pour RAG/KAG
    def rag_generate(self, query, context_documents, model='teacher'):
        # Génération avec contexte documentaire
        
    def kag_generate(self, query, knowledge_nodes, model='teacher'):
        # Génération avec contexte du graphe de connaissances
        
    def hybrid_generate(self, query, context_documents, knowledge_nodes, model='teacher'):
        # Génération hybride RAG-KAG
```

### 3.2 Interfaces TypeScript (NestJS)

#### 3.2.1 Structure de dossiers
```
src/rag-kag/apis/python-models/
├── dto/                          # Data Transfer Objects
│   ├── teacher.dto.ts            # DTOs pour TeacherModel
│   ├── image.dto.ts              # DTOs pour ImageTeacherModel
│   └── common.dto.ts             # DTOs communs
├── python-api.service.ts         # Service principal d'API Python
├── python-api.module.ts          # Module NestJS
├── python-api.interface.ts       # Interfaces partagées
└── resilience.service.ts         # Service de résilience (circuit breaker)
```

#### 3.2.2 Interfaces principales

**PythonApiService**

```typescript
@Injectable()
export class PythonApiService {
  constructor(
    private readonly configService: ConfigService,
    private readonly httpService: HttpService,
    private readonly resilienceService: ResilienceService,
    @Inject(LOGGER_TOKEN) private readonly logger: ILogger
  ) {
    // Initialisation du service
  }

  // Méthodes principales d'interaction avec l'API Python
  public async evaluateResponse(response: string, options?: EvaluateOptions): Promise<EvaluationResult> {
    // Appelle l'API Python pour évaluer une réponse
  }

  public async synthesizeDebate(perspectiveA: string, perspectiveB: string, history: string[], options?: SynthesizeOptions): Promise<SynthesisResult> {
    // Appelle l'API Python pour synthétiser un débat
  }

  public async generateImage(prompt: string, options?: ImageGenerationOptions): Promise<ImageGenerationResult> {
    // Appelle l'API Python pour générer une image
  }

  // Méthodes RAG/KAG
  public async storeDocuments(documents: Document[]): Promise<DocumentStoreResult> {
    // Stocke des documents pour RAG
  }

  public async retrieveDocuments(query: string, options?: RetrieveOptions): Promise<RetrieveResult> {
    // Récupère des documents pertinents
  }

  public async generateWithRag(query: string, options?: RagGenerateOptions): Promise<GenerationResult> {
    // Génère une réponse avec RAG
  }

  public async generateWithKag(query: string, options?: KagGenerateOptions): Promise<GenerationResult> {
    // Génère une réponse avec KAG
  }

  public async generateHybrid(query: string, options?: HybridGenerateOptions): Promise<GenerationResult> {
    // Génère une réponse hybride RAG-KAG
  }

  // Méthodes de vérification de santé
  public async checkHealth(): Promise<boolean> {
    // Vérifie si l'API Python est disponible
  }

  public async getHealthStats(): Promise<HealthStats> {
    // Récupère des statistiques détaillées sur l'API Python
  }

  // Méthode utilitaire généralisée
  private async executeWithRetry<T>(fn: () => Promise<T>, options?: RetryOptions): Promise<T> {
    // Exécute une fonction avec retry et circuit breaker
  }
}
```

**DTOs**

```typescript
// teacher.dto.ts
export interface EvaluateRequestDto {
  response: string;
  context?: {
    topic?: string;
    domain?: string;
  };
  model?: string; // Optionnel, par défaut 'qwen25'
}

export interface EvaluateResponseDto {
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

// image.dto.ts
export interface GenerateImageRequestDto {
  prompt: string;
  negative_prompt?: string;
  config?: {
    width?: number;
    height?: number;
    num_inference_steps?: number;
    guidance_scale?: number;
  };
}

export interface GenerateImageResponseDto {
  image_url: string;
  quality_score: number;
  metadata: {
    model_used: string;
    processing_time: number;
    version: string;
  }
}
```

## 4. Plan d'implémentation par phases

### Phase 1: Configuration et architecture de base (3 jours)

1. **Jour 1: Configuration initiale**
   - Mettre en place l'environnement Python avec les dépendances nécessaires
   - Initialiser l'application FastAPI et ses middlewares
   - Implémenter les endpoints de santé (/health)
   - Configurer le logging structuré
   - Mettre en place les structures de dossiers

2. **Jour 2: Implémentation du ModelManager**
   - Développer la classe ModelManager
   - Implémenter la gestion des modèles (chargement/déchargement)
   - Configurer l'optimisation GPU
   - Mettre en place le système de cache

3. **Jour 3: Exposition des modèles basiques**
   - Implémenter les routes pour le TeacherModel
   - Implémenter les routes pour l'ImageTeacherModel
   - Ajouter la documentation OpenAPI
   - Mettre en place les validations de données

### Phase 2: Intégration RAG-KAG (4 jours)

4. **Jour 4: Implémentation de RAG**
   - Développer le système de vectorisation des documents
   - Créer le stockage vectoriel pour les embeddings
   - Implémenter le retriever pour la recherche sémantique
   - Ajouter les endpoints RAG

5. **Jour 5: Implémentation de KAG**
   - Développer le système de synchronisation du graphe de connaissances
   - Implémenter le stockage local du graphe
   - Créer les mécanismes d'extraction de connaissances
   - Ajouter les endpoints KAG

6. **Jour 6: Intégration hybride RAG-KAG**
   - Développer les mécanismes de génération hybride
   - Implémenter la validation des sorties avec le graphe
   - Créer les mécanismes d'évaluation de confiance
   - Ajouter les endpoints hybrides

7. **Jour 7: Systèmes de feedback**
   - Implémenter le système de collection des feedbacks
   - Créer les mécanismes d'apprentissage à partir des feedbacks
   - Développer les endpoints de feedback
   - Mettre en place les mécanismes de notification

### Phase 3: Monitoring et optimisation (3 jours)

8. **Jour 8: Métriques et monitoring**
   - Mettre en place l'intégration Prometheus
   - Développer les collecteurs de métriques spécifiques
   - Configurer les alertes
   - Implémenter les endpoints de métriques

9. **Jour 9: Optimisation des performances**
   - Configurer le chargement paresseux des modèles
   - Implémenter la quantification pour économiser la mémoire
   - Optimiser les pipelines de traitement
   - Mettre en place la limitation de débit (rate limiting)

10. **Jour 10: Déploiement et conteneurisation**
    - Créer les Dockerfiles pour l'API Python
    - Configurer le docker-compose pour l'environnement complet
    - Mettre en place les configurations pour différents environnements
    - Préparer les scripts de démarrage et d'initialisation

### Phase 4: Intégration avec NestJS (4 jours)

11. **Jour 11: Création des interfaces TypeScript**
    - Définir les DTOs pour tous les modèles
    - Créer les interfaces pour tous les endpoints
    - Documenter les API avec JsDoc
    - Configurer les validations

12. **Jour 12: Implémentation du PythonApiService**
    - Développer le service principal d'API
    - Implémenter les méthodes pour tous les endpoints
    - Configurer les timeouts et retries
    - Mettre en place la gestion des erreurs

13. **Jour 13: Implémentation du système de résilience**
    - Développer le ResilienceService
    - Implémenter les circuit breakers
    - Configurer les mécanismes de fallback
    - Mettre en place la détection d'anomalies

14. **Jour 14: Intégration avec le graphe de connaissances NestJS**
    - Développer les mécanismes de synchronisation bidirectionnelle
    - Implémenter les méthodes d'enrichissement du graphe
    - Configurer les mises à jour automatiques
    - Tester l'intégration complète

### Phase 5: Tests et documentation (3 jours)

15. **Jour 15: Tests unitaires**
    - Écrire les tests pour ModelManager
    - Tester les endpoints FastAPI
    - Tester les services NestJS
    - Mettre en place les mocks nécessaires

16. **Jour 16: Tests d'intégration**
    - Écrire les tests d'intégration Python-NestJS
    - Tester les scénarios RAG-KAG complets
    - Vérifier le comportement en cas de défaillance
    - Valider les performances

17. **Jour 17: Documentation complète**
    - Finaliser la documentation OpenAPI
    - Créer des exemples d'utilisation pour NestJS
    - Documenter les processus de déploiement
    - Préparer les guides d'utilisation

## 5. Risques et mitigations

| Risque | Impact | Probabilité | Mitigation |
|--------|--------|------------|------------|
| Latence élevée entre NestJS et l'API Python | Élevé | Moyen | Mise en cache agressive, chargement anticipé des modèles |
| Consommation excessive de mémoire GPU | Élevé | Élevé | Quantification des modèles, déchargement des modèles inutilisés |
| Incohérence des types entre Python et TypeScript | Moyen | Élevé | Génération automatique des types depuis OpenAPI |
| Défaillance de l'API Python | Élevé | Faible | Circuit breakers, mécanismes de fallback, monitoring |
| Difficulté d'intégration avec le graphe de connaissances | Moyen | Moyen | Tests extensifs, synchronisation contrôlée |

## 6. Livrables attendus

1. **Code source**
   - API Python FastAPI complète
   - Services NestJS d'intégration
   - Tests unitaires et d'intégration

2. **Documentation**
   - Documentation OpenAPI complète
   - Guide d'utilisation pour les développeurs NestJS
   - Guide de déploiement et de maintenance

3. **Conteneurs Docker**
   - Images Docker pour l'API Python
   - Configuration docker-compose
   - Scripts de déploiement

4. **Métriques et monitoring**
   - Dashboards Grafana
   - Alertes configurées
   - Documentation de surveillance

## 7. Estimation des ressources

- **Développeurs**: 2 (1 spécialiste Python ML, 1 développeur TypeScript/NestJS)
- **Durée totale**: 17 jours ouvrables
- **Ressources matérielles**: 
  - Serveur avec GPU (au moins 8 GB VRAM) pour le développement et les tests
  - Serveur CI/CD pour l'intégration continue
- **Budget logiciel**: Essentiellement des outils open-source (PyTorch, FastAPI, NestJS)

## 8. Prochaines étapes immédiates

1. **Mettre en place l'environnement de développement**
   - Installer les dépendances Python
   - Configurer l'environnement virtuel
   - Mettre en place VSCode avec les extensions appropriées

2. **Créer le squelette de l'API Python**
   - Initialiser la structure de dossiers
   - Mettre en place l'application FastAPI
   - Configurer les middlewares essentiels

3. **Implémenter le ModelManager basique**
   - Charger les modèles TeacherModel et ImageTeacherModel
   - Mettre en place l'interface unifiée
   - Tester le chargement des modèles

4. **Définir les contrats d'API**
   - Spécifier tous les endpoints et leurs paramètres
   - Documenter les formats de réponse
   - Préparer les schémas de validation 