# Projet RAG-KAG

## Vue d'ensemble

RAG-KAG est un système avancé qui intègre des technologies de frontend moderne, de backend robuste et de services d'intelligence artificielle pour fournir une solution complète de traitement et d'analyse de données.

Le projet s'appuie sur une architecture à trois niveaux:
- **Frontend**: Vue 3 avec TypeScript, Pinia et Vue Router
- **Backend**: NestJS avec une architecture Domain-Driven Design
- **Services ML**: Modèles Python pour le traitement et l'analyse des données avec RAG (Retrieval Augmented Generation) et KAG (Knowledge Augmented Generation)

## Structure du projet

```
py-ml/
├── src/
│   ├── ml_service/             # Services ML en Python
│   │   ├── api/                # API FastAPI exposant les modèles ML
│   │   ├── models/             # Modèles ML (TeacherModel, ImageTeacherModel)
│   │   ├── rag/                # Composants RAG (vectorizer, document_store, retriever)
│   │   └── ...                 # Autres modules (monitoring, utils, etc.)
│   ├── frontend/               # Application Vue 3 (à venir)
│   └── backend/                # API NestJS (à venir)
├── Dockerfile                  # Configuration de déploiement Docker
├── docker-compose.yml          # Configuration pour déploiement local
├── requirements.txt            # Dépendances Python
├── PYTHON_INTEGRATION_GUIDE.md # Guide d'intégration Python détaillé
├── PLAN_IMPLEMENTATION_DETAILLE.md # Plan d'implémentation détaillé
├── ROADMAP.md                  # Feuille de route du projet
└── README.md                   # Documentation principale
```

## Prérequis

- Node.js 18+
- Python 3.10+
- npm ou yarn
- pip
- Docker et Docker Compose (pour le déploiement avec conteneurs)
- Accès à des ressources GPU (optionnel mais recommandé)

## Installation

### Option 1: Installation directe

#### Services ML (Python)

```bash
# Créer un environnement virtuel
python -m venv venv
source venv/bin/activate  # Sur Windows: venv\Scripts\activate

# Installer les dépendances
pip install -r requirements.txt
```

#### Backend (NestJS)

```bash
# Installation des dépendances
cd src/backend
npm install
```

#### Frontend (Vue 3)

```bash
# Installation des dépendances
cd src/frontend
npm install
```

### Option 2: Utilisation de Docker

```bash
# Construction et démarrage des conteneurs
docker-compose up --build
```

## Démarrage

### Option 1: Démarrage direct

#### Services ML

```bash
# Activer l'environnement virtuel si ce n'est pas déjà fait
source venv/bin/activate  # Sur Windows: venv\Scripts\activate

# Démarrer l'API ML
cd src
python -m uvicorn ml_service.api.main:app --host 0.0.0.0 --port 8000 --reload
```

#### Backend

```bash
cd src/backend
npm run start:dev
```

#### Frontend

```bash
cd src/frontend
npm run dev
```

### Option 2: Utilisation de Docker

```bash
docker-compose up
```

## API ML

L'API ML expose plusieurs endpoints pour interagir avec les modèles ML:

### Gestion des modèles

- `GET /api/v1/models/list`: Liste tous les modèles disponibles
- `POST /api/v1/models/teacher/evaluate`: Évalue une réponse avec le TeacherModel
- `POST /api/v1/models/teacher/synthesize`: Synthétise un débat avec le TeacherModel
- `POST /api/v1/models/image/generate`: Génère une image avec l'ImageTeacherModel

### RAG (Retrieval Augmented Generation)

- `POST /api/v1/rag/documents/store`: Stocke et vectorise des documents
- `POST /api/v1/rag/retrieve`: Récupère les documents pertinents pour une requête
- `POST /api/v1/rag/generate`: Génère une réponse basée sur les documents récupérés
- `GET /api/v1/rag/namespaces`: Liste tous les espaces de noms disponibles
- `DELETE /api/v1/rag/documents/{namespace}`: Supprime tous les documents d'un espace de noms

### Vérification de santé

- `GET /api/v1/health`: Vérification basique de l'état de l'API
- `GET /api/v1/health/stats`: Statistiques détaillées sur l'API et le système
- `GET /api/v1/health/ping`: Simple endpoint ping-pong pour vérifier que l'API est réactive

## Documentation de l'API

La documentation Swagger de l'API est disponible à l'adresse suivante:

```
http://localhost:8000/docs
```

## Exemples d'utilisation

### Stockage de documents pour RAG

```python
import requests
import json

url = "http://localhost:8000/api/v1/rag/documents/store"
payload = {
    "documents": [
        {
            "content": "RAG (Retrieval Augmented Generation) est une technique qui améliore les modèles de langage en leur permettant d'accéder à des connaissances externes.",
            "metadata": {
                "source": "documentation",
                "topic": "RAG"
            }
        },
        {
            "content": "KAG (Knowledge Augmented Generation) utilise des graphes de connaissances pour enrichir les réponses des modèles de langage.",
            "metadata": {
                "source": "documentation",
                "topic": "KAG"
            }
        }
    ],
    "namespace": "techniques"
}
headers = {"Content-Type": "application/json"}

response = requests.post(url, data=json.dumps(payload), headers=headers)
print(response.json())
```

### Génération avec RAG

```python
import requests
import json

url = "http://localhost:8000/api/v1/rag/generate"
payload = {
    "query": "Quelle est la différence entre RAG et KAG?",
    "namespace": "techniques",
    "top_k": 3,
    "model": "teacher"
}
headers = {"Content-Type": "application/json"}

response = requests.post(url, data=json.dumps(payload), headers=headers)
print(response.json())
```

## Documentation

- [Guide d'intégration Python](./PYTHON_INTEGRATION_GUIDE.md) - Détails sur l'intégration des modèles ML
- [Plan d'implémentation détaillé](./PLAN_IMPLEMENTATION_DETAILLE.md) - Plan détaillé de l'implémentation
- [Roadmap](./ROADMAP.md) - Plan de développement du projet

## Contribution

Veuillez consulter le fichier CONTRIBUTING.md pour les directives de contribution (à venir).

## License

[MIT](LICENSE) 