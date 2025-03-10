# Guide d'intégration Python pour le projet RAG-KAG

Ce document détaille la méthodologie d'intégration des modèles Python dans notre architecture NestJS pour le système RAG-KAG. Il s'agit d'un guide pratique de mise en œuvre destiné aux développeurs du projet.

## 1. Architecture de l'intégration

### 1.1 Vue d'ensemble

L'intégration repose sur trois composants principaux :

```
+----------------+      REST API      +----------------+      Python      +----------------+
|                |  <-------------->  |                |  <------------>  |                |
|  NestJS API    |                    |  Python API    |                  |  Modèles ML    |
|  (TypeScript)  |                    |  (Flask/Fast)  |                  |  (TensorFlow)  |
|                |                    |                |                  |                |
+----------------+                    +----------------+                  +----------------+
```

- **NestJS API** : Notre application principale écrite en TypeScript
- **Python API** : Un service intermédiaire exposant les fonctionnalités ML via REST
- **Modèles ML** : Les modèles TensorFlow/PyTorch exécutés dans un environnement Python

### 1.2 Responsabilités

- **NestJS API** :
  - Gestion des requêtes utilisateur
  - Orchestration du flux de travail
  - Mise en cache des résultats
  - Gestion de la résilience et des fallbacks

- **Python API** :
  - Chargement et gestion des modèles
  - Prétraitement et post-traitement des données
  - Exécution des inférences et entraînements
  - Gestion des ressources GPU/CPU

- **Modèles ML** :
  - Exécution des opérations mathématiques
  - Inférence et prédiction
  - Entraînement et distillation

## 2. Implémentation côté NestJS

### 2.1 Services d'intégration

Nous utilisons un service dédié (`PythonApiService`) qui encapsule toutes les interactions avec l'API Python :

```typescript
// src/rag-kag/apis/python-api.service.ts
import { Injectable, Inject } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import { LOGGER_TOKEN, ILogger } from '../utils/logger-tokens';
import axios, { AxiosInstance } from 'axios';

@Injectable()
export class PythonApiService {
  // Implémentation...
}
```

### 2.2 Interfaces TypeScript

Pour assurer la cohérence, nous définissons des interfaces claires pour les requêtes et réponses :

```typescript
// Options d'entraînement
export interface TrainingRequestOptions {
  model: string;
  epochs?: number;
  batchSize?: number;
  learningRate?: number;
  // ...autres options
}

// Résultat d'entraînement
export interface TrainingResult {
  success: boolean;
  model: string;
  accuracy: number;
  loss: number;
  // ...autres métriques
}

// Options d'inférence
export interface InferenceRequestOptions {
  temperature?: number;
  maxTokens?: number;
  // ...autres options
}

// Résultat d'inférence
export interface InferenceResult {
  text: string;
  tokensUsed: number;
  // ...autres informations
}
```

### 2.3 Intégration avec les services existants

Mise à jour du `ModelTrainingService` pour utiliser la nouvelle API Python :

```typescript
// src/rag-kag/apis/model-training.service.ts
@Injectable()
export class ModelTrainingService implements OnModuleInit {
  constructor(
    // ...autres dépendances
    private readonly pythonApiService: PythonApiService
  ) {}

  public async forceTrainModel(modelName: string): Promise<boolean> {
    try {
      const result = await this.pythonApiService.trainModel(modelName, {
        epochs: 5,
        batchSize: 32,
        saveToDisk: true
      });

      if (result.success) {
        // Mise à jour des statistiques et du graphe de connaissances
        this.storeTrainingResultInGraph(modelName, result, {
          lastTraining: new Date(),
          examples: result.trainedExamples,
          accuracy: result.accuracy,
          loss: result.loss
        });
        
        return true;
      }
      
      return false;
    } catch (error) {
      this.logger.error(`Erreur lors de l'entraînement du modèle ${modelName}`, { error });
      return false;
    }
  }
}
```

### 2.4 Gestion de la résilience

Pour la gestion des erreurs et retries, implémentation d'une méthode `executeWithRetry` dans le `ResilienceService` :

```typescript
// src/rag-kag/utils/resilience.service.ts
export interface RetryOptions {
  maxRetries: number;
  retryCondition?: (error: any) => boolean;
  onRetry?: (error: any, attempt: number) => void;
  backoffFactor?: number;
}

@Injectable()
export class ResilienceService {
  // ...

  public async executeWithRetry<T>(
    fn: () => Promise<T>,
    options: RetryOptions
  ): Promise<T> {
    let lastError: any;
    const { maxRetries, retryCondition, onRetry, backoffFactor = 1.5 } = options;

    for (let attempt = 1; attempt <= maxRetries + 1; attempt++) {
      try {
        return await fn();
      } catch (error) {
        lastError = error;
        
        // Dernière tentative échouée
        if (attempt > maxRetries) {
          break;
        }
        
        // Vérifier si on doit réessayer pour cette erreur
        if (retryCondition && !retryCondition(error)) {
          break;
        }
        
        // Notification de retry
        if (onRetry) {
          onRetry(error, attempt);
        }
        
        // Attente exponentielle
        const delayMs = 1000 * Math.pow(backoffFactor, attempt - 1);
        await new Promise(resolve => setTimeout(resolve, delayMs));
      }
    }
    
    throw lastError;
  }
}
```

## 3. Implémentation côté Python

### 3.1 Structure de l'API Python

L'API Python doit exposer au minimum les endpoints suivants :

```
GET  /health                      # Vérification de l'état de l'API
POST /train                       # Entraînement d'un modèle
POST /generate                    # Génération/inférence
GET  /models/{modelName}/metrics  # Récupération des métriques d'un modèle
```

### 3.2 Spécification de l'API

#### 3.2.1 Endpoint `/health`

**Réponse** :
```json
{
  "status": "ok",
  "version": "1.0.0",
  "models": ["phi-3-mini", "llama-3-8b", "mistral-7b-fr"],
  "gpu_available": true,
  "memory_available": "12GB"
}
```

#### 3.2.2 Endpoint `/train`

**Requête** :
```json
{
  "model": "phi-3-mini",
  "epochs": 5,
  "batchSize": 32,
  "learningRate": 5e-5,
  "validationSplit": 0.1,
  "maxExamples": 1000,
  "useCache": true,
  "saveToDisk": true,
  "outputDirectory": "checkpoints/phi-3-mini"
}
```

**Réponse** :
```json
{
  "success": true,
  "model": "phi-3-mini",
  "trainedExamples": 1000,
  "epochs": 5,
  "accuracy": 0.87,
  "loss": 0.32,
  "validationAccuracy": 0.85,
  "validationLoss": 0.34,
  "trainingTime": 3600,
  "modelSize": 3500000000,
  "timestamp": "2023-03-08T12:34:56Z",
  "checkpointPath": "checkpoints/phi-3-mini/checkpoint_1234",
  "metrics": {
    "perplexity": 3.2,
    "f1_score": 0.89
  }
}
```

#### 3.2.3 Endpoint `/generate`

**Requête** :
```json
{
  "prompt": "Expliquez le fonctionnement d'un système RAG-KAG",
  "model": "phi-3-mini",
  "temperature": 0.7,
  "maxTokens": 1000,
  "topP": 0.95,
  "topK": 40,
  "repetitionPenalty": 1.1,
  "stop": ["###", "FIN"]
}
```

**Réponse** :
```json
{
  "text": "Un système RAG-KAG combine deux approches complémentaires...",
  "logprobs": [0.98, 0.87, 0.76],
  "tokensUsed": 450,
  "generationTime": 2.3,
  "model": "phi-3-mini"
}
```

#### 3.2.4 Endpoint `/models/{modelName}/metrics`

**Réponse** :
```json
{
  "model": "phi-3-mini",
  "parameters": 3500000000,
  "lastTrainingDate": "2023-03-08T12:34:56Z",
  "totalTrainingTime": 86400,
  "trainingIterations": 5,
  "averageAccuracy": 0.87,
  "latestEvaluation": {
    "timestamp": "2023-03-09T10:00:00Z",
    "metrics": {
      "perplexity": 3.2,
      "f1_score": 0.89,
      "rouge": 0.76
    }
  }
}
```

### 3.3 Implémentation suggérée (Python)

```python
# app.py (Flask)
from flask import Flask, request, jsonify
from model_manager import ModelManager

app = Flask(__name__)
model_manager = ModelManager()

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "ok",
        "version": "1.0.0",
        "models": model_manager.list_available_models(),
        "gpu_available": model_manager.is_gpu_available(),
        "memory_available": model_manager.get_available_memory()
    })

@app.route('/train', methods=['POST'])
def train_model():
    data = request.json
    result = model_manager.train_model(
        model_name=data.get('model'),
        epochs=data.get('epochs', 3),
        batch_size=data.get('batchSize', 32),
        learning_rate=data.get('learningRate', 5e-5),
        # ...autres paramètres
    )
    return jsonify(result)

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    result = model_manager.generate(
        prompt=data.get('prompt'),
        model_name=data.get('model'),
        temperature=data.get('temperature', 0.7),
        max_tokens=data.get('maxTokens', 500),
        # ...autres paramètres
    )
    return jsonify(result)

@app.route('/models/<model_name>/metrics', methods=['GET'])
def get_model_metrics(model_name):
    metrics = model_manager.get_metrics(model_name)
    return jsonify(metrics)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

## 4. Configuration et déploiement

### 4.1 Variables d'environnement

#### 4.1.1 Côté NestJS

```dotenv
# .env
PYTHON_API_URL=http://localhost:5000
PYTHON_API_KEY=dev-key
PYTHON_API_TIMEOUT=30000
```

#### 4.1.2 Côté Python

```dotenv
# .env.python
MODEL_CACHE_DIR=./models
HUGGINGFACE_TOKEN=hf_xxxxx
USE_GPU=true
MAX_MEMORY=8GB
```

### 4.2 Docker Compose

Pour faciliter le déploiement, utilisation de Docker Compose :

```yaml
# docker-compose.yml
version: '3'

services:
  nestjs-api:
    build:
      context: .
      dockerfile: Dockerfile.node
    ports:
      - "3001:3001"
    environment:
      - NODE_ENV=production
      - PYTHON_API_URL=http://python-api:5000
    depends_on:
      - python-api

  python-api:
    build:
      context: ./python
      dockerfile: Dockerfile.python
    ports:
      - "5000:5000"
    volumes:
      - ./models:/app/models
    environment:
      - MODEL_CACHE_DIR=/app/models
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

## 5. Roadmap d'implémentation

### 5.1 Étape 1 : Services d'intégration NestJS

- [x] Créer `PythonApiService` avec interfaces TypeScript
- [ ] Mettre à jour `ResilienceService` avec méthode `executeWithRetry`
- [ ] Ajouter le service au module CommonModule
- [ ] Créer les fichiers de configuration et variables d'environnement

### 5.2 Étape 2 : Mise à jour des services existants

- [ ] Mettre à jour `ModelTrainingService.forceTrainModel()` pour utiliser PythonApiService
- [ ] Mettre à jour `HouseModelService` pour utiliser PythonApiService
- [ ] Mettre à jour `ModelEvaluationService` pour utiliser PythonApiService

### 5.3 Étape 3 : Implémentation Python

- [ ] Créer le squelette de l'API Flask/FastAPI
- [ ] Implémenter la classe ModelManager pour gérer les modèles
- [ ] Implémenter les endpoints d'API
- [ ] Ajouter la gestion d'erreurs et logging

### 5.3.1 Étape 3.1 : Intégration des frameworks ML

- [ ] Configurer l'environnement PyTorch avec support GPU/CPU
- [ ] Implémenter la classe HuggingFaceModelHandler pour les modèles pré-entraînés
- [ ] Créer la classe RAGVectorizer avec sentence-transformers pour les embeddings
- [ ] Développer des utilitaires NumPy optimisés pour la recherche vectorielle
- [ ] Intégrer Pandas pour l'analyse et préparation des données d'entraînement
- [ ] Mettre en place les adaptateurs pour modèles personnalisés
- [ ] Implémenter les fonctions d'optimisation des performances GPU/CPU

### 5.4 Étape 4 : Tests et validation

- [ ] Créer des tests unitaires pour PythonApiService
- [ ] Créer des tests d'intégration entre NestJS et Python API
- [ ] Valider les performances et la résilience
- [ ] Documenter les API avec Swagger/OpenAPI

### 5.5 Étape 5 : Optimisations et monitoring

- [ ] Ajouter la mise en cache des réponses
- [ ] Configurer le monitoring (métriques, logs)
- [ ] Optimiser les performances (taille des batches, compression)
- [ ] Ajouter des circuit-breakers et fallbacks

## 6. Problèmes potentiels et solutions

### 6.1 Gestion des processus Python

**Problème** : Les processus Python peuvent consommer beaucoup de mémoire avec les modèles LLM.

**Solution** : 
- Limiter le nombre d'instances de modèles chargées simultanément
- Implémenter un système de déchargement des modèles inactifs
- Utiliser des techniques comme la quantification (8-bit, 4-bit)

### 6.2 Timeouts et requêtes longues

**Problème** : L'entraînement des modèles peut prendre plusieurs heures.

**Solution** :
- Utiliser un système de tâches asynchrones (Celery, Bull)
- Implémenter des webhooks pour les notifications de fin d'entraînement
- Mettre en place un système de statut pour les tâches longues

### 6.3 Synchronisation des versions

**Problème** : Maintenir la synchronisation entre les interfaces TypeScript et l'API Python.

**Solution** :
- Générer automatiquement les types TypeScript à partir des schémas OpenAPI
- Implémenter des tests de contrat pour valider la compatibilité
- Versionner explicitement les API

## 7. Références et ressources

### 7.1 Bibliothèques et frameworks

- **TypeScript/NestJS** :
  - [Axios](https://axios-http.com/) - Client HTTP
  - [Nestjs Config](https://docs.nestjs.com/techniques/configuration) - Gestion de configuration
  - [class-validator](https://github.com/typestack/class-validator) - Validation

- **Python/Flask** :
  - [Flask](https://flask.palletsprojects.com/) - Framework web
  - [Flask-RESTful](https://flask-restful.readthedocs.io/) - Extension REST
  - [Transformers](https://huggingface.co/docs/transformers/index) - Modèles LLM
  - [TensorFlow](https://www.tensorflow.org/) - Framework ML

### 7.2 Documentation

- [HuggingFace Model Training](https://huggingface.co/docs/transformers/training)
- [TensorFlow Model Optimization](https://www.tensorflow.org/model_optimization)
- [NestJS HTTP Module](https://docs.nestjs.com/techniques/http-module)
- [Flask Production Deployment](https://flask.palletsprojects.com/en/2.0.x/deploying/)

### 7.3 Exemples de code

- [Exemple complet d'API Flask pour ML](https://github.com/example/flask-ml-api)
- [Intégration TensorFlow avec NestJS](https://github.com/example/nestjs-tensorflow) 

## 8. Implémentation détaillée côté Python

### 8.1 Structure du projet Python

Pour maintenir une application Python bien organisée, nous recommandons la structure suivante:

```
python-api/
├── app.py                     # Point d'entrée principal de l'application
├── config.py                  # Configuration et variables d'environnement
├── requirements.txt           # Dépendances Python
├── Dockerfile                 # Configuration Docker
├── .env                       # Variables d'environnement (non versionné)
├── tests/                     # Tests unitaires et d'intégration
│   ├── test_app.py
│   ├── test_model_manager.py
│   └── ...
├── models/                    # Répertoire de stockage des modèles
│   ├── phi-3-mini/
│   ├── llama-3-8b/
│   └── mistral-7b-fr/
└── src/                       # Code source principal
    ├── __init__.py
    ├── model_manager.py       # Gestionnaire des modèles
    ├── training.py            # Logique d'entraînement
    ├── inference.py           # Logique d'inférence
    ├── monitoring.py          # Surveillance et métriques
    ├── utils.py               # Utilitaires divers
    └── schemas.py             # Schémas de validation (Pydantic)
```

### 8.2 Classe ModelManager

Le `ModelManager` est responsable du chargement, de la gestion et de l'utilisation des modèles. Voici une implémentation plus détaillée:

```python
# src/model_manager.py
import os
import time
import logging
from typing import Dict, List, Optional, Any, Union
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class ModelManager:
    def __init__(self, model_cache_dir: str = './models'):
        self.model_cache_dir = model_cache_dir
        self.models: Dict[str, Any] = {}
        self.tokenizers: Dict[str, Any] = {}
        self.model_metrics: Dict[str, Dict[str, Any]] = {}
        self.supported_models = ['phi-3-mini', 'llama-3-8b', 'mistral-7b-fr']
        self.logger = logging.getLogger('model_manager')
        
        # Configuration GPU
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.logger.info(f"Using device: {self.device}")
        
        # Suivi des modèles chargés et leurs statistiques
        for model_name in self.supported_models:
            self.model_metrics[model_name] = {
                'lastTrainingDate': None,
                'totalTrainingTime': 0,
                'trainingIterations': 0,
                'averageAccuracy': 0.0,
                'latestEvaluation': None
            }
    
    def list_available_models(self) -> List[str]:
        """Renvoie la liste des modèles supportés"""
        return self.supported_models
    
    def is_gpu_available(self) -> bool:
        """Vérifie si le GPU est disponible"""
        return self.device == 'cuda'
    
    def get_available_memory(self) -> str:
        """Renvoie la quantité de mémoire disponible"""
        if not self.is_gpu_available():
            return "CPU only"
        
        free_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
        return f"{free_memory / 1024**3:.2f}GB"
    
    def load_model(self, model_name: str) -> bool:
        """Charge un modèle en mémoire s'il n'est pas déjà chargé"""
        if model_name not in self.supported_models:
            self.logger.error(f"Model {model_name} not supported")
            return False
        
        if model_name in self.models:
            self.logger.info(f"Model {model_name} already loaded")
            return True
        
        try:
            self.logger.info(f"Loading model {model_name}...")
            model_path = os.path.join(self.model_cache_dir, model_name)
            
            # Vérifier si le modèle existe localement, sinon le télécharger
            if not os.path.exists(model_path):
                self.logger.info(f"Model {model_name} not found locally, downloading...")
                os.makedirs(model_path, exist_ok=True)
            
            # Charger le modèle et le tokenizer
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                cache_dir=model_path,
                device_map=self.device,
                torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32
            )
            tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=model_path)
            
            self.models[model_name] = model
            self.tokenizers[model_name] = tokenizer
            self.logger.info(f"Model {model_name} loaded successfully")
            
            return True
        except Exception as e:
            self.logger.error(f"Error loading model {model_name}: {str(e)}")
            return False
    
    def unload_model(self, model_name: str) -> bool:
        """Décharge un modèle de la mémoire"""
        if model_name not in self.models:
            return True
        
        try:
            del self.models[model_name]
            del self.tokenizers[model_name]
            
            # Force garbage collection
            import gc
            gc.collect()
            
            if self.is_gpu_available():
                torch.cuda.empty_cache()
                
            self.logger.info(f"Model {model_name} unloaded")
            return True
        except Exception as e:
            self.logger.error(f"Error unloading model {model_name}: {str(e)}")
            return False
    
    def train_model(self, 
                    model_name: str, 
                    epochs: int = 3, 
                    batch_size: int = 32, 
                    learning_rate: float = 5e-5,
                    validation_split: float = 0.1,
                    max_examples: Optional[int] = None,
                    use_cache: bool = True,
                    save_to_disk: bool = True,
                    output_directory: Optional[str] = None) -> Dict[str, Any]:
        """
        Entraîne un modèle avec les paramètres spécifiés
        Renvoie les métriques d'entraînement
        """
        if model_name not in self.supported_models:
            return {
                "success": False,
                "model": model_name,
                "message": f"Model {model_name} not supported"
            }
        
        start_time = time.time()
        self.logger.info(f"Starting training for model {model_name}")
        
        try:
            # TODO: Implémenter la logique d'entraînement réelle
            # Ce serait un processus qui utilise les datasets, configure les optimiseurs,
            # et exécute l'entraînement avec les hyperparamètres spécifiés
            
            # Pour l'instant, simulons l'entraînement pour démonstration
            time.sleep(2)  # Simuler l'entraînement
            
            # Résultats simulés
            accuracy = 0.87
            loss = 0.32
            validation_accuracy = 0.85
            validation_loss = 0.34
            trained_examples = 1000
            
            training_time = time.time() - start_time
            
            # Mettre à jour les métriques
            self.model_metrics[model_name]["lastTrainingDate"] = time.strftime("%Y-%m-%dT%H:%M:%SZ")
            self.model_metrics[model_name]["totalTrainingTime"] += training_time
            self.model_metrics[model_name]["trainingIterations"] += 1
            self.model_metrics[model_name]["averageAccuracy"] = (
                (self.model_metrics[model_name]["averageAccuracy"] * 
                 (self.model_metrics[model_name]["trainingIterations"] - 1) + 
                 accuracy) / self.model_metrics[model_name]["trainingIterations"]
            )
            
            # Enregistrer le modèle si demandé
            checkpoint_path = None
            if save_to_disk:
                output_dir = output_directory or os.path.join(self.model_cache_dir, model_name)
                checkpoint_path = os.path.join(output_dir, f"checkpoint_{int(time.time())}")
                os.makedirs(checkpoint_path, exist_ok=True)
                
                # Simuler l'enregistrement du modèle
                with open(os.path.join(checkpoint_path, "model_info.txt"), "w") as f:
                    f.write(f"Model: {model_name}\nAccuracy: {accuracy}\nLoss: {loss}")
                
                self.logger.info(f"Model saved to {checkpoint_path}")
            
            # Construire et renvoyer la réponse
            return {
                "success": True,
                "model": model_name,
                "trainedExamples": trained_examples,
                "epochs": epochs,
                "accuracy": accuracy,
                "loss": loss,
                "validationAccuracy": validation_accuracy,
                "validationLoss": validation_loss,
                "trainingTime": training_time,
                "modelSize": 3500000000,  # Simulé
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "checkpointPath": checkpoint_path,
                "metrics": {
                    "perplexity": 3.2,
                    "f1_score": 0.89
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error training model {model_name}: {str(e)}")
            return {
                "success": False,
                "model": model_name,
                "message": str(e),
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ")
            }
    
    def generate(self, 
                prompt: str, 
                model_name: str, 
                temperature: float = 0.7, 
                max_tokens: int = 500,
                top_p: float = 0.95,
                top_k: int = 40,
                repetition_penalty: float = 1.1,
                stop: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Génère une réponse à partir du modèle et du prompt donnés
        """
        if model_name not in self.supported_models:
            return {
                "success": False,
                "message": f"Model {model_name} not supported"
            }
        
        # Charger le modèle s'il n'est pas déjà chargé
        if model_name not in self.models:
            success = self.load_model(model_name)
            if not success:
                return {
                    "success": False,
                    "message": f"Failed to load model {model_name}"
                }
        
        start_time = time.time()
        self.logger.info(f"Generating with model {model_name}")
        
        try:
            model = self.models[model_name]
            tokenizer = self.tokenizers[model_name]
            
            # Encoding du prompt
            inputs = tokenizer(prompt, return_tensors="pt").to(self.device)
            
            # Génération
            generation_config = {
                "max_new_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "repetition_penalty": repetition_penalty,
                "do_sample": temperature > 0,
                "pad_token_id": tokenizer.eos_token_id
            }
            
            # Ajouter les tokens d'arrêt si spécifiés
            if stop:
                stop_token_ids = [tokenizer.encode(s, add_special_tokens=False)[0] for s in stop]
                generation_config["eos_token_id"] = stop_token_ids
            
            # Génération
            with torch.no_grad():
                output = model.generate(**inputs, **generation_config)
            
            # Décodage et suppression du prompt
            generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
            prompt_text = tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)
            if generated_text.startswith(prompt_text):
                generated_text = generated_text[len(prompt_text):].strip()
            
            # Calcul des metrics
            tokens_used = len(output[0]) - len(inputs["input_ids"][0])
            generation_time = time.time() - start_time
            
            # Renvoyer le résultat
            return {
                "text": generated_text,
                "tokensUsed": tokens_used,
                "generationTime": generation_time,
                "model": model_name
            }
        
        except Exception as e:
            self.logger.error(f"Error generating with model {model_name}: {str(e)}")
            return {
                "success": False,
                "message": str(e)
            }
    
    def get_metrics(self, model_name: str) -> Dict[str, Any]:
        """Renvoie les métriques du modèle spécifié"""
        if model_name not in self.supported_models:
            return {
                "success": False,
                "message": f"Model {model_name} not supported"
            }
        
        # Récupérer les métriques de base
        metrics = self.model_metrics.get(model_name, {})
        
        # Ajouter des informations supplémentaires
        return {
            "model": model_name,
            "parameters": 3500000000 if model_name == "phi-3-mini" else 8000000000 if model_name == "llama-3-8b" else 7000000000,
            "lastTrainingDate": metrics.get("lastTrainingDate"),
            "totalTrainingTime": metrics.get("totalTrainingTime", 0),
            "trainingIterations": metrics.get("trainingIterations", 0),
            "averageAccuracy": metrics.get("averageAccuracy", 0.0),
            "latestEvaluation": metrics.get("latestEvaluation", {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "metrics": {
                    "perplexity": 3.2,
                    "f1_score": 0.89,
                    "rouge": 0.76
                }
            })
        } 

### 8.3 Implémentation de l'application Flask

Voici une implémentation complète de l'application Flask qui expose les endpoints nécessaires:

```python
# app.py
import os
import logging
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from src.model_manager import ModelManager
from src.schemas import (
    TrainingRequest, 
    GenerationRequest,
    validate_request
)

# Charger les variables d'environnement
load_dotenv()

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.getenv('LOG_FILE', 'python-api.log'))
    ]
)

logger = logging.getLogger(__name__)

# Initialisation de l'application Flask
app = Flask(__name__)

# Initialisation du gestionnaire de modèles
model_manager = ModelManager(
    model_cache_dir=os.getenv('MODEL_CACHE_DIR', './models')
)

# Middleware de journalisation des requêtes
@app.before_request
def log_request_info():
    if request.path != '/health':  # Exclure les checks de santé pour réduire le bruit
        logger.info(f"Request: {request.method} {request.path}")
        if request.json:
            logger.debug(f"Request JSON: {request.json}")

# Middleware de gestion d'erreurs
@app.errorhandler(Exception)
def handle_exception(e):
    logger.error(f"Unhandled exception: {str(e)}", exc_info=True)
    return jsonify({
        "success": False,
        "message": str(e)
    }), 500

# Routes de l'API
@app.route('/health', methods=['GET'])
def health_check():
    """Endpoint de vérification de l'état de l'API"""
    return jsonify({
        "status": "ok",
        "version": os.getenv('API_VERSION', '1.0.0'),
        "models": model_manager.list_available_models(),
        "gpu_available": model_manager.is_gpu_available(),
        "memory_available": model_manager.get_available_memory()
    })

@app.route('/train', methods=['POST'])
def train_model():
    """Endpoint pour l'entraînement d'un modèle"""
    # Valider la requête
    data, validation_error = validate_request(request, TrainingRequest)
    if validation_error:
        logger.warning(f"Validation error: {validation_error}")
        return jsonify({
            "success": False,
            "message": validation_error
        }), 400
    
    # Extraire les paramètres de la requête
    model_name = data.model
    
    # Vérifier que le modèle est supporté
    if model_name not in model_manager.list_available_models():
        return jsonify({
            "success": False,
            "message": f"Model {model_name} not supported"
        }), 400
    
    # Entraîner le modèle
    result = model_manager.train_model(
        model_name=model_name,
        epochs=data.epochs,
        batch_size=data.batchSize,
        learning_rate=data.learningRate,
        validation_split=data.validationSplit,
        max_examples=data.maxExamples,
        use_cache=data.useCache,
        save_to_disk=data.saveToDisk,
        output_directory=data.outputDirectory
    )
    
    # Traitement du résultat
    if result.get("success", False):
        return jsonify(result), 200
    else:
        return jsonify(result), 500

@app.route('/generate', methods=['POST'])
def generate():
    """Endpoint pour la génération de texte"""
    # Valider la requête
    data, validation_error = validate_request(request, GenerationRequest)
    if validation_error:
        logger.warning(f"Validation error: {validation_error}")
        return jsonify({
            "success": False,
            "message": validation_error
        }), 400
    
    # Extraire les paramètres de la requête
    model_name = data.model
    prompt = data.prompt
    
    # Vérifier que le modèle est supporté
    if model_name not in model_manager.list_available_models():
        return jsonify({
            "success": False,
            "message": f"Model {model_name} not supported"
        }), 400
    
    # Générer la réponse
    result = model_manager.generate(
        prompt=prompt,
        model_name=model_name,
        temperature=data.temperature,
        max_tokens=data.maxTokens,
        top_p=data.topP,
        top_k=data.topK,
        repetition_penalty=data.repetitionPenalty,
        stop=data.stop
    )
    
    # Traitement du résultat
    if "success" in result and not result["success"]:
        return jsonify(result), 500
    else:
        return jsonify(result), 200

@app.route('/models/<model_name>/metrics', methods=['GET'])
def get_model_metrics(model_name):
    """Endpoint pour récupérer les métriques d'un modèle"""
    # Vérifier que le modèle est supporté
    if model_name not in model_manager.list_available_models():
        return jsonify({
            "success": False,
            "message": f"Model {model_name} not supported"
        }), 400
    
    # Récupérer les métriques
    metrics = model_manager.get_metrics(model_name)
    
    return jsonify(metrics), 200

@app.route('/train/status/<task_id>', methods=['GET'])
def get_task_status(task_id):
    """Endpoint pour vérifier le statut d'une tâche d'entraînement"""
    from src.celery_app import train_model_task
    
    # Récupérer la tâche
    task = train_model_task.AsyncResult(task_id)
    
    # Construire la réponse en fonction du statut
    response = {"taskId": task_id}
    
    if task.state == 'PENDING':
        response.update({
            "status": "PENDING",
            "message": "La tâche est en attente de démarrage"
        })
    elif task.state == 'PROGRESS':
        response.update({
            "status": "PROGRESS",
            "progress": task.info.get('progress', 0),
            "model": task.info.get('model')
        })
    elif task.state == 'SUCCESS':
        response.update({
            "status": "SUCCESS",
            "result": task.result
        })
    elif task.state == 'FAILURE':
        response.update({
            "status": "FAILURE",
            "message": str(task.info.get('message', 'Unknown error')),
            "model": task.info.get('model')
        })
    else:
        response.update({
            "status": task.state,
            "message": "Statut inconnu"
        })
    
    return jsonify(response), 200

# Démarrage de l'application
if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('DEBUG', 'False').lower() == 'true'
    
    logger.info(f"Starting Python API on port {port} (debug={debug})")
    logger.info(f"Available models: {model_manager.list_available_models()}")
    logger.info(f"GPU available: {model_manager.is_gpu_available()}")
    
    app.run(host='0.0.0.0', port=port, debug=debug)
```

### 8.4 Validation des données avec Pydantic

Pour assurer la robustesse de l'API, nous utilisons Pydantic pour valider les données entrantes:

```python
# src/schemas.py
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field, validator
from flask import Request

class TrainingRequest(BaseModel):
    """Schéma de validation pour les requêtes d'entraînement"""
    model: str
    epochs: int = Field(default=3, ge=1, le=100)
    batchSize: int = Field(default=32, ge=1, le=256)
    learningRate: float = Field(default=5e-5, ge=1e-7, le=1e-2)
    validationSplit: float = Field(default=0.1, ge=0.0, le=0.5)
    maxExamples: Optional[int] = Field(default=None, ge=1)
    useCache: bool = Field(default=True)
    saveToDisk: bool = Field(default=True)
    outputDirectory: Optional[str] = None
    
    @validator('model')
    def model_must_be_valid(cls, v):
        valid_models = ['phi-3-mini', 'llama-3-8b', 'mistral-7b-fr']
        if v not in valid_models:
            raise ValueError(f"Model must be one of: {', '.join(valid_models)}")
        return v

class GenerationRequest(BaseModel):
    """Schéma de validation pour les requêtes de génération"""
    prompt: str = Field(min_length=1, max_length=10000)
    model: str
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    maxTokens: int = Field(default=500, ge=1, le=2048)
    topP: float = Field(default=0.95, ge=0.0, le=1.0)
    topK: int = Field(default=40, ge=0, le=100)
    repetitionPenalty: float = Field(default=1.1, ge=1.0, le=2.0)
    stop: Optional[List[str]] = None
    
    @validator('model')
    def model_must_be_valid(cls, v):
        valid_models = ['phi-3-mini', 'llama-3-8b', 'mistral-7b-fr']
        if v not in valid_models:
            raise ValueError(f"Model must be one of: {', '.join(valid_models)}")
        return v
    
    @validator('prompt')
    def prompt_must_not_be_empty(cls, v):
        if not v or v.isspace():
            raise ValueError("Prompt cannot be empty or whitespace")
        return v

def validate_request(request: Request, schema_class):
    """
    Valide une requête HTTP selon un schéma Pydantic
    Renvoie les données validées et une erreur éventuelle
    """
    try:
        # Récupérer les données JSON
        json_data = request.get_json()
        if not json_data:
            return None, "Missing JSON data"
        
        # Valider les données avec le schéma
        validated_data = schema_class(**json_data)
        return validated_data, None
    except Exception as e:
        return None, str(e)
```

### 8.5 Gestion des tâches longues d'entraînement

Pour gérer les tâches d'entraînement qui peuvent prendre beaucoup de temps, nous pouvons utiliser Celery pour exécuter ces tâches en arrière-plan:

```python
# src/celery_app.py
import os
from celery import Celery
from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv()

# Configuration de Celery
redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
celery_app = Celery('model_training', broker=redis_url, backend=redis_url)

# Configuration des tâches
celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='Europe/Paris',
    enable_utc=True,
    worker_concurrency=1,  # Pour éviter de surcharger le GPU
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    task_track_started=True,
    task_time_limit=86400,  # 24h max
    task_soft_time_limit=82800  # 23h
)

@celery_app.task(bind=True)
def train_model_task(self, model_name, **kwargs):
    """Tâche d'entraînement de modèle en arrière-plan"""
    from src.model_manager import ModelManager
    
    # Mise à jour du statut
    self.update_state(state='PROGRESS', meta={'model': model_name, 'progress': 0})
    
    # Initialisation du gestionnaire de modèles
    model_manager = ModelManager(
        model_cache_dir=os.getenv('MODEL_CACHE_DIR', './models')
    )
    
    # Entraînement du modèle
    try:
        # Simulation de progression
        for i in range(1, 10):
            import time
            time.sleep(5)  # Simuler du travail
            self.update_state(state='PROGRESS', meta={'model': model_name, 'progress': i * 10})
        
        # Entraînement réel
        result = model_manager.train_model(model_name, **kwargs)
        return result
    except Exception as e:
        # En cas d'erreur, mettre à jour le statut
        self.update_state(
            state='FAILURE',
            meta={
                'model': model_name,
                'success': False,
                'message': str(e)
            }
        )
        raise e
```

Modifications à apporter à l'application Flask pour utiliser Celery:

```python
# Dans app.py, ajouter l'endpoint pour les tâches asynchrones

@app.route('/train/async', methods=['POST'])
def train_model_async():
    """Endpoint pour l'entraînement asynchrone d'un modèle"""
    # Valider la requête
    data, validation_error = validate_request(request, TrainingRequest)
    if validation_error:
        logger.warning(f"Validation error: {validation_error}")
        return jsonify({
            "success": False,
            "message": validation_error
        }), 400
    
    # Extraire les paramètres de la requête
    model_name = data.model
    
    # Lancer la tâche asynchrone
    from src.celery_app import train_model_task
    task = train_model_task.delay(
        model_name=model_name,
        epochs=data.epochs,
        batch_size=data.batchSize,
        learning_rate=data.learningRate,
        validation_split=data.validationSplit,
        max_examples=data.maxExamples,
        use_cache=data.useCache,
        save_to_disk=data.saveToDisk,
        output_directory=data.outputDirectory
    )
    
    # Renvoyer l'ID de la tâche
    return jsonify({
        "success": True,
        "taskId": task.id,
        "model": model_name,
        "status": "PENDING"
    }), 202

@app.route('/train/status/<task_id>', methods=['GET'])
def get_task_status(task_id):
    """Endpoint pour vérifier le statut d'une tâche d'entraînement"""
    from src.celery_app import train_model_task
    
    # Récupérer la tâche
    task = train_model_task.AsyncResult(task_id)
    
    # Construire la réponse en fonction du statut
    response = {"taskId": task_id}
    
    if task.state == 'PENDING':
        response.update({
            "status": "PENDING",
            "message": "La tâche est en attente de démarrage"
        })
    elif task.state == 'PROGRESS':
        response.update({
            "status": "PROGRESS",
            "progress": task.info.get('progress', 0),
            "model": task.info.get('model')
        })
    elif task.state == 'SUCCESS':
        response.update({
            "status": "SUCCESS",
            "result": task.result
        })
    elif task.state == 'FAILURE':
        response.update({
            "status": "FAILURE",
            "message": str(task.info.get('message', 'Unknown error')),
            "model": task.info.get('model')
        })
    else:
        response.update({
            "status": task.state,
            "message": "Statut inconnu"
        })
    
    return jsonify(response), 200 

### 8.6 Configuration Docker et déploiement

Pour faciliter le déploiement, voici les fichiers de configuration Docker nécessaires:

#### 8.6.1 Dockerfile pour l'API Python

```dockerfile
# Dockerfile.python
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Installation des dépendances système
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Définir le répertoire de travail
WORKDIR /app

# Copier les fichiers de dépendances et installer
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copier le code source
COPY . .

# Créer le répertoire pour les modèles
RUN mkdir -p /app/models

# Variables d'environnement
ENV PYTHONUNBUFFERED=1 \
    MODEL_CACHE_DIR=/app/models \
    PORT=5000

# Exposer le port
EXPOSE 5000

# Commande de démarrage
CMD ["python3", "app.py"]
```

#### 8.6.2 Dockerfile pour les workers Celery

```dockerfile
# Dockerfile.celery
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Installation des dépendances système
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Définir le répertoire de travail
WORKDIR /app

# Copier les fichiers de dépendances et installer
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copier le code source
COPY . .

# Créer le répertoire pour les modèles
RUN mkdir -p /app/models

# Variables d'environnement
ENV PYTHONUNBUFFERED=1 \
    MODEL_CACHE_DIR=/app/models \
    C_FORCE_ROOT=true

# Commande de démarrage
CMD ["celery", "-A", "src.celery_app", "worker", "--loglevel=info"]
```

#### 8.6.3 docker-compose.yml complet

```yaml
version: '3.8'

services:
  nestjs-api:
    build:
      context: .
      dockerfile: Dockerfile.node
    ports:
      - "3001:3001"
    environment:
      - NODE_ENV=production
      - PYTHON_API_URL=http://python-api:5000
      - PYTHON_API_KEY=${PYTHON_API_KEY}
    depends_on:
      - python-api
    networks:
      - app-network
    restart: unless-stopped

  python-api:
    build:
      context: ./python-api
      dockerfile: Dockerfile.python
    ports:
      - "5000:5000"
    environment:
      - MODEL_CACHE_DIR=/app/models
      - REDIS_URL=redis://redis:6379/0
      - API_VERSION=1.0.0
      - API_KEY=${PYTHON_API_KEY}
      - DEBUG=false
    volumes:
      - model-data:/app/models
    depends_on:
      - redis
    networks:
      - app-network
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    restart: unless-stopped

  celery-worker:
    build:
      context: ./python-api
      dockerfile: Dockerfile.celery
    environment:
      - MODEL_CACHE_DIR=/app/models
      - REDIS_URL=redis://redis:6379/0
    volumes:
      - model-data:/app/models
    depends_on:
      - redis
      - python-api
    networks:
      - app-network
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data
    networks:
      - app-network
    restart: unless-stopped

networks:
  app-network:
    driver: bridge

volumes:
  model-data:
    driver: local
  redis-data:
    driver: local
```

### 8.7 Fichiers de configuration et dépendances

#### 8.7.1 requirements.txt

```
# API Flask et utilitaires
flask==2.3.3
flask-cors==4.0.0
python-dotenv==1.0.0
pydantic==2.5.0
requests==2.31.0
celery==5.3.4
redis==5.0.1

# Machine Learning
torch==2.1.0
transformers==4.36.0
accelerate==0.25.0
datasets==2.14.6
scikit-learn==1.3.2

# Utilitaires
pillow==10.1.0
numpy==1.26.0
pandas==2.1.1
tqdm==4.66.1
matplotlib==3.8.0
python-jose==3.3.0 # Pour JWT

# Logging et monitoring
prometheus-client==0.17.1
python-logging-loki==0.3.1
opentelemetry-api==1.20.0
opentelemetry-sdk==1.20.0
opentelemetry-exporter-otlp==1.20.0
```

#### 8.7.2 Fichier de configuration .env

```dotenv
# Configuration de l'API
PORT=5000
DEBUG=false
API_VERSION=1.0.0
API_KEY=your-secure-api-key-here
LOG_FILE=python-api.log

# Configuration du modèle
MODEL_CACHE_DIR=./models
HUGGINGFACE_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# Configuration de Celery
REDIS_URL=redis://localhost:6379/0

# Configuration GPU
CUDA_VISIBLE_DEVICES=0
```

#### 8.7.3 Fichier de configuration pour gunicorn (production)

```python
# gunicorn_config.py
import multiprocessing

# Configuration de base
bind = "0.0.0.0:5000"
workers = 1  # Pour les modèles ML, généralement 1 seul worker pour éviter la duplication en mémoire
worker_class = "gthread"
threads = 4
timeout = 300
keepalive = 5

# Configuration de logging
accesslog = "-"  # stdout
errorlog = "-"   # stderr
loglevel = "info"

# Configuration de performance
worker_tmp_dir = "/dev/shm"  # Utiliser la mémoire partagée pour les fichiers temporaires
max_requests = 1000
max_requests_jitter = 50

# Préchargement de l'application
preload_app = True

# Hook de démarrage pour initialiser des ressources
def on_starting(server):
    server.log.info("Démarrage du serveur gunicorn pour l'API Python")

# Hook de fin pour libérer des ressources
def on_exit(server):
    server.log.info("Arrêt du serveur gunicorn")
```

#### 8.7.4 Commande de démarrage (production)

Pour démarrer l'application en production avec gunicorn:

```bash
#!/bin/bash
# start_api.sh
set -e

echo "Démarrage de l'API Python en mode production"

# Activation de l'environnement virtuel si nécessaire
# source /app/venv/bin/activate

# Définition des variables d'environnement
export $(grep -v '^#' .env | xargs)

# Démarrage avec gunicorn
gunicorn --config gunicorn_config.py app:app
```

## 9. Intégration avec le système de monitoring

Pour suivre les performances et l'état de santé de l'API Python, il est recommandé d'intégrer un système de monitoring. Voici un exemple d'intégration avec Prometheus et Grafana:

```python
# src/monitoring.py
import time
import threading
from prometheus_client import Counter, Gauge, Histogram, Summary, start_http_server

# Métriques Prometheus
API_REQUESTS = Counter('python_api_requests_total', 'Nombre total de requêtes', ['endpoint', 'method', 'status'])
API_REQUEST_LATENCY = Histogram('python_api_request_latency_seconds', 'Latence des requêtes', ['endpoint'])
MODEL_INFERENCE_LATENCY = Histogram('model_inference_latency_seconds', 'Latence d\'inférence par modèle', ['model'])
MODEL_TRAINING_DURATION = Histogram('model_training_duration_seconds', 'Durée d\'entraînement par modèle', ['model'])
MODEL_MEMORY_USAGE = Gauge('model_memory_usage_bytes', 'Utilisation mémoire par modèle', ['model'])
GPU_MEMORY_USAGE = Gauge('gpu_memory_usage_bytes', 'Utilisation mémoire GPU', ['device'])
GPU_UTILIZATION = Gauge('gpu_utilization_percent', 'Utilisation GPU en pourcentage', ['device'])
ACTIVE_MODELS = Gauge('active_models', 'Nombre de modèles chargés en mémoire')

def start_metrics_server(port=8000):
    """Démarre le serveur Prometheus metrics"""
    start_http_server(port)
    
def monitor_gpu_usage():
    """Surveille l'utilisation du GPU et met à jour les métriques"""
    try:
        import torch
        if not torch.cuda.is_available():
            return
            
        def update_gpu_metrics():
            while True:
                for i in range(torch.cuda.device_count()):
                    # Mémoire GPU
                    memory_allocated = torch.cuda.memory_allocated(i)
                    memory_reserved = torch.cuda.memory_reserved(i)
                    GPU_MEMORY_USAGE.labels(device=f"cuda:{i}").set(memory_allocated)
                    
                    # Utilisation GPU (nécessite pynvml)
                    try:
                        import pynvml
                        pynvml.nvmlInit()
                        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                        GPU_UTILIZATION.labels(device=f"cuda:{i}").set(utilization.gpu)
                    except:
                        pass
                        
                time.sleep(15)  # Mise à jour toutes les 15 secondes
                
        thread = threading.Thread(target=update_gpu_metrics, daemon=True)
        thread.start()
    except:
        pass
        
def track_request(endpoint, method, status_code, duration):
    """Enregistre les métriques pour une requête"""
    API_REQUESTS.labels(endpoint=endpoint, method=method, status=status_code).inc()
    API_REQUEST_LATENCY.labels(endpoint=endpoint).observe(duration)
    
def track_inference(model_name, duration):
    """Enregistre les métriques pour une inférence"""
    MODEL_INFERENCE_LATENCY.labels(model=model_name).observe(duration)
    
def track_training(model_name, duration):
    """Enregistre les métriques pour un entraînement"""
    MODEL_TRAINING_DURATION.labels(model=model_name).observe(duration)
    
def update_model_count(count):
    """Met à jour le nombre de modèles actifs"""
    ACTIVE_MODELS.set(count)
```

Intégration dans Flask avec un middleware:

```python
# Ajouter dans app.py
from src.monitoring import (
    start_metrics_server, 
    track_request, 
    monitor_gpu_usage
)

# Démarrer le serveur de métriques
start_metrics_server(port=int(os.getenv('METRICS_PORT', 8000)))

# Démarrer la surveillance GPU
monitor_gpu_usage()

# Middleware pour suivre les requêtes
@app.before_request
def before_request():
    request.start_time = time.time()

@app.after_request
def after_request(response):
    if hasattr(request, 'start_time'):
        duration = time.time() - request.start_time
        track_request(
            endpoint=request.path,
            method=request.method,
            status_code=response.status_code,
            duration=duration
        )
    return response

## 10. Tests et maintenance

### 10.1 Tests unitaires

Voici un exemple de tests unitaires pour le `ModelManager` :

```python
# tests/test_model_manager.py
import unittest
import os
import tempfile
from unittest.mock import patch, MagicMock
from src.model_manager import ModelManager

class TestModelManager(unittest.TestCase):
    def setUp(self):
        """Configuration avant chaque test"""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.model_cache_dir = self.temp_dir.name
        
        # Créer un ModelManager avec le répertoire temporaire
        self.model_manager = ModelManager(model_cache_dir=self.model_cache_dir)
        
        # Mock pour torch.cuda.is_available
        self.cuda_patch = patch('torch.cuda.is_available', return_value=False)
        self.mock_cuda = self.cuda_patch.start()
        
    def tearDown(self):
        """Nettoyage après chaque test"""
        self.temp_dir.cleanup()
        self.cuda_patch.stop()
        
    def test_list_available_models(self):
        """Test de la liste des modèles disponibles"""
        models = self.model_manager.list_available_models()
        self.assertEqual(len(models), 3)
        self.assertIn('phi-3-mini', models)
        self.assertIn('llama-3-8b', models)
        self.assertIn('mistral-7b-fr', models)
        
    def test_is_gpu_available(self):
        """Test de la détection du GPU"""
        # Test avec GPU non disponible (mock actif)
        self.assertFalse(self.model_manager.is_gpu_available())
        
        # Test avec GPU disponible
        self.mock_cuda.return_value = True
        self.assertTrue(self.model_manager.is_gpu_available())
        
    @patch('transformers.AutoModelForCausalLM.from_pretrained')
    @patch('transformers.AutoTokenizer.from_pretrained')
    def test_load_model(self, mock_tokenizer, mock_model):
        """Test du chargement d'un modèle"""
        # Configuration des mocks
        mock_model.return_value = MagicMock()
        mock_tokenizer.return_value = MagicMock()
        
        # Test avec un modèle supporté
        result = self.model_manager.load_model('phi-3-mini')
        self.assertTrue(result)
        self.assertIn('phi-3-mini', self.model_manager.models)
        
        # Vérifier que from_pretrained a été appelé
        mock_model.assert_called_once()
        mock_tokenizer.assert_called_once()
        
    def test_load_unsupported_model(self):
        """Test du chargement d'un modèle non supporté"""
        result = self.model_manager.load_model('unsupported-model')
        self.assertFalse(result)
        self.assertNotIn('unsupported-model', self.model_manager.models)
        
    @patch('src.model_manager.ModelManager.load_model')
    def test_generate(self, mock_load):
        """Test de la génération de texte"""
        # Configuration des mocks
        mock_load.return_value = True
        self.model_manager.models = {
            'phi-3-mini': MagicMock()
        }
        self.model_manager.tokenizers = {
            'phi-3-mini': MagicMock()
        }
        
        # Mock pour tokenizer et modèle
        tokenizer_mock = self.model_manager.tokenizers['phi-3-mini']
        model_mock = self.model_manager.models['phi-3-mini']
        
        # Configuration pour simuler l'encodage/décodage
        tokenizer_mock.encode.return_value = [1, 2, 3]
        tokenizer_mock.return_value = {'input_ids': torch.tensor([[1, 2, 3]])}
        model_mock.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])
        tokenizer_mock.decode.side_effect = ['Test prompt', 'Test prompt generated text']
        
        # Appel à la fonction generate
        result = self.model_manager.generate('Test prompt', 'phi-3-mini')
        
        # Vérification des résultats
        self.assertEqual(result['text'], 'generated text')
        self.assertEqual(result['model'], 'phi-3-mini')
        self.assertIn('tokensUsed', result)
        self.assertIn('generationTime', result)

if __name__ == '__main__':
    unittest.main()
```

### 10.2 Tests d'intégration

Voici un exemple de tests d'intégration pour l'API Flask :

```python
# tests/test_api.py
import unittest
import json
import os
from app import app
from unittest.mock import patch, MagicMock

class TestAPI(unittest.TestCase):
    def setUp(self):
        """Configuration avant chaque test"""
        self.app = app.test_client()
        self.app.testing = True
        
    @patch('src.model_manager.ModelManager.list_available_models')
    @patch('src.model_manager.ModelManager.is_gpu_available')
    @patch('src.model_manager.ModelManager.get_available_memory')
    def test_health_endpoint(self, mock_memory, mock_gpu, mock_models):
        """Test de l'endpoint /health"""
        # Configuration des mocks
        mock_models.return_value = ['phi-3-mini', 'llama-3-8b']
        mock_gpu.return_value = True
        mock_memory.return_value = "8.5GB"
        
        # Appel à l'endpoint
        response = self.app.get('/health')
        
        # Vérification de la réponse
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data['status'], 'ok')
        self.assertIn('models', data)
        self.assertEqual(len(data['models']), 2)
        self.assertTrue(data['gpu_available'])
        
    @patch('src.model_manager.ModelManager.train_model')
    def test_train_endpoint(self, mock_train):
        """Test de l'endpoint /train"""
        # Configuration du mock
        mock_train.return_value = {
            "success": True,
            "model": "phi-3-mini",
            "accuracy": 0.87,
            "loss": 0.32
        }
        
        # Données de la requête
        data = {
            "model": "phi-3-mini",
            "epochs": 5,
            "batchSize": 32
        }
        
        # Appel à l'endpoint
        response = self.app.post('/train', 
                               data=json.dumps(data),
                               content_type='application/json')
        
        # Vérification de la réponse
        self.assertEqual(response.status_code, 200)
        result = json.loads(response.data)
        self.assertTrue(result['success'])
        self.assertEqual(result['model'], 'phi-3-mini')
        
    @patch('src.model_manager.ModelManager.generate')
    def test_generate_endpoint(self, mock_generate):
        """Test de l'endpoint /generate"""
        # Configuration du mock
        mock_generate.return_value = {
            "text": "Ceci est une réponse générée",
            "tokensUsed": 10,
            "generationTime": 0.5,
            "model": "phi-3-mini"
        }
        
        # Données de la requête
        data = {
            "prompt": "Génère un texte",
            "model": "phi-3-mini",
            "temperature": 0.7
        }
        
        # Appel à l'endpoint
        response = self.app.post('/generate', 
                               data=json.dumps(data),
                               content_type='application/json')
        
        # Vérification de la réponse
        self.assertEqual(response.status_code, 200)
        result = json.loads(response.data)
        self.assertEqual(result['text'], "Ceci est une réponse générée")
        self.assertEqual(result['model'], 'phi-3-mini')
        
    def test_generate_endpoint_bad_request(self):
        """Test de l'endpoint /generate avec une requête invalide"""
        # Données de la requête manquantes
        data = {
            "temperature": 0.7
        }
        
        # Appel à l'endpoint
        response = self.app.post('/generate', 
                               data=json.dumps(data),
                               content_type='application/json')
        
        # Vérification de la réponse
        self.assertEqual(response.status_code, 400)
        result = json.loads(response.data)
        self.assertFalse(result['success'])

if __name__ == '__main__':
    unittest.main()
```

### 10.3 Maintenance et mise à jour

Pour maintenir l'intégration Python à jour et fonctionnelle à long terme, il est recommandé de :

1. **Tester régulièrement** l'intégration avec des scénarios automatisés
2. **Surveiller les performances** via le dashboard Grafana
3. **Mettre à jour les dépendances** en suivant une stratégie gérée :

```bash
# Mise à jour des dépendances avec vérification des compatibilités
pip-compile --upgrade requirements.in
# Installation des versions précises
pip install -r requirements.txt
```

4. **Gérer les versions** des modèles HuggingFace en utilisant des tags spécifiques :

```python
def load_model(model_name, revision="v1.0.0"):
    """Charger un modèle avec une version spécifique"""
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        revision=revision,
        # ...autres paramètres
    )
    return model
```

5. **Créer des snapshots périodiques** des poids des modèles et de la base de connaissances :

```python
def create_model_snapshot(model_name):
    """Crée un snapshot du modèle actuel"""
    timestamp = int(time.time())
    snapshot_dir = f"snapshots/{model_name}/{timestamp}"
    os.makedirs(snapshot_dir, exist_ok=True)
    
    # Sauvegarde du modèle
    model = model_manager.models.get(model_name)
    if model:
        model.save_pretrained(snapshot_dir)
        
    # Sauvegarde des métriques
    metrics = model_manager.get_metrics(model_name)
    with open(f"{snapshot_dir}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
        
    return snapshot_dir
```

### 10.4 Sécurité et Authentification

Pour protéger l'API Python :

```python
# src/auth.py
import os
import jwt
from functools import wraps
from flask import request, jsonify
from datetime import datetime, timedelta

def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization')
        
        if not token:
            return jsonify({
                'success': False,
                'message': 'Token is missing'
            }), 401
        
        # Format: "Bearer <token>"
        if token.startswith('Bearer '):
            token = token[7:]
            
        try:
            api_key = os.getenv('API_KEY')
            if token != api_key:
                raise ValueError("Invalid token")
        except Exception as e:
            return jsonify({
                'success': False,
                'message': 'Invalid token'
            }), 401
            
        return f(*args, **kwargs)
    
    return decorated
```

Intégration dans l'API :

```python
# Ajouter dans app.py
from src.auth import token_required

# Protéger les endpoints sensibles
@app.route('/train', methods=['POST'])
@token_required
def train_model():
    # ...

@app.route('/generate', methods=['POST'])
@token_required
def generate():
    # ...
```

### 10.5 Documentation de l'API

Utiliser Swagger/OpenAPI pour documenter l'API :

```python
# Ajouter dans app.py
from flask_swagger_ui import get_swaggerui_blueprint

# Configuration Swagger
SWAGGER_URL = '/api/docs'
API_URL = '/static/swagger.json'

# Création du blueprint Swagger
swagger_ui_blueprint = get_swaggerui_blueprint(
    SWAGGER_URL,
    API_URL,
    config={
        'app_name': "Python API Documentation"
    }
)

# Enregistrement du blueprint
app.register_blueprint(swagger_ui_blueprint, url_prefix=SWAGGER_URL)
```

Avec un fichier swagger.json :

```json
{
  "openapi": "3.0.0",
  "info": {
    "title": "API Python pour les modèles ML",
    "description": "API d'entraînement et d'inférence pour les modèles de ML",
    "version": "1.0.0"
  },
  "servers": [
    {
      "url": "http://localhost:5000",
      "description": "Serveur local"
    }
  ],
  "components": {
    "securitySchemes": {
      "BearerAuth": {
        "type": "http",
        "scheme": "bearer"
      }
    }
  },
  "security": [
    {
      "BearerAuth": []
    }
  ],
  "paths": {
    "/health": {
      "get": {
        "summary": "Vérifier l'état de l'API",
        "responses": {
          "200": {
            "description": "L'API est opérationnelle",
            "content": {
              "application/json": {
                "schema": {
                  "type": "object",
                  "properties": {
                    "status": { "type": "string" },
                    "version": { "type": "string" },
                    "models": { "type": "array", "items": { "type": "string" } },
                    "gpu_available": { "type": "boolean" },
                    "memory_available": { "type": "string" }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}
```

## 11. Références supplémentaires

### 11.1 Ressources HuggingFace

- [Guide de fine-tuning PEFT](https://huggingface.co/docs/peft/index)
- [LoRA pour l'adaptation de modèles](https://huggingface.co/docs/diffusers/training/lora)
- [Quantization avec bitsandbytes](https://huggingface.co/blog/4bit-transformers-bitsandbytes)

### 11.2 TensorFlow.js pour l'optimisation

Si nécessaire, possibilité d'utiliser TensorFlow.js pour exécuter certains modèles légers côté client :

```javascript
// Exemple d'intégration côté client
import * as tf from '@tensorflow/tfjs';

async function loadModel() {
  const model = await tf.loadLayersModel('path/to/model.json');
  return model;
}

async function predict(input) {
  const model = await loadModel();
  const tensor = tf.tensor2d([input]);
  const prediction = model.predict(tensor);
  return prediction.dataSync();
}
```

### 11.3 Ressources d'optimisation

- [ONNX Runtime](https://onnxruntime.ai/) pour l'optimisation des inférences
- [TensorRT](https://developer.nvidia.com/tensorrt) pour l'accélération GPU avancée
- [Triton Inference Server](https://github.com/triton-inference-server/server) pour le déploiement à grande échelle

## 12. Aspects additionnels de l'intégration

### 12.1 Intégration avec le graphe de connaissances

Le stockage des résultats d'entraînement dans le graphe de connaissances est une étape cruciale pour maintenir la traçabilité des modèles. Voici comment cela est implémenté :

```typescript
// Fonction de stockage des résultats dans le graphe de connaissances
private storeTrainingResultInGraph(
  model: string,
  result: TrainingResult,
  stats: { lastTraining: Date, examples: number, accuracy?: number, loss?: number }
): void {
  if (!this.knowledgeGraph) {
    this.logger.warn(`Graphe de connaissances non disponible pour stocker les résultats d'entraînement`);
    return;
  }

  try {
    // Création d'un nœud pour l'entraînement
    const trainingNodeId = `training_${model}_${new Date().getTime()}`;
    const modelNodeId = `model_${model}`;
    
    // Ajout des nœuds
    this.knowledgeGraph.addNode(trainingNodeId, 'TrainingResult', {
      model: model,
      timestamp: new Date().toISOString(),
      accuracy: stats.accuracy || result.accuracy,
      loss: stats.loss || result.loss,
      examples: stats.examples || result.trainedExamples,
      trainingTime: result.trainingTime
    });
    
    // Vérification si le nœud du modèle existe, sinon le créer
    const modelNodeExists = await this.knowledgeGraph.nodeExists(modelNodeId);
    if (!modelNodeExists) {
      this.knowledgeGraph.addNode(modelNodeId, 'Model', {
        name: model,
        parameters: model === 'phi-3-mini' ? 3500000000 : model === 'llama-3-8b' ? 8000000000 : 7000000000,
        lastUpdated: new Date().toISOString()
      });
    } else {
      // Mise à jour du modèle
      this.knowledgeGraph.updateNode(modelNodeId, {
        lastUpdated: new Date().toISOString()
      });
    }
    
    // Création de la relation entre le modèle et l'entraînement
    this.knowledgeGraph.addEdge(
      modelNodeId,
      'HAS_TRAINING_RESULT',
      trainingNodeId,
      0.9,
      {
        bidirectional: true,
        weight: 0.8
      }
    );
  } catch (error) {
    this.logger.error(`Erreur lors du stockage des résultats d'entraînement dans le graphe: ${error.message}`, {
      error: error.stack
    });
  }
}
```

Pour interroger l'historique d'entraînement d'un modèle :

```typescript
async getModelTrainingHistory(modelName: string): Promise<any[]> {
  if (!this.knowledgeGraph) {
    return [];
  }
  
  const modelNodeId = `model_${modelName}`;
  
  // Requête pour obtenir tous les résultats d'entraînement liés au modèle
  const query = `
    MATCH (m:Model {name: $modelName})-[:HAS_TRAINING_RESULT]->(t:TrainingResult)
    RETURN t
    ORDER BY t.timestamp DESC
    LIMIT 10
  `;
  
  return this.knowledgeGraph.executeQuery(query, { modelName });
}
```

### 12.2 Optimisation avancée des modèles

#### 12.2.1 Quantification des modèles

La quantification peut réduire significativement l'empreinte mémoire des modèles LLM :

```python
# Chargement d'un modèle quantifié en 4-bit
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
import torch

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)

model = AutoModelForCausalLM.from_pretrained(
    "mistral-7b-fr",
    device_map="auto",
    quantization_config=quantization_config
)
```

#### 12.2.2 Pruning des modèles

Le pruning permet d'éliminer les poids non essentiels :

```python
# Exemple de pruning avec PyTorch
from torch.nn.utils import prune

# Appliquer un pruning basé sur la magnitude
prune.l1_unstructured(model.decoder.layers[0].self_attn.q_proj, name="weight", amount=0.2)

# Rendre le pruning permanent
prune.remove(model.decoder.layers[0].self_attn.q_proj, "weight")
```

### 12.3 Gestion des files d'attente pour les entraînements parallèles

Pour gérer plusieurs demandes d'entraînement simultanées, nous pouvons implémenter un système de file d'attente :

```python
# src/queue_manager.py
import threading
import queue
import time
from typing import Dict, Any, Callable, List

class TrainingQueue:
    def __init__(self, max_concurrent_trainings=1):
        self.task_queue = queue.PriorityQueue()
        self.active_tasks = {}
        self.max_concurrent = max_concurrent_trainings
        self.lock = threading.Lock()
        self.worker_thread = threading.Thread(target=self._process_queue, daemon=True)
        self.worker_thread.start()
    
    def add_task(self, model_name: str, params: Dict[str, Any], priority: int = 5) -> str:
        """Ajouter une tâche à la file d'attente avec une priorité (1 = haute, 10 = basse)"""
        task_id = f"task_{int(time.time())}_{model_name}"
        
        with self.lock:
            self.active_tasks[task_id] = {
                "model": model_name,
                "params": params,
                "status": "queued",
                "queued_at": time.time(),
                "started_at": None,
                "completed_at": None,
                "result": None,
                "error": None
            }
            
        # Ajouter à la file d'attente prioritaire
        self.task_queue.put((priority, task_id))
        return task_id
    
    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """Obtenir le statut d'une tâche"""
        with self.lock:
            if task_id not in self.active_tasks:
                return {"status": "unknown", "task_id": task_id}
            return {**self.active_tasks[task_id], "task_id": task_id}
    
    def get_all_tasks(self) -> List[Dict[str, Any]]:
        """Obtenir la liste de toutes les tâches actives"""
        with self.lock:
            return [
                {**task, "task_id": task_id}
                for task_id, task in self.active_tasks.items()
            ]
    
    def _process_queue(self):
        """Traiter la file d'attente en continu"""
        while True:
            # Compter les tâches en cours
            running_tasks = 0
            with self.lock:
                for task in self.active_tasks.values():
                    if task["status"] == "running":
                        running_tasks += 1
            
            # Si nous avons atteint le maximum de tâches concurrentes, attendre
            if running_tasks >= self.max_concurrent:
                time.sleep(5)
                continue
            
            try:
                # Récupérer la tâche suivante
                priority, task_id = self.task_queue.get(block=False)
                
                with self.lock:
                    if task_id not in self.active_tasks:
                        self.task_queue.task_done()
                        continue
                    
                    # Mettre à jour le statut
                    self.active_tasks[task_id]["status"] = "running"
                    self.active_tasks[task_id]["started_at"] = time.time()
                    
                    # Récupérer les informations de la tâche
                    model_name = self.active_tasks[task_id]["model"]
                    params = self.active_tasks[task_id]["params"]
                
                # Exécuter la tâche dans un thread séparé
                thread = threading.Thread(
                    target=self._run_task,
                    args=(task_id, model_name, params),
                    daemon=True
                )
                thread.start()
                
                self.task_queue.task_done()
            except queue.Empty:
                # Aucune tâche dans la file d'attente, attendre
                time.sleep(1)
    
    def _run_task(self, task_id: str, model_name: str, params: Dict[str, Any]):
        """Exécuter une tâche d'entraînement"""
        try:
            from src.model_manager import ModelManager
            
            # Créer un gestionnaire de modèles
            model_manager = ModelManager()
            
            # Exécuter l'entraînement
            result = model_manager.train_model(model_name, **params)
            
            # Mettre à jour le statut
            with self.lock:
                if task_id in self.active_tasks:
                    self.active_tasks[task_id]["status"] = "completed"
                    self.active_tasks[task_id]["completed_at"] = time.time()
                    self.active_tasks[task_id]["result"] = result
        except Exception as e:
            # En cas d'erreur, mettre à jour le statut
            with self.lock:
                if task_id in self.active_tasks:
                    self.active_tasks[task_id]["status"] = "failed"
                    self.active_tasks[task_id]["completed_at"] = time.time()
                    self.active_tasks[task_id]["error"] = str(e)

# Utilisation dans l'API Flask
training_queue = TrainingQueue(max_concurrent_trainings=2)

@app.route('/train/queue', methods=['POST'])
@token_required
def queue_training():
    data, validation_error = validate_request(request, TrainingRequest)
    if validation_error:
        return jsonify({"success": False, "message": validation_error}), 400
    
    # Ajouter à la file d'attente
    priority = request.json.get('priority', 5)
    task_id = training_queue.add_task(data.model, data.dict(), priority)
    
    return jsonify({
        "success": True,
        "taskId": task_id,
        "model": data.model,
        "status": "queued"
    }), 202
```

### 12.4 Configuration spécifique pour différents types de GPUs

#### 12.4.1 Support de ROCm (AMD)

Pour les GPUs AMD, utilisez cette configuration :

```python
# Détection du type de GPU et configuration
def configure_gpu_environment():
    """Configure l'environnement en fonction du GPU disponible"""
    import torch
    import os
    
    if torch.cuda.is_available():
        # NVIDIA CUDA
        device_name = torch.cuda.get_device_name(0)
        device_count = torch.cuda.device_count()
        return {
            "type": "cuda",
            "count": device_count,
            "name": device_name,
            "memory": f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f}GB"
        }
    elif hasattr(torch, 'hip') and torch.hip.is_available():
        # AMD ROCm
        device_count = torch.hip.device_count()
        return {
            "type": "rocm",
            "count": device_count,
            "name": "AMD GPU",
            "memory": "N/A"  # ROCm ne fournit pas facilement cette information
        }
    else:
        # CPU only
        return {
            "type": "cpu",
            "count": os.cpu_count(),
            "name": "CPU Only",
            "memory": "N/A"
        }

# Utilisation pour charger un modèle
def load_model_with_gpu_config(model_name, model_cache_dir):
    """Charge un modèle en tenant compte du type de GPU disponible"""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    gpu_config = configure_gpu_environment()
    
    # Options de chargement communes
    load_options = {
        "cache_dir": model_cache_dir,
        "torch_dtype": torch.float16
    }
    
    # Ajuster les options selon le type de GPU
    if gpu_config["type"] == "cuda":
        # NVIDIA CUDA
        load_options["device_map"] = "auto"
    elif gpu_config["type"] == "rocm":
        # AMD ROCm
        os.environ["HF_REMOTES_OFFLINE"] = "1"  # Éviter certains problèmes avec ROCm
        if model_name == "phi-3-mini":
            # Certains modèles fonctionnent mieux avec des configurations spécifiques sur ROCm
            load_options["low_cpu_mem_usage"] = True
    else:
        # CPU only
        load_options["device_map"] = "cpu"
        load_options["torch_dtype"] = torch.float32  # Pas de float16 sur CPU
    
    # Charger le modèle
    model = AutoModelForCausalLM.from_pretrained(model_name, **load_options)
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=model_cache_dir)
    
    return model, tokenizer, gpu_config
```

### 12.5 Cycle de vie complet des événements

Voici comment est géré le cycle de vie complet des événements liés à l'entraînement des modèles :

```typescript
// Dans ModelTrainingService
public async forceTrainModel(modelName: string): Promise<boolean> {
  if (!this.distilledModels.includes(modelName)) {
    throw new Error(`Le modèle ${modelName} n'est pas un modèle distillé valide`);
  }
  
  try {
    // Si le service Python API n'est pas disponible
    if (!this.pythonApiService) {
      this.logger.warn(`Service Python API non disponible pour l'entraînement du modèle ${modelName}`);
      return false;
    }
    
    // Vérifier si le service est disponible
    if (!this.pythonApiService.isAvailable()) {
      this.logger.warn(`API Python non disponible pour l'entraînement du modèle ${modelName}`);
      return false;
    }
    
    // Vérifier si un entraînement est déjà en cours
    if (this.isTrainingInProgress) {
      this.logger.warn(`Un entraînement est déjà en cours. Impossible de démarrer l'entraînement de ${modelName}`);
      return false;
    }
    
    // Marquer l'entraînement comme en cours
    this.isTrainingInProgress = true;
    
    // Émettre un événement de début d'entraînement
    if (this.eventBus) {
      this.eventBus.emit({
        type: RagKagEventType.MODEL_TRAINING_STARTED,
        source: 'ModelTrainingService',
        payload: { model: modelName }
      });
    }
    
    this.logger.info(`Démarrage de l'entraînement forcé pour le modèle ${modelName}`);
    
    // Appeler l'API Python pour démarrer l'entraînement
    const result = await this.pythonApiService.trainModel(modelName, {
      epochs: 5,
      batchSize: 32,
      saveToDisk: true
    });
    
    // Marquer l'entraînement comme terminé
    this.isTrainingInProgress = false;
    
    if (result.success) {
      this.logger.info(`Entraînement réussi pour le modèle ${modelName}`, { 
        accuracy: result.accuracy,
        loss: result.loss 
      });
      
      // Mettre à jour les statistiques
      this.trainingStats.set(modelName, {
        lastTraining: new Date(),
        examples: result.trainedExamples,
        accuracy: result.accuracy,
        loss: result.loss
      });
      
      // Stocker les résultats dans le graphe de connaissances
      this.storeTrainingResultInGraph(modelName, result, {
        lastTraining: new Date(),
        examples: result.trainedExamples,
        accuracy: result.accuracy,
        loss: result.loss
      });
      
      // Émettre un événement de fin d'entraînement
      if (this.eventBus) {
        this.eventBus.emit({
          type: RagKagEventType.MODEL_TRAINING_COMPLETED,
          source: 'ModelTrainingService',
          payload: { 
            model: modelName,
            result: {
              accuracy: result.accuracy,
              loss: result.loss,
              trainedExamples: result.trainedExamples
            }
          }
        });
      }
      
      return true;
    } else {
      this.logger.error(`Échec de l'entraînement pour le modèle ${modelName}`, { 
        message: result.message 
      });
      
      // Émettre un événement d'échec d'entraînement
      if (this.eventBus) {
        this.eventBus.emit({
          type: RagKagEventType.MODEL_TRAINING_FAILED,
          source: 'ModelTrainingService',
          payload: { 
            model: modelName,
            error: result.message
          }
        });
      }
      
      return false;
    }
  } catch (error) {
    // Marquer l'entraînement comme terminé en cas d'erreur
    this.isTrainingInProgress = false;
    
    this.logger.error(`Erreur lors de l'entraînement du modèle ${modelName}`, { 
      error: error.message 
    });
    
    // Émettre un événement d'erreur
    if (this.eventBus) {
      this.eventBus.emit({
        type: RagKagEventType.MODEL_TRAINING_FAILED,
        source: 'ModelTrainingService',
        payload: { 
          model: modelName,
          error: error.message
        }
      });
    }
    
    return false;
  }
}
```

L'abonnement aux événements peut être implémenté comme suit :

```typescript
// Dans un service de journalisation ou de surveillance
constructor(private readonly eventBus: EventBusService) {
  // S'abonner aux événements d'entraînement de modèle
  this.eventBus.subscribe(RagKagEventType.MODEL_TRAINING_STARTED, this.handleTrainingStarted.bind(this));
  this.eventBus.subscribe(RagKagEventType.MODEL_TRAINING_COMPLETED, this.handleTrainingCompleted.bind(this));
  this.eventBus.subscribe(RagKagEventType.MODEL_TRAINING_FAILED, this.handleTrainingFailed.bind(this));
}

private handleTrainingStarted(event: RagKagEvent): void {
  console.log(`Entraînement démarré pour le modèle ${event.payload.model}`);
  // Logique de notification, mise à jour de l'interface utilisateur, etc.
}

private handleTrainingCompleted(event: RagKagEvent): void {
  console.log(`Entraînement terminé pour le modèle ${event.payload.model}`);
  console.log(`Précision: ${event.payload.result.accuracy}, Perte: ${event.payload.result.loss}`);
  // Logique de notification, mise à jour de l'interface utilisateur, etc.
}

private handleTrainingFailed(event: RagKagEvent): void {
  console.error(`Échec de l'entraînement pour le modèle ${event.payload.model}`);
  console.error(`Erreur: ${event.payload.error}`);
  // Logique de notification, mise à jour de l'interface utilisateur, etc.
}
```

## 13. Intégration avec les Circuit Breakers

### 13.1 Endpoint de santé robuste

Le endpoint `/health` est crucial pour les circuit breakers côté NestJS. Il doit:
- Renvoyer rapidement (< 200ms)
- Indiquer l'état réel des ressources (GPU, modèles)
- Inclure des métriques essentielles

Voici un exemple d'implémentation optimisée pour les circuit breakers :

```python
@app.route('/health', methods=['GET'])
def health_check():
    """Endpoint de vérification santé pour circuit breaker"""
    # Éviter les opérations coûteuses qui pourraient ralentir la réponse
    gpu_status = model_manager.is_gpu_available_cached()
    
    response = {
        "status": "ok",
        "timestamp": int(time.time()),
        "version": "1.0.0",
        "models": model_manager.get_cached_model_list(),
        "gpu_available": gpu_status,
        "service_uptime": get_uptime_seconds()
    }
    
    # Facultatif: ajouter des métriques système basiques
    try:
        import psutil
        response["system"] = {
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            "memory_percent": psutil.virtual_memory().percent
        }
    except ImportError:
        pass
        
    return jsonify(response)
```

### 13.2 Gestion des erreurs compatibles

Pour que les circuit breakers fonctionnent correctement:
- Utiliser des codes HTTP appropriés (500 pour erreurs serveur, 400 pour requêtes invalides)
- Inclure des messages d'erreur structurés
- Ne pas bloquer les requêtes trop longtemps

Exemple de gestionnaire d'erreurs global :

```python
@app.errorhandler(Exception)
def handle_exception(e):
    """Gestionnaire d'erreur global compatible avec les circuit breakers"""
    app.logger.error(f"Erreur non gérée: {str(e)}", exc_info=True)
    
    if isinstance(e, HTTPException):
        # Erreurs HTTP connues (comme 404, 405, etc.)
        response = e.get_response()
        response.data = json.dumps({
            "success": False,
            "code": e.code,
            "name": e.name,
            "message": e.description,
        })
        response.content_type = "application/json"
        return response
    
    # Pour toutes les autres exceptions non gérées
    return jsonify({
        "success": False,
        "code": 500,
        "name": e.__class__.__name__,
        "message": str(e),
        "timestamp": int(time.time())
    }), 500
```

### 13.3 Configuration côté NestJS

Dans le service `ResilienceService` de NestJS, la configuration pour l'API Python devrait ressembler à :

```typescript
// Création du circuit breaker pour l'API Python
this.createCircuitBreaker('python-api', {
  failureThreshold: 3,        // 3 échecs consécutifs déclenchent l'ouverture
  resetTimeout: 30000,        // 30 secondes avant tentative de fermeture  
  fallbackResponse: null,     // Pas de fallback par défaut
  healthCheckInterval: 5000,  // Vérification toutes les 5 secondes en état ouvert
  healthCheckUrl: `${this.configService.get('PYTHON_API_URL')}/health`,
  timeout: 2000               // Timeout pour la vérification de santé
});
```

## 14. Synchronisation avec le Knowledge Graph

### 14.1 Endpoint de synchronisation

Implémenter un endpoint `/knowledge/sync` qui:
- Accepte les nœuds et relations du Knowledge Graph NestJS
- Persiste ces données localement
- Confirme quels éléments ont été synchronisés

Voici une implémentation recommandée :

```python
# src/knowledge_graph.py
class KnowledgeGraphSync:
    def __init__(self, db_path="./data/knowledge_graph.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.init_db()
        
    def init_db(self):
        """Initialise la base de données SQLite pour stocker le graphe"""
        import sqlite3
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Créer les tables si elles n'existent pas
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS nodes (
            id TEXT PRIMARY KEY,
            label TEXT NOT NULL,
            properties TEXT NOT NULL,
            last_updated INTEGER NOT NULL
        )
        ''')
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS relationships (
            id TEXT PRIMARY KEY,
            source_id TEXT NOT NULL,
            target_id TEXT NOT NULL,
            type TEXT NOT NULL,
            properties TEXT NOT NULL,
            weight REAL NOT NULL,
            last_updated INTEGER NOT NULL,
            FOREIGN KEY (source_id) REFERENCES nodes (id),
            FOREIGN KEY (target_id) REFERENCES nodes (id)
        )
        ''')
        
        conn.commit()
        conn.close()
    
    def sync_nodes(self, nodes):
        """Synchronise les nœuds du graphe de connaissances"""
        import sqlite3
        import json
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        synced_count = 0
        timestamp = int(time.time())
        
        for node in nodes:
            node_id = node.get('id')
            if not node_id:
                continue
                
            # Convertir les propriétés en JSON
            properties = json.dumps(node.get('properties', {}))
            
            # Vérifier si le nœud existe déjà
            cursor.execute("SELECT id FROM nodes WHERE id = ?", (node_id,))
            exists = cursor.fetchone()
            
            if exists:
                # Mettre à jour le nœud existant
                cursor.execute(
                    "UPDATE nodes SET label = ?, properties = ?, last_updated = ? WHERE id = ?",
                    (node.get('label', 'Node'), properties, timestamp, node_id)
                )
            else:
                # Insérer un nouveau nœud
                cursor.execute(
                    "INSERT INTO nodes (id, label, properties, last_updated) VALUES (?, ?, ?, ?)",
                    (node_id, node.get('label', 'Node'), properties, timestamp)
                )
            
            synced_count += 1
        
        conn.commit()
        conn.close()
        
        return synced_count
    
    def sync_relationships(self, relationships):
        """Synchronise les relations du graphe de connaissances"""
        import sqlite3
        import json
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        synced_count = 0
        timestamp = int(time.time())
        
        for rel in relationships:
            rel_id = rel.get('id')
            source_id = rel.get('sourceId')
            target_id = rel.get('targetId')
            
            if not (rel_id and source_id and target_id):
                continue
                
            # Convertir les propriétés en JSON
            properties = json.dumps(rel.get('properties', {}))
            
            # Vérifier si la relation existe déjà
            cursor.execute("SELECT id FROM relationships WHERE id = ?", (rel_id,))
            exists = cursor.fetchone()
            
            if exists:
                # Mettre à jour la relation existante
                cursor.execute(
                    "UPDATE relationships SET type = ?, properties = ?, weight = ?, last_updated = ? WHERE id = ?",
                    (rel.get('type', 'RELATED_TO'), properties, rel.get('weight', 1.0), timestamp, rel_id)
                )
            else:
                # Insérer une nouvelle relation
                cursor.execute(
                    "INSERT INTO relationships (id, source_id, target_id, type, properties, weight, last_updated) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (rel_id, source_id, target_id, rel.get('type', 'RELATED_TO'), properties, rel.get('weight', 1.0), timestamp)
                )
            
            synced_count += 1
        
        conn.commit()
        conn.close()
        
        return synced_count
    
    def get_knowledge_for_model(self, model_name):
        """Récupère les connaissances pertinentes pour un modèle spécifique"""
        import sqlite3
        import json
        
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Exemple: récupérer les nœuds liés au modèle spécifié
        cursor.execute('''
        SELECT n.* FROM nodes n
        JOIN relationships r ON n.id = r.target_id
        JOIN nodes model ON model.id = r.source_id
        WHERE model.properties LIKE ?
        ORDER BY r.weight DESC
        LIMIT 100
        ''', (f'%"name":"{model_name}"%',))
        
        nodes = []
        for row in cursor.fetchall():
            properties = json.loads(row['properties'])
            nodes.append({
                'id': row['id'],
                'label': row['label'],
                'properties': properties
            })
        
        conn.close()
        
        return {
            'nodes': nodes,
            'count': len(nodes)
        }

# Dans app.py
@app.route('/knowledge/sync', methods=['POST'])
def sync_knowledge():
    """Synchronise le Knowledge Graph local avec les données NestJS"""
    data = request.json
    
    if not data:
        return jsonify({"success": False, "message": "Données manquantes"}), 400
    
    kg_sync = KnowledgeGraphSync()
    
    nodes_count = 0
    relationships_count = 0
    
    # Synchroniser les nœuds
    if 'nodes' in data:
        nodes_count = kg_sync.sync_nodes(data['nodes'])
    
    # Synchroniser les relations
    if 'relationships' in data:
        relationships_count = kg_sync.sync_relationships(data['relationships'])
    
    return jsonify({
        "success": True,
        "nodes_synced": nodes_count,
        "relationships_synced": relationships_count,
        "timestamp": int(time.time())
    })
```

### 14.2 Utilisation pour le fine-tuning

Le Knowledge Graph améliore le fine-tuning des modèles:

```python
def prepare_training_data_with_knowledge(model_name, training_config):
    """Prépare des données d'entraînement enrichies avec KG"""
    kg_sync = KnowledgeGraphSync()
    
    # Récupérer les données standard
    base_training_data = get_base_training_data(training_config)
    
    # Enrichir avec les connaissances du graphe
    knowledge = kg_sync.get_knowledge_for_model(model_name)
    
    enhanced_data = []
    for example in base_training_data:
        # Format original de l'exemple
        query = example.get('input', '')
        expected_output = example.get('output', '')
        
        # Rechercher des connaissances pertinentes
        relevant_nodes = find_relevant_knowledge(query, knowledge['nodes'])
        
        if relevant_nodes:
            # Enrichir l'entrée avec les connaissances structurées
            knowledge_context = format_knowledge_context(relevant_nodes)
            enhanced_query = f"Knowledge: {knowledge_context}\n\nQuery: {query}"
            
            # Ajouter l'exemple enrichi
            enhanced_data.append({
                'input': enhanced_query,
                'output': expected_output,
                'enhanced': True
            })
        else:
            # Garder l'exemple original si pas de connaissance pertinente
            enhanced_data.append(example)
    
    return enhanced_data
```

### 14.3 Validation des sorties avec Knowledge Graph

Utiliser le Knowledge Graph pour valider les réponses générées:

```python
def validate_response_with_knowledge(query, generated_response, confidence_threshold=0.7):
    """Valide une réponse générée contre le graphe de connaissances"""
    kg_sync = KnowledgeGraphSync()
    
    # Extraire les faits de la réponse générée
    extracted_facts = extract_facts_from_text(generated_response)
    
    validation_results = []
    for fact in extracted_facts:
        # Vérifier le fait contre le graphe de connaissances
        kg_validation = validate_fact_against_kg(fact, kg_sync)
        
        validation_results.append({
            'fact': fact,
            'validated': kg_validation['validated'],
            'confidence': kg_validation['confidence'],
            'supporting_nodes': kg_validation['supporting_nodes']
        })
    
    # Calculer le score global de fiabilité
    validated_facts = [r for r in validation_results if r['validated'] and r['confidence'] >= confidence_threshold]
    validation_score = len(validated_facts) / max(1, len(extracted_facts))
    
    return {
        'validation_score': validation_score,
        'fact_validations': validation_results,
        'is_reliable': validation_score >= 0.8  # Seuil de fiabilité
    }
```

## 15. Système de callbacks pour le fine-tuning

### 15.1 Tâches asynchrones avec retours d'état

Plutôt qu'un modèle bloquant, l'API Python doit implémenter:
- Un système de tâches avec identifiants uniques
- Des callbacks HTTP vers NestJS à chaque étape clé
- Un stockage d'état local pour la reprise sur erreur

Implémentation recommandée:

```python
# src/training_manager.py
import threading
import time
import uuid
import json
import os
import requests
from datetime import datetime

class TrainingManager:
    def __init__(self, storage_dir="./data/training_tasks"):
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)
        self.active_tasks = {}
        self.load_persisted_tasks()
    
    def load_persisted_tasks(self):
        """Charge les tâches persistées depuis le disque"""
        try:
            task_files = [f for f in os.listdir(self.storage_dir) if f.endswith('.json')]
            for task_file in task_files:
                with open(os.path.join(self.storage_dir, task_file), 'r') as f:
                    task_data = json.load(f)
                    task_id = task_data.get('task_id')
                    if task_id and task_data.get('status') in ['PENDING', 'RUNNING']:
                        self.active_tasks[task_id] = task_data
        except Exception as e:
            print(f"Erreur lors du chargement des tâches: {str(e)}")
    
    def create_task(self, model_name, training_config, callback_url=None):
        """Crée une nouvelle tâche de fine-tuning"""
        task_id = f"finetune_{uuid.uuid4()}"
        
        task = {
            'task_id': task_id,
            'model_name': model_name,
            'training_config': training_config,
            'callback_url': callback_url,
            'status': 'PENDING',
            'progress': 0,
            'created_at': int(time.time()),
            'updated_at': int(time.time()),
            'logs': [],
            'result': None
        }
        
        # Persister la tâche
        self._persist_task(task)
        
        # Ajouter à la liste des tâches actives
        self.active_tasks[task_id] = task
        
        # Démarrer la tâche dans un thread séparé
        threading.Thread(
            target=self._run_training_task,
            args=(task_id,),
            daemon=True
        ).start()
        
        return task_id
    
    def get_task_status(self, task_id):
        """Récupère le statut d'une tâche"""
        if task_id in self.active_tasks:
            return self.active_tasks[task_id]
        
        # Si pas en mémoire, essayer de charger depuis le disque
        task_path = os.path.join(self.storage_dir, f"{task_id}.json")
        if os.path.exists(task_path):
            with open(task_path, 'r') as f:
                return json.load(f)
        
        return None
    
    def _persist_task(self, task):
        """Persiste une tâche sur le disque"""
        task_path = os.path.join(self.storage_dir, f"{task['task_id']}.json")
        with open(task_path, 'w') as f:
            json.dump(task, f, indent=2)
    
    def _send_callback(self, task, event_type="progress"):
        """Envoie un callback HTTP vers NestJS"""
        if not task.get('callback_url'):
            return False
        
        callback_data = {
            'task_id': task['task_id'],
            'model_name': task['model_name'],
            'status': task['status'],
            'progress': task['progress'],
            'event_type': event_type,
            'timestamp': int(time.time())
        }
        
        # Ajouter le résultat si disponible
        if task.get('result') and event_type == "completed":
            callback_data['result'] = task['result']
        
        # Ajouter l'erreur si disponible
        if task.get('error') and event_type == "failed":
            callback_data['error'] = task['error']
        
        try:
            response = requests.post(
                task['callback_url'],
                json=callback_data,
                headers={'Content-Type': 'application/json'},
                timeout=10
            )
            return response.status_code >= 200 and response.status_code < 300
        except Exception as e:
            print(f"Erreur lors de l'envoi du callback: {str(e)}")
            return False
    
    def _update_task_status(self, task_id, status, progress=None, message=None, result=None, error=None):
        """Met à jour le statut d'une tâche et envoie un callback"""
        if task_id not in self.active_tasks:
            return False
        
        task = self.active_tasks[task_id]
        task['status'] = status
        task['updated_at'] = int(time.time())
        
        if progress is not None:
            task['progress'] = progress
        
        if message:
            task['logs'].append({
                'timestamp': int(time.time()),
                'message': message
            })
        
        if result:
            task['result'] = result
        
        if error:
            task['error'] = error
        
        # Persister la mise à jour
        self._persist_task(task)
        
        # Envoyer un callback si URL disponible
        event_type = "progress"
        if status == "COMPLETED":
            event_type = "completed"
        elif status == "FAILED":
            event_type = "failed"
        
        self._send_callback(task, event_type)
        
        return True
    
    def _run_training_task(self, task_id):
        """Exécute une tâche de fine-tuning"""
        if task_id not in self.active_tasks:
            return
        
        task = self.active_tasks[task_id]
        model_name = task['model_name']
        config = task['training_config']
        
        try:
            # Mettre à jour le statut
            self._update_task_status(
                task_id, 
                "RUNNING", 
                progress=0, 
                message=f"Démarrage du fine-tuning pour {model_name}"
            )
            
            # Charger le modèle et les données
            self._update_task_status(
                task_id, 
                "RUNNING", 
                progress=10, 
                message="Chargement du modèle et préparation des données"
            )
            
            # TODO: Implémenter la logique réelle de fine-tuning
            # Simuler les étapes pour l'exemple
            for step in range(1, 11):
                time.sleep(1)  # Simuler le travail
                progress = step * 10
                self._update_task_status(
                    task_id, 
                    "RUNNING", 
                    progress=progress, 
                    message=f"Étape {step}/10: Entraînement en cours ({progress}%)"
                )
            
            # Simuler un résultat de fine-tuning réussi
            result = {
                "accuracy": 0.92,
                "loss": 0.08,
                "training_time": time.time() - task['created_at'],
                "model_path": f"models/{model_name}_finetuned_{int(time.time())}"
            }
            
            # Mettre à jour avec succès
            self._update_task_status(
                task_id, 
                "COMPLETED", 
                progress=100, 
                message="Fine-tuning terminé avec succès",
                result=result
            )
            
        except Exception as e:
            error_message = str(e)
            print(f"Erreur lors du fine-tuning {task_id}: {error_message}")
            
            # Mettre à jour avec échec
            self._update_task_status(
                task_id, 
                "FAILED", 
                message=f"Erreur: {error_message}",
                error=error_message
            )

# À ajouter dans app.py
training_manager = TrainingManager()

@app.route('/finetune/start', methods=['POST'])
def start_finetune():
    """Démarre un fine-tuning asynchrone avec callbacks"""
    data = request.json
    
    if not data or 'model' not in data:
        return jsonify({
            "success": False,
            "message": "Le paramètre 'model' est requis"
        }), 400
    
    model_name = data['model']
    callback_url = data.get('callback_url')
    training_config = data.get('config', {})
    
    # Vérifier que le modèle est valide
    if model_name not in model_manager.supported_models:
        return jsonify({
            "success": False,
            "message": f"Modèle non supporté: {model_name}"
        }), 400
    
    # Créer la tâche de fine-tuning
    task_id = training_manager.create_task(
        model_name=model_name,
        training_config=training_config,
        callback_url=callback_url
    )
    
    return jsonify({
        "success": True,
        "task_id": task_id,
        "model": model_name,
        "status": "PENDING"
    })

@app.route('/finetune/status/<task_id>', methods=['GET'])
def get_finetune_status(task_id):
    """Récupère le statut d'une tâche de fine-tuning"""
    task = training_manager.get_task_status(task_id)
    
    if not task:
        return jsonify({
            "success": False,
            "message": f"Tâche inconnue: {task_id}"
        }), 404
    
    return jsonify({
        "success": True,
        "task_id": task_id,
        "status": task['status'],
        "progress": task['progress'],
        "model": task['model_name'],
        "logs": task.get('logs', [])[-5:],  # 5 derniers logs
        "created_at": task['created_at'],
        "updated_at": task['updated_at'],
        "result": task.get('result'),
        "error": task.get('error')
    })
```

### 15.2 Structure des données de callback

Pour chaque notification de progression, l'API Python enverra des données structurées à NestJS:

```json
{
  "task_id": "finetune_123456789",
  "model_name": "phi-3-mini",
  "status": "RUNNING",
  "progress": 45,
  "event_type": "progress",
  "timestamp": 1678271543,
  "logs": [
    {"timestamp": 1678271530, "message": "Démarrage du fine-tuning"}
  ]
}
```

Pour les callbacks de fin de tâche:

```json
{
  "task_id": "finetune_123456789",
  "model_name": "phi-3-mini",
  "status": "COMPLETED",
  "progress": 100,
  "event_type": "completed",
  "timestamp": 1678271543,
  "result": {
    "accuracy": 0.92,
    "loss": 0.08,
    "training_time": 125.3,
    "model_path": "models/phi-3-mini_finetuned_1678271543"
  }
}
```

Pour les callbacks d'erreur:

```json
{
  "task_id": "finetune_123456789",
  "model_name": "phi-3-mini",
  "status": "FAILED",
  "progress": 35,
  "event_type": "failed",
  "timestamp": 1678271543,
  "error": "GPU out of memory: CUDA error: out of memory"
}
```

### 15.3 Implémentation côté NestJS

Exemple de méthode de callback dans un contrôleur NestJS:

```typescript
@Post('model-training/callback')
async handleTrainingCallback(@Body() callbackData: any) {
  this.logger.log(`Callback reçu: ${callbackData.event_type} pour la tâche ${callbackData.task_id}`);
  
  // Mettre à jour l'état dans la base de données
  await this.modelTrainingService.updateTrainingStatus(
    callbackData.task_id,
    callbackData.model_name,
    callbackData.status,
    callbackData.progress
  );
  
  // Si l'entraînement est terminé, stocker les résultats dans le graphe de connaissances
  if (callbackData.status === 'COMPLETED' && callbackData.result) {
    await this.modelTrainingService.storeTrainingResultInGraph(
      callbackData.model_name,
      callbackData.result,
      {
        lastTraining: new Date(),
        examples: callbackData.result.examples || 0,
        accuracy: callbackData.result.accuracy,
        loss: callbackData.result.loss
      }
    );
    
    // Émettre un événement
    this.eventBus.emit({
      type: RagKagEventType.MODEL_TRAINING_COMPLETED,
      source: 'PythonApiService',
      payload: {
        model: callbackData.model_name,
        result: callbackData.result
      }
    });
  }
  
  // Si l'entraînement a échoué
  if (callbackData.status === 'FAILED') {
    this.eventBus.emit({
      type: RagKagEventType.MODEL_TRAINING_FAILED,
      source: 'PythonApiService',
      payload: {
        model: callbackData.model_name,
        error: callbackData.error
      }
    });
  }
  
  return { success: true };
}
```

## 16. Intégration complète RAG-KAG

L'API Python ne doit pas être une simple couche de service, mais un participant actif dans l'architecture RAG-KAG.

### 16.1 Architecture RAG (Retrieval Augmented Generation)

#### 16.1.1 Endpoints spécifiques RAG

```python
@app.route('/rag/documents/store', methods=['POST'])
def store_documents():
    """Stocke des documents pour la récupération contextuelle RAG"""
    data = request.json
    
    if not data or 'documents' not in data:
        return jsonify({
            "success": False,
            "message": "Paramètre 'documents' manquant"
        }), 400
    
    documents = data['documents']
    collection_name = data.get('collection', 'default')
    
    # Stocker et vectoriser les documents
    result = rag_manager.store_documents(documents, collection_name)
    
    return jsonify({
        "success": True,
        "stored_count": result['stored_count'],
        "indexing_time": result['indexing_time'],
        "collection": collection_name
    })

@app.route('/rag/retrieve', methods=['POST'])
def retrieve_documents():
    """Récupère les documents pertinents pour une requête"""
    data = request.json
    
    if not data or 'query' not in data:
        return jsonify({
            "success": False,
            "message": "Paramètre 'query' manquant"
        }), 400
    
    query = data['query']
    collection_name = data.get('collection', 'default')
    top_k = data.get('top_k', 5)
    
    # Récupérer les documents pertinents
    results = rag_manager.retrieve_documents(query, collection_name, top_k)
    
    return jsonify({
        "success": True,
        "query": query,
        "documents": results
    })

@app.route('/rag/generate', methods=['POST'])
def rag_generate():
    """Génère une réponse basée sur les documents récupérés"""
    data = request.json
    
    if not data or 'query' not in data:
        return jsonify({
            "success": False,
            "message": "Paramètre 'query' manquant"
        }), 400
    
    query = data['query']
    model_name = data.get('model', 'phi-3-mini')
    collection_name = data.get('collection', 'default')
    top_k = data.get('top_k', 5)
    
    # Récupérer les documents pertinents
    documents = rag_manager.retrieve_documents(query, collection_name, top_k)
    
    # Générer une réponse avec le contexte des documents
    response = model_manager.rag_augmented_generate(
        query=query,
        context_documents=documents,
        model_name=model_name
    )
    
    return jsonify({
        "success": True,
        "query": query,
        "answer": response['text'],
        "model": model_name,
        "documents": [doc['metadata'] for doc in documents],
        "generation_time": response.get('generationTime', 0)
    })
```

#### 16.1.2 Classe `RAGManager` pour gérer les documents et la récupération

```python
# src/rag_manager.py
import time
import os
import json
import numpy as np
from typing import List, Dict, Any

class RAGManager:
    def __init__(self, vectors_dir="./data/vectors"):
        self.vectors_dir = vectors_dir
        os.makedirs(vectors_dir, exist_ok=True)
        self.collections = self._load_collections()
        
    def _load_collections(self):
        """Charge les collections existantes"""
        collections = {}
        
        collection_dirs = [d for d in os.listdir(self.vectors_dir) 
                          if os.path.isdir(os.path.join(self.vectors_dir, d))]
        
        for collection_name in collection_dirs:
            collection_path = os.path.join(self.vectors_dir, collection_name)
            metadata_path = os.path.join(collection_path, "metadata.json")
            vectors_path = os.path.join(collection_path, "vectors.npy")
            
            if os.path.exists(metadata_path) and os.path.exists(vectors_path):
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    
                    vectors = np.load(vectors_path)
                    
                    collections[collection_name] = {
                        'metadata': metadata,
                        'vectors': vectors
                    }
                except Exception as e:
                    print(f"Erreur lors du chargement de la collection {collection_name}: {str(e)}")
        
        return collections
    
    def store_documents(self, documents: List[Dict[str, Any]], collection_name: str = "default"):
        """Stocke et vectorise des documents pour la récupération"""
        start_time = time.time()
        
        # Créer le répertoire de la collection si nécessaire
        collection_dir = os.path.join(self.vectors_dir, collection_name)
        os.makedirs(collection_dir, exist_ok=True)
        
        # Charger les documents et métadonnées existants si la collection existe
        existing_metadata = []
        if collection_name in self.collections:
            existing_metadata = self.collections[collection_name]['metadata']
        
        # Traiter les nouveaux documents
        new_metadata = []
        new_vectors = []
        
        for doc in documents:
            # Extraire le contenu et les métadonnées
            content = doc.get('content', '')
            metadata = {
                'id': doc.get('id', f"doc_{len(existing_metadata) + len(new_metadata)}"),
                'title': doc.get('title', ''),
                'source': doc.get('source', ''),
                'timestamp': doc.get('timestamp', int(time.time())),
                'content': content[:200] + "..." if len(content) > 200 else content  # Résumé du contenu
            }
            
            # Vectoriser le document
            # TODO: Utiliser une vraie vectorisation avec sentence-transformers ou similaire
            # Simulons des vecteurs pour l'exemple
            vector = np.random.random(384)  # 384 dimensions comme exemple
            
            new_metadata.append(metadata)
            new_vectors.append(vector)
        
        # Combiner avec les données existantes
        all_metadata = existing_metadata + new_metadata
        
        if collection_name in self.collections:
            all_vectors = np.vstack([
                self.collections[collection_name]['vectors'], 
                np.array(new_vectors)
            ])
        else:
            all_vectors = np.array(new_vectors)
        
        # Sauvegarder
        metadata_path = os.path.join(collection_dir, "metadata.json")
        vectors_path = os.path.join(collection_dir, "vectors.npy")
        
        with open(metadata_path, 'w') as f:
            json.dump(all_metadata, f, indent=2)
        
        np.save(vectors_path, all_vectors)
        
        # Mettre à jour la collection en mémoire
        self.collections[collection_name] = {
            'metadata': all_metadata,
            'vectors': all_vectors
        }
        
        indexing_time = time.time() - start_time
        
        return {
            'stored_count': len(new_metadata),
            'total_count': len(all_metadata),
            'indexing_time': indexing_time
        }
    
    def retrieve_documents(self, query: str, collection_name: str = "default", top_k: int = 5):
        """Récupère les documents les plus pertinents pour une requête"""
        if collection_name not in self.collections:
            return []
        
        collection = self.collections[collection_name]
        
        # TODO: Vectoriser la requête avec le même modèle utilisé pour les documents
        query_vector = np.random.random(384)  # Simulation
        
        # Calculer les similarités (produit scalaire normalisé = similarité cosinus)
        similarities = np.dot(collection['vectors'], query_vector) / (
            np.linalg.norm(collection['vectors'], axis=1) * np.linalg.norm(query_vector)
        )
        
        # Trier et récupérer les documents les plus similaires
        most_similar_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in most_similar_indices:
            metadata = collection['metadata'][idx]
            results.append({
                'metadata': metadata,
                'similarity': float(similarities[idx])
            })
        
        return results
```

### 16.2 Architecture KAG (Knowledge Augmented Generation)

#### 16.2.1 Endpoints spécifiques KAG

```python
@app.route('/kag/generate', methods=['POST'])
def kag_generate():
    """Génère une réponse basée sur le graphe de connaissances"""
    data = request.json
    
    if not data or 'query' not in data:
        return jsonify({
            "success": False,
            "message": "Paramètre 'query' manquant"
        }), 400
    
    query = data['query']
    model_name = data.get('model', 'phi-3-mini')
    
    # Extraire les connaissances pertinentes du graphe
    kg_sync = KnowledgeGraphSync()
    knowledge = kg_sync.get_relevant_knowledge(query)
    
    # Générer une réponse avec les connaissances structurées
    response = model_manager.kag_enhanced_generate(
        query=query,
        knowledge_graph_data=knowledge,
        model_name=model_name
    )
    
    return jsonify({
        "success": True,
        "query": query,
        "answer": response['text'],
        "model": model_name,
        "knowledge_nodes": len(knowledge['nodes']),
        "knowledge_relationships": len(knowledge.get('relationships', [])),
        "generation_time": response.get('generationTime', 0)
    })

@app.route('/kag/verify', methods=['POST'])
def verify_with_knowledge():
    """Vérifie une réponse générée contre le graphe de connaissances"""
    data = request.json
    
    if not data or 'text' not in data:
        return jsonify({
            "success": False,
            "message": "Paramètre 'text' manquant"
        }), 400
    
    text = data['text']
    query = data.get('query', '')
    confidence_threshold = data.get('confidence_threshold', 0.7)
    
    # Valider le texte contre le graphe de connaissances
    validation_result = validate_response_with_knowledge(
        query=query,
        generated_response=text,
        confidence_threshold=confidence_threshold
    )
    
    return jsonify({
        "success": True,
        "validation_score": validation_result['validation_score'],
        "is_reliable": validation_result['is_reliable'],
        "validated_facts": len([f for f in validation_result['fact_validations'] if f['validated']]),
        "total_facts": len(validation_result['fact_validations']),
        "fact_validations": validation_result['fact_validations']
    })
```

### 16.3 Architecture hybride RAG-KAG

Combinaison des deux approches pour une génération optimale :

```python
@app.route('/hybrid/generate', methods=['POST'])
def hybrid_generate():
    """Génère une réponse en combinant RAG et KAG"""
    data = request.json
    
    if not data or 'query' not in data:
        return jsonify({
            "success": False,
            "message": "Paramètre 'query' manquant"
        }), 400
    
    query = data['query']
    model_name = data.get('model', 'phi-3-mini')
    collection_name = data.get('collection', 'default')
    top_k = data.get('top_k', 5)
    
    # 1. Récupérer les documents pertinents (RAG)
    rag_documents = rag_manager.retrieve_documents(query, collection_name, top_k)
    
    # 2. Extraire les connaissances pertinentes du graphe (KAG)
    kg_sync = KnowledgeGraphSync()
    knowledge = kg_sync.get_relevant_knowledge(query)
    
    # 3. Générer une réponse hybride
    hybrid_context = {
        "documents": rag_documents,
        "knowledge": knowledge
    }
    
    response = model_manager.hybrid_augmented_generate(
        query=query,
        hybrid_context=hybrid_context,
        model_name=model_name
    )
    
    # 4. Valider la réponse contre le graphe de connaissances
    validation_result = validate_response_with_knowledge(
        query=query,
        generated_response=response['text'],
        confidence_threshold=0.7
    )
    
    return jsonify({
        "success": True,
        "query": query,
        "answer": response['text'],
        "model": model_name,
        "rag_documents": len(rag_documents),
        "kg_nodes": len(knowledge['nodes']),
        "validation_score": validation_result['validation_score'],
        "is_reliable": validation_result['is_reliable'],
        "generation_time": response.get('generationTime', 0)
    })
```

### 16.4 Monitoring spécifique à RAG-KAG

Endpoints pour surveiller spécifiquement les performances RAG-KAG :

```python
@app.route('/monitoring/rag/metrics', methods=['GET'])
def get_rag_metrics():
    """Retourne les métriques RAG"""
    collection_name = request.args.get('collection', 'default')
    
    if collection_name not in rag_manager.collections:
        return jsonify({
            "success": False,
            "message": f"Collection {collection_name} non trouvée"
        }), 404
    
    collection = rag_manager.collections[collection_name]
    
    return jsonify({
        "success": True,
        "collection": collection_name,
        "document_count": len(collection['metadata']),
        "vector_dimensions": collection['vectors'].shape[1] if len(collection['vectors']) > 0 else 0,
        "storage_size_kb": os.path.getsize(os.path.join(rag_manager.vectors_dir, collection_name, "vectors.npy")) / 1024,
        "sources": list(set(doc.get('source', '') for doc in collection['metadata']))
    })

@app.route('/monitoring/kag/metrics', methods=['GET'])
def get_kag_metrics():
    """Retourne les métriques KAG"""
    kg_sync = KnowledgeGraphSync()
    metrics = kg_sync.get_metrics()
    
    return jsonify({
        "success": True,
        "node_count": metrics['node_count'],
        "relationship_count": metrics['relationship_count'],
        "node_types": metrics['node_types'],
        "relationship_types": metrics['relationship_types'],
        "last_sync": metrics['last_sync'],
        "storage_size_kb": metrics['storage_size_kb']
    })
```

### 16.5 Traçabilité des requêtes dans l'architecture RAG-KAG

Pour le suivi de bout en bout des requêtes :

```python
@app.before_request
def add_trace_id():
    """Ajoute un ID de traçage à chaque requête"""
    trace_id = request.headers.get('X-Trace-ID')
    if not trace_id:
        trace_id = f"trace_{uuid.uuid4()}"
    
    # Stocker dans le contexte global pour cette requête
    g.trace_id = trace_id
    g.start_time = time.time()

@app.after_request
def add_trace_header(response):
    """Ajoute l'ID de traçage à la réponse"""
    trace_id = getattr(g, 'trace_id', None)
    if trace_id:
        response.headers['X-Trace-ID'] = trace_id
        
        # Calculer le temps de traitement
        start_time = getattr(g, 'start_time', None)
        if start_time:
            processing_time = time.time() - start_time
            response.headers['X-Processing-Time'] = str(processing_time)
    
    return response
```

## 17. Frameworks ML et intégration

Cette section détaille l'utilisation des frameworks ML spécifiques dans notre API Python pour l'architecture RAG-KAG.

### 17.1 PyTorch - Framework principal

PyTorch est le framework de base pour tous nos modèles en raison de sa flexibilité et de ses performances.

```python
# Configuration PyTorch optimisée
import torch

def configure_pytorch_environment():
    """Configure l'environnement PyTorch de manière optimale"""
    # Performance sur CPU
    torch.set_num_threads(8)  # Ajuster en fonction du nombre de cœurs disponibles
    
    # Mode d'inférence
    torch.set_grad_enabled(False)  # Désactiver le calcul de gradient pour l'inférence
    
    # Configuration CUDA/GPU
    if torch.cuda.is_available():
        # Optimisation mémoire GPU
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        
        # Afficher les informations GPU
        device_count = torch.cuda.device_count()
        for i in range(device_count):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")
        
        return torch.device("cuda")
    else:
        return torch.device("cpu")

# Utilisation pour charger des modèles avec mixed precision
def load_model_with_mixed_precision(model_path, device):
    """Charge un modèle avec mixed precision pour optimiser les performances"""
    # Définir le type de données en fonction du device
    dtype = torch.float16 if device.type == 'cuda' else torch.float32
    
    # Charger le modèle avec le bon type de données
    with torch.cuda.amp.autocast(enabled=device.type == 'cuda'):
        model = torch.load(model_path, map_location=device)
        model = model.to(device).to(dtype)
        model.eval()  # Passer en mode évaluation
    
    return model
```

### 17.2 NumPy - Traitement vectoriel efficace

NumPy est utilisé pour toutes les opérations vectorielles de base, en particulier dans le contexte RAG.

```python
# Optimisations NumPy pour la similarité cosinus
import numpy as np

class VectorStore:
    def __init__(self, dimensions=768):
        self.vectors = None  # Matrice vectorielle de forme (n_docs, dimensions)
        self.ids = []  # IDs correspondant aux vecteurs
        self.dimensions = dimensions
    
    def add_vectors(self, new_vectors, new_ids):
        """Ajoute des vecteurs de manière efficace"""
        new_vectors = np.asarray(new_vectors, dtype=np.float32)
        
        # Normaliser les vecteurs (pour la similarité cosinus)
        new_vectors_normalized = new_vectors / np.linalg.norm(new_vectors, axis=1, keepdims=True)
        
        if self.vectors is None:
            self.vectors = new_vectors_normalized
        else:
            self.vectors = np.vstack([self.vectors, new_vectors_normalized])
        
        self.ids.extend(new_ids)
    
    def find_similar(self, query_vector, top_k=5):
        """Recherche efficace des vecteurs les plus similaires"""
        query_vector = np.asarray(query_vector, dtype=np.float32)
        query_norm = query_vector / np.linalg.norm(query_vector)
        
        # Calcul de similarité cosinus optimisé avec numpy
        similarities = np.dot(self.vectors, query_norm)
        
        # Récupérer les indices des top_k plus grandes similarités
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = [
            {"id": self.ids[idx], "similarity": float(similarities[idx])}
            for idx in top_indices
        ]
        
        return results
```

### 17.3 Pandas - Traitement de données structurées

Pandas est utilisé pour la préparation des données, la validation et l'analyse des résultats.

```python
# Utilisation de Pandas pour la préparation des données d'entraînement
import pandas as pd
from typing import Dict, List

def prepare_training_data(raw_data: List[Dict], model_name: str) -> pd.DataFrame:
    """Prépare les données d'entraînement à partir de données brutes"""
    # Convertir en DataFrame
    df = pd.DataFrame(raw_data)
    
    # Filtrer les données nulles ou vides
    df = df.dropna(subset=['input', 'output'])
    df = df[df['input'].str.strip().str.len() > 0]
    df = df[df['output'].str.strip().str.len() > 0]
    
    # Ajouter des colonnes utiles
    df['input_length'] = df['input'].str.len()
    df['output_length'] = df['output'].str.len()
    df['model'] = model_name
    df['created_at'] = pd.Timestamp.now()
    
    # Statistiques sur les données
    print(f"Nombre d'exemples: {len(df)}")
    print(f"Longueur d'entrée moyenne: {df['input_length'].mean():.1f}")
    print(f"Longueur de sortie moyenne: {df['output_length'].mean():.1f}")
    
    return df

def analyze_training_results(metrics_history: List[Dict]) -> pd.DataFrame:
    """Analyse les résultats d'entraînement avec Pandas"""
    metrics_df = pd.DataFrame(metrics_history)
    
    # Convertir les timestamps en datetime
    if 'timestamp' in metrics_df.columns:
        metrics_df['timestamp'] = pd.to_datetime(metrics_df['timestamp'], unit='s')
    
    # Calculer les statistiques glissantes
    if len(metrics_df) > 5:
        metrics_df['loss_ma5'] = metrics_df['loss'].rolling(5).mean()
        metrics_df['accuracy_ma5'] = metrics_df['accuracy'].rolling(5).mean()
    
    # Identifier les tendances
    if len(metrics_df) > 1:
        metrics_df['loss_trend'] = metrics_df['loss'].diff().apply(lambda x: 'improving' if x < 0 else 'worsening')
    
    return metrics_df
```

### 17.4 Hugging Face - Modèles et tokenizers

Hugging Face est notre principale source de modèles et d'outils pour la gestion des LLMs.

```python
# Utilisation des modèles Hugging Face pour RAG-KAG
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

class HuggingFaceModelHandler:
    def __init__(self, cache_dir="./models"):
        self.cache_dir = cache_dir
        self.loaded_models = {}
        self.loaded_tokenizers = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def load_model(self, model_name, quantization="4bit"):
        """Charge un modèle Hugging Face avec quantification optimale"""
        if model_name in self.loaded_models:
            return self.loaded_models[model_name], self.loaded_tokenizers[model_name]
        
        # Configuration de quantification pour réduire l'empreinte mémoire
        if quantization == "4bit" and self.device.type == 'cuda':
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True
            )
        elif quantization == "8bit" and self.device.type == 'cuda':
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True
            )
        else:
            quantization_config = None
        
        # Charger le modèle avec la configuration appropriée
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                cache_dir=os.path.join(self.cache_dir, model_name),
                device_map="auto" if self.device.type == 'cuda' else None,
                quantization_config=quantization_config,
                torch_dtype=torch.float16 if self.device.type == 'cuda' else torch.float32
            )
            
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=os.path.join(self.cache_dir, model_name)
            )
            
            # S'assurer que le tokenizer a un pad_token
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            self.loaded_models[model_name] = model
            self.loaded_tokenizers[model_name] = tokenizer
            
            return model, tokenizer
            
        except Exception as e:
            print(f"Erreur lors du chargement du modèle {model_name}: {str(e)}")
            return None, None
    
    def generate_rag_kag_response(self, prompt, model_name="mistral-7b-instruct-v0.2", 
                                context_docs=None, knowledge_nodes=None, 
                                temperature=0.7, max_tokens=512):
        """Génère une réponse en utilisant RAG et KAG"""
        # Charger le modèle si nécessaire
        model, tokenizer = self.load_model(model_name)
        if not model or not tokenizer:
            return {"error": f"Impossible de charger le modèle {model_name}"}
        
        # Construire le prompt enrichi
        enriched_prompt = prompt
        
        # Ajouter le contexte RAG si disponible
        if context_docs:
            rag_context = "\n\n".join([f"Document {i+1}: {doc.get('content', '')}" 
                                       for i, doc in enumerate(context_docs)])
            enriched_prompt = f"Contexte:\n{rag_context}\n\nQuestion: {prompt}"
        
        # Ajouter les connaissances KAG si disponibles
        if knowledge_nodes:
            kag_context = "\n".join([f"- {node.get('label', '')}: {node.get('properties', {}).get('description', '')}"
                                    for node in knowledge_nodes])
            enriched_prompt = f"{enriched_prompt}\n\nConnaissances pertinentes:\n{kag_context}"
        
        # Générer la réponse
        input_ids = tokenizer.encode(enriched_prompt, return_tensors="pt").to(self.device)
        
        # Configuration de génération avec les paramètres fournis
        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=0.9,
                do_sample=temperature > 0,
                pad_token_id=tokenizer.pad_token_id
            )
        
        # Décoder la sortie et nettoyer
        full_output = tokenizer.decode(output[0], skip_special_tokens=True)
        
        # Extraire uniquement la réponse (sans le prompt)
        response = full_output[len(enriched_prompt):].strip()
        
        return {
            "text": response,
            "model": model_name,
            "tokensUsed": len(output[0]),
            "generationTime": 0.0  # À remplacer par le temps réel
        }
```

### 17.5 Intégration avec sentence-transformers pour l'embedding

Pour les embeddings vectoriels RAG :

```python
# Utilisation de sentence-transformers pour RAG
from sentence_transformers import SentenceTransformer
import numpy as np
import torch

class RAGVectorizer:
    def __init__(self, model_name="all-MiniLM-L6-v2", cache_dir="./models"):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._load_model()
    
    def _load_model(self):
        """Charge le modèle d'embedding"""
        try:
            self.model = SentenceTransformer(
                self.model_name, 
                cache_folder=os.path.join(self.cache_dir, "sentence_transformers"),
                device=self.device
            )
            print(f"Modèle d'embedding {self.model_name} chargé sur {self.device}")
        except Exception as e:
            print(f"Erreur lors du chargement du modèle d'embedding: {str(e)}")
    
    def encode_documents(self, texts, batch_size=32, show_progress=True):
        """Encode une liste de textes en vecteurs"""
        if not self.model:
            raise ValueError("Le modèle d'embedding n'est pas chargé")
            
        # Normalisation de texte basique
        normalized_texts = [text.replace("\n", " ").strip() for text in texts]
        
        # Encodage par lots pour économiser la mémoire
        embeddings = self.model.encode(
            normalized_texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True  # Normalisation L2 pour la similarité cosinus
        )
        
        return embeddings
    
    def encode_query(self, query):
        """Encode une requête en vecteur"""
        if not self.model:
            raise ValueError("Le modèle d'embedding n'est pas chargé")
            
        query_embedding = self.model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        return query_embedding
```

### 17.6 Optimisation des performances et de l'utilisation GPU

```python
# Optimisation des performances GPU
import gc
import torch
import psutil
import os

def optimize_gpu_memory():
    """Libère la mémoire GPU non utilisée"""
    gc.collect()
    torch.cuda.empty_cache()
    
    # Afficher l'utilisation actuelle
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            print(f"GPU {i}: Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")

def get_system_resources():
    """Récupère les ressources système disponibles"""
    resources = {
        "cpu_percent": psutil.cpu_percent(),
        "memory_percent": psutil.virtual_memory().percent,
        "memory_available_gb": psutil.virtual_memory().available / 1024**3
    }
    
    if torch.cuda.is_available():
        resources["gpu_count"] = torch.cuda.device_count()
        resources["gpu_info"] = []
        
        for i in range(resources["gpu_count"]):
            resources["gpu_info"].append({
                "name": torch.cuda.get_device_name(i),
                "memory_allocated_gb": torch.cuda.memory_allocated(i) / 1024**3,
                "memory_reserved_gb": torch.cuda.memory_reserved(i) / 1024**3
            })
    
    return resources

def adaptive_batch_size(available_memory_gb, model_size_gb, overhead_factor=1.5):
    """Calcule une taille de batch adaptative en fonction de la mémoire disponible"""
    # Estimation de la mémoire nécessaire par élément de batch
    memory_per_item = model_size_gb * overhead_factor
    
    # Calcul du batch size maximal
    max_batch_size = int(available_memory_gb / memory_per_item)
    
    # Assurer un minimum de 1 et un maximum raisonnable
    batch_size = max(1, min(32, max_batch_size))
    
    return batch_size
```

### 17.7 Installation des dépendances

Voici le fichier `requirements.txt` complet pour toutes ces bibliothèques:

```
# Base Flask et utilitaires
flask==2.3.3
flask-cors==4.0.0
python-dotenv==1.0.0
gunicorn==21.2.0
pydantic==2.5.0
requests==2.31.0

# Gestion des tâches asynchrones
celery==5.3.4
redis==5.0.1

# Machine Learning - PyTorch et Hugging Face
torch==2.1.0
transformers==4.36.0
accelerate==0.25.0
bitsandbytes==0.41.1
sentence-transformers==2.2.2
optimum==1.14.0
peft==0.6.0

# Traitement de données
numpy==1.26.0
pandas==2.1.1
scikit-learn==1.3.2
matplotlib==3.8.0
scipy==1.11.3

# Traitement de texte
nltk==3.8.1
spacy==3.7.1

# Analyse d'images (si nécessaire)
pillow==10.1.0

# Monitoring et performances
prometheus-client==0.17.1
psutil==5.9.5
py-spy==0.3.14

# Logging et tracing
opentelemetry-api==1.20.0
opentelemetry-sdk==1.20.0
opentelemetry-exporter-otlp==1.20.0

# Sécurité
python-jose==3.3.0
passlib==1.7.4
```

### 17.8 Adaptation aux modèles maison

Pour l'intégration de modèles maison:

```python
# Intégration pour modèles personnalisés
class CustomModelAdapter:
    """Adaptateur pour modèles personnalisés non-HuggingFace"""
    
    def __init__(self, model_dir="./custom_models"):
        self.model_dir = model_dir
        self.models = {}
        os.makedirs(model_dir, exist_ok=True)
    
    def register_model(self, model_name, model_class, model_config):
        """Enregistre un modèle personnalisé"""
        self.models[model_name] = {
            "class": model_class,
            "config": model_config,
            "instance": None
        }
        
        return True
    
    def load_model(self, model_name):
        """Charge un modèle personnalisé"""
        if model_name not in self.models:
            raise ValueError(f"Modèle {model_name} non enregistré")
        
        if self.models[model_name]["instance"] is not None:
            return self.models[model_name]["instance"]
        
        try:
            model_class = self.models[model_name]["class"]
            model_config = self.models[model_name]["config"]
            
            model = model_class(**model_config)
            model.load(os.path.join(self.model_dir, model_name))
            
            self.models[model_name]["instance"] = model
            return model
        except Exception as e:
            raise Exception(f"Erreur lors du chargement du modèle {model_name}: {str(e)}")
    
    def generate(self, model_name, prompt, **kwargs):
        """Génère une réponse avec un modèle personnalisé"""
        model = self.load_model(model_name)
        
        start_time = time.time()
        response = model.generate(prompt, **kwargs)
        generation_time = time.time() - start_time
        
        # Adapter la sortie au format standard
        return {
            "text": response,
            "model": model_name,
            "generationTime": generation_time
        }
    
    def save_model(self, model_name):
        """Sauvegarde un modèle personnalisé"""
        if model_name not in self.models or self.models[model_name]["instance"] is None:
            raise ValueError(f"Modèle {model_name} non chargé")
        
        model = self.models[model_name]["instance"]
        model.save(os.path.join(self.model_dir, model_name))
        
        return True
```

Cette section fournit les détails d'implémentation essentiels pour intégrer efficacement les frameworks ML dans l'architecture RAG-KAG.

## 18. Techniques d'apprentissage avancées et AGI

Cette section présente les techniques d'apprentissage rapide, les architectures LSTM, et les approches AGI à intégrer dans l'API Python.

### 18.1 Fast Learner - Apprentissage avec peu d'exemples

Les techniques d'apprentissage rapide permettent d'adapter des modèles pré-entraînés avec très peu d'exemples.

```python
# src/fast_learning.py
import os
import time
import torch
import numpy as np
from typing import List, Dict, Any, Optional, Union
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import (
    get_peft_model,
    LoraConfig,
    prepare_model_for_kbit_training,
    TaskType
)
from datasets import Dataset

class FastLearner:
    """Implémentation d'apprentissage rapide avec LoRA et transfer learning"""
    
    def __init__(self, 
                 base_model_name: str = "mistral-7b-instruct-v0.2",
                 cache_dir: str = "./models",
                 lora_r: int = 16,
                 lora_alpha: int = 32,
                 lora_dropout: float = 0.05):
        """
        Initialise un Fast Learner basé sur LoRA
        
        Args:
            base_model_name: Modèle de base à adapter
            cache_dir: Répertoire de cache des modèles
            lora_r: Rang de la matrice d'adaptation LoRA
            lora_alpha: Facteur d'échelle alpha pour LoRA
            lora_dropout: Taux de dropout pour LoRA
        """
        self.base_model_name = base_model_name
        self.cache_dir = cache_dir
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self.peft_config = None
        
    def prepare_base_model(self):
        """Charge et prépare le modèle de base pour l'adaptation LoRA"""
        try:
            print(f"Chargement du modèle de base {self.base_model_name}...")
            
            # Configuration pour la quantification 4-bit
            quantization_config = None
            if self.device.type == 'cuda':
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True
                )
            
            # Chargement du modèle pré-entraîné
            self.model = AutoModelForCausalLM.from_pretrained(
                self.base_model_name,
                cache_dir=os.path.join(self.cache_dir, self.base_model_name),
                quantization_config=quantization_config,
                device_map="auto" if self.device.type == 'cuda' else None,
                trust_remote_code=True
            )
            
            # Chargement du tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.base_model_name,
                cache_dir=os.path.join(self.cache_dir, self.base_model_name),
                trust_remote_code=True
            )
            
            # S'assurer que le tokenizer a un pad_token
            if not self.tokenizer.pad_token:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Préparation du modèle pour l'entraînement avec quantification
            if self.device.type == 'cuda':
                self.model = prepare_model_for_kbit_training(self.model)
            
            # Configuration LoRA
            self.peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=self.lora_r,
                lora_alpha=self.lora_alpha,
                lora_dropout=self.lora_dropout,
                target_modules=self._get_target_modules(),
            )
            
            # Appliquer LoRA au modèle
            self.model = get_peft_model(self.model, self.peft_config)
            
            # Afficher les paramètres entraînables vs. totaux
            self._print_trainable_parameters()
            
            return True
        
        except Exception as e:
            print(f"Erreur lors de la préparation du modèle: {str(e)}")
            return False
            
    def _get_target_modules(self) -> List[str]:
        """Détermine les modules cibles pour LoRA en fonction du modèle de base"""
        if "llama" in self.base_model_name.lower():
            return ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "down_proj", "up_proj"]
        elif "mistral" in self.base_model_name.lower():
            return ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "down_proj", "up_proj"]
        elif "phi" in self.base_model_name.lower():
            return ["Wqkv", "out_proj", "fc1", "fc2"]
        else:
            # Valeur par défaut qui fonctionne souvent
            return ["q_proj", "v_proj", "k_proj", "o_proj"]
        
    def _print_trainable_parameters(self):
        """Affiche le nombre de paramètres entraînables vs. total"""
        trainable_params = 0
        all_param = 0
        
        for _, param in self.model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
                
        print(
            f"Paramètres entraînables: {trainable_params:,} ({100 * trainable_params / all_param:.2f}% du total)"
        )
        print(f"Tous les paramètres: {all_param:,}")
    
    def format_instruction_dataset(self, examples: List[Dict[str, str]]) -> Dataset:
        """
        Formate les exemples dans un format adapté pour l'instruction tuning
        
        Args:
            examples: Liste de dictionnaires avec 'input' et 'output'
        
        Returns:
            Dataset formaté
        """
        formatted_examples = []
        
        for ex in examples:
            instruction = ex.get('input', '')
            response = ex.get('output', '')
            
            # Format d'instruction standard
            formatted_text = f"[INST] {instruction} [/INST] {response}"
            
            formatted_examples.append({
                "text": formatted_text
            })
        
        # Création du dataset
        return Dataset.from_list(formatted_examples)
    
    def tokenize_dataset(self, dataset: Dataset) -> Dataset:
        """Tokenize le dataset pour l'entraînement"""
        max_length = 1024  # Adapter en fonction des besoins
        
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors=None
            )
        
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["text"]
        )
        
        return tokenized_dataset
    
    def train(self, 
              examples: List[Dict[str, str]], 
              epochs: int = 3, 
              batch_size: int = 4, 
              learning_rate: float = 2e-4,
              checkpoint_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Entraîne le modèle sur les exemples fournis (few-shot learning)
        
        Args:
            examples: Liste de dictionnaires avec 'input' et 'output'
            epochs: Nombre d'époques d'entraînement
            batch_size: Taille du batch
            learning_rate: Taux d'apprentissage
            checkpoint_dir: Répertoire où sauvegarder le modèle adapté
        
        Returns:
            Métriques d'entraînement
        """
        start_time = time.time()
        
        # Préparer le modèle si ce n'est pas déjà fait
        if self.model is None:
            success = self.prepare_base_model()
            if not success:
                return {
                    "success": False,
                    "message": "Échec de la préparation du modèle de base"
                }
        
        try:
            # Formater et tokenizer les exemples
            dataset = self.format_instruction_dataset(examples)
            tokenized_dataset = self.tokenize_dataset(dataset)
            
            # Importer après la préparation pour éviter des imports inutiles si la préparation échoue
            from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
            
            # Collecter les données pour l'entraînement
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer, 
                mlm=False  # Causal LM, pas de masquage
            )
            
            # Déterminer le répertoire de sortie
            if checkpoint_dir:
                output_dir = checkpoint_dir
            else:
                timestamp = int(time.time())
                model_name = self.base_model_name.split("/")[-1]
                output_dir = f"./checkpoints/fast_learner_{model_name}_{timestamp}"
                
            os.makedirs(output_dir, exist_ok=True)
            
            # Configurer l'entraînement
            training_args = TrainingArguments(
                output_dir=output_dir,
                per_device_train_batch_size=batch_size,
                gradient_accumulation_steps=4,  # Permet d'utiliser des batchs plus grands virtuellement
                learning_rate=learning_rate,
                num_train_epochs=epochs,
                weight_decay=0.01,
                logging_steps=10,
                save_steps=100,
                save_total_limit=2,  # Garder uniquement les 2 meilleurs checkpoints
                remove_unused_columns=True,
                push_to_hub=False,
                report_to=None,  # Désactiver Wandb, etc.
                lr_scheduler_type="cosine",
                warmup_ratio=0.05,
                optim="paged_adamw_8bit" if self.device.type == 'cuda' else "adamw_torch"
            )
            
            # Créer et lancer l'entraîneur
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=tokenized_dataset,
                data_collator=data_collator,
                tokenizer=self.tokenizer
            )
            
            print(f"Début de l'entraînement avec {len(examples)} exemples pour {epochs} époques...")
            trainer.train()
            
            # Sauvegarder le modèle adapté
            self.model.save_pretrained(output_dir)
            self.tokenizer.save_pretrained(output_dir)
            
            # Mesurer le temps d'entraînement
            training_time = time.time() - start_time
            
            # Retourner les métriques d'entraînement
            loss = trainer.state.log_history[-1].get('loss', 0)
            return {
                "success": True,
                "model": self.base_model_name,
                "trained_examples": len(examples),
                "epochs": epochs,
                "loss": loss,
                "training_time": training_time,
                "checkpoint_path": output_dir
            }
            
        except Exception as e:
            print(f"Erreur lors de l'entraînement: {str(e)}")
            import traceback
            traceback.print_exc()
            
            return {
                "success": False,
                "model": self.base_model_name,
                "message": str(e)
            }
    
    def generate(self, 
                 prompt: str,
                 max_tokens: int = 512,
                 temperature: float = 0.7) -> Dict[str, Any]:
        """
        Génère une réponse à partir du modèle adapté
        
        Args:
            prompt: Texte d'entrée
            max_tokens: Nombre maximum de tokens à générer
            temperature: Température pour l'échantillonnage (0.0=déterministe)
        
        Returns:
            Dict contenant la réponse générée et des métadonnées
        """
        if self.model is None or self.tokenizer is None:
            return {
                "error": "Modèle non initialisé. Appelez d'abord prepare_base_model()"
            }
        
        # Formater le prompt comme une instruction pour les modèles chat
        if "[INST]" not in prompt:
            formatted_prompt = f"[INST] {prompt} [/INST]"
        else:
            formatted_prompt = prompt
        
        start_time = time.time()
        
        try:
            # Tokéniser l'entrée
            inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.device)
            input_ids_len = inputs.input_ids.shape[1]
            
            # Générer la sortie
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=temperature > 0.0,
                    top_p=0.9,
                    repetition_penalty=1.1,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            
            # Décoder la sortie
            full_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extraire uniquement la partie générée (après le prompt)
            # Cette logique peut varier selon les modèles
            if "[/INST]" in full_output:
                generated_output = full_output.split("[/INST]", 1)[1].strip()
            else:
                input_text = self.tokenizer.decode(inputs.input_ids[0], skip_special_tokens=True)
                if full_output.startswith(input_text):
                    generated_output = full_output[len(input_text):].strip()
                else:
                    generated_output = full_output.strip()
            
            # Calculer le temps de génération
            generation_time = time.time() - start_time
            
            # Calculer les tokens utilisés
            tokens_used = outputs.shape[1] - input_ids_len
            
            return {
                "text": generated_output,
                "tokensUsed": tokens_used,
                "generationTime": generation_time,
                "model": self.base_model_name
            }
            
        except Exception as e:
            print(f"Erreur lors de la génération: {str(e)}")
            return {
                "error": str(e)
            }
```

### 18.2 Endpoint API pour le Fast Learning

Implémentation de l'endpoint FastAPI pour l'apprentissage rapide :

```python
# Dans app.py
from src.fast_learning import FastLearner

# Créer une instance globale
fast_learner = FastLearner()

@app.route('/fastlearn/train', methods=['POST'])
def fast_learn_train():
    """Endpoint d'apprentissage rapide avec peu d'exemples"""
    data = request.json
    
    if not data or 'examples' not in data:
        return jsonify({
            "success": False,
            "message": "Paramètre 'examples' manquant"
        }), 400
    
    examples = data['examples']
    model_name = data.get('model', 'mistral-7b-instruct-v0.2')
    epochs = data.get('epochs', 3)
    batch_size = data.get('batch_size', 4)
    learning_rate = data.get('learning_rate', 2e-4)
    
    # Configurer le modèle de base si différent
    if fast_learner.base_model_name != model_name:
        fast_learner.base_model_name = model_name
        fast_learner.model = None
        fast_learner.tokenizer = None
    
    # Lancer l'entraînement
    result = fast_learner.train(
        examples=examples,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate
    )
    
    if result.get("success", False):
        return jsonify(result), 200
    else:
        return jsonify(result), 500

@app.route('/fastlearn/generate', methods=['POST'])
def fast_learn_generate():
    """Génère une réponse avec le modèle adapté par apprentissage rapide"""
    data = request.json
    
    if not data or 'prompt' not in data:
        return jsonify({
            "success": False,
            "message": "Paramètre 'prompt' manquant"
        }), 400
    
    prompt = data['prompt']
    max_tokens = data.get('max_tokens', 512)
    temperature = data.get('temperature', 0.7)
    
    # Vérifier que le modèle est chargé
    if fast_learner.model is None:
        if not fast_learner.prepare_base_model():
            return jsonify({
                "success": False,
                "message": "Échec du chargement du modèle. Veuillez d'abord entraîner un modèle."
            }), 500
    
    # Générer la réponse
    result = fast_learner.generate(
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature
    )
    
    if "error" in result:
        return jsonify({
            "success": False,
            "message": result["error"]
        }), 500
    else:
        return jsonify({
            "success": True,
            **result
        }), 200
```

### 18.3 Architectures LSTM pour l'apprentissage séquentiel

Les modèles LSTM (Long Short-Term Memory) sont particulièrement adaptés pour l'apprentissage de séquences temporelles et la modélisation de dépendances à long terme.

```python
# src/lstm_models.py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Any, Tuple, Optional
import time
import json
import os

class TimeSeriesDataset(Dataset):
    """Dataset personnalisé pour données de séries temporelles"""
    
    def __init__(self, sequences, targets):
        self.sequences = sequences
        self.targets = targets
        
    def __len__(self):
        return len(self.sequences)
        
    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]

class LSTMModel(nn.Module):
    """Modèle LSTM pour prédiction et classification de séquences"""
    
    def __init__(self, 
                 input_size: int, 
                 hidden_size: int = 128, 
                 num_layers: int = 2, 
                 output_size: int = 1, 
                 dropout: float = 0.2,
                 bidirectional: bool = False):
        """
        Initialise un modèle LSTM
        
        Args:
            input_size: Taille de chaque entrée de séquence
            hidden_size: Nombre d'unités dans les couches cachées LSTM
            num_layers: Nombre de couches LSTM empilées
            output_size: Taille de la sortie
            dropout: Taux de dropout pour régularisation
            bidirectional: Si True, utilise un LSTM bidirectionnel
        """
        super(LSTMModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.directions = 2 if bidirectional else 1
        
        # Couche LSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Couche de sortie
        self.fc = nn.Linear(hidden_size * self.directions, output_size)
        
    def forward(self, x):
        # Initialiser l'état caché
        h0 = torch.zeros(self.num_layers * self.directions, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * self.directions, x.size(0), self.hidden_size).to(x.device)
        
        # Forward pass LSTM
        out, (hidden, _) = self.lstm(x, (h0, c0))
        
        # Récupérer uniquement la sortie de la dernière séquence
        if self.bidirectional:
            # Concaténer les dernières sorties de chaque direction
            hidden_forward = hidden[self.num_layers-1, :, :]
            hidden_backward = hidden[self.num_layers*2-1, :, :]
            final_hidden = torch.cat((hidden_forward, hidden_backward), dim=1)
        else:
            final_hidden = hidden[-1, :, :]
            
        # Passer par la couche linéaire
        output = self.fc(final_hidden)
        
        return output

class DeepLSTMManager:
    """Gestionnaire pour l'entraînement et l'inférence avec des modèles LSTM"""
    
    def __init__(self, model_dir: str = "./lstm_models"):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.config = {}
        self.scaler = None
        
    def create_model(self, 
                     input_size: int, 
                     hidden_size: int = 128, 
                     num_layers: int = 2, 
                     output_size: int = 1, 
                     dropout: float = 0.2,
                     bidirectional: bool = False) -> LSTMModel:
        """Crée un nouveau modèle LSTM avec la configuration spécifiée"""
        self.model = LSTMModel(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            output_size=output_size,
            dropout=dropout,
            bidirectional=bidirectional
        ).to(self.device)
        
        # Stocker la configuration pour référence ultérieure
        self.config = {
            "type": "lstm",
            "input_size": input_size,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "output_size": output_size,
            "dropout": dropout,
            "bidirectional": bidirectional
        }
        
        return self.model
    
    def prepare_data(self, 
                     sequences: List[List[float]], 
                     targets: List[float], 
                     test_size: float = 0.2,
                     batch_size: int = 32,
                     normalize: bool = True) -> Tuple[DataLoader, DataLoader, Dict]:
        """
        Prépare les données pour l'entraînement et l'évaluation
        
        Args:
            sequences: Liste de séquences d'entrée
            targets: Liste de cibles correspondantes
            test_size: Proportion des données à utiliser pour le test
            batch_size: Taille des batchs pour l'entraînement
            normalize: Si True, normalise les données
            
        Returns:
            train_loader, test_loader, preprocessing_info
        """
        # Convertir en arrays numpy
        sequences = np.array(sequences, dtype=np.float32)
        targets = np.array(targets, dtype=np.float32)
        
        # Normaliser les données si demandé
        preprocessing_info = {}
        if normalize:
            from sklearn.preprocessing import StandardScaler
            
            # Créer et ajuster le scaler
            original_shape = sequences.shape
            flattened = sequences.reshape(-1, sequences.shape[-1])
            
            self.scaler = StandardScaler()
            flattened_scaled = self.scaler.fit_transform(flattened)
            sequences = flattened_scaled.reshape(original_shape)
            
            # Stocker les infos de prétraitement
            preprocessing_info["scaler_mean"] = self.scaler.mean_.tolist()
            preprocessing_info["scaler_scale"] = self.scaler.scale_.tolist()
        
        # Diviser en ensembles d'entraînement et de test
        test_idx = int(len(sequences) * (1 - test_size))
        train_sequences, test_sequences = sequences[:test_idx], sequences[test_idx:]
        train_targets, test_targets = targets[:test_idx], targets[test_idx:]
        
        # Convertir en tensors PyTorch
        train_sequences = torch.tensor(train_sequences, dtype=torch.float32)
        test_sequences = torch.tensor(test_sequences, dtype=torch.float32)
        
        if len(train_targets.shape) == 1:
            train_targets = train_targets.reshape(-1, 1)
            test_targets = test_targets.reshape(-1, 1)
            
        train_targets = torch.tensor(train_targets, dtype=torch.float32)
        test_targets = torch.tensor(test_targets, dtype=torch.float32)
        
        # Créer les datasets
        train_dataset = TimeSeriesDataset(train_sequences, train_targets)
        test_dataset = TimeSeriesDataset(test_sequences, test_targets)
        
        # Créer les dataloaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # Stocker des statistiques supplémentaires
        preprocessing_info["train_size"] = len(train_sequences)
        preprocessing_info["test_size"] = len(test_sequences)
        preprocessing_info["sequence_length"] = sequences.shape[1]
        preprocessing_info["feature_size"] = sequences.shape[2]
        
        return train_loader, test_loader, preprocessing_info
    
    def train(self, 
              train_loader: DataLoader, 
              test_loader: Optional[DataLoader] = None,
              epochs: int = 50,
              learning_rate: float = 0.001,
              patience: int = 10,
              model_name: str = "lstm_model") -> Dict[str, Any]:
        """
        Entraîne le modèle LSTM
        
        Args:
            train_loader: DataLoader pour les données d'entraînement
            test_loader: DataLoader pour les données de test (validation)
            epochs: Nombre maximal d'époques
            learning_rate: Taux d'apprentissage
            patience: Nombre d'époques sans amélioration avant arrêt précoce
            model_name: Nom du modèle pour la sauvegarde
            
        Returns:
            Historique et métriques d'entraînement
        """
        if self.model is None:
            raise ValueError("Le modèle n'a pas été créé. Appelez create_model d'abord.")
        
        start_time = time.time()
        
        # Définir le critère et l'optimiseur
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Pour l'arrêt précoce
        best_loss = float('inf')
        no_improve_epochs = 0
        best_model_state = None
        
        # Historique d'entraînement
        history = {
            "train_loss": [],
            "test_loss": [] if test_loader else None,
            "epochs": 0
        }
        
        # Boucle d'entraînement
        self.model.train()
        for epoch in range(epochs):
            train_loss = 0.0
            
            for sequences, targets in train_loader:
                # Déplacer les données vers le device
                sequences = sequences.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                outputs = self.model(sequences)
                loss = criterion(outputs, targets)
                
                # Backward pass et optimisation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * sequences.size(0)
            
            # Calculer la perte moyenne
            train_loss = train_loss / len(train_loader.dataset)
            history["train_loss"].append(train_loss)
            
            # Évaluer sur l'ensemble de test si fourni
            test_loss = None
            if test_loader:
                test_loss = self._evaluate(test_loader, criterion)
                history["test_loss"].append(test_loss)
                
                # Vérifier pour l'arrêt précoce
                if test_loss < best_loss:
                    best_loss = test_loss
                    no_improve_epochs = 0
                    best_model_state = self.model.state_dict()
                else:
                    no_improve_epochs += 1
                    
                if no_improve_epochs >= patience:
                    print(f"Arrêt précoce à l'époque {epoch+1}")
                    break
            else:
                # Utiliser la perte d'entraînement pour l'arrêt précoce
                if train_loss < best_loss:
                    best_loss = train_loss
                    no_improve_epochs = 0
                    best_model_state = self.model.state_dict()
                else:
                    no_improve_epochs += 1
                    
                if no_improve_epochs >= patience:
                    print(f"Arrêt précoce à l'époque {epoch+1}")
                    break
            
            # Afficher les statistiques
            print(f"Époque {epoch+1}/{epochs}, Train Loss: {train_loss:.6f}" + 
                  (f", Test Loss: {test_loss:.6f}" if test_loss is not None else ""))
        
        # Restaurer le meilleur modèle
        if best_model_state:
            self.model.load_state_dict(best_model_state)
        
        # Mettre à jour le compte final d'époques
        history["epochs"] = epoch + 1
        
        # Sauvegarder le modèle
        self.save_model(model_name)
        
        # Calculer le temps d'entraînement
        history["training_time"] = time.time() - start_time
        
        return history
    
    def _evaluate(self, dataloader: DataLoader, criterion: nn.Module) -> float:
        """Évaluer le modèle sur un dataloader"""
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for sequences, targets in dataloader:
                sequences = sequences.to(self.device)
                targets = targets.to(self.device)
                
                outputs = self.model(sequences)
                loss = criterion(outputs, targets)
                
                total_loss += loss.item() * sequences.size(0)
        
        # Repasser en mode entraînement
        self.model.train()
        
        return total_loss / len(dataloader.dataset)
    
    def predict(self, sequences: List[List[float]]) -> np.ndarray:
        """
        Utilise le modèle entraîné pour prédire sur de nouvelles séquences
        
        Args:
            sequences: Liste de séquences d'entrée
            
        Returns:
            Prédictions
        """
        if self.model is None:
            raise ValueError("Le modèle n'a pas été entraîné ou chargé.")
        
        # Convertir en numpy
        sequences = np.array(sequences, dtype=np.float32)
        
        # Normaliser si un scaler existe
        if self.scaler:
            original_shape = sequences.shape
            flattened = sequences.reshape(-1, sequences.shape[-1])
            flattened_scaled = self.scaler.transform(flattened)
            sequences = flattened_scaled.reshape(original_shape)
        
        # Convertir en tensor
        sequences = torch.tensor(sequences, dtype=torch.float32).to(self.device)
        
        # Prédiction
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(sequences)
        
        # Retourner comme numpy
        return predictions.cpu().numpy()
    
    def save_model(self, model_name: str) -> str:
        """
        Sauvegarde le modèle et sa configuration
        
        Args:
            model_name: Nom du modèle
            
        Returns:
            Chemin du répertoire de sauvegarde
        """
        if self.model is None:
            raise ValueError("Le modèle n'a pas été créé ou entraîné.")
        
        # Créer le répertoire pour ce modèle
        model_dir = os.path.join(self.model_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)
        
        # Sauvegarder l'état du modèle
        model_path = os.path.join(model_dir, "model.pt")
        torch.save(self.model.state_dict(), model_path)
        
        # Sauvegarder la configuration
        config_path = os.path.join(model_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(self.config, f)
        
        # Sauvegarder le scaler si disponible
        if self.scaler:
            import pickle
            scaler_path = os.path.join(model_dir, "scaler.pkl")
            with open(scaler_path, "wb") as f:
                pickle.dump(self.scaler, f)
        
        return model_dir
    
    def load_model(self, model_name: str) -> bool:
        """
        Charge un modèle sauvegardé
        
        Args:
            model_name: Nom du modèle
            
        Returns:
            True si le chargement a réussi, False sinon
        """
        model_dir = os.path.join(self.model_dir, model_name)
        
        try:
            # Charger la configuration
            config_path = os.path.join(model_dir, "config.json")
            with open(config_path, "r") as f:
                self.config = json.load(f)
            
            # Créer le modèle
            self.model = LSTMModel(
                input_size=self.config["input_size"],
                hidden_size=self.config["hidden_size"],
                num_layers=self.config["num_layers"],
                output_size=self.config["output_size"],
                dropout=self.config["dropout"],
                bidirectional=self.config["bidirectional"]
            ).to(self.device)
            
            # Charger l'état du modèle
            model_path = os.path.join(model_dir, "model.pt")
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
            
            # Charger le scaler s'il existe
            scaler_path = os.path.join(model_dir, "scaler.pkl")
            if os.path.exists(scaler_path):
                import pickle
                with open(scaler_path, "rb") as f:
                    self.scaler = pickle.load(f)
            
            return True
            
        except Exception as e:
            print(f"Erreur lors du chargement du modèle {model_name}: {str(e)}")
            return False
```

### 18.4 Endpoints API pour les modèles LSTM

Implémentation des endpoints pour l'accès aux modèles LSTM depuis l'API :

```python
# Dans app.py
from src.lstm_models import DeepLSTMManager

# Créer une instance globale
lstm_manager = DeepLSTMManager()

@app.route('/lstm/create', methods=['POST'])
def lstm_create():
    """Crée un nouveau modèle LSTM avec la configuration spécifiée"""
    data = request.json
    
    if not data:
        return jsonify({
            "success": False,
            "message": "Données de configuration manquantes"
        }), 400
    
    try:
        # Extraire les paramètres
        input_size = data.get('input_size')
        if not input_size:
            return jsonify({
                "success": False,
                "message": "Le paramètre 'input_size' est obligatoire"
            }), 400
            
        # Créer le modèle avec les paramètres fournis
        lstm_manager.create_model(
            input_size=input_size,
            hidden_size=data.get('hidden_size', 128),
            num_layers=data.get('num_layers', 2),
            output_size=data.get('output_size', 1),
            dropout=data.get('dropout', 0.2),
            bidirectional=data.get('bidirectional', False)
        )
        
        return jsonify({
            "success": True,
            "message": "Modèle LSTM créé avec succès",
            "config": lstm_manager.config
        }), 200
        
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Erreur lors de la création du modèle: {str(e)}"
        }), 500

@app.route('/lstm/train', methods=['POST'])
def lstm_train():
    """Entraîne un modèle LSTM sur des données séquentielles"""
    data = request.json
    
    if not data or 'sequences' not in data or 'targets' not in data:
        return jsonify({
            "success": False,
            "message": "Les paramètres 'sequences' et 'targets' sont obligatoires"
        }), 400
    
    try:
        # Extraire les données
        sequences = data['sequences']
        targets = data['targets']
        model_name = data.get('model_name', f"lstm_model_{int(time.time())}")
        
        # Préparer les données
        train_loader, test_loader, preprocessing_info = lstm_manager.prepare_data(
            sequences=sequences,
            targets=targets,
            test_size=data.get('test_size', 0.2),
            batch_size=data.get('batch_size', 32),
            normalize=data.get('normalize', True)
        )
        
        # Entraîner le modèle
        history = lstm_manager.train(
            train_loader=train_loader,
            test_loader=test_loader,
            epochs=data.get('epochs', 50),
            learning_rate=data.get('learning_rate', 0.001),
            patience=data.get('patience', 10),
            model_name=model_name
        )
        
        # Combiner les résultats
        result = {
            "success": True,
            "model_name": model_name,
            "epochs_trained": history["epochs"],
            "final_train_loss": history["train_loss"][-1],
            "training_time": history["training_time"],
            "preprocessing_info": preprocessing_info
        }
        
        if history["test_loss"]:
            result["final_test_loss"] = history["test_loss"][-1]
        
        return jsonify(result), 200
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            "success": False,
            "message": f"Erreur lors de l'entraînement: {str(e)}"
        }), 500

@app.route('/lstm/predict', methods=['POST'])
def lstm_predict():
    """Prédit avec un modèle LSTM sur de nouvelles séquences"""
    data = request.json
    
    if not data or 'sequences' not in data:
        return jsonify({
            "success": False,
            "message": "Le paramètre 'sequences' est obligatoire"
        }), 400
    
    try:
        # Charger le modèle si spécifié
        model_name = data.get('model_name')
        if model_name and not lstm_manager.model:
            success = lstm_manager.load_model(model_name)
            if not success:
                return jsonify({
                    "success": False,
                    "message": f"Impossible de charger le modèle {model_name}"
                }), 404
        
        # Vérifier qu'un modèle est disponible
        if not lstm_manager.model:
            return jsonify({
                "success": False,
                "message": "Aucun modèle n'est disponible. Créez ou chargez un modèle d'abord."
            }), 400
        
        # Faire la prédiction
        sequences = data['sequences']
        predictions = lstm_manager.predict(sequences)
        
        return jsonify({
            "success": True,
            "predictions": predictions.tolist(),
            "model_config": lstm_manager.config
        }), 200
        
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Erreur lors de la prédiction: {str(e)}"
        }), 500

@app.route('/lstm/models', methods=['GET'])
def lstm_list_models():
    """Liste les modèles LSTM disponibles"""
    try:
        models = []
        for model_name in os.listdir(lstm_manager.model_dir):
            model_path = os.path.join(lstm_manager.model_dir, model_name)
            if os.path.isdir(model_path):
                # Lire la configuration
                config_path = os.path.join(model_path, "config.json")
                if os.path.exists(config_path):
                    with open(config_path, "r") as f:
                        config = json.load(f)
                    
                    # Ajouter les informations du modèle
                    models.append({
                        "name": model_name,
                        "config": config,
                        "created_at": os.path.getctime(model_path)
                    })
        
        return jsonify({
            "success": True,
            "models": models
        }), 200
        
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Erreur lors de la récupération des modèles: {str(e)}"
        }), 500
```

### 18.5 Conversion et traitement de documents

Pour traiter et convertir différents formats de documents (PDF, DOCX, etc.) en texte exploitable:

```python
# src/document_processing.py
import os
import time
import logging
import tempfile
from typing import List, Dict, Any, Optional, Union
import pandas as pd
import numpy as np
import json
import re
from datetime import datetime

# Dépendances pour la conversion de documents
try:
    import fitz  # PyMuPDF
    import docx
    from bs4 import BeautifulSoup
    import csv
    import xml.etree.ElementTree as ET
    import zipfile
    from PIL import Image
    import pytesseract
    HAS_DOC_LIBRARIES = True
except ImportError:
    HAS_DOC_LIBRARIES = False
    print("Certaines bibliothèques de traitement de documents ne sont pas disponibles. Installez-les avec:"
          " pip install pymupdf python-docx beautifulsoup4 pillow pytesseract")

class DocumentProcessor:
    """Classe pour la conversion et le traitement de divers formats de documents"""
    
    def __init__(self, storage_dir: str = "./processed_documents"):
        """
        Initialise le processeur de documents
        
        Args:
            storage_dir: Répertoire pour stocker les documents traités
        """
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)
        
        self.logger = logging.getLogger("document_processor")
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
        
        # Vérifier la disponibilité des bibliothèques de traitement de documents
        if not HAS_DOC_LIBRARIES:
            self.logger.warning("Certaines fonctionnalités de traitement de documents ne seront pas disponibles")
    
    def process_file(self, file_path: str) -> Dict[str, Any]:
        """
        Traite un fichier et l'extrait en texte en fonction de son type
        
        Args:
            file_path: Chemin du fichier à traiter
            
        Returns:
            Dict contenant le texte extrait et des métadonnées
        """
        if not os.path.exists(file_path):
            raise ValueError(f"Le fichier {file_path} n'existe pas")
        
        # Déterminer l'extension du fichier
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()
        
        # Traiter en fonction du type de fichier
        try:
            if ext == '.pdf':
                return self._process_pdf(file_path)
            elif ext == '.docx':
                return self._process_docx(file_path)
            elif ext == '.txt':
                return self._process_text(file_path)
            elif ext == '.csv':
                return self._process_csv(file_path)
            elif ext == '.html' or ext == '.htm':
                return self._process_html(file_path)
            elif ext == '.xml':
                return self._process_xml(file_path)
            elif ext in ['.jpg', '.jpeg', '.png', '.tiff', '.bmp', '.gif']:
                return self._process_image(file_path)
            else:
                raise ValueError(f"Format de fichier non pris en charge: {ext}")
        except Exception as e:
            self.logger.error(f"Erreur lors du traitement du fichier {file_path}: {str(e)}")
            raise
    
    def _process_pdf(self, file_path: str) -> Dict[str, Any]:
        """Extrait le texte et les métadonnées d'un fichier PDF"""
        if not HAS_DOC_LIBRARIES:
            raise ImportError("PyMuPDF (fitz) est requis pour traiter les fichiers PDF")
        
        self.logger.info(f"Traitement du fichier PDF: {file_path}")
        
        start_time = time.time()
        result = {
            "file_path": file_path,
            "file_type": "pdf",
            "processing_time": 0,
            "page_count": 0,
            "metadata": {},
            "text": ""
        }
        
        try:
            # Ouvrir le PDF
            pdf_document = fitz.open(file_path)
            result["page_count"] = len(pdf_document)
            
            # Extraire les métadonnées
            metadata = pdf_document.metadata
            if metadata:
                result["metadata"] = {
                    "title": metadata.get("title", ""),
                    "author": metadata.get("author", ""),
                    "subject": metadata.get("subject", ""),
                    "creator": metadata.get("creator", ""),
                    "producer": metadata.get("producer", ""),
                    "creation_date": metadata.get("creationDate", ""),
                    "modification_date": metadata.get("modDate", "")
                }
            
            # Extraire le texte page par page
            full_text = []
            for page_num, page in enumerate(pdf_document):
                text = page.get_text()
                full_text.append(text)
                
                # Ajouter le numéro de page pour faciliter la référence
                if len(full_text) > 1:
                    full_text[-1] = f"\n--- Page {page_num + 1} ---\n{text}"
            
            result["text"] = "\n".join(full_text)
            result["processing_time"] = time.time() - start_time
            
            return result
            
        except Exception as e:
            self.logger.error(f"Erreur lors du traitement du PDF: {str(e)}")
            raise
    
    def _process_docx(self, file_path: str) -> Dict[str, Any]:
        """Extrait le texte et les métadonnées d'un fichier DOCX"""
        if not HAS_DOC_LIBRARIES:
            raise ImportError("python-docx est requis pour traiter les fichiers DOCX")
        
        self.logger.info(f"Traitement du fichier DOCX: {file_path}")
        
        start_time = time.time()
        result = {
            "file_path": file_path,
            "file_type": "docx",
            "processing_time": 0,
            "metadata": {},
            "text": ""
        }
        
        try:
            # Ouvrir le document
            doc = docx.Document(file_path)
            
            # Extraire les métadonnées
            core_properties = doc.core_properties
            if core_properties:
                result["metadata"] = {
                    "title": core_properties.title if hasattr(core_properties, 'title') else "",
                    "author": core_properties.author if hasattr(core_properties, 'author') else "",
                    "subject": core_properties.subject if hasattr(core_properties, 'subject') else "",
                    "created": str(core_properties.created) if hasattr(core_properties, 'created') else "",
                    "modified": str(core_properties.modified) if hasattr(core_properties, 'modified') else ""
                }
            
            # Extraire le texte
            full_text = []
            for para in doc.paragraphs:
                full_text.append(para.text)
            
            # Extraire le texte des tableaux
            for table in doc.tables:
                for row in table.rows:
                    row_text = []
                    for cell in row.cells:
                        if cell.text.strip():
                            row_text.append(cell.text.strip())
                    if row_text:
                        full_text.append(" | ".join(row_text))
            
            result["text"] = "\n".join(full_text)
            result["processing_time"] = time.time() - start_time
            
            return result
            
        except Exception as e:
            self.logger.error(f"Erreur lors du traitement du DOCX: {str(e)}")
            raise
    
    def _process_text(self, file_path: str) -> Dict[str, Any]:
        """Extrait le texte d'un fichier texte simple"""
        self.logger.info(f"Traitement du fichier texte: {file_path}")
        
        start_time = time.time()
        result = {
            "file_path": file_path,
            "file_type": "txt",
            "processing_time": 0,
            "metadata": {
                "created": str(datetime.fromtimestamp(os.path.getctime(file_path))),
                "modified": str(datetime.fromtimestamp(os.path.getmtime(file_path))),
                "size": os.path.getsize(file_path)
            },
            "text": ""
        }
        
        try:
            # Lire le contenu du fichier
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                result["text"] = f.read()
            
            result["processing_time"] = time.time() - start_time
            return result
            
        except Exception as e:
            self.logger.error(f"Erreur lors du traitement du fichier texte: {str(e)}")
            raise
    
    def _process_csv(self, file_path: str) -> Dict[str, Any]:
        """Extrait le texte et les données d'un fichier CSV"""
        if not HAS_DOC_LIBRARIES:
            raise ImportError("pandas est requis pour traiter les fichiers CSV")
        
        self.logger.info(f"Traitement du fichier CSV: {file_path}")
        
        start_time = time.time()
        result = {
            "file_path": file_path,
            "file_type": "csv",
            "processing_time": 0,
            "metadata": {
                "created": str(datetime.fromtimestamp(os.path.getctime(file_path))),
                "modified": str(datetime.fromtimestamp(os.path.getmtime(file_path))),
                "size": os.path.getsize(file_path)
            },
            "text": "",
            "data": []
        }
        
        try:
            # Lire le CSV avec pandas
            df = pd.read_csv(file_path)
            
            # Convertir en texte lisible
            text_parts = []
            
            # En-têtes
            headers = " | ".join(df.columns.tolist())
            text_parts.append(headers)
            text_parts.append("-" * len(headers))
            
            # Données (limiter à 100 lignes pour le texte)
            row_limit = min(100, len(df))
            for _, row in df.head(row_limit).iterrows():
                text_parts.append(" | ".join(str(val) for val in row.values))
            
            if len(df) > row_limit:
                text_parts.append(f"... et {len(df) - row_limit} lignes supplémentaires ...")
            
            result["text"] = "\n".join(text_parts)
            
            # Stocker les données sous forme de liste de dict
            result["data"] = df.head(1000).to_dict(orient='records')  # Limiter à 1000 lignes
            result["metadata"]["row_count"] = len(df)
            result["metadata"]["column_count"] = len(df.columns)
            result["metadata"]["columns"] = df.columns.tolist()
            
            result["processing_time"] = time.time() - start_time
            return result
            
        except Exception as e:
            self.logger.error(f"Erreur lors du traitement du fichier CSV: {str(e)}")
            raise
    
    def _process_html(self, file_path: str) -> Dict[str, Any]:
        """Extrait le texte d'un fichier HTML"""
        if not HAS_DOC_LIBRARIES:
            raise ImportError("BeautifulSoup4 est requis pour traiter les fichiers HTML")
        
        self.logger.info(f"Traitement du fichier HTML: {file_path}")
        
        start_time = time.time()
        result = {
            "file_path": file_path,
            "file_type": "html",
            "processing_time": 0,
            "metadata": {
                "created": str(datetime.fromtimestamp(os.path.getctime(file_path))),
                "modified": str(datetime.fromtimestamp(os.path.getmtime(file_path))),
                "size": os.path.getsize(file_path)
            },
            "text": ""
        }
        
        try:
            # Lire le fichier HTML
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                html_content = f.read()
            
            # Parser le HTML
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Extraire le titre
            if soup.title:
                result["metadata"]["title"] = soup.title.string
            
            # Extraire les métadonnées
            meta_tags = soup.find_all('meta')
            meta_data = {}
            for tag in meta_tags:
                if tag.get('name'):
                    meta_data[tag.get('name')] = tag.get('content', '')
                elif tag.get('property'):
                    meta_data[tag.get('property')] = tag.get('content', '')
            
            result["metadata"]["meta_tags"] = meta_data
            
            # Extraire le texte
            # Supprimer les scripts et les styles
            for script in soup(["script", "style"]):
                script.extract()
            
            # Extraire le texte
            text = soup.get_text()
            
            # Nettoyer le texte
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = '\n'.join(chunk for chunk in chunks if chunk)
            
            result["text"] = text
            result["processing_time"] = time.time() - start_time
            
            return result
            
        except Exception as e:
            self.logger.error(f"Erreur lors du traitement du fichier HTML: {str(e)}")
            raise
    
    def _process_xml(self, file_path: str) -> Dict[str, Any]:
        """Extrait le texte d'un fichier XML"""
        if not HAS_DOC_LIBRARIES:
            raise ImportError("xml.etree.ElementTree est requis pour traiter les fichiers XML")
        
        self.logger.info(f"Traitement du fichier XML: {file_path}")
        
        start_time = time.time()
        result = {
            "file_path": file_path,
            "file_type": "xml",
            "processing_time": 0,
            "metadata": {
                "created": str(datetime.fromtimestamp(os.path.getctime(file_path))),
                "modified": str(datetime.fromtimestamp(os.path.getmtime(file_path))),
                "size": os.path.getsize(file_path)
            },
            "text": ""
        }
        
        try:
            # Parser le fichier XML
            tree = ET.parse(file_path)
            root = tree.getroot()
            
            # Extraire le texte récursivement
            def extract_text(element, indent=0):
                result = []
                
                # Ajouter le texte de l'élément actuel
                if element.text and element.text.strip():
                    result.append(f"{' ' * indent}<{element.tag}> {element.text.strip()}")
                else:
                    result.append(f"{' ' * indent}<{element.tag}>")
                
                # Traiter les attributs
                if element.attrib:
                    for key, value in element.attrib.items():
                        result.append(f"{' ' * (indent+2)}@{key}: {value}")
                
                # Traiter les enfants
                for child in element:
                    result.extend(extract_text(child, indent+2))
                
                # Ajouter le texte après les enfants
                if element.tail and element.tail.strip():
                    result.append(f"{' ' * indent}{element.tail.strip()}")
                
                return result
            
            text_lines = extract_text(root)
            result["text"] = "\n".join(text_lines)
            
            # Extraire des métadonnées de base
            result["metadata"]["root_tag"] = root.tag
            result["metadata"]["namespace"] = root.tag.split('}')[0] + '}' if '}' in root.tag else ""
            
            result["processing_time"] = time.time() - start_time
            return result
            
        except Exception as e:
            self.logger.error(f"Erreur lors du traitement du fichier XML: {str(e)}")
            raise
    
    def _process_image(self, file_path: str) -> Dict[str, Any]:
        """Extrait le texte d'une image via OCR"""
        if not HAS_DOC_LIBRARIES:
            raise ImportError("PIL et pytesseract sont requis pour traiter les fichiers image")
        
        self.logger.info(f"Traitement de l'image: {file_path}")
        
        start_time = time.time()
        result = {
            "file_path": file_path,
            "file_type": "image",
            "processing_time": 0,
            "metadata": {
                "created": str(datetime.fromtimestamp(os.path.getctime(file_path))),
                "modified": str(datetime.fromtimestamp(os.path.getmtime(file_path))),
                "size": os.path.getsize(file_path)
            },
            "text": ""
        }
        
        try:
            # Ouvrir l'image
            image = Image.open(file_path)
            
            # Extraire les métadonnées
            result["metadata"]["format"] = image.format
            result["metadata"]["mode"] = image.mode
            result["metadata"]["size"] = f"{image.width}x{image.height}"
            
            # Extraire le texte avec OCR
            text = pytesseract.image_to_string(image)
            result["text"] = text
            
            result["processing_time"] = time.time() - start_time
            return result
            
        except Exception as e:
            self.logger.error(f"Erreur lors du traitement de l'image: {str(e)}")
            raise
    
    def process_and_store(self, file_path: str) -> Dict[str, Any]:
        """
        Traite un fichier et stocke les résultats
        
        Args:
            file_path: Chemin du fichier à traiter
            
        Returns:
            Dict avec les informations du traitement et l'emplacement des résultats
        """
        file_name = os.path.basename(file_path)
        name, _ = os.path.splitext(file_name)
        
        # Créer un ID unique basé sur le nom et l'heure
        unique_id = f"{name}_{int(time.time())}"
        output_dir = os.path.join(self.storage_dir, unique_id)
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            # Traiter le fichier
            result = self.process_file(file_path)
            
            # Sauvegarder le texte extrait
            text_file = os.path.join(output_dir, f"{name}.txt")
            with open(text_file, 'w', encoding='utf-8') as f:
                f.write(result["text"])
            
            # Sauvegarder les métadonnées et les infos de traitement
            meta_file = os.path.join(output_dir, f"{name}_meta.json")
            meta_data = {
                "original_file": file_path,
                "processed_at": datetime.now().isoformat(),
                "processing_time": result["processing_time"],
                "file_type": result["file_type"],
                "metadata": result["metadata"]
            }
            
            with open(meta_file, 'w', encoding='utf-8') as f:
                json.dump(meta_data, f, indent=2)
            
            # Si des données structurées sont disponibles, les sauvegarder aussi
            if "data" in result:
                data_file = os.path.join(output_dir, f"{name}_data.json")
                with open(data_file, 'w', encoding='utf-8') as f:
                    json.dump(result["data"], f, indent=2)
            
            # Ajouter les chemins de sortie au résultat
            result["output_dir"] = output_dir
            result["text_file"] = text_file
            result["meta_file"] = meta_file
            
            return result
        
        except Exception as e:
            self.logger.error(f"Erreur lors du traitement et stockage de {file_path}: {str(e)}")
            
            # Sauvegarder les détails de l'erreur
            error_file = os.path.join(output_dir, "error_log.txt")
            with open(error_file, 'w', encoding='utf-8') as f:
                f.write(f"Erreur lors du traitement de {file_path}:\n{str(e)}")
            
            raise
```

### 18.6 Endpoint API pour le traitement de documents

Implémentation de l'API pour traiter différents types de documents:

```python
# Dans app.py
import os
from werkzeug.utils import secure_filename
from src.document_processing import DocumentProcessor

# Créer une instance du processeur de documents
document_processor = DocumentProcessor()

# Configuration pour le téléchargement de fichiers
UPLOAD_FOLDER = './uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'txt', 'csv', 'html', 'htm', 'xml', 'jpg', 'jpeg', 'png'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/documents/process', methods=['POST'])
def process_document():
    """Endpoint pour traiter un document téléchargé"""
    # Vérifier qu'un fichier est présent dans la requête
    if 'file' not in request.files:
        return jsonify({
            "success": False,
            "message": "Aucun fichier trouvé dans la requête"
        }), 400
    
    file = request.files['file']
    
    # Vérifier que le fichier a un nom
    if file.filename == '':
        return jsonify({
            "success": False,
            "message": "Aucun fichier sélectionné"
        }), 400
    
    if file and allowed_file(file.filename):
        try:
            # Sécuriser le nom du fichier
            filename = secure_filename(file.filename)
            
            # Sauvegarder le fichier
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(file_path)
            
            # Traiter le fichier
            result = document_processor.process_and_store(file_path)
            
            # Renvoyer les résultats importants
            response = {
                "success": True,
                "file_name": filename,
                "file_type": result["file_type"],
                "processing_time": result["processing_time"],
                "text_length": len(result["text"]),
                "text_preview": result["text"][:500] + "..." if len(result["text"]) > 500 else result["text"],
                "output_dir": result["output_dir"],
                "metadata": result["metadata"]
            }
            
            return jsonify(response), 200
            
        except Exception as e:
            return jsonify({
                "success": False,
                "message": f"Erreur lors du traitement du document: {str(e)}"
            }), 500
    
    return jsonify({
        "success": False,
        "message": f"Format de fichier non autorisé. Formats supportés: {', '.join(ALLOWED_EXTENSIONS)}"
    }), 400

@app.route('/documents/list', methods=['GET'])
def list_processed_documents():
    """Liste les documents traités"""
    try:
        documents = []
        
        # Parcourir les dossiers de documents traités
        for doc_id in os.listdir(document_processor.storage_dir):
            doc_path = os.path.join(document_processor.storage_dir, doc_id)
            
            if os.path.isdir(doc_path):
                # Chercher le fichier de métadonnées (premier trouvé)
                meta_files = [f for f in os.listdir(doc_path) if f.endswith('_meta.json')]
                
                if meta_files:
                    meta_file = os.path.join(doc_path, meta_files[0])
                    
                    with open(meta_file, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                    
                    # Trouver le fichier texte correspondant
                    text_files = [f for f in os.listdir(doc_path) if f.endswith('.txt') and not f.startswith('error_log')]
                    
                    text_preview = ""
                    if text_files:
                        text_file = os.path.join(doc_path, text_files[0])
                        with open(text_file, 'r', encoding='utf-8') as f:
                            text = f.read(1000)  # Lire max 1000 caractères
                            text_preview = text + "..." if len(text) >= 1000 else text
                    
                    # Ajouter les infos du document
                    doc_info = {
                        "id": doc_id,
                        "original_file": os.path.basename(metadata["original_file"]),
                        "processed_at": metadata["processed_at"],
                        "file_type": metadata["file_type"],
                        "text_preview": text_preview,
                        "directory": doc_path
                    }
                    
                    documents.append(doc_info)
        
        return jsonify({
            "success": True,
            "documents": documents
        }), 200
        
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Erreur lors de la récupération des documents: {str(e)}"
        }), 500

@app.route('/documents/<doc_id>/text', methods=['GET'])
def get_document_text(doc_id):
    """Récupère le texte intégral d'un document traité"""
    try:
        doc_path = os.path.join(document_processor.storage_dir, doc_id)
        
        if not os.path.isdir(doc_path):
            return jsonify({
                "success": False,
                "message": f"Document {doc_id} non trouvé"
            }), 404
        
        # Trouver le fichier texte (premier fichier .txt qui n'est pas un log d'erreur)
        text_files = [f for f in os.listdir(doc_path) if f.endswith('.txt') and not f.startswith('error_log')]
        
        if not text_files:
            return jsonify({
                "success": False,
                "message": f"Texte non disponible pour le document {doc_id}"
            }), 404
        
        # Lire le fichier texte
        text_file = os.path.join(doc_path, text_files[0])
        with open(text_file, 'r', encoding='utf-8') as f:
            text_content = f.read()
        
        return jsonify({
            "success": True,
            "document_id": doc_id,
            "text": text_content
        }), 200
        
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Erreur lors de la récupération du texte: {str(e)}"
        }), 500

### 18.7 Boucles de feedback et automatisation des emails

L'un des aspects cruciaux d'un système d'IA est la capacité à collecter des retours utilisateurs et à automatiser les communications. Implémentons ces fonctionnalités:

```python
# src/feedback_system.py
import os
import json
import time
import uuid
import logging
import smtplib
import threading
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
from typing import List, Dict, Any, Optional, Union, Callable
from datetime import datetime, timedelta

class FeedbackManager:
    """Gestionnaire de boucles de feedback et de notifications automatisées"""
    
    def __init__(self, storage_dir: str = "./feedback_data", email_config: Optional[Dict[str, Any]] = None):
        """
        Initialise le gestionnaire de feedback
        
        Args:
            storage_dir: Répertoire de stockage des feedbacks
            email_config: Configuration pour les emails (None pour désactiver)
        """
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)
        
        # Configuration email
        self.email_enabled = email_config is not None
        self.email_config = email_config or {}
        
        # Configuration du logger
        self.logger = logging.getLogger("feedback_manager")
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
        
        # État interne
        self.feedback_callbacks = []
        self.scheduled_emails = []
        self.email_thread = None
        self.stop_email_thread = threading.Event()
        
        # Démarrer le thread de traitement des emails si activé
        if self.email_enabled:
            self._start_email_thread()
    
    def _start_email_thread(self):
        """Démarre le thread de traitement des emails programmés"""
        self.stop_email_thread.clear()
        self.email_thread = threading.Thread(target=self._process_email_queue, daemon=True)
        self.email_thread.start()
        self.logger.info("Thread de traitement des emails démarré")
    
    def _process_email_queue(self):
        """Traite la file d'attente des emails programmés"""
        while not self.stop_email_thread.is_set():
            now = datetime.now()
            
            # Vérifier les emails à envoyer
            emails_to_send = [email for email in self.scheduled_emails if email["scheduled_time"] <= now]
            
            # Envoyer les emails
            for email in emails_to_send:
                try:
                    self._send_email(
                        recipients=email["recipients"],
                        subject=email["subject"],
                        body=email["body"],
                        html_body=email.get("html_body"),
                        attachments=email.get("attachments", [])
                    )
                    self.logger.info(f"Email programmé envoyé à {', '.join(email['recipients'])}")
                except Exception as e:
                    self.logger.error(f"Erreur lors de l'envoi d'un email programmé: {str(e)}")
                
                # Supprimer l'email de la file d'attente
                self.scheduled_emails = [e for e in self.scheduled_emails if e["id"] != email["id"]]
            
            # Attendre avant la prochaine vérification
            time.sleep(60)  # Vérifier chaque minute
    
    def stop(self):
        """Arrête le thread de traitement des emails"""
        if self.email_thread:
            self.stop_email_thread.set()
            self.email_thread.join(timeout=5)
            self.logger.info("Thread de traitement des emails arrêté")
    
    def register_feedback_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """
        Enregistre une fonction de callback pour les nouveaux feedbacks
        
        Args:
            callback: Fonction appelée avec le feedback comme argument
        """
        self.feedback_callbacks.append(callback)
        self.logger.info(f"Nouveau callback de feedback enregistré, total: {len(self.feedback_callbacks)}")
    
    def store_feedback(self, feedback_data: Dict[str, Any]) -> str:
        """
        Stocke un feedback utilisateur
        
        Args:
            feedback_data: Données du feedback
            
        Returns:
            ID unique du feedback
        """
        # Ajouter des métadonnées
        feedback_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        
        enriched_feedback = {
            "id": feedback_id,
            "timestamp": timestamp,
            "data": feedback_data
        }
        
        # Sauvegarder dans un fichier JSON
        feedback_file = os.path.join(self.storage_dir, f"{feedback_id}.json")
        with open(feedback_file, 'w', encoding='utf-8') as f:
            json.dump(enriched_feedback, f, indent=2)
        
        self.logger.info(f"Feedback {feedback_id} enregistré")
        
        # Appeler les callbacks
        for callback in self.feedback_callbacks:
            try:
                callback(enriched_feedback)
            except Exception as e:
                self.logger.error(f"Erreur dans un callback de feedback: {str(e)}")
        
        return feedback_id
    
    def get_feedback(self, feedback_id: str) -> Optional[Dict[str, Any]]:
        """
        Récupère un feedback par son ID
        
        Args:
            feedback_id: ID du feedback
            
        Returns:
            Données du feedback ou None si non trouvé
        """
        feedback_file = os.path.join(self.storage_dir, f"{feedback_id}.json")
        
        if not os.path.exists(feedback_file):
            return None
        
        try:
            with open(feedback_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Erreur lors de la lecture du feedback {feedback_id}: {str(e)}")
            return None
    
    def get_all_feedback(self, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """
        Récupère tous les feedbacks dans une période donnée
        
        Args:
            start_date: Date de début (None pour aucune limite)
            end_date: Date de fin (None pour aucune limite)
            
        Returns:
            Liste des feedbacks
        """
        feedbacks = []
        
        for filename in os.listdir(self.storage_dir):
            if not filename.endswith('.json'):
                continue
            
            try:
                with open(os.path.join(self.storage_dir, filename), 'r', encoding='utf-8') as f:
                    feedback = json.load(f)
                    
                # Filtrer par date si nécessaire
                if start_date or end_date:
                    feedback_date = datetime.fromisoformat(feedback["timestamp"])
                    
                    if start_date and feedback_date < start_date:
                        continue
                    
                    if end_date and feedback_date > end_date:
                        continue
                
                feedbacks.append(feedback)
                
            except Exception as e:
                self.logger.error(f"Erreur lors de la lecture du fichier {filename}: {str(e)}")
        
        # Trier par date
        return sorted(feedbacks, key=lambda x: x["timestamp"], reverse=True)
    
    def _send_email(self, recipients: List[str], subject: str, body: str, html_body: Optional[str] = None, attachments: List[str] = []) -> bool:
        """
        Envoie un email
        
        Args:
            recipients: Liste des destinataires
            subject: Sujet de l'email
            body: Corps du message (texte)
            html_body: Corps du message (HTML, optionnel)
            attachments: Liste des chemins de fichiers à joindre
            
        Returns:
            True si envoyé avec succès, False sinon
        """
        if not self.email_enabled:
            self.logger.warning("L'envoi d'email est désactivé")
            return False
        
        try:
            # Créer le message
            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = self.email_config.get('sender', 'noreply@example.com')
            msg['To'] = ', '.join(recipients)
            
            # Ajouter la version texte
            msg.attach(MIMEText(body, 'plain'))
            
            # Ajouter la version HTML si fournie
            if html_body:
                msg.attach(MIMEText(html_body, 'html'))
            
            # Ajouter les pièces jointes
            for attachment_path in attachments:
                if os.path.exists(attachment_path):
                    with open(attachment_path, 'rb') as f:
                        attachment = MIMEApplication(f.read())
                    
                    attachment_name = os.path.basename(attachment_path)
                    attachment.add_header('Content-Disposition', f'attachment; filename="{attachment_name}"')
                    msg.attach(attachment)
            
            # Configurer le serveur SMTP
            smtp_server = self.email_config.get('smtp_server', 'localhost')
            smtp_port = self.email_config.get('smtp_port', 25)
            use_ssl = self.email_config.get('use_ssl', False)
            use_tls = self.email_config.get('use_tls', False)
            
            # Se connecter au serveur
            if use_ssl:
                server = smtplib.SMTP_SSL(smtp_server, smtp_port)
            else:
                server = smtplib.SMTP(smtp_server, smtp_port)
            
            if use_tls:
                server.starttls()
            
            # S'authentifier si nécessaire
            if 'username' in self.email_config and 'password' in self.email_config:
                server.login(self.email_config['username'], self.email_config['password'])
            
            # Envoyer l'email
            server.send_message(msg)
            server.quit()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Erreur lors de l'envoi de l'email: {str(e)}")
            return False
    
    def schedule_email(self, recipients: List[str], subject: str, body: str, 
                      scheduled_time: datetime, html_body: Optional[str] = None, 
                      attachments: List[str] = []) -> Optional[str]:
        """
        Programme l'envoi d'un email
        
        Args:
            recipients: Liste des destinataires
            subject: Sujet de l'email
            body: Corps du message (texte)
            scheduled_time: Date et heure d'envoi
            html_body: Corps du message (HTML, optionnel)
            attachments: Liste des chemins de fichiers à joindre
            
        Returns:
            ID de l'email programmé ou None si échoué
        """
        if not self.email_enabled:
            self.logger.warning("L'envoi d'email est désactivé")
            return None
        
        try:
            email_id = str(uuid.uuid4())
            
            self.scheduled_emails.append({
                "id": email_id,
                "recipients": recipients,
                "subject": subject,
                "body": body,
                "html_body": html_body,
                "attachments": attachments,
                "scheduled_time": scheduled_time,
                "created_at": datetime.now()
            })
            
            self.logger.info(f"Email programmé pour {scheduled_time.isoformat()}")
            return email_id
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la programmation de l'email: {str(e)}")
            return None
    
    def generate_report(self, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Génère un rapport sur les feedbacks
        
        Args:
            start_date: Date de début (None pour dernière semaine)
            end_date: Date de fin (None pour maintenant)
            
        Returns:
            Rapport avec statistiques et analyses
        """
        # Définir la période par défaut (dernière semaine)
        if end_date is None:
            end_date = datetime.now()
        
        if start_date is None:
            start_date = end_date - timedelta(days=7)
        
        # Récupérer les feedbacks pour la période
        feedbacks = self.get_all_feedback(start_date, end_date)
        
        if not feedbacks:
            return {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "feedback_count": 0,
                "average_rating": None,
                "summary": "Aucun feedback disponible pour cette période"
            }
        
        # Analyser les feedbacks
        ratings = [fb["data"].get("rating") for fb in feedbacks if "rating" in fb["data"]]
        valid_ratings = [r for r in ratings if r is not None]
        
        # Catégoriser les commentaires (très basique)
        comments = [fb["data"].get("comment", "") for fb in feedbacks if "comment" in fb["data"]]
        positive_count = sum(1 for c in comments if any(word in c.lower() for word in ["bien", "super", "excellent", "bravo"]))
        negative_count = sum(1 for c in comments if any(word in c.lower() for word in ["problème", "mauvais", "erreur", "bug"]))
        
        # Générer le rapport
        report = {
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "feedback_count": len(feedbacks),
            "average_rating": sum(valid_ratings) / len(valid_ratings) if valid_ratings else None,


            nstructions détaillées pour ajouter des sections sur les boucles de feedback et l'automatisation des emails
1. Section sur les boucles de feedback
Créez une nouvelle section 18.7 intitulée "Boucles de feedback" dans le document PYTHON_INTEGRATION_GUIDE.md. Cette section devrait inclure:
Structure recommandée
18.7.1 Architecture des boucles de feedback
18.7.2 Implémentation du FeedbackManager
18.7.3 API pour la collecte et le traitement des feedbacks
18.7.4 Intégration avec le système d'entraînement
Contenu clé à inclure
Architecture des boucles de feedback:
Un diagramme ou description montrant comment les feedbacks utilisateurs circulent dans le système
Comment les feedbacks sont utilisés pour améliorer les modèles
Types de feedback supportés (binaire, échelle, texte libre, etc.)
Implémentation du FeedbackManager:
Une classe Python qui gère la collecte, le stockage et l'analyse des feedbacks
Méthodes pour collecter, filtrer et prioriser les feedbacks
Intégration avec la base de données pour le stockage persistant
Logique pour transformer les feedbacks en données d'entraînement
API de feedback:
Endpoints REST pour soumettre différents types de feedback
Endpoints pour obtenir des statistiques et analyses sur les feedbacks
Sécurité et validation des données de feedback
Intégration avec le système d'entraînement:
Comment les feedbacks sont utilisés pour générer de nouvelles données d'entraînement
Mécanisme pour déclencher automatiquement le ré-entraînement basé sur les feedbacks
Évaluation de l'impact des feedbacks sur la qualité du modèle
2. Section sur l'automatisation des emails
Créez une nouvelle section 18.8 intitulée "Automatisation des emails" dans le document:
Structure recommandée
18.8.1 Architecture du système d'emails
18.8.2 Implémentation du EmailManager
18.8.3 Templates et personnalisation
18.8.4 API pour l'envoi d'emails automatisés
18.8.5 Intégration avec le système de notification d'événements
Contenu clé à inclure
Architecture du système d'emails:
Comment le système d'emails s'intègre avec les autres composants
Gestion des files d'attente pour les emails
Mécanismes de throttling et rate limiting
Implémentation du EmailManager:
Classe Python pour gérer les templates, la personnalisation et l'envoi d'emails
Gestion des pièces jointes et du contenu multimédia
Support pour HTML et texte brut
Gestion des rebonds et des erreurs d'envoi
Templates et personnalisation:
Système de templates avec Jinja2
Variables de substitution et logique conditionnelle
Stockage et versionnement des templates
API pour l'envoi d'emails:
Endpoints pour programmer l'envoi d'emails
Options pour les emails récurrents ou déclenchés par des événements
Métriques et suivi des emails envoyés
Intégration avec le système d'événements:
Comment les emails sont déclenchés par des événements du système (utiliser le RagKagEventType)
Abonnement à des événements spécifiques pour l'envoi d'emails
3. Instructions d'intégration avec le code existant
Intégrations côté NestJS:
Mise à jour du EventBusService:
Ajouter des types d'événements pour les feedbacks et les notifications par email:
FEEDBACK_RECEIVED
FEEDBACK_PROCESSED
EMAIL_QUEUED
EMAIL_SENT
EMAIL_FAILED
Créer un service de feedback:
Implémenter un FeedbackService dans le module rag-kag/core
Ce service devrait communiquer avec l'API Python pour soumettre des feedbacks
Il devrait émettre des événements via EventBusService
Créer un service d'email:
Implémenter un EmailService dans le module rag-kag/core
Utiliser ce service pour envoyer des emails via l'API Python
S'intégrer avec EventBusService pour les notifications
Intégration avec le ModelTrainingService:
1. Mise à jour pour utiliser les feedbacks:
Ajouter une méthode trainWithFeedback() qui utilise les données de feedback
Configurer un seuil de feedback pour déclencher un entraînement automatique
S'assurer que les résultats sont stockés dans le graphe de connaissances
4. Considérations de sécurité et performance
1. Sécurité:
Validation stricte des entrées pour les feedbacks et les templates d'email
Protection contre les injections et le spam
Gestion sécurisée des PII (informations personnelles identifiables)
Performance:
Traitement asynchrone des feedbacks
File d'attente pour les emails avec priorités
Mécanismes de repli en cas d'indisponibilité des services
5. Tests et validation
1. Tests unitaires:
Tests pour les services de feedback et d'email
Mocks pour les API externes
Tests d'intégration:
Vérification du flux complet depuis la soumission du feedback jusqu'à l'entraînement
Validation du système d'email de bout en bout