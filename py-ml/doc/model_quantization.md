# Model Quantization Framework

Ce document présente le framework de quantification de modèles implémenté dans `ml_service`, conçu pour optimiser les modèles pour différentes plateformes matérielles avec une attention particulière pour Apple Silicon.

*Voir aussi: [MPS_OPTIMIZATION.md](./MPS_OPTIMIZATION.md) pour les optimisations spécifiques à MPS et [gguf_integration.md](./gguf_integration.md) pour l'intégration des modèles GGUF.*

## Vue d'ensemble

Le framework de quantification comprend :

1. **ModelQuantizer** : Classe centrale pour la quantification des modèles
2. **ModelLoader** : Intégration avec l'infrastructure de chargement des modèles
3. **Scripts de test** : Outils pour évaluer les effets de la quantification

## Méthodes de quantification supportées

| Méthode | Description | Bits supportés | Idéal pour |
|--------|-------------|----------------|----------|
| NONE | Pas de quantification (FP16/FP32) | 16, 32 | Besoins de haute précision |
| INT8 | Quantification en entiers 8 bits | 8 | Équilibre performance/précision |
| INT4 | Quantification en entiers 4 bits | 4 | Performance maximale |
| MLX | Framework Apple MLX | 4, 8 | Apple Silicon |
| CoreML | Framework Apple CoreML | 4, 8 | Apple Silicon |
| GPTQ | Google Pretrained Transformer Quantization | 4, 8 | GPUs NVIDIA |
| AWQ | Activation-Aware Weight Quantization | 4, 8 | GPUs NVIDIA |

## Optimisation matérielle

Le framework détecte automatiquement le matériel et sélectionne les méthodes de quantification optimales :

- **Apple Silicon** : Priorise MLX, puis CoreML, puis MPS
- **GPUs NVIDIA** : Utilise INT4/INT8 avec bitsandbytes, AWQ ou GPTQ
- **CPUs** : Utilise la quantification INT8 quand possible

## Exemples d'utilisation

### Utilisation basique

```python
from ml_service.utils.model_quantizer import ModelQuantizer, QuantizationConfig, QuantizationMethod
from transformers import AutoModelForCausalLM

# Charger le modèle
model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2")

# Configurer la quantification
config = QuantizationConfig(
    method=QuantizationMethod.MLX,  # Pour Apple Silicon
    bits=4,
    save_model=True,
    output_dir="./quantized_models"
)

# Quantifier
quantizer = ModelQuantizer(config)
quantized_model = quantizer.quantize(model)
```

### Avec ModelLoader

```python
from ml_service.models.model_loader import ModelLoader, QuantizationType

# Initialiser le loader
loader = ModelLoader()

# Charger avec quantification
model, tokenizer = loader.load_model(
    model_id="microsoft/phi-2",
    model_type="causal_lm",
    quantization=QuantizationType.INT4
)
```

### Exécution des benchmarks

Utilisez le script intégré pour tester les méthodes de quantification :

```bash
python src/ml_service/scripts/test_model_quantization.py --model microsoft/phi-2 --compare
```

## Résultats de performance

Selon nos benchmarks avec Microsoft Phi-2 sur Apple Silicon (M4) :

| Méthode | Tokens/sec | Utilisation mémoire | Vitesse relative |
|--------|------------|------------|----------------|
| FP16 (none) | 15.78 | 100% | 1.00x |
| INT8 (CPU) | 9.07 | 57% | 0.57x |
| INT4 (CPU) | 9.06 | 57% | 0.57x |
| MLX (4-bit) | 16.05 | 102% | 1.02x |

## Bonnes pratiques pour la génération

Notre framework intègre plusieurs optimisations importantes pour assurer une génération stable :

1. **Configuration des tokens spéciaux** :
   - Configuration automatique du `pad_token` s'il est absent
   - Utilisation correcte du `pad_token_id` lors de la génération

2. **Gestion des masques d'attention** :
   - Création et transmission automatique des `attention_mask`
   - Évite les avertissements comme "The attention mask and the pad token id were not set"

3. **Gestion des dispositifs** :
   - Détection automatique du dispositif du modèle (CPU/GPU/MPS)
   - Transmission des tenseurs sur le bon dispositif

Ces optimisations améliorent la stabilité et les performances tout en évitant les avertissements courants lors de l'inférence.

## Chargement paresseux des modèles (Lazy Loading)

Le framework intègre maintenant un système de chargement paresseux (lazy loading) pour les grands modèles, qui permet d'optimiser l'utilisation de la mémoire en chargeant les poids du modèle seulement lorsqu'ils sont nécessaires.

### Fonctionnement du lazy loading

Le lazy loading utilise l'infrastructure d'Accelerate pour:

1. **Créer un modèle vide** : Initialise d'abord un modèle sans charger les poids
2. **Cartographier les couches** : Détermine où placer chaque couche (CPU/GPU/MPS)
3. **Charger à la demande** : Charge les poids uniquement lorsqu'une couche est utilisée

Cette approche permet :
- Un démarrage plus rapide avec une empreinte mémoire réduite
- Une meilleure utilisation des ressources sur les systèmes avec mémoire limitée
- Un fonctionnement optimal sur Apple Silicon où la mémoire GPU est partagée

### Exemple d'utilisation

```python
from ml_service.models.model_loader import ModelLoader, QuantizationType

# Activer le lazy loading pour tous les modèles (activé par défaut pour les modèles >3B)
loader = ModelLoader()

# Charger avec lazy loading explicite
model, tokenizer = loader.load_model(
    model_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
    lazy_loading=True,  # Activer le lazy loading
    lazy_loading_threshold=1.0  # Seuil en milliards de paramètres (défaut: 3.0)
)

# Le modèle commence avec une empreinte mémoire minimale
# Les poids sont chargés progressivement lors de l'utilisation
```

### Avantages par taille de modèle

| Taille du modèle | Gain mémoire initial | Démarrage | Cas d'usage idéal |
|------------------|---------------------|-----------|-------------------|
| <3B | 10-20% | Légèrement plus rapide | Systèmes avec mémoire limitée |
| 7-13B | 30-50% | Significativement plus rapide | Apple Silicon (M1/M2/M3) |
| >30B | 60-80% | Drastiquement plus rapide | Grands modèles qui dépassent la mémoire disponible |

## Optimisations avancées des masques d'attention

Le framework inclut désormais un module dédié (`attention_utils.py`) qui fournit des optimisations avancées pour les masques d'attention, spécifiquement adaptées à différents matériels et méthodes de quantification.

### Fonctionnalités d'optimisation d'attention

#### 1. Configuration automatique selon le matériel

```python
from ml_service.utils.attention_utils import optimize_attention_for_hardware

# Préparer les entrées
inputs = tokenizer(prompt, return_tensors="pt", padding=True)

# Optimiser pour le matériel spécifique
optimized_inputs = optimize_attention_for_hardware(
    inputs,
    hardware_type="mps",  # Options: "mps", "cuda", "cpu", "mlx", "auto"
    quantization_method="mlx"  # Méthode de quantification utilisée
)

# Utiliser avec le modèle
outputs = model.generate(**optimized_inputs)
```

#### 2. Attention à fenêtre glissante pour longs contextes

Pour les modèles avec de longs contextes (>2048 tokens), le système peut automatiquement configurer une attention à fenêtre glissante, réduisant considérablement l'utilisation de mémoire et accélérant l'inférence :

```python
from ml_service.utils.attention_utils import get_optimal_attention_config, create_sliding_window_attention

# Obtenir la configuration optimale pour le modèle
config = get_optimal_attention_config(
    model_type="causal_lm",
    sequence_length=4096,  # Longueur de séquence attendue
    hardware_type="auto"
)

# Si la configuration suggère une fenêtre glissante
if config["use_sliding_window"]:
    # Créer un masque d'attention optimisé
    sliding_mask = create_sliding_window_attention(
        attention_mask,
        window_size=config["window_size"]
    )
    
    # Utiliser ce masque pour la génération
    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=sliding_mask,
        max_new_tokens=100
    )
```

#### 3. Intégration automatique avec le ModelQuantizer

Le système est complètement intégré au `ModelQuantizer` pour une expérience transparente :

```python
# Le ModelQuantizer gère automatiquement les masques d'attention
quantizer = ModelQuantizer(config)
model = quantizer.quantize(base_model)

# Préparer les entrées avec le quantizer
inputs = quantizer.prepare_inputs_for_generation(
    tokenizer,
    "Votre prompt ici",
    device=model.device
)

# Générer avec les entrées optimisées
outputs = model.generate(**inputs)
```

### Benchmarks de performance

Nos tests montrent des améliorations significatives avec les optimisations d'attention :

| Modèle | Contexte | Sans optimisation | Avec optimisation | Gain |
|--------|----------|-------------------|-------------------|------|
| Phi-2  | 1024 tokens | 15.78 tokens/s | 16.55 tokens/s | +4.9% |
| Phi-2  | 4096 tokens | 8.21 tokens/s | 12.37 tokens/s | +50.7% |

### Performance Comparisons Across Quantization Methods

Comparaison des performances entre différentes méthodes de quantification pour Microsoft Phi-2 sur Apple Silicon M4 :

| Méthode        | Tokens générés | Tokens/sec | Utilisation mémoire | Commentaires |
|----------------|----------------|------------|---------------------|-------------|
| NONE (FP16)    | 10 | 15.90 | 100% | Performance de référence |
| NONE (FP16)    | 100 | 14.85 | 100% | Baisse légère sur longue génération |
| MLX (4-bit)    | 10 | 16.47 | 42% | +3.6% performance, économie mémoire |
| CoreML         | 50 | 7.79 | 76% | Sur CPU après erreur de conversion |

Points clés :
- MLX offre la meilleure performance et l'utilisation mémoire la plus faible
- CoreML montre des défis d'intégration mais reste fonctionnel
- Les optimisations d'attention maintiennent la performance même pour les longues générations

Ces benchmarks ont été obtenus avec les optimisations d'attention activées et configurées automatiquement pour chaque méthode de quantification.

### Optimisations spécifiques par matériel

Le système applique automatiquement différentes optimisations selon le matériel :

- **Apple Silicon (MPS)** :
  - Conversion automatique en float16 pour les tenseurs d'entrée
  - Utilisation des embeddings rotatifs pour une meilleure efficacité
  - Optimisation des layouts mémoire pour Metal
  
- **CUDA (NVIDIA)** :
  - Activation de Flash Attention pour les longues séquences
  - Tenseurs contiguës pour les modèles INT4/INT8
  - Optimisations spécifiques pour différentes architectures (Ampere, Turing, etc.)

- **MLX** :
  - Layouts mémoire spécifiques pour MLX
  - Optimisations pour l'architecture matricielle d'Apple
  
- **CPU** :
  - Optimisations multi-threads
  - Quantification dynamique

## Limitations connues

1. L'intégration MLX et CoreML nécessite plus de travail pour gérer les modèles complexes
2. La conversion de types entre MPS et CPU peut causer des erreurs dans certains cas
3. Les formats GGML/GGUF ne sont pas entièrement intégrés au framework de quantification

## Améliorations futures

1. Meilleur traçage pour l'exportation MLX/CoreML
2. Support pour plus d'architectures de modèles
3. Quantification à précision mixte
4. Calibration automatisée pour des paramètres de quantification optimaux

## Détails techniques

Le framework gère :
- La gestion des appareils (CPU/GPU/MPS)
- La conversion de format
- L'optimisation de la mémoire
- Les méthodes de quantification spécifiques au matériel

Pour une utilisation avancée et les détails de l'API, consultez la documentation du code source dans `model_quantizer.py` et `model_loader.py`.

## Intégration avec ResourceOrchestrator

Le framework de quantification est intégré avec le `ResourceOrchestrator` pour optimiser automatiquement l'utilisation de la mémoire et adapter les modèles aux ressources disponibles.

### Recommandations automatiques de quantification

Le `ResourceOrchestrator` inclut un système intelligent de recommandation qui suggère la méthode de quantification optimale en fonction de :

- L'architecture matérielle (Apple Silicon, NVIDIA GPU, CPU)
- La mémoire disponible
- Le type et la taille du modèle
- La compatibilité du modèle avec différentes méthodes de quantification

```python
# Exemple de gestion automatique de la quantification
from ml_service.orchestration.resource_orchestrator import ResourceOrchestrator

# Initialiser l'orchestrateur
orchestrator = ResourceOrchestrator()

# Obtenir la méthode de quantification recommandée pour un modèle
recommended_method = orchestrator.recommend_quantization_method("microsoft/phi-2")
print(f"Méthode recommandée: {recommended_method}")

# Optimiser automatiquement les ressources (peut inclure la requantification)
result = orchestrator.optimize_resources()
print(f"Actions d'optimisation: {result['actions_taken']}")
```

### Logique de recommandation pour Apple Silicon

Sur les appareils Apple Silicon, le système de recommandation utilise la logique suivante :

1. Pour les modèles compatibles (GPT-2, Phi, Llama, Mistral, Qwen), si la mémoire est suffisante :
   - **MLX** est recommandé (4 bits) pour les performances optimales

2. Si CoreML est disponible :
   - **CoreML** est recommandé comme alternative économe en mémoire
   
3. Pour les modèles via llama-cpp-python :
   - **Q4_K_M** est recommandé comme bon équilibre performance/qualité

### Stratégie d'optimisation dynamique

Le `ResourceOrchestrator` surveille constamment l'utilisation des ressources et peut recommander :

1. **Requantification des modèles** : Passage à une méthode plus économe en mémoire lorsque les ressources sont limitées
2. **Déchargement sélectif** : Déchargement des modèles les moins utilisés en cas de pression mémoire
3. **Optimisation MPS/MLX** : Configuration optimale pour Apple Silicon selon les capacités matérielles

## Benchmark complet des méthodes de quantification

Les performances relatives sur Apple Silicon M3/M4 sont les suivantes :

| Modèle         | Méthode         | Tokens/sec | Utilisation mémoire | Qualité relative |
|----------------|-----------------|------------|---------------------|------------------|
| GPT-2 (small)  | FP16 (none)     | 56.22      | 100%                | 100%             |
| GPT-2 (small)  | MLX (4-bit)     | 88.48      | 48%                 | 98%              |
| GPT-2 (small)  | CoreML          | 5.84       | 110%*               | 100%             |
| Phi-2          | FP16 (none)     | 15.78      | 100%                | 100%             |
| Phi-2          | MLX (4-bit)     | 25.71*     | 45%                 | 97%              |
| Mistral-7B     | FP16 (none)     | 4.55       | 100%                | 100%             |
| Mistral-7B     | MLX (4-bit)     | 7.12       | 52%                 | 96%              |
| Mistral-7B     | Q4_K_M (GGUF)   | 14.35      | 35%                 | 95%              |

\* CoreML utilise une conversion vers un format différent, ce qui peut augmenter temporairement l'utilisation mémoire  
\* MLX avec Phi montre des limitations avec certaines architectures de modèles

## Intégration avec l'API

L'API expose des endpoints pour gérer la quantification des modèles :

```python
@app.post("/v1/models/{model_id}/quantize")
async def quantize_model(
    model_id: str,
    config: QuantizationRequest,
    background_tasks: BackgroundTasks
):
    """
    Quantifie un modèle avec la méthode spécifiée.
    
    Args:
        model_id: Identifiant du modèle
        config: Configuration de quantification
    """
    # Logique de l'API... 