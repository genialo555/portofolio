# ResourceOrchestrator - Gestion intelligente des ressources système

## Introduction

Le `ResourceOrchestrator` est un composant central du système RAG/KAG responsable de l'allocation optimale des ressources système (CPU, RAM, GPU) entre les différents composants (modèles, agents, etc.). Il implémente une gestion dynamique des ressources avec des optimisations spécifiques pour Apple Silicon.

## Fonctionnalités clés

- **Surveillance continue** des ressources système (CPU, RAM, GPU/MPS)
- **Optimisation automatique** pour différentes architectures matérielles
- **Recommandations de quantification** adaptées au matériel et aux modèles
- **Gestion du cycle de vie des modèles** en fonction de l'utilisation des ressources
- **Optimisations spécifiques pour Apple Silicon** (M1/M2/M3/M4)

## Architecture

```
ResourceOrchestrator
├── Surveillance des ressources
│   ├── check_resources() - Vérification des ressources disponibles
│   └── get_metrics() - Métriques d'utilisation et d'optimisation
├── Optimisation des ressources
│   ├── optimize_resources() - Optimisation automatique
│   ├── recommend_quantization_method() - Méthodes de quantification adaptées
│   └── can_load_model() - Évaluation de la faisabilité
└── Intégrations
    ├── model_manager - Gestion des modèles
    ├── model_lifecycle - Cycle de vie des modèles
    └── memory_manager - Gestion optimisée de la mémoire
```

## Optimisations pour Apple Silicon

Le `ResourceOrchestrator` implémente plusieurs optimisations spécifiques pour les puces Apple M-series :

1. **Détection automatique** des capacités matérielles (M1/M2/M3/M4)
2. **Utilisation optimisée de MPS** (Metal Performance Shaders)
3. **Support MLX** pour les opérations tensorielles
4. **Recommandations de quantification spécifiques** à Apple Silicon
5. **Gestion adaptative de la mémoire partagée** CPU/GPU

### Détection et configuration automatique

```python
# Détection automatique Apple Silicon
is_apple_silicon = platform.processor() == 'arm' and platform.system() == 'Darwin'
has_mps = is_apple_silicon and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()

# Détection du modèle de puce
if is_apple_silicon:
    # Configuration spécifique selon le modèle
    if "M1" in platform.processor():
        # Configuration M1
        metal_config = {"n_gpu_layers": 24}
    elif "M2" in platform.processor():
        # Configuration M2
        metal_config = {"n_gpu_layers": 32}
    else:  # M3/M4 et plus
        # Configuration M3/M4
        metal_config = {"n_gpu_layers": 48, "offload_kqv": True}
```

## Recommandations de quantification

Le système de recommandation utilise différentes heuristiques selon le matériel :

### Pour Apple Silicon

```python
def recommend_quantification_apple_silicon(model_id, available_memory):
    # Pour les modèles compatibles avec MLX
    if is_mlx_compatible(model_id) and (available_memory > required_memory * 0.8):
        return "mlx"  # MLX 4-bit est optimal sur M-series
    
    # Alternative: CoreML si disponible
    if coreml_available:
        return "coreml"
    
    # Pour llama-cpp avec Metal
    if is_gguf_model(model_id):
        return "q4_k_m"  # Bon équilibre performance/qualité
    
    # Par défaut
    return "int8"
```

### Pour NVIDIA GPUs

```python
def recommend_quantification_cuda(model_id, available_memory):
    if available_memory > required_memory * 0.6:
        return "int8"  # Bon équilibre pour les GPUs NVIDIA
    else:
        return "int4"  # Plus économe en mémoire
```

## Optimisation des ressources

L'optimisation des ressources suit une approche en plusieurs étapes :

1. **Nettoyage de base** : GC et libération des caches
2. **Requantification** : Passage à des méthodes plus économes si nécessaire
3. **Déchargement de modèles** : Déchargement sélectif basé sur l'utilisation

```python
# Exemple d'utilisation
orchestrator = ResourceOrchestrator()
result = orchestrator.optimize_resources()

print(f"Actions réalisées: {result['actions_taken']}")
print(f"Mémoire libérée: {result['memory_before'] - result['memory_after']} MB")
```

## Intégration avec l'API

```python
@app.get("/v1/system/resources")
async def get_system_resources():
    """Retourne l'état actuel des ressources système."""
    orchestrator = get_resource_orchestrator()
    return orchestrator.check_resources()

@app.post("/v1/system/optimize")
async def optimize_system_resources():
    """Optimise l'utilisation des ressources système."""
    orchestrator = get_resource_orchestrator()
    return orchestrator.optimize_resources()
```

## Métriques et surveillance

Le `ResourceOrchestrator` collecte plusieurs métriques clés :

- Nombre d'optimisations effectuées
- Nombre de modèles déchargés
- Mémoire récupérée (CPU et GPU)
- Temps d'inactivité des modèles
- Utilisation moyenne des ressources

Ces métriques sont utilisées pour :
- Ajuster les seuils d'optimisation
- Améliorer les politiques de déchargement
- Fournir des insights sur l'utilisation du système

## Bonnes pratiques d'utilisation

1. **Vérifier régulièrement les ressources** avant le chargement de nouveaux modèles
2. **Utiliser can_load_model()** pour vérifier la faisabilité avant chargement
3. **Configurer les seuils appropriés** selon votre matériel
4. **Optimiser en période de faible utilisation** pour minimiser l'impact

## Exemple d'utilisation complet

```python
from ml_service.orchestration.resource_orchestrator import ResourceOrchestrator

# Initialiser avec des seuils personnalisés
orchestrator = ResourceOrchestrator(
    memory_threshold=0.8,  # 80% d'utilisation mémoire avant optimisation
    gpu_memory_threshold=0.85  # 85% d'utilisation GPU avant optimisation
)

# Vérifier si un modèle peut être chargé
model_id = "mistralai/Mistral-7B-v0.1"
estimated_size_mb = 14000  # Taille estimée en MB

if orchestrator.can_load_model(model_id, estimated_size_mb):
    # Obtenir la méthode de quantification recommandée
    recommended_method = orchestrator.recommend_quantization_method(model_id)
    
    print(f"Chargement du modèle {model_id} avec quantification {recommended_method}")
    # Logique de chargement...
else:
    # Optimiser les ressources et réessayer
    result = orchestrator.optimize_resources()
    
    if result["optimized"] and orchestrator.can_load_model(model_id, estimated_size_mb):
        print(f"Après optimisation, chargement possible avec actions: {result['actions_taken']}")
        # Logique de chargement...
    else:
        print("Ressources insuffisantes même après optimisation")
```

## Conclusion

Le `ResourceOrchestrator` joue un rôle central dans l'optimisation des performances du système RAG/KAG, particulièrement sur les architectures Apple Silicon. Sa gestion intelligente des ressources et ses recommandations de quantification permettent d'obtenir les meilleures performances possibles sur différentes plateformes matérielles. 