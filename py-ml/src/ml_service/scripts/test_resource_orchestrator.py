#!/usr/bin/env python3
"""
Test du ResourceOrchestrator amélioré.

Ce script teste le ResourceOrchestrator et ses recommandations de quantification
sur différentes plateformes matérielles, en particulier Apple Silicon.
"""

import os
import sys
import logging
import time
import argparse
from typing import Dict, Any, List
import platform
import torch

# Configurer le logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger("test_resource_orchestrator")

# Ajouter le répertoire parent au PYTHONPATH
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(parent_dir)

from src.ml_service.orchestration.resource_orchestrator import ResourceOrchestrator, get_resource_orchestrator
from src.ml_service.api.model_manager import ModelManager, get_model_manager
from src.ml_service.orchestration.model_lifecycle import ModelLifecycleManager, get_model_lifecycle_manager

def print_system_info():
    """Affiche les informations système."""
    logger.info(f"Système: {platform.system()} {platform.release()}")
    logger.info(f"Processeur: {platform.processor()}")
    logger.info(f"Python: {platform.python_version()}")
    
    if torch.cuda.is_available():
        logger.info(f"GPU CUDA: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA version: {torch.version.cuda}")
    
    is_apple_silicon = platform.processor() == "arm" and platform.system() == "Darwin"
    if is_apple_silicon:
        has_mps = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
        logger.info(f"Apple Silicon: Oui - MPS disponible: {has_mps}")
    
    logger.info(f"Torch version: {torch.__version__}")

def test_resource_check():
    """Teste la vérification des ressources."""
    logger.info("=== Test de la vérification des ressources ===")
    
    orchestrator = get_resource_orchestrator()
    resources = orchestrator.check_resources(force=True)
    
    logger.info(f"Mémoire totale: {resources['memory_total_mb']/1024:.2f} GB")
    logger.info(f"Mémoire utilisée: {resources['memory_used_mb']/1024:.2f} GB ({resources['memory_used_percent']:.1f}%)")
    
    if 'gpu_memory_total_mb' in resources:
        logger.info(f"Mémoire GPU totale: {resources['gpu_memory_total_mb']/1024:.2f} GB")
        logger.info(f"Mémoire GPU utilisée: {resources['gpu_memory_used_mb']/1024:.2f} GB ({resources['gpu_memory_used_percent']:.1f}%)")

def test_quantization_recommendations():
    """Teste les recommandations de quantification pour différents modèles."""
    logger.info("=== Test des recommandations de quantification ===")
    
    # Modèles populaires à tester
    test_models = [
        "gpt2",
        "facebook/opt-350m",
        "microsoft/phi-2",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "mistralai/Mistral-7B-v0.1",
        "meta-llama/Llama-2-7b",
        "microsoft/phi-3-mini"
    ]
    
    orchestrator = get_resource_orchestrator()
    
    # Créer des mocks d'informations modèles
    mock_model_info = {
        "gpt2": {
            "id": "gpt2",
            "model_type": "causal_lm",
            "model_family": "gpt2",
            "size_mb": 500,
            "backend": "transformers",
            "available_libraries": ["transformers", "mlx", "coremltools"]
        },
        "facebook/opt-350m": {
            "id": "facebook/opt-350m",
            "model_type": "causal_lm",
            "model_family": "opt",
            "size_mb": 750,
            "backend": "transformers",
            "available_libraries": ["transformers", "mlx", "coremltools"]
        },
        "microsoft/phi-2": {
            "id": "microsoft/phi-2",
            "model_type": "causal_lm",
            "model_family": "phi",
            "size_mb": 2500,
            "backend": "transformers",
            "available_libraries": ["transformers", "mlx", "coremltools"]
        },
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B": {
            "id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
            "model_type": "causal_lm",
            "model_family": "qwen",
            "size_mb": 3000,
            "backend": "transformers",
            "available_libraries": ["transformers", "mlx", "coremltools"]
        },
        "mistralai/Mistral-7B-v0.1": {
            "id": "mistralai/Mistral-7B-v0.1",
            "model_type": "causal_lm",
            "model_family": "mistral",
            "size_mb": 14000,
            "backend": "transformers",
            "available_libraries": ["transformers", "mlx", "coremltools"]
        },
        "meta-llama/Llama-2-7b": {
            "id": "meta-llama/Llama-2-7b",
            "model_type": "causal_lm",
            "model_family": "llama",
            "size_mb": 14000,
            "backend": "transformers",
            "available_libraries": ["transformers", "mlx", "coremltools"]
        },
        "microsoft/phi-3-mini": {
            "id": "microsoft/phi-3-mini",
            "model_type": "causal_lm",
            "model_family": "phi",
            "size_mb": 7000,
            "backend": "transformers",
            "available_libraries": ["transformers", "mlx", "coremltools"]
        }
    }
    
    # Pour chaque modèle, simuler les informations et obtenir les recommandations
    for model_id in test_models:
        # Simuler un modèle chargé
        if model_id in mock_model_info:
            # Monkey-patch pour les tests
            if not hasattr(orchestrator.model_manager, "get_model_info"):
                orchestrator.model_manager.get_model_info = lambda x: mock_model_info.get(x, {})
            if not hasattr(orchestrator.model_manager, "is_model_loaded"):
                orchestrator.model_manager.is_model_loaded = lambda x: x in mock_model_info
                
            # Obtenir la recommandation
            recommendation = orchestrator.recommend_quantization_method(model_id)
            logger.info(f"Modèle: {model_id} -> Quantification recommandée: {recommendation}")
        else:
            logger.warning(f"Modèle {model_id} non trouvé dans les mocks")

def test_optimize_resources():
    """Teste l'optimisation des ressources avec différents scénarios."""
    logger.info("=== Test de l'optimisation des ressources ===")
    
    orchestrator = get_resource_orchestrator()
    
    # Vérifier les ressources avant
    resources_before = orchestrator.check_resources(force=True)
    logger.info(f"Ressources avant optimisation: {resources_before['memory_used_mb']/1024:.2f} GB "
               f"({resources_before['memory_used_percent']:.1f}%)")
    
    # Exécuter l'optimisation
    result = orchestrator.optimize_resources()
    
    # Afficher les résultats
    logger.info(f"Optimisation réussie: {result['optimized']}")
    logger.info(f"Actions réalisées: {result['actions_taken']}")
    logger.info(f"Mémoire avant: {result['memory_before']/1024:.2f} GB")
    logger.info(f"Mémoire après: {result['memory_after']/1024:.2f} GB")
    
    # Afficher les métriques
    metrics = orchestrator.get_metrics()
    logger.info(f"Métriques: {metrics}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test du ResourceOrchestrator')
    parser.add_argument('--resources', action='store_true', help='Tester la vérification des ressources')
    parser.add_argument('--recommendations', action='store_true', help='Tester les recommandations de quantification')
    parser.add_argument('--optimize', action='store_true', help='Tester l\'optimisation des ressources')
    parser.add_argument('--all', action='store_true', help='Exécuter tous les tests')
    
    args = parser.parse_args()
    
    # Si aucun argument spécifique, exécuter tous les tests
    if not (args.resources or args.recommendations or args.optimize):
        args.all = True
    
    print_system_info()
    
    if args.resources or args.all:
        test_resource_check()
    
    if args.recommendations or args.all:
        test_quantization_recommendations()
    
    if args.optimize or args.all:
        test_optimize_resources()
    
    logger.info("Tests terminés.") 