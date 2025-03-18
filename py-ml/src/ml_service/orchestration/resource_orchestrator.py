"""
Orchestrateur de ressources système.

Ce module fournit la classe ResourceOrchestrator qui coordonne
l'allocation des ressources système (mémoire, GPU, CPU) entre
les différents composants du système: modèles, agents, etc.
"""

import logging
import time
import os
import psutil
import torch
import platform
import gc
from typing import Dict, List, Any, Optional, Union, Tuple
from functools import lru_cache

from ..config import settings
from ..api.model_manager import ModelManager, get_model_manager
from ..agents.agent_manager import AgentManager, get_agent_manager

logger = logging.getLogger("ml_api.orchestration.resource_orchestrator")

class ResourceOrchestrator:
    """
    Orchestrateur de ressources système.
    
    Cette classe coordonne l'allocation des ressources système (CPU, RAM, GPU)
    entre les différents composants qui en ont besoin: modèles, agents, etc.
    """
    
    def __init__(self, 
                model_manager: Optional[ModelManager] = None,
                agent_manager: Optional[AgentManager] = None,
                memory_threshold: float = 0.85,
                gpu_memory_threshold: float = 0.9):
        """
        Initialise l'orchestrateur de ressources.
        
        Args:
            model_manager: Gestionnaire de modèles
            agent_manager: Gestionnaire d'agents
            memory_threshold: Seuil d'utilisation mémoire (0-1) avant optimisation
            gpu_memory_threshold: Seuil d'utilisation GPU (0-1) avant optimisation
        """
        self.model_manager = model_manager or get_model_manager()
        self.agent_manager = agent_manager or get_agent_manager()
        self.memory_threshold = memory_threshold
        self.gpu_memory_threshold = gpu_memory_threshold
        
        # Détecter Apple Silicon
        self.is_apple_silicon = platform.processor() == 'arm' and platform.system() == 'Darwin'
        self.has_mps = False
        
        if self.is_apple_silicon:
            self.has_mps = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
            logger.info(f"Apple Silicon détecté, MPS disponible: {self.has_mps}")
        
        # État du système
        self.system_info = self._get_system_info()
        
        # Dernière vérification des ressources
        self.last_check = None
        self.last_check_resources = None
        
        # Métriques
        self.metrics = {
            "optimizations_performed": 0,
            "models_unloaded": 0,
            "memory_reclaimed_mb": 0,
            "gpu_memory_reclaimed_mb": 0
        }
        
        logger.info(f"ResourceOrchestrator initialisé, système: {self.system_info['os']}")
    
    def _get_system_info(self) -> Dict[str, Any]:
        """
        Récupère les informations sur le système.
        
        Returns:
            Informations système
        """
        info = {
            "os": f"{platform.system()} {platform.release()}",
            "cpu_model": platform.processor() or "Unknown",
            "cpu_cores": psutil.cpu_count(logical=False),
            "cpu_threads": psutil.cpu_count(logical=True),
            "total_memory_gb": psutil.virtual_memory().total / (1024 ** 3),
            "has_gpu": False,
            "gpu_info": None,
            "total_gpu_memory_gb": 0
        }
        
        # Détecter le GPU
        if torch.cuda.is_available():
            info["has_gpu"] = True
            info["gpu_info"] = torch.cuda.get_device_name(0)
            info["total_gpu_memory_gb"] = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        elif self.has_mps:
            info["has_gpu"] = True
            info["gpu_info"] = "Apple Silicon GPU (MPS)"
            # MPS n'expose pas directement la mémoire totale
            # On utilise une estimation basée sur la RAM totale
            info["total_gpu_memory_gb"] = info["total_memory_gb"] * 0.7  # Estimation
        
        return info
    
    def check_resources(self, force: bool = False) -> Dict[str, Any]:
        """
        Vérifie l'état actuel des ressources système.
        
        Args:
            force: Forcer une nouvelle vérification même si une récente existe
            
        Returns:
            État des ressources
        """
        current_time = time.time()
        
        # Utiliser les résultats en cache si récents
        if not force and self.last_check and (current_time - self.last_check) < 5:
            return self.last_check_resources
        
        # Récupérer l'utilisation CPU
        cpu_percent = psutil.cpu_percent(interval=0.1)
        
        # Récupérer l'utilisation mémoire
        memory = psutil.virtual_memory()
        memory_usage = memory.percent / 100.0
        
        # Par défaut, pas d'informations GPU
        gpu_memory_usage = 0.0
        gpu_memory_used_mb = 0
        gpu_memory_total_mb = 0
        
        # Vérifier l'utilisation GPU si disponible
        if torch.cuda.is_available():
            # CUDA
            gpu_memory_used = torch.cuda.memory_allocated(0)
            gpu_memory_total = torch.cuda.get_device_properties(0).total_memory
            gpu_memory_usage = gpu_memory_used / gpu_memory_total
            gpu_memory_used_mb = gpu_memory_used / (1024 ** 2)
            gpu_memory_total_mb = gpu_memory_total / (1024 ** 2)
        elif self.has_mps:
            # MPS - pas d'API directe pour la mémoire, estimation
            # On utilise la charge globale comme indicateur
            activity = psutil.getloadavg()[0] / psutil.cpu_count()
            gpu_memory_usage = min(0.95, activity)  # Estimation
            gpu_memory_used_mb = 0  # Pas disponible directement
            gpu_memory_total_mb = 0  # Pas disponible directement
        
        # Compter les modèles chargés
        models_loaded = len(self.model_manager.get_loaded_models())
        
        # Compter les agents actifs (si disponible)
        agents_active = 0
        if hasattr(self.agent_manager, 'get_active_agents'):
            agents_active = len(self.agent_manager.get_active_agents())
        
        # Déterminer si des seuils sont atteints
        memory_warning = memory_usage > self.memory_threshold
        gpu_memory_warning = gpu_memory_usage > self.gpu_memory_threshold if gpu_memory_total_mb > 0 else False
        
        # Construire le résultat
        result = {
            "cpu_usage": cpu_percent,
            "memory_usage": memory_usage,
            "memory_used_mb": memory.used / (1024 ** 2),
            "memory_total_mb": memory.total / (1024 ** 2),
            "gpu_memory_usage": gpu_memory_usage,
            "gpu_memory_used_mb": gpu_memory_used_mb,
            "gpu_memory_total_mb": gpu_memory_total_mb,
            "models_loaded": models_loaded,
            "agents_active": agents_active,
            "memory_warning": memory_warning,
            "gpu_memory_warning": gpu_memory_warning,
            "system_needs_optimization": memory_warning or gpu_memory_warning
        }
        
        # Mettre à jour le cache
        self.last_check = current_time
        self.last_check_resources = result
        
        return result
    
    def _cleanup_mps_memory(self) -> None:
        """
        Nettoie spécifiquement la mémoire MPS sur Apple Silicon.
        Cette fonction est cruciale car MPS ne libère pas automatiquement
        toute la mémoire après les opérations contrairement à CUDA.
        """
        if not (self.is_apple_silicon and self.has_mps):
            return
        
        try:
            # Force synchronization and clear caches
            if hasattr(torch.mps, 'synchronize'):
                torch.mps.synchronize()
            
            # Empty cache si disponible
            if hasattr(torch.mps, 'empty_cache'):
                torch.mps.empty_cache()
            
            # Alternative pour les versions de PyTorch qui n'ont pas empty_cache pour MPS
            dummy_tensors = []
            for _ in range(5):
                # Créer et libérer des tenseurs pour forcer le garbage collector
                dummy = torch.ones(1000, 1000, device='mps')
                dummy_tensors.append(dummy)
            
            # Libérer les tensors et forcer le garbage collection
            dummy_tensors.clear()
            gc.collect()
            
            logger.debug("Nettoyage mémoire MPS effectué")
            
        except Exception as e:
            logger.warning(f"Erreur lors du nettoyage de la mémoire MPS: {str(e)}")
    
    def optimize_resources(self) -> Dict[str, Any]:
        """
        Optimise l'utilisation des ressources système.
        
        Returns:
            Résultats de l'optimisation
        """
        # Vérifier l'état actuel des ressources
        resources = self.check_resources(force=True)
        
        # Résultat par défaut
        result = {
            "optimized": False,
            "actions_taken": [],
            "memory_before": resources["memory_used_mb"],
            "memory_after": resources["memory_used_mb"],
            "gpu_memory_before": resources.get("gpu_memory_used_mb", 0),
            "gpu_memory_after": resources.get("gpu_memory_used_mb", 0)
        }
        
        # Si l'utilisation est sous les seuils, ne rien faire
        if (resources["memory_used_mb"] < self.memory_threshold * resources["memory_total_mb"] and
            resources.get("gpu_memory_used_mb", 0) < self.gpu_memory_threshold * resources.get("gpu_memory_total_mb", 1)):
            logger.debug("Ressources suffisantes, pas d'optimisation nécessaire")
            return result
        
        logger.info(f"Optimisation des ressources nécessaire: "
                   f"Mémoire {resources['memory_used_mb']:.1f} MB / {resources['memory_total_mb']:.1f} MB, "
                   f"GPU {resources.get('gpu_memory_used_mb', 0):.1f} MB / {resources.get('gpu_memory_total_mb', 0):.1f} MB")
        
        # Actions possibles
        actions_taken = []
        
        # 1. Nettoyage de base
        gc.collect()
        if hasattr(torch, 'cuda') and torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif self.has_mps:
            self._cleanup_mps_memory()
        
        actions_taken.append("Nettoyage de base (GC + cache GPU)")
        
        # 2. Remplacer les modèles par des versions quantifiées si disponible
        loaded_models = self.model_manager.get_loaded_models_info()
        for model_info in loaded_models:
            model_id = model_info["id"]
            
            # Vérifier si ce modèle supporte la quantification
            if not self.model_manager.is_model_loaded(model_id):
                continue
            
            # Obtenir la méthode de quantification recommandée
            current_quantization = model_info.get("quantization", "none")
            recommended_quantization = self.recommend_quantization_method(model_id)
            
            # Si la méthode recommandée est différente et plus efficace en mémoire
            if recommended_quantization != current_quantization:
                logger.info(f"Remplacement du modèle {model_id} par version quantifiée "
                           f"({current_quantization} -> {recommended_quantization})")
                
                try:
                    # Décharger la version actuelle
                    self.model_manager.unload_model(model_id)
                    
                    # Recharger avec quantification
                    self.model_manager.load_model(
                        model_id, 
                        quantization=recommended_quantization
                    )
                    
                    actions_taken.append(f"Quantification du modèle {model_id} "
                                       f"({current_quantization} -> {recommended_quantization})")
                except Exception as e:
                    logger.error(f"Erreur lors de la quantification du modèle {model_id}: {str(e)}")
        
        # 3. Si nécessaire, décharger les modèles les moins utilisés
        if resources["memory_used_mb"] > self.memory_threshold * resources["memory_total_mb"]:
            # Obtenir la liste des modèles chargés
            loaded_models = self.model_manager.get_loaded_models_info()
            
            # Si plus d'un modèle est chargé, décharger le moins utilisé
            if len(loaded_models) > 1:
                # Trier par utilisation (du moins au plus utilisé)
                loaded_models.sort(key=lambda x: x.get("usage_count", 0))
                
                # Décharger le moins utilisé
                model_to_unload = loaded_models[0]["id"]
                logger.info(f"Déchargement du modèle le moins utilisé: {model_to_unload}")
                
                if self.model_manager.unload_model(model_to_unload):
                    actions_taken.append(f"Déchargement du modèle {model_to_unload}")
                    self.metrics["models_unloaded"] += 1
        
        # Vérifier l'état après optimisation
        resources_after = self.check_resources(force=True)
        
        # Mettre à jour les métriques
        self.metrics["optimizations_performed"] += 1
        self.metrics["memory_reclaimed_mb"] += (
            resources["memory_used_mb"] - resources_after["memory_used_mb"]
        )
        if "gpu_memory_used_mb" in resources and "gpu_memory_used_mb" in resources_after:
            self.metrics["gpu_memory_reclaimed_mb"] += (
                resources["gpu_memory_used_mb"] - resources_after["gpu_memory_used_mb"]
            )
        
        # Résultat final
        result["optimized"] = True
        result["actions_taken"] = actions_taken
        result["memory_after"] = resources_after["memory_used_mb"]
        result["gpu_memory_after"] = resources_after.get("gpu_memory_used_mb", 0)
        
        logger.info(f"Optimisation terminée: "
                   f"Mémoire {result['memory_before']:.1f} MB -> {result['memory_after']:.1f} MB, "
                   f"GPU {result['gpu_memory_before']:.1f} MB -> {result['gpu_memory_after']:.1f} MB")
        
        return result
    
    def recommend_quantization_method(self, model_id: str) -> str:
        """
        Recommande une méthode de quantification optimale pour un modèle.
        
        Args:
            model_id: Identifiant du modèle
            
        Returns:
            Méthode de quantification recommandée (none, int8, int4, mlx, coreml)
        """
        # Récupérer les informations du modèle
        model_info = self.model_manager.get_model_info(model_id) if self.model_manager.is_model_loaded(model_id) else None
        
        if not model_info:
            return "none"  # Par défaut, pas de quantification
        
        # Vérifier si le modèle est compatible avec MLX
        model_type = model_info.get("model_type", "")
        model_family = model_info.get("model_family", "")
        
        # Estimer la taille du modèle (en MB)
        model_size_mb = model_info.get("size_mb", 0)
        if model_size_mb == 0:
            # Estimation grossière basée sur le type de modèle
            if "7b" in model_id.lower():
                model_size_mb = 7000
            elif "13b" in model_id.lower():
                model_size_mb = 13000
            elif "70b" in model_id.lower():
                model_size_mb = 70000
            else:
                model_size_mb = 3000  # Valeur par défaut pour les petits modèles
        
        # Vérifier la mémoire disponible (en MB)
        resources = self.check_resources()
        available_memory_mb = resources["total_memory_mb"] - resources["memory_used_mb"]
        
        # Sur Apple Silicon
        if self.is_apple_silicon:
            # Vérifier la compatibilité MLX
            # MLX fonctionne bien avec gpt2, phi, llama, mistral
            compatible_with_mlx = any(family in model_id.lower() 
                                      for family in ["gpt2", "phi", "llama", "mistral", "qwen"])
            
            if compatible_with_mlx and "llama-cpp-python" not in model_info.get("backend", ""):
                # Si mémoire suffisante et compatible MLX
                if model_size_mb < available_memory_mb * 0.8:
                    return "mlx"  # MLX est généralement le plus rapide sur Apple Silicon
            
            # Si CoreML est disponible (alternative plus économe en mémoire)
            if "coremltools" in model_info.get("available_libraries", []):
                return "coreml"
            
            # Sinon, options spécifiques à llama-cpp
            if "llama-cpp-python" in model_info.get("backend", ""):
                # La quantification Q4_K_M est généralement un bon équilibre
                return "q4_k_m"
        
        # Sur GPU NVIDIA
        elif hasattr(torch, 'cuda') and torch.cuda.is_available():
            # Si mémoire suffisante pour FP16
            if model_size_mb * 0.6 < available_memory_mb:  # 60% de la taille originale
                return "int8"  # Bon équilibre performance/qualité
            else:
                return "int4"  # Plus économe en mémoire
        
        # Sur CPU
        else:
            # Privilégier la performance
            if model_size_mb < available_memory_mb * 0.5:
                return "none"  # Modèle complet
            else:
                return "int8"  # Économie de mémoire
        
        # Par défaut
        return "none"
    
    def can_load_model(self, model_id: str, model_size_mb: Optional[int] = None) -> bool:
        """
        Vérifie si un modèle peut être chargé sans dépasser les seuils.
        
        Args:
            model_id: Identifiant du modèle
            model_size_mb: Taille estimée du modèle en MB
            
        Returns:
            True si le modèle peut être chargé, False sinon
        """
        # Si le modèle est déjà chargé, autoriser
        if self.model_manager.is_model_loaded(model_id):
            return True
        
        # Vérifier l'état actuel des ressources
        resources = self.check_resources()
        
        # Si on approche déjà des limites, refuser
        if resources.get("system_needs_optimization", False):
            logger.warning(f"Impossible de charger le modèle {model_id}: système déjà proche des limites")
            return False
        
        # Si la taille du modèle n'est pas fournie, estimation conservatrice
        if model_size_mb is None:
            # Vérifier si on a des métadonnées sur ce modèle
            # TODO: Intégrer avec ModelLifecycleManager pour obtenir les tailles
            model_size_mb = 1000  # Valeur par défaut (1 GB)
        
        # Estimation de la mémoire nécessaire (modèle + overhead)
        required_memory = model_size_mb * 1.5  # +50% pour l'overhead
        
        # Mémoire disponible
        available_memory_mb = resources["memory_total_mb"] * (1 - self.memory_threshold)
        current_used_memory_mb = resources["memory_used_mb"]
        actual_available_mb = max(0, resources["memory_total_mb"] - current_used_memory_mb)
        
        # Vérifier si assez de mémoire
        if required_memory > actual_available_mb:
            logger.warning(f"Mémoire insuffisante pour charger le modèle {model_id} "
                         f"({required_memory:.1f} MB nécessaires, {actual_available_mb:.1f} MB disponibles)")
            return False
        
        # Si GPU disponible, vérifier aussi la mémoire GPU
        if resources["gpu_memory_total_mb"] > 0:
            gpu_available_mb = resources["gpu_memory_total_mb"] * (1 - self.gpu_memory_threshold)
            current_gpu_used_mb = resources["gpu_memory_used_mb"]
            actual_gpu_available_mb = max(0, resources["gpu_memory_total_mb"] - current_gpu_used_mb)
            
            # Pour MPS, l'estimation est moins précise
            if self.has_mps:
                # Sur Apple Silicon, la RAM et VRAM sont partagées, donc on est plus conservatif
                if self.is_apple_silicon and required_memory > actual_gpu_available_mb * 0.8:
                    logger.warning(f"Mémoire MPS probablement insuffisante pour charger le modèle {model_id}")
                    return False
            else:
                # Pour CUDA, on peut être plus précis
                if required_memory > actual_gpu_available_mb:
                    logger.warning(f"Mémoire GPU insuffisante pour charger le modèle {model_id} "
                                f"({required_memory:.1f} MB nécessaires, {actual_gpu_available_mb:.1f} MB disponibles)")
                    return False
        
        return True
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Récupère les métriques de l'orchestrateur.
        
        Returns:
            Métriques de l'orchestrateur
        """
        # Vérifier les ressources actuelles
        resources = self.check_resources()
        
        # Copier les métriques
        metrics = dict(self.metrics)
        
        # Ajouter des métriques supplémentaires
        metrics.update({
            "current_memory_usage": resources["memory_usage"],
            "current_gpu_memory_usage": resources["gpu_memory_usage"],
            "current_cpu_usage": resources["cpu_usage"] / 100.0,
            "system_info": self.system_info
        })
        
        return metrics


@lru_cache()
def get_resource_orchestrator() -> ResourceOrchestrator:
    """
    Récupère l'instance singleton de l'orchestrateur de ressources.
    
    Returns:
        Instance de ResourceOrchestrator
    """
    return ResourceOrchestrator() 