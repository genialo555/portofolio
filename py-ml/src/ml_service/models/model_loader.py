"""
Module de chargement des modèles ML.

Ce module fournit des classes et fonctions pour charger et quantifier efficacement
les modèles de machine learning, en optimisant l'utilisation des ressources.
"""

import os
import logging
import torch
from typing import Dict, Any, Optional, List, Union, Tuple
from pathlib import Path
from functools import lru_cache
import time
import gc
import numpy as np
from enum import Enum
import platform

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
    BitsAndBytesConfig,
)

try:
    import bitsandbytes as bnb
    HAS_BITSANDBYTES = True
except ImportError:
    HAS_BITSANDBYTES = False

try:
    from accelerate import init_empty_weights, infer_auto_device_map, load_checkpoint_and_dispatch
    HAS_ACCELERATE = True
except ImportError:
    HAS_ACCELERATE = False

# Vérification pour CoreML Tools sur les appareils Apple Silicon
HAS_COREML = False
IS_APPLE_SILICON = platform.processor() == 'arm' and platform.system() == 'Darwin'
if IS_APPLE_SILICON:
    try:
        import coremltools as ct
        HAS_COREML = True
    except ImportError:
        HAS_COREML = False

from ..config import settings
from ..utils.memory_manager import get_memory_manager

logger = logging.getLogger("ml_api.models.loader")

class QuantizationType(str, Enum):
    """Types de quantification supportés."""
    NONE = "none"          # Pas de quantification
    INT8 = "int8"          # Quantification 8-bit
    INT4 = "int4"          # Quantification 4-bit
    GPTQ = "gptq"          # GPTQ (Google's Pretrained Transformer Quantization)
    GGML = "ggml"          # GGML (format de quantification de llama.cpp)
    AWQINT4 = "awq-int4"   # Activation-aware Weight Quantization (4-bit)
    AWQINT8 = "awq-int8"   # Activation-aware Weight Quantization (8-bit)
    COREML = "coreml"      # CoreML (optimisé pour Apple Silicon)


class ModelLoader:
    """
    Gestionnaire de modèles ML.
    
    Cette classe est responsable de:
    - Charger et décharger des modèles
    - Gérer les quantifications et les configurations
    - Optimiser l'utilisation des ressources
    """
    
    def __init__(self, 
                model_cache_dir: Optional[str] = None,
                device: Optional[str] = None,
                max_gpu_memory: Optional[int] = None):
        """
        Initialise le chargeur de modèles.
        
        Args:
            model_cache_dir: Répertoire de cache pour les modèles
            device: Dispositif à utiliser (auto, cpu, cuda:0, etc.)
            max_gpu_memory: Mémoire GPU maximale à utiliser (en GB)
        """
        self.model_cache_dir = model_cache_dir or str(settings.MODEL_PATH)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_gpu_memory = max_gpu_memory
        
        # Dictionnaire pour suivre les modèles chargés
        self.loaded_models: Dict[str, Dict[str, Any]] = {}
        
        # Obtenir le gestionnaire de mémoire virtuelle
        self.memory_manager = get_memory_manager(cache_dir=self.model_cache_dir)
        
        logger.info(f"ModelLoader initialisé avec cache: {self.model_cache_dir}, device: {self.device}")
        
        # Métriques
        self.metrics = {
            "total_load_time": 0,
            "model_loads": 0,
            "successful_loads": 0,
            "failed_loads": 0,
            "cache_hits": 0,
        }
        
        # Créer le répertoire de cache s'il n'existe pas
        os.makedirs(self.model_cache_dir, exist_ok=True)
    
    def load_model(self, 
                 model_id: str, 
                 model_type: str = "causal_lm", 
                 quantization: QuantizationType = QuantizationType.NONE,
                 trust_remote_code: bool = False,
                 use_cache: bool = True,
                 use_virtual_memory: bool = True,
                 offload_layers: bool = False,
                 **kwargs) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """
        Charge un modèle et son tokenizer.
        
        Args:
            model_id: Identifiant du modèle à charger
            model_type: Type de modèle (causal_lm, seq2seq, etc.)
            quantization: Type de quantification à utiliser
            trust_remote_code: Si True, autorise l'exécution de code distant
            use_cache: Si True, utilise le cache pour les modèles déjà chargés
            use_virtual_memory: Si True, utilise la mémoire virtuelle pour les grands modèles
            offload_layers: Si True, active l'offloading automatique des couches
            **kwargs: Arguments supplémentaires pour le chargement
            
        Returns:
            Tuple contenant le modèle et le tokenizer
        """
        # Vérifier si le modèle est déjà chargé
        cache_key = f"{model_id}_{model_type}_{quantization}"
        if use_cache and cache_key in self.loaded_models:
            logger.info(f"Utilisation du modèle {model_id} depuis le cache")
            return self.loaded_models[cache_key]["model"], self.loaded_models[cache_key]["tokenizer"]
        
        try:
            logger.info(f"Chargement du modèle {model_id} (type: {model_type}, quantization: {quantization})")
            
            # Utiliser la mémoire virtuelle si demandé
            if use_virtual_memory:
                # Obtenir la configuration optimisée pour le modèle
                vram_config = self.memory_manager.optimize_model_loading(
                    model_id=model_id, 
                    quantization=quantization
                )
                
                # Fusionner avec les kwargs existants
                for key, value in vram_config.items():
                    if key not in kwargs:
                        kwargs[key] = value
                        
                logger.info(f"Configuration VRAM virtuelle appliquée: {vram_config}")
            
            # Configuration de quantification
            quantization_config = self._configure_quantization(quantization)
            
            # Charger d'abord le tokenizer qui est plus léger
            tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                cache_dir=self.model_cache_dir,
                trust_remote_code=trust_remote_code
            )
            
            # Charger le modèle en fonction de son type
            if model_type == "causal_lm":
                model = self._load_causal_lm(
                    model_id=model_id,
                    quantization_config=quantization_config,
                    trust_remote_code=trust_remote_code,
                    **kwargs
                )
            else:
                raise ValueError(f"Type de modèle non supporté: {model_type}")
            
            # Mettre le modèle en mode évaluation
            model.eval()
            
            # Stocker le modèle dans le cache
            self.loaded_models[cache_key] = {
                "model": model,
                "tokenizer": tokenizer,
                "loaded_at": time.time(),
                "type": model_type,
                "quantization": quantization
            }
            
            # Offloading automatique des couches si demandé pour les grands modèles
            if offload_layers and use_virtual_memory:
                self._configure_layer_offloading(model)
            
            return model, tokenizer
            
        except Exception as e:
            logger.error(f"Erreur lors du chargement du modèle {model_id}: {str(e)}")
            raise
    
    def _load_causal_lm(self, 
                       model_id: str,
                       quantization_config: Optional[Dict[str, Any]] = None,
                       trust_remote_code: bool = False,
                       **kwargs) -> PreTrainedModel:
        """
        Charge un modèle de langage causal (comme GPT, LLaMA, etc.).
        
        Args:
            model_id: Identifiant du modèle
            quantization_config: Configuration de quantification
            trust_remote_code: Autoriser l'exécution de code distant
            **kwargs: Arguments supplémentaires
        
        Returns:
            Modèle chargé
        """
        model_kwargs = kwargs.get("model_kwargs", {})
        
        # Vérifier si on utilise Apple Silicon pour des optimisations spécifiques
        is_apple_silicon = IS_APPLE_SILICON
        has_coreml = HAS_COREML
        
        # Utiliser CoreML pour la quantification sur les appareils Apple Silicon si approprié
        if is_apple_silicon and has_coreml and quantization_config is not None and quantization_config.get("type") in ["int4", "int8", "coreml"]:
            logger.info("Utilisation de CoreML pour la quantification sur Apple Silicon")
            
            # Configuration pour utiliser CoreML après le chargement
            memory_manager = get_memory_manager()
            model_kwargs["post_load_hook"] = memory_manager.apple_coreml_quantization
            
            # Utiliser float16 pour le modèle initial
            model_kwargs["torch_dtype"] = torch.float16
        # Configurer pour BitsAndBytes si nécessaire (sur les plateformes non-Apple ou si CoreML n'est pas disponible)
        elif quantization_config is not None and quantization_config.get("type") in ["int8", "int4"]:
            if not HAS_BITSANDBYTES:
                logger.warning("La bibliothèque bitsandbytes n'est pas installée. Utilisation du modèle non quantifié.")
            elif not torch.cuda.is_available() and is_apple_silicon:
                logger.warning("Quantification bitsandbytes non disponible sur Apple Silicon. Utilisation du modèle en float16.")
                model_kwargs["torch_dtype"] = torch.float16
            else:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=quantization_config.get("type") == "int4",
                    load_in_8bit=quantization_config.get("type") == "int8",
                    llm_int8_threshold=6.0,
                    llm_int8_has_fp16_weight=False,
                    bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
                model_kwargs["quantization_config"] = bnb_config
        
        # Configurer le device mapping si nécessaire
        if is_apple_silicon and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device_map = "auto"  # Hugging Face gérera l'allocation entre CPU et MPS
            logger.info("Configuration pour utiliser MPS (Metal Performance Shaders) sur Apple Silicon")
        else:
            device_map = "auto" if torch.cuda.is_available() else None
        
        model_kwargs["device_map"] = kwargs.get("device_map", device_map)
        
        # Configurer le type de données (dtype)
        if torch.cuda.is_available():
            torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        elif is_apple_silicon and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            torch_dtype = torch.float16  # float16 est bien supporté par le Neural Engine
        else:
            torch_dtype = torch.float32
        
        model_kwargs["torch_dtype"] = kwargs.get("torch_dtype", torch_dtype)
        
        # Charger le modèle
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            cache_dir=self.model_cache_dir,
            trust_remote_code=trust_remote_code,
            **model_kwargs
        )
        
        return model
    
    def _configure_quantization(self, quantization: QuantizationType) -> Optional[Dict[str, Any]]:
        """
        Configure les paramètres de quantification.
        
        Args:
            quantization: Type de quantification
        
        Returns:
            Configuration de quantification ou None
        """
        if quantization == QuantizationType.NONE:
            return None
            
        if quantization == QuantizationType.INT8:
            if IS_APPLE_SILICON and HAS_COREML:
                return {"type": "coreml", "bits": 8}
            elif not HAS_BITSANDBYTES:
                logger.warning("La bibliothèque bitsandbytes n'est pas installée. La quantification INT8 ne sera pas appliquée.")
                return None
            return {"type": "int8"}
            
        elif quantization == QuantizationType.INT4:
            if IS_APPLE_SILICON and HAS_COREML:
                return {"type": "coreml", "bits": 4}
            elif not HAS_BITSANDBYTES:
                logger.warning("La bibliothèque bitsandbytes n'est pas installée. La quantification INT4 ne sera pas appliquée.")
                return None
            return {"type": "int4"}
            
        elif quantization == QuantizationType.GPTQ:
            return {"type": "gptq"}
            
        elif quantization == QuantizationType.GGML:
            return {"type": "ggml"}
            
        elif quantization == QuantizationType.AWQINT4:
            return {"type": "awq", "bits": 4}
            
        elif quantization == QuantizationType.AWQINT8:
            return {"type": "awq", "bits": 8}
            
        elif quantization == QuantizationType.COREML:
            if not (IS_APPLE_SILICON and HAS_COREML):
                logger.warning("CoreML n'est disponible que sur Apple Silicon avec coremltools installé.")
                return None
            return {"type": "coreml", "bits": 4}
        
        return None
    
    def unload_model(self, model_id: str, model_type: str = "causal_lm",
                    quantization: QuantizationType = QuantizationType.NONE) -> bool:
        """
        Décharge un modèle de la mémoire.
        
        Args:
            model_id: Identifiant du modèle
            model_type: Type de modèle
            quantization: Type de quantification
        
        Returns:
            True si le modèle a été déchargé, False sinon
        """
        cache_key = f"{model_id}_{model_type}_{quantization}"
        
        if cache_key in self.loaded_models:
            # Récupérer les références au modèle et au tokenizer
            model_info = self.loaded_models[cache_key]
            model = model_info.get("model")
            
            # Supprimer le modèle du cache
            del self.loaded_models[cache_key]
            
            # Forcer la libération de la mémoire
            if model is not None:
                del model
            
            # Forcer le garbage collector
            gc.collect()
            
            # Libérer la mémoire CUDA si nécessaire
            if self.device == "cuda":
                torch.cuda.empty_cache()
            
            logger.info(f"Modèle {model_id} déchargé avec succès")
            return True
            
        logger.info(f"Le modèle {model_id} n'était pas chargé")
        return False
    
    def get_model_info(self, model_id: str, model_type: str = "causal_lm",
                      quantization: QuantizationType = QuantizationType.NONE) -> Optional[Dict[str, Any]]:
        """
        Récupère les informations sur un modèle chargé.
        
        Args:
            model_id: Identifiant du modèle
            model_type: Type de modèle
            quantization: Type de quantification
        
        Returns:
            Informations sur le modèle ou None s'il n'est pas chargé
        """
        cache_key = f"{model_id}_{model_type}_{quantization}"
        
        return self.loaded_models.get(cache_key)
    
    def list_loaded_models(self) -> List[Dict[str, Any]]:
        """
        Liste tous les modèles actuellement chargés.
        
        Returns:
            Liste des informations sur les modèles chargés
        """
        result = []
        for cache_key, model_info in self.loaded_models.items():
            model_id, model_type, quant = cache_key.rsplit("_", 2)
            
            # Extraire les informations basiques
            info = {
                "model_id": model_id,
                "model_type": model_type,
                "quantization": quant,
                "device": str(model_info.get("device", "unknown")),
                "loaded_at": model_info.get("loaded_at", 0)
            }
            
            # Ajouter des informations sur la taille du modèle si disponible
            model = model_info.get("model")
            if model:
                try:
                    param_size = sum(p.numel() for p in model.parameters()) / 1e6  # En millions
                    info["parameters"] = f"{param_size:.2f}M"
                except:
                    pass
            
            result.append(info)
            
        return result
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Récupère les métriques du chargeur de modèles.
        
        Returns:
            Métriques du chargeur
        """
        metrics = dict(self.metrics)
        
        # Ajouter des métriques supplémentaires
        metrics["avg_load_time"] = self.metrics["total_load_time"] / max(1, self.metrics["successful_loads"])
        metrics["success_rate"] = self.metrics["successful_loads"] / max(1, self.metrics["model_loads"])
        metrics["loaded_models_count"] = len(self.loaded_models)
        
        # Ajouter des infos sur l'utilisation GPU si disponible
        if torch.cuda.is_available():
            metrics["gpu_memory_allocated"] = torch.cuda.memory_allocated() / 1024 / 1024 / 1024  # En Go
            metrics["gpu_memory_reserved"] = torch.cuda.memory_reserved() / 1024 / 1024 / 1024   # En Go
            metrics["gpu_utilization"] = torch.cuda.utilization()
        
        return metrics
    
    def _configure_layer_offloading(self, model):
        """
        Configure l'offloading automatique des couches pour un modèle.
        
        Cette fonction identifie les couches qui peuvent être déchargées et rechargées à la demande.
        """
        # Chercher les couches d'attention et MLP qui peuvent être offloadées
        transformer_layers = []
        
        # Parcourir la structure du modèle pour trouver les couches transformers
        if hasattr(model, "transformer") and hasattr(model.transformer, "layers"):
            # Structure typique des modèles LLaMA, OPT, etc.
            transformer_layers = model.transformer.layers
            prefix = "transformer.layers"
        elif hasattr(model, "model") and hasattr(model.model, "layers"):
            # Structure alternative
            transformer_layers = model.model.layers
            prefix = "model.layers"
        
        if not transformer_layers:
            logger.warning("Structure de modèle non reconnue pour l'offloading de couches")
            return
        
        # Configuré pour offloader les couches du milieu (gardant les premières et dernières en mémoire)
        num_layers = len(transformer_layers)
        layers_to_offload = []
        
        # Garder les 2 premières et 2 dernières couches en mémoire pour des performances optimales
        # Offloader celles du milieu
        if num_layers > 6:  # Seulement si assez de couches
            for i in range(2, num_layers - 2):
                layers_to_offload.append(f"{prefix}.{i}")
                
        logger.info(f"Configuration des couches pour offloading: {layers_to_offload}")
        
        # Stocker la configuration pour l'offloading
        model._offloadable_layers = layers_to_offload


def create_r1_teacher_model_loader() -> ModelLoader:
    """
    Crée un chargeur pour le modèle R1 quantifié à utiliser comme teacher.
    
    Returns:
        ModelLoader configuré pour le modèle R1
    """
    # ID du modèle R1 
    model_id = "anthropic/claude-3-sonnet-20240229"  # Simulé avec un modèle disponible
    
    # Pour un modèle réellement disponible, on pourrait utiliser:
    # model_id = "meta-llama/Llama-2-7b-chat-hf"
    
    loader = ModelLoader()
    
    # Configurer et précharger le modèle si possible
    try:
        model, tokenizer = loader.load_model(
            model_id=model_id,
            model_type="causal_lm",
            quantization=QuantizationType.INT8,  # Quantification 8-bit pour économiser la mémoire
            trust_remote_code=True,
            use_cache=True
        )
        logger.info(f"Modèle R1 teacher chargé avec succès: {model_id}")
    except Exception as e:
        logger.warning(f"Impossible de précharger le modèle R1 teacher: {str(e)}")
    
    return loader


def create_phi4_distilled_model_loader() -> ModelLoader:
    """
    Crée un chargeur pour le modèle Phi-4 distillé.
    
    Returns:
        ModelLoader configuré pour le modèle Phi-4
    """
    # ID du modèle Phi-4 distillé
    model_id = "microsoft/Phi-4"  # Version simulée
    
    # Pour un modèle réellement disponible, on pourrait utiliser:
    # model_id = "microsoft/phi-2"
    
    loader = ModelLoader()
    
    # Configurer et précharger le modèle si possible
    try:
        model, tokenizer = loader.load_model(
            model_id=model_id,
            model_type="causal_lm",
            quantization=QuantizationType.INT4,  # Quantification 4-bit pour les modèles plus petits
            trust_remote_code=True,
            use_cache=True
        )
        logger.info(f"Modèle Phi-4 distillé chargé avec succès: {model_id}")
    except Exception as e:
        logger.warning(f"Impossible de précharger le modèle Phi-4 distillé: {str(e)}")
    
    return loader


def create_qwen32b_model_loader() -> ModelLoader:
    """
    Crée un chargeur pour le modèle Qwen 32B.
    
    Returns:
        ModelLoader configuré pour le modèle Qwen 32B
    """
    # ID du modèle Qwen 32B
    model_id = "Qwen/Qwen1.5-32B"
    
    loader = ModelLoader()
    
    # Configurer et précharger le modèle si possible
    try:
        model, tokenizer = loader.load_model(
            model_id=model_id,
            model_type="causal_lm",
            quantization=QuantizationType.INT4,  # Quantification 4-bit nécessaire pour ce grand modèle
            trust_remote_code=True,
            use_cache=True,
            # Configuration supplémentaire pour optimiser la mémoire
            max_gpu_memory="80%",  # Utilise 80% de la mémoire GPU disponible
        )
        logger.info(f"Modèle Qwen 32B chargé avec succès: {model_id}")
    except Exception as e:
        logger.warning(f"Impossible de précharger le modèle Qwen 32B: {str(e)}")
    
    return loader


@lru_cache()
def get_model_loader() -> ModelLoader:
    """
    Fonction pour obtenir une instance singleton du chargeur de modèles.
    
    Returns:
        Instance du chargeur de modèles
    """
    return ModelLoader() 