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

from ..config import settings

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


class ModelLoader:
    """
    Classe pour charger et gérer les modèles ML.
    
    Cette classe s'occupe du chargement des modèles, de leur quantification
    et de la gestion optimale des ressources (CPU/GPU).
    """
    
    def __init__(self, 
                model_cache_dir: Optional[str] = None,
                device: Optional[str] = None,
                max_gpu_memory: Optional[int] = None):
        """
        Initialise le chargeur de modèles.
        
        Args:
            model_cache_dir: Répertoire de cache pour les modèles
            device: Dispositif de calcul ('cpu', 'cuda', 'mps', 'auto')
            max_gpu_memory: Mémoire GPU maximale à utiliser (en Go)
        """
        self.model_cache_dir = model_cache_dir or str(settings.MODEL_PATH)
        
        # Déterminer le device disponible
        if device == "auto" or device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        # Configurer la mémoire GPU maximale
        self.max_gpu_memory = max_gpu_memory
        if self.max_gpu_memory is None:
            if torch.cuda.is_available():
                # Utiliser 90% de la mémoire GPU disponible par défaut
                self.max_gpu_memory = int(torch.cuda.get_device_properties(0).total_memory / 1024 / 1024 / 1024 * 0.9)
            else:
                self.max_gpu_memory = 0
                
        # Dictionnaire pour garder une trace des modèles chargés
        self.loaded_models: Dict[str, Dict[str, Any]] = {}
        
        # Métriques
        self.metrics = {
            "total_load_time": 0,
            "model_loads": 0,
            "successful_loads": 0,
            "failed_loads": 0,
            "cache_hits": 0,
        }
        
        logger.info(f"ModelLoader initialisé avec device={self.device}, max_gpu_memory={self.max_gpu_memory}Go")
        logger.info(f"Répertoire de cache des modèles: {self.model_cache_dir}")
        
        # Créer le répertoire de cache s'il n'existe pas
        os.makedirs(self.model_cache_dir, exist_ok=True)
    
    def load_model(self, 
                 model_id: str, 
                 model_type: str = "causal_lm", 
                 quantization: QuantizationType = QuantizationType.NONE,
                 trust_remote_code: bool = False,
                 use_cache: bool = True,
                 **kwargs) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """
        Charge un modèle transformers avec son tokenizer.
        
        Args:
            model_id: Identifiant du modèle (nom Hugging Face ou chemin local)
            model_type: Type de modèle ('causal_lm', 'seq2seq_lm', etc.)
            quantization: Type de quantification à utiliser
            trust_remote_code: Autoriser l'exécution de code distant
            use_cache: Utiliser le cache pour les modèles déjà chargés
            **kwargs: Arguments supplémentaires pour le chargement du modèle
        
        Returns:
            Tuple (modèle, tokenizer)
        """
        cache_key = f"{model_id}_{model_type}_{quantization.value}"
        
        # Vérifier si le modèle est déjà chargé
        if use_cache and cache_key in self.loaded_models:
            logger.info(f"Modèle {model_id} déjà chargé, utilisation depuis le cache")
            self.metrics["cache_hits"] += 1
            return (
                self.loaded_models[cache_key]["model"],
                self.loaded_models[cache_key]["tokenizer"]
            )
        
        start_time = time.time()
        self.metrics["model_loads"] += 1
        
        try:
            # Configurer la quantification
            quantization_config = self._configure_quantization(quantization)
            
            logger.info(f"Chargement du modèle {model_id} (type: {model_type}, quantization: {quantization.value})")
            
            # Charger le tokenizer en premier
            tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                cache_dir=self.model_cache_dir,
                trust_remote_code=trust_remote_code,
                padding_side="left",  # Pour le chat
                **kwargs.get("tokenizer_kwargs", {})
            )
            
            # Configurer padding token si nécessaire
            if tokenizer.pad_token is None:
                if tokenizer.eos_token is not None:
                    tokenizer.pad_token = tokenizer.eos_token
                else:
                    tokenizer.pad_token = tokenizer.eos_token = "</s>"
            
            # Charger le modèle avec la quantification appropriée
            if model_type == "causal_lm":
                model = self._load_causal_lm(
                    model_id=model_id,
                    quantization_config=quantization_config,
                    trust_remote_code=trust_remote_code,
                    **kwargs
                )
            else:
                raise ValueError(f"Type de modèle non supporté: {model_type}")
            
            # Mettre en cache le modèle chargé
            self.loaded_models[cache_key] = {
                "model": model,
                "tokenizer": tokenizer,
                "loaded_at": time.time(),
                "device": model.device,
                "quantization": quantization.value
            }
            
            load_time = time.time() - start_time
            self.metrics["total_load_time"] += load_time
            self.metrics["successful_loads"] += 1
            
            logger.info(f"Modèle {model_id} chargé avec succès en {load_time:.2f}s sur {model.device}")
            
            return model, tokenizer
        
        except Exception as e:
            logger.error(f"Erreur lors du chargement du modèle {model_id}: {str(e)}")
            self.metrics["failed_loads"] += 1
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
        
        # Configurer pour BitsAndBytes si nécessaire
        if quantization_config is not None and quantization_config.get("type") in ["int8", "int4"]:
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
        device_map = "auto" if self.device == "cuda" else self.device
        model_kwargs["device_map"] = kwargs.get("device_map", device_map)
        
        # Configurer le type de données (dtype)
        if self.device == "cuda":
            torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
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
            if not HAS_BITSANDBYTES:
                logger.warning("La bibliothèque bitsandbytes n'est pas installée. La quantification INT8 ne sera pas appliquée.")
                return None
            return {"type": "int8"}
            
        elif quantization == QuantizationType.INT4:
            if not HAS_BITSANDBYTES:
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
        cache_key = f"{model_id}_{model_type}_{quantization.value}"
        
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
        cache_key = f"{model_id}_{model_type}_{quantization.value}"
        
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


@lru_cache()
def get_model_loader() -> ModelLoader:
    """
    Fonction pour obtenir une instance singleton du chargeur de modèles.
    
    Returns:
        Instance du chargeur de modèles
    """
    return ModelLoader() 