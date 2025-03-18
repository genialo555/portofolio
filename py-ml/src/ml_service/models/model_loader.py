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
import datetime  # Import manquant pour les métadonnées de conversion CoreML
import json
import warnings
import traceback
from dataclasses import dataclass
import importlib

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
    BitsAndBytesConfig,
)

# Import du nouveau quantificateur de modèles
try:
    from ..utils.model_quantizer import (
        QuantizationMethod, 
        ComputePrecision, 
        ModelQuantizer,
        get_optimal_quantization_config
    )
    HAS_MODEL_QUANTIZER = True
except ImportError:
    HAS_MODEL_QUANTIZER = False

# Dependency management
AVAILABLE_LIBS = {
    "transformers": True,  # Base requirement, should always be available
    "llama-cpp": False,    # Will be set to True if import successful
    "auto-gptq": False,    # Will be set to True if import successful
    "optimum": False,      # Will be set to True if import successful
    "coreml": False,       # Will be set to True if import successful
    "bitsandbytes": False,
    "accelerate": False,
    "coremltools": False,
    "mlx": False,
}

# Vérifier les bibliothèques disponibles
try:
    import bitsandbytes as bnb
    AVAILABLE_LIBS["bitsandbytes"] = True
except ImportError:
    logging.warning("La bibliothèque 'bitsandbytes' n'est pas installée. Les quantifications INT4/INT8 ne seront pas disponibles.")

try:
    from accelerate import init_empty_weights, infer_auto_device_map, load_checkpoint_and_dispatch
    AVAILABLE_LIBS["accelerate"] = True
except ImportError:
    logging.warning("La bibliothèque 'accelerate' n'est pas installée. Certaines fonctionnalités seront limitées.")

try:
    import auto_gptq
    AVAILABLE_LIBS["auto-gptq"] = True
    logging.info("La bibliothèque 'auto-gptq' est disponible")
except ImportError:
    logging.warning("La bibliothèque 'auto-gptq' n'est pas installée. La quantification GPTQ ne sera pas disponible.")

try:
    import llama_cpp
    AVAILABLE_LIBS["llama-cpp"] = True
    logging.info("La bibliothèque 'llama-cpp-python' est disponible avec accélération Metal")
except ImportError:
    logging.warning("La bibliothèque 'llama-cpp-python' n'est pas installée. Les modèles GGUF ne seront pas supportés.")
    
try:
    import awq
    AVAILABLE_LIBS["awq"] = True
    logging.info("La bibliothèque 'awq' est disponible")
except ImportError:
    logging.warning("La bibliothèque 'awq' n'est pas installée. La quantification AWQ ne sera pas disponible.")

try:
    import coremltools
    AVAILABLE_LIBS["coremltools"] = True
except ImportError:
    logging.warning("La bibliothèque 'coremltools' n'est pas installée. Les optimisations pour Apple Silicon seront limitées.")
    
# Vérifier optimum[coreml]
try:
    from optimum.exporters.coreml import CoreMLConfig
    AVAILABLE_LIBS["optimum_coreml"] = True
except ImportError:
    logging.warning("La bibliothèque 'optimum[coreml]' n'est pas installée. L'export CoreML ne sera pas disponible.")

try:
    import mlx
    AVAILABLE_LIBS["mlx"] = True
except ImportError:
    pass

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
        
        # Détection optimisée du device sur Apple Silicon
        if platform.system() == "Darwin" and platform.processor() == "arm" and device is None:
            self.device = "mps" if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else "cpu"
            logger.info(f"Détection Apple Silicon: utilisation de {self.device}")
        else:
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
        
        # Journaliser les bibliothèques disponibles
        available_libs_str = ", ".join([k for k, v in AVAILABLE_LIBS.items() if v])
        logger.info(f"Bibliothèques disponibles: {available_libs_str}")
    
    def _check_library_availability(self, library_name):
        """
        Vérifie si une bibliothèque est disponible.
        
        Args:
            library_name: Nom de la bibliothèque à vérifier
            
        Returns:
            bool: True si la bibliothèque est disponible, False sinon
        """
        try:
            importlib.import_module(library_name)
            return True
        except ImportError:
            return False
    
    def _check_available_libraries(self):
        """Vérifie les bibliothèques disponibles pour les différentes fonctionnalités."""
        available_libs = []
        
        # Bibliothèques principales
        if self._check_library_availability("transformers"):
            available_libs.append("transformers")
        
        if self._check_library_availability("llama_cpp_python") or self._check_library_availability("llama_cpp"):
            available_libs.append("llama-cpp")
        
        # Bibliothèques de quantification
        if self._check_library_availability("bitsandbytes"):
            available_libs.append("bitsandbytes")
        
        if self._check_library_availability("accelerate"):
            available_libs.append("accelerate")
        
        # Bibliothèques spécifiques à Apple
        if self._check_library_availability("coremltools"):
            available_libs.append("coremltools")
        
        if self._check_library_availability("mlx"):
            available_libs.append("mlx")
        
        self.available_libraries = available_libs
        self.logger.info(f"Bibliothèques disponibles: {', '.join(available_libs)}")
    
    def load_model(self, 
                 model_id_or_path: str, 
                 model_type: str = "causal_lm",
                 quantization: QuantizationType = QuantizationType.NONE,
                 trust_remote_code: bool = False,
                 use_cache: bool = True,
                 use_virtual_memory: bool = True,
                 offload_layers: bool = False,
                 lazy_loading: bool = True,
                 lazy_loading_threshold: float = 3.0,
                 **kwargs) -> Tuple[Optional[Any], Optional[PreTrainedTokenizer]]:
        """
        Charge un modèle.
        
        Args:
            model_id_or_path: Identifiant ou chemin du modèle
            model_type: Type de modèle
            quantization: Type de quantification
            trust_remote_code: Si True, autorise l'exécution de code distant
            use_cache: Si True, utilise le cache pour les modèles déjà chargés
            use_virtual_memory: Si True, utilise la mémoire virtuelle pour les grands modèles
            offload_layers: Si True, active l'offloading automatique des couches
            lazy_loading: Si True, active le chargement paresseux pour les grands modèles
            lazy_loading_threshold: Seuil en milliards de paramètres pour activer le lazy loading
            **kwargs: Arguments supplémentaires
            
        Returns:
            Tuple contenant le modèle et le tokenizer (peut être None pour les modèles GGUF)
        """
        # Mettre à jour les métriques
        self.metrics["model_loads"] += 1
        start_time = time.time()
        
        # Vérifier les dépendances pour la quantification demandée
        self._check_quantization_dependencies(quantization)
        
        # Vérifier si le modèle est déjà chargé
        cache_key = f"{model_id_or_path}_{model_type}_{quantization}"
        if use_cache and cache_key in self.loaded_models:
            logger.info(f"Utilisation du modèle {model_id_or_path} depuis le cache")
            self.metrics["cache_hits"] += 1
            return self.loaded_models[cache_key]["model"], self.loaded_models[cache_key]["tokenizer"]
        
        try:
            logger.info(f"Chargement du modèle {model_id_or_path} (type: {model_type}, quantization: {quantization})")
            
            # Si le modèle est un chemin de fichier GGML/GGUF, utiliser llama-cpp
            if (quantization == QuantizationType.GGML or 
                str(model_id_or_path).endswith((".gguf", ".ggml", ".bin")) and
                os.path.exists(str(model_id_or_path))):
                
                if not AVAILABLE_LIBS["llama-cpp"]:
                    logger.error("Impossible de charger un modèle GGML/GGUF sans llama-cpp-python")
                    logger.error("Installez-le avec: pip install llama-cpp-python")
                    return None, None
                
                try:
                    logger.info(f"Chargement du modèle GGML/GGUF depuis {model_id_or_path}")
                    start_time = time.time()
                    model = self._load_ggml_model(str(model_id_or_path), **kwargs)
                    
                    # Enregistrer les infos du modèle
                    model_info = {
                        "model_id": os.path.basename(model_id_or_path),
                        "model_type": "ggml",
                        "quantization": str(quantization),
                        "load_time": time.time() - start_time,
                        "model": model
                    }
                    
                    self.loaded_models[cache_key] = model_info
                    return model, None
                    
                except Exception as e:
                    logger.error(f"Erreur lors du chargement du modèle GGML/GGUF: {str(e)}")
                    return None, None
            
            # Utiliser la mémoire virtuelle si demandé
            if use_virtual_memory:
                # Obtenir la configuration optimisée pour le modèle
                vram_config = self.memory_manager.optimize_model_loading(
                    model_id=model_id_or_path, 
                    quantization=quantization
                )
                
                # Fusionner avec les kwargs existants
                for key, value in vram_config.items():
                    if key not in kwargs:
                        kwargs[key] = value
                        
                logger.info(f"Configuration VRAM virtuelle appliquée: {vram_config}")
            
            # Configuration de quantification
            quantization_config = self._configure_quantization(quantization)
            
            # Ajouter les paramètres de chargement paresseux aux kwargs
            kwargs['use_lazy_loading'] = lazy_loading
            kwargs['lazy_loading_threshold'] = lazy_loading_threshold
            
            # Charger d'abord le tokenizer qui est plus léger
            tokenizer = AutoTokenizer.from_pretrained(
                model_id_or_path,
                cache_dir=self.model_cache_dir,
                trust_remote_code=trust_remote_code
            )
            
            # Charger le modèle en fonction de son type
            if model_type == "causal_lm":
                model = self._load_causal_lm(
                    model_id=model_id_or_path,
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
                "quantization": quantization,
                "lazy_loaded": getattr(model, "_lazy_loaded", False)
            }
            
            # Offloading automatique des couches si demandé pour les grands modèles
            if offload_layers and use_virtual_memory:
                self._configure_layer_offloading(model)
            
            # Mettre à jour les métriques
            load_time = time.time() - start_time
            self.metrics["total_load_time"] += load_time
            self.metrics["successful_loads"] += 1
            
            # Ajouter des informations sur le lazy loading dans le log
            lazy_load_status = "avec lazy loading" if getattr(model, "_lazy_loaded", False) else "chargement standard"
            logger.info(f"Modèle {model_id_or_path} chargé en {load_time:.2f} secondes ({lazy_load_status})")
            
            return model, tokenizer
            
        except Exception as e:
            # Mettre à jour les métriques d'échec
            self.metrics["failed_loads"] += 1
            logger.error(f"Erreur lors du chargement du modèle {model_id_or_path}: {str(e)}")
            raise
    
    def _check_quantization_dependencies(self, quantization: QuantizationType):
        """
        Vérifie si les dépendances nécessaires pour la quantification demandée sont disponibles.
        Génère des erreurs significatives si les dépendances manquent.
        
        Args:
            quantization: Type de quantification demandée
        """
        if quantization == QuantizationType.NONE:
            return
            
        if quantization in (QuantizationType.INT8, QuantizationType.INT4):
            # Sur Apple Silicon, on privilégie CoreML
            if platform.system() == "Darwin" and platform.processor() == "arm":
                if not AVAILABLE_LIBS["coremltools"]:
                    logger.warning(
                        f"Quantification {quantization} sur Apple Silicon: coremltools non disponible. "
                        f"Pour une meilleure performance, installez: pip install coremltools"
                    )
            # Sinon on a besoin de bitsandbytes
            elif not AVAILABLE_LIBS["bitsandbytes"]:
                logger.warning(
                    f"La quantification {quantization} nécessite bitsandbytes. "
                    f"Installez-le avec: pip install bitsandbytes"
                )
                
        elif quantization == QuantizationType.GPTQ and not AVAILABLE_LIBS["auto-gptq"]:
            logger.warning(
                "La quantification GPTQ nécessite auto-gptq. "
                "Installez-le avec: pip install auto-gptq"
            )
            
        elif quantization == QuantizationType.GGML and not AVAILABLE_LIBS["llama-cpp"]:
            logger.warning(
                "La quantification GGML nécessite llama-cpp-python. "
                "Installez-le avec: pip install llama-cpp-python"
            )
            
        elif quantization in (QuantizationType.AWQINT4, QuantizationType.AWQINT8) and not AVAILABLE_LIBS["awq"]:
            logger.warning(
                "La quantification AWQ nécessite awq. "
                "Installez-le avec: pip install awq"
            )
            
        elif quantization == QuantizationType.COREML:
            if not (platform.system() == "Darwin" and platform.processor() == "arm" and AVAILABLE_LIBS["coremltools"]):
                logger.warning("CoreML n'est disponible que sur Apple Silicon avec coremltools installé.")
                return None
            return {"type": "coreml", "bits": 4}
        
        return None
    
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
        try:
            # Vérifier si nous sommes sur Apple Silicon et si les bibliothèques nécessaires sont disponibles
            is_apple_silicon = platform.system() == "Darwin" and platform.processor() == "arm"
            has_coremltools = self._check_library_availability("coremltools")
            has_mlx = self._check_library_availability("mlx")
            
            # Vérifier si nous utilisons le ModelQuantizer
            use_model_quantizer = HAS_MODEL_QUANTIZER and quantization_config is not None
            
            device_map = kwargs.pop('device_map', 'auto')
            torch_dtype = kwargs.pop('torch_dtype', None)
            low_cpu_mem_usage = kwargs.pop('low_cpu_mem_usage', True if device_map != 'cpu' else None)
            
            # Paramètre de chargement paresseux (lazy loading)
            use_lazy_loading = kwargs.pop('use_lazy_loading', True)
            # Seuil pour activer le chargement paresseux automatiquement pour les grands modèles
            lazy_loading_threshold = kwargs.pop('lazy_loading_threshold', 3)  # en milliards de paramètres
            
            # Déterminer le dtype optimal pour cette machine
            if torch_dtype is None:
                if is_apple_silicon and hasattr(torch, 'float16'):
                    torch_dtype = torch.float16
                elif hasattr(torch, 'bfloat16') and torch.cuda.is_available():
                    torch_dtype = torch.bfloat16
                else:
                    torch_dtype = torch.float32
            
            # Configurer les paramètres pour HF Accelerate en fonction du matériel
            if is_apple_silicon:
                device_map = 'mps' if torch.backends.mps.is_available() else 'cpu'
                
            # Construire les paramètres de chargement généraux
            model_kwargs = {
                "device_map": device_map,
                "torch_dtype": torch_dtype,
                "trust_remote_code": trust_remote_code,
                "low_cpu_mem_usage": low_cpu_mem_usage,
                **kwargs
            }
            
            # Appliquer les optimisations spécifiques à Apple Silicon
            if is_apple_silicon:
                apple_silicon_config = self.memory_manager.get_apple_silicon_config()
                
                # Ne pas passer ces options directement au modèle
                apple_silicon_opts = {
                    'use_fp16', 'use_metal', 'optimized_attention', 
                    'batch_size', 'stream_attention'
                }
                
                # Extraire les options spécifiques pour l'initialisation du modèle
                apple_config_for_model = {k: v for k, v in apple_silicon_config.items() 
                                         if k not in apple_silicon_opts}
                
                # Stocker les options d'optimisation pour usage ultérieur
                self.apple_silicon_optimizations = {k: v for k, v in apple_silicon_config.items()
                                                  if k in apple_silicon_opts}
                
                # Mettre à jour les arguments du modèle avec les options compatibles seulement
                model_kwargs.update(apple_config_for_model)
                
                logger.info(f"Optimisations Apple Silicon configurées: {self.apple_silicon_optimizations}")
            
            # Configurer les options de lazy loading
            if use_lazy_loading:
                lazy_device_map = self.memory_manager.get_lazy_device_map(
                    model_id, 
                    threshold=lazy_loading_threshold
                )
                model_kwargs.update(lazy_device_map)
                logger.info(f"Configuration de lazy loading appliquée: {lazy_device_map}")
            
            # Charger le modèle avec la quantification demandée
            if use_model_quantizer:
                # Si nous utilisons ModelQuantizer, charger le modèle sans quantification d'abord
                try:
                    logger.info(f"Chargement du modèle avec ModelQuantizer - étape 1: chargement standard")
                    
                    # Filtrer les arguments incompatibles avec le chargement de modèle
                    clean_model_kwargs = {}
                    for k, v in model_kwargs.items():
                        # Exclure les options spécifiques à apple_silicon non compatibles
                        if k not in ['use_fp16', 'use_metal', 'optimized_attention', 
                                    'batch_size', 'stream_attention']:
                            clean_model_kwargs[k] = v
                    
                    # Charger le modèle avec les arguments nettoyés
                    model = AutoModelForCausalLM.from_pretrained(
                        model_id,
                        **clean_model_kwargs
                    )
                    
                    if quantization_config and "type" in quantization_config:
                        # Mapper la méthode de quantification
                        quantization_method = quantization_config["type"]
                        bits = quantization_config.get("bits", 8)
                        
                        # Créer la configuration pour le quantizer
                        from ..utils.model_quantizer import QuantizationConfig
                        quant_config = QuantizationConfig(
                            method=quantization_method,
                            bits=bits
                        )
                        
                        # Instancier et appliquer le quantizer
                        logger.info(f"Application du ModelQuantizer avec méthode: {quantization_method}")
                        quantizer = ModelQuantizer(quant_config)
                        model = quantizer.quantize(model)
                        
                        # Configurer les attributs spéciaux pour le modèle
                        model.quantizer = quantizer
                        logger.info(f"Modèle quantifié avec succès via ModelQuantizer: {type(model).__name__}")
                    
                    return model
                    
                except Exception as e:
                    logger.error(f"Erreur lors de la quantification avec ModelQuantizer: {e}")
                    error_msg = traceback.format_exc()
                    logger.debug(f"Détails de l'erreur: {error_msg}")
                    
                    # Fallback au chargement standard
                    logger.warning("Fallback au chargement standard sans ModelQuantizer")
                    use_model_quantizer = False
            
            # Chargement standard (sans ModelQuantizer)
            logger.info(f"Chargement standard du modèle: {model_id}")
            
            # Filtrer les arguments incompatibles avec le chargement de modèle
            clean_model_kwargs = {}
            for k, v in model_kwargs.items():
                # Exclure les options spécifiques à apple_silicon non compatibles
                if k not in ['use_fp16', 'use_metal', 'optimized_attention', 
                            'batch_size', 'stream_attention']:
                    clean_model_kwargs[k] = v
            
            try:
                # Charger le modèle avec les arguments nettoyés
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    **clean_model_kwargs
                )
                logger.info(f"Modèle chargé avec succès: {type(model).__name__}")
                return model
                
            except Exception as e:
                logger.error(f"Erreur lors du chargement du modèle {model_id}: {e}")
                raise
                
        except Exception as e:
            logger.error(f"Erreur lors du chargement du modèle {model_id}: {str(e)}")
            raise
    
    def _lazy_load_model(self, model_id: str, model_kwargs: Dict[str, Any]) -> PreTrainedModel:
        """
        Charge un modèle avec la technique de chargement paresseux (lazy loading).
        
        Cette méthode utilise accelerate pour charger les poids du modèle de manière sélective,
        ce qui permet d'économiser de la mémoire en ne chargeant que les parties du modèle
        qui sont nécessaires lors de leur première utilisation.
        
        Args:
            model_id: Identifiant du modèle
            model_kwargs: Arguments pour le chargement du modèle
            
        Returns:
            Le modèle chargé avec lazy loading
        """
        if not AVAILABLE_LIBS["accelerate"]:
            raise ImportError("La bibliothèque 'accelerate' est requise pour le lazy loading")
        
        logger.info(f"Chargement paresseux (lazy loading) du modèle {model_id}")
        
        from transformers import AutoConfig, AutoModelForCausalLM
        
        # Extraire les arguments spécifiques
        device_map = model_kwargs.get('device_map', 'auto')
        torch_dtype = model_kwargs.get('torch_dtype', None)
        trust_remote_code = model_kwargs.get('trust_remote_code', False)
        
        # Étape 1: Charger la configuration du modèle
        config = AutoConfig.from_pretrained(
            model_id,
            trust_remote_code=trust_remote_code
        )
        
        # Étape 2: Créer un modèle avec des poids vides
        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(
                config,
                trust_remote_code=trust_remote_code
            )
        
        # Étape 3: Inférer la cartographie optimale des appareils
        if device_map == 'auto':
            device_map = infer_auto_device_map(
                model,
                max_memory=None,  # Utiliser les valeurs par défaut de HF
                dtype=torch_dtype
            )
            
            # Optimisations pour les appareils spécifiques
            is_apple_silicon = platform.system() == "Darwin" and platform.processor() == "arm"
            if is_apple_silicon:
                # Adapter device_map pour MPS
                for layer_name in device_map:
                    if device_map[layer_name] == 'cuda':
                        device_map[layer_name] = 'mps'
            
            logger.info(f"Carte de dispositifs calculée: {device_map}")
        
        # Étape 4: Charger les poids et effectuer la répartition
        model = load_checkpoint_and_dispatch(
            model,
            model_id,
            device_map=device_map,
            offload_folder=self.memory_manager.offload_folder,
            dtype=torch_dtype,
            no_split_module_classes=model._no_split_modules if hasattr(model, "_no_split_modules") else None
        )
        
        logger.info(f"Modèle {model_id} chargé avec lazy loading")
        
        # Stocker des métadonnées supplémentaires
        model._lazy_loaded = True
        model._lazy_loading_info = {
            "model_id": model_id,
            "device_map": device_map,
            "loaded_at": time.time()
        }
        
        return model
        
    def _get_model_info_from_huggingface(self, model_id: str) -> Dict[str, Any]:
        """
        Récupère les informations sur un modèle depuis HuggingFace Hub.
        
        Args:
            model_id: Identifiant du modèle HuggingFace
            
        Returns:
            Dictionnaire contenant les informations du modèle
        """
        try:
            # Tenter d'abord avec la bibliothèque huggingface_hub si disponible
            try:
                from huggingface_hub import model_info
                info = model_info(model_id)
                
                # Extraire les informations pertinentes
                result = {
                    "model_id": model_id,
                    "parameters": info.config.get("n_params", 0),
                    "parameters_billion": info.config.get("n_params", 0) / 1e9 if info.config.get("n_params") else 0,
                    "architecture": info.config.get("architectures", ["Unknown"])[0],
                    "model_type": info.config.get("model_type", "Unknown"),
                }
                
                return result
            except (ImportError, Exception) as e:
                logger.debug(f"Erreur lors de l'utilisation de huggingface_hub: {e}")
                
            # Méthode alternative: charger le config.json directement
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(model_id)
            
            # Estimation du nombre de paramètres à partir du config.json
            n_params = 0
            if hasattr(config, "num_parameters"):
                n_params = config.num_parameters
            elif hasattr(config, "n_params"):
                n_params = config.n_params
            elif hasattr(config, "d_model") and hasattr(config, "num_hidden_layers"):
                # Estimation grossière pour les modèles transformers
                d_model = config.d_model
                n_layers = config.num_hidden_layers
                vocab_size = getattr(config, "vocab_size", 50000)
                n_params = (12 * d_model * d_model * n_layers) + (vocab_size * d_model)
            
            result = {
                "model_id": model_id,
                "parameters": n_params,
                "parameters_billion": n_params / 1e9 if n_params else 0,
                "architecture": getattr(config, "architectures", ["Unknown"])[0] if hasattr(config, "architectures") else "Unknown",
                "model_type": config.model_type if hasattr(config, "model_type") else "Unknown",
            }
            
            return result
            
        except Exception as e:
            logger.warning(f"Impossible de récupérer les informations du modèle {model_id}: {e}")
            return {"model_id": model_id, "parameters_billion": 0, "error": str(e)}
    
    def _configure_quantization(self, quantization: QuantizationType) -> Optional[Dict[str, Any]]:
        """
        Configure les paramètres de quantification.
        
        Args:
            quantization: Type de quantification
        
        Returns:
            Configuration de quantification ou None
        """
        # Utiliser le nouveau ModelQuantizer si disponible
        if HAS_MODEL_QUANTIZER:
            try:
                # Mapping entre QuantizationType et QuantizationMethod
                method_mapping = {
                    QuantizationType.NONE: "none",
                    QuantizationType.INT8: "int8",
                    QuantizationType.INT4: "int4",
                    QuantizationType.GPTQ: "gptq",
                    QuantizationType.AWQINT4: "awq-int4",
                    QuantizationType.AWQINT8: "awq-int8",
                    QuantizationType.COREML: "coreml",
                }
                
                if quantization == QuantizationType.NONE:
                    return None
                
                # Obtenir la méthode de quantification correspondante
                if quantization in method_mapping:
                    quant_method = method_mapping[quantization]
                    
                    # Obtenir une configuration optimisée pour le matériel actuel
                    config = get_optimal_quantization_config()
                    
                    # Remplacer la méthode par celle demandée
                    config.method = quant_method
                    
                    # Ajuster les bits selon le type de quantification
                    if quantization in [QuantizationType.INT4, QuantizationType.AWQINT4]:
                        config.bits = 4
                    elif quantization in [QuantizationType.INT8, QuantizationType.AWQINT8]:
                        config.bits = 8
                    
                    # Sur Apple Silicon, privilégier MLX si disponible
                    is_apple_silicon = platform.system() == "Darwin" and platform.processor() == "arm"
                    if is_apple_silicon:
                        if AVAILABLE_LIBS["mlx"] and quantization in [QuantizationType.INT4, QuantizationType.INT8]:
                            config.method = "mlx"
                            logger.info(f"Utilisation de MLX pour la quantification {quantization} sur Apple Silicon")
                        elif AVAILABLE_LIBS["coremltools"] and quantization in [QuantizationType.INT4, QuantizationType.INT8, QuantizationType.COREML]:
                            config.method = "coreml"
                            logger.info(f"Utilisation de CoreML pour la quantification {quantization} sur Apple Silicon")
                    
                    # Retourner un dictionnaire compatible avec l'interface existante
                    return {
                        "type": config.method,
                        "bits": config.bits,
                        "compute_precision": str(config.compute_precision),
                        "quant_config": config.__dict__,
                        "use_model_quantizer": True
                    }
                
                logger.info(f"Utilisation du ModelQuantizer pour la quantification {quantization}")
                
            except Exception as e:
                logger.warning(f"Erreur lors de l'utilisation du ModelQuantizer: {e}")
                logger.info("Fallback vers l'ancienne méthode de quantification")
        
        # Méthode originale (fallback)
        if quantization == QuantizationType.NONE:
            return None
            
        if quantization == QuantizationType.INT8:
            if platform.system() == "Darwin" and platform.processor() == "arm" and AVAILABLE_LIBS["coremltools"]:
                return {"type": "coreml", "bits": 8}
            elif not AVAILABLE_LIBS["bitsandbytes"]:
                logger.warning("La bibliothèque bitsandbytes n'est pas installée. La quantification INT8 ne sera pas appliquée.")
                return None
            return {"type": "int8"}
            
        elif quantization == QuantizationType.INT4:
            if platform.system() == "Darwin" and platform.processor() == "arm" and AVAILABLE_LIBS["coremltools"]:
                return {"type": "coreml", "bits": 4}
            elif not AVAILABLE_LIBS["bitsandbytes"]:
                logger.warning("La bibliothèque bitsandbytes n'est pas installée. La quantification INT4 ne sera pas appliquée.")
                return None
            return {"type": "int4"}
            
        elif quantization == QuantizationType.GPTQ:
            if not AVAILABLE_LIBS["auto-gptq"]:
                logger.warning("La bibliothèque auto-gptq n'est pas installée. La quantification GPTQ ne sera pas appliquée.")
                return None
            return {"type": "gptq"}
            
        elif quantization == QuantizationType.GGML:
            if not AVAILABLE_LIBS["llama-cpp"]:
                logger.warning("La bibliothèque llama-cpp n'est pas installée. La quantification GGML ne sera pas appliquée.")
                return None
            return {"type": "ggml"}
            
        elif quantization == QuantizationType.AWQINT4:
            if not AVAILABLE_LIBS["awq"]:
                logger.warning("La bibliothèque AWQ n'est pas installée. La quantification AWQ ne sera pas appliquée.")
                return None
            return {"type": "awq", "bits": 4}
            
        elif quantization == QuantizationType.AWQINT8:
            if not AVAILABLE_LIBS["awq"]:
                logger.warning("La bibliothèque AWQ n'est pas installée. La quantification AWQ ne sera pas appliquée.")
                return None
            return {"type": "awq", "bits": 8}
            
        elif quantization == QuantizationType.COREML:
            if not (platform.system() == "Darwin" and platform.processor() == "arm" and AVAILABLE_LIBS["coremltools"]):
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
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Libérer la mémoire MPS si nécessaire
            if platform.system() == "Darwin" and platform.processor() == "arm" and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                if hasattr(torch.mps, 'empty_cache'):
                    torch.mps.empty_cache()
            
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
            # Extraction correcte des composants de la clé
            parts = cache_key.split("_")
            # Prendre la dernière partie comme quantification
            quant = parts[-1]
            # Prendre l'avant-dernière partie comme type de modèle
            model_type = parts[-2]
            # Le reste est l'ID du modèle (peut contenir des underscores)
            model_id = "_".join(parts[:-2])
            
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
        
        # Éviter la division par zéro
        if self.metrics["successful_loads"] > 0:
            metrics["avg_load_time"] = self.metrics["total_load_time"] / self.metrics["successful_loads"]
        else:
            metrics["avg_load_time"] = 0
            
        metrics["success_rate"] = self.metrics["successful_loads"] / max(1, self.metrics["model_loads"])
        metrics["loaded_models_count"] = len(self.loaded_models)
        
        # Ajouter des infos sur l'utilisation GPU si disponible
        if torch.cuda.is_available():
            metrics["gpu_memory_allocated"] = torch.cuda.memory_allocated() / 1024 / 1024 / 1024  # En Go
            metrics["gpu_memory_reserved"] = torch.cuda.memory_reserved() / 1024 / 1024 / 1024   # En Go
            
            # Certaines versions de PyTorch n'ont pas torch.cuda.utilization()
            try:
                metrics["gpu_utilization"] = torch.cuda.utilization()
            except AttributeError:
                # Alternative pour obtenir l'utilisation GPU
                try:
                    import pynvml
                    pynvml.nvmlInit()
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    metrics["gpu_utilization"] = utilization.gpu
                except (ImportError, Exception) as e:
                    metrics["gpu_utilization"] = "Non disponible"
        
        # Ajouter des infos sur l'utilisation MPS si disponible (Apple Silicon)
        if platform.system() == "Darwin" and platform.processor() == "arm" and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            # macOS ne fournit pas d'API directe pour l'utilisation MPS via PyTorch
            metrics["using_mps"] = True
        
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

    def _load_ggml_model(self, 
                       model_path: str,
                       **kwargs) -> "Llama":
        """
        Charge un modèle GGML/GGUF (format optimisé pour llama.cpp).
        
        Args:
            model_path: Chemin vers le fichier modèle GGML/GGUF
            **kwargs: Paramètres supplémentaires pour le modèle
            
        Returns:
            Instance de Llama
            
        Raises:
            ImportError: Si llama-cpp-python n'est pas installé
        """
        if not AVAILABLE_LIBS["llama-cpp"]:
            raise ImportError("llama-cpp-python est requis pour charger les modèles GGML/GGUF")
        
        # Paramètres par défaut optimisés pour Apple Silicon
        if platform.system() == "Darwin" and platform.processor() == "arm":
            # Utilise Metal pour accélérer l'inférence sur Apple Silicon
            # n_gpu_layers=-1 utilise Metal pour toutes les couches possibles
            kwargs.setdefault("n_gpu_layers", -1)
            logger.info(f"Configuration pour Metal (Apple Silicon): n_gpu_layers={kwargs['n_gpu_layers']}")
            
            # Optimisations spécifiques pour Metal
            if "metal_device" not in kwargs:
                # Si plusieurs GPU Metal sont disponibles, utilisez le premier par défaut
                kwargs.setdefault("metal_device", 0)
            
            # Vérifier si la mémoire GPU est limitée sur ce modèle d'Apple Silicon
            import subprocess
            try:
                # Cette commande récupère la mémoire GPU sur macOS
                result = subprocess.run(['system_profiler', 'SPDisplaysDataType'], capture_output=True, text=True)
                output = result.stdout
                
                # Chercher la ligne contenant "VRAM" pour estimer la mémoire disponible
                import re
                vram_match = re.search(r'VRAM \(Total\):\s*(\d+)\s*MB', output)
                if vram_match:
                    vram_mb = int(vram_match.group(1))
                    logger.info(f"Détection mémoire GPU: {vram_mb} MB")
                    
                    # Si la mémoire est limitée, réduire le nombre de couches GPU
                    if vram_mb < 4000:  # Moins de 4 GB
                        logger.warning("Mémoire GPU limitée détectée, réduction des couches GPU")
                        # Ne mettre que les premiers layers sur GPU
                        kwargs["n_gpu_layers"] = min(kwargs["n_gpu_layers"], 32)
                        
                        # Adapter la taille de batch pour éviter les OOM
                        if "n_batch" not in kwargs or kwargs["n_batch"] > 256:
                            kwargs["n_batch"] = 256
                            logger.info("Ajustement de n_batch à 256 pour économiser la mémoire GPU")
            except Exception as e:
                logger.warning(f"Impossible de détecter la mémoire GPU: {e}. Configuration par défaut utilisée.")
            
            # En cas de problème avec Metal, ces options peuvent aider
            kwargs.setdefault("use_mmap", False)  # Souvent plus stable sur macOS
            kwargs.setdefault("use_mlock", False)  # Évite les problèmes de permissions mémoire
        
        # Paramétrer le nombre de threads en fonction du CPU
        import multiprocessing
        cpu_count = multiprocessing.cpu_count()
        
        # Sur Apple Silicon, optimiser pour les cœurs de performance vs efficacité
        if platform.system() == "Darwin" and platform.processor() == "arm":
            # Détecter les modèles M1/M2/M3 qui ont des configurations différentes de cœurs
            try:
                import platform
                cpu_info = platform.processor()
                
                # Distinguer différentes configurations
                if "M1" in cpu_info:
                    # M1: 4P+4E ou 8P+2E selon modèle
                    performance_cores = 4 if "Pro" not in cpu_info and "Max" not in cpu_info else 8
                elif "M2" in cpu_info:
                    # M2: 4P+4E, 8P+2E ou 12P selon modèle
                    performance_cores = 4 if "Pro" not in cpu_info and "Max" not in cpu_info else (8 if "Pro" in cpu_info else 12)
                elif "M3" in cpu_info:
                    # M3: 4P+4E, 10P+2E, 12P+4E, etc.
                    performance_cores = 4 if "Pro" not in cpu_info and "Max" not in cpu_info else (10 if "Pro" in cpu_info else 12)
                else:
                    # Fallback: considérer la moitié des cœurs comme performance
                    performance_cores = max(1, cpu_count // 2)
                
                # Utiliser principalement les cœurs de performance
                # et laisser les cœurs d'efficacité pour le système
                n_threads = max(1, performance_cores)
                logger.info(f"Configuration optimisée pour le processeur Apple {cpu_info}: {n_threads} threads")
            except Exception as e:
                logger.warning(f"Erreur lors de la détection des cœurs: {e}")
                # Fallback à la configuration standard
                n_threads = max(1, cpu_count - 2)
        else:
            # Sur les autres plateformes, utiliser presque tous les cœurs
            n_threads = max(1, cpu_count - 1)
        
        kwargs.setdefault("n_threads", n_threads)
        
        # Optimisations générales pour la mémoire
        kwargs.setdefault("offload_kqv", True)  # Améliore la gestion mémoire
        
        # Configuration de la taille du contexte si non spécifiée
        kwargs.setdefault("n_ctx", 2048)
        
        # Configuration de la taille du batch pour de meilleures performances
        kwargs.setdefault("n_batch", 512)
        
        logger.info(f"Chargement du modèle GGML/GGUF depuis {model_path}")
        logger.info(f"Paramètres: n_threads={kwargs['n_threads']}, n_ctx={kwargs['n_ctx']}, n_batch={kwargs.get('n_batch')}")
        
        # Importation uniquement si nécessaire
        from llama_cpp import Llama
        
        # Chargement du modèle avec gestion d'erreur
        try:
            model = Llama(model_path=model_path, **kwargs)
            logger.info(f"Modèle GGML/GGUF chargé avec succès: {model_path}")
            return model
        except Exception as e:
            logger.error(f"Erreur lors du chargement du modèle GGML/GGUF {model_path}: {str(e)}")
            
            # Tentative de récupération avec des paramètres plus conservateurs
            if kwargs.get("n_gpu_layers", 0) > 0:
                logger.info("Tentative de récupération: désactivation de l'accélération GPU")
                kwargs["n_gpu_layers"] = 0
                try:
                    model = Llama(model_path=model_path, **kwargs)
                    logger.info(f"Modèle GGML/GGUF chargé en mode CPU: {model_path}")
                    return model
                except Exception as e2:
                    logger.error(f"Échec de la récupération: {str(e2)}")
            
            raise RuntimeError(f"Impossible de charger le modèle GGML/GGUF: {str(e)}")

    def _optimize_for_device(self, model: PreTrainedModel) -> PreTrainedModel:
        """
        Optimise un modèle pour le périphérique actuel.
        
        Args:
            model: Modèle à optimiser
            
        Returns:
            Modèle optimisé
        """
        # Optimisations pour Apple Silicon
        if platform.system() == "Darwin" and platform.processor() == "arm" and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            # Vérifier si c'est un modèle PyTorch natif ou llama-cpp
            if not isinstance(model, PreTrainedModel):
                logger.info("Modèle non-PyTorch détecté, pas d'optimisation MPS nécessaire")
                return model
                
            logger.info("Optimisation du modèle pour Apple Silicon (MPS)")
            model = model.to('mps')
            
            # Optimisations supplémentaires pour MPS
            for param in model.parameters():
                if param.requires_grad:
                    # Utiliser le format float16 pour améliorer les performances sur MPS
                    if hasattr(param, 'to') and callable(param.to):
                        param.to(dtype=torch.float16)
            
            torch.mps.empty_cache()
            
        # Optimisations pour CUDA
        elif torch.cuda.is_available():
            # Vérifier si c'est un modèle PyTorch natif
            if not isinstance(model, PreTrainedModel):
                logger.info("Modèle non-PyTorch détecté, pas d'optimisation CUDA nécessaire")
                return model
                
            logger.info("Optimisation du modèle pour CUDA")
            model = model.to('cuda')
            torch.cuda.empty_cache()
            
        return model


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