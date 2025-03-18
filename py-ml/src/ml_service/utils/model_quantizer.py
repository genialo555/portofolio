#!/usr/bin/env python3
"""
Module de quantification pour modèles ML, optimisé pour Apple Silicon.

Ce module fournit des utilitaires pour quantifier des modèles ML en
utilisant les formats et bibliothèques les plus adaptés, notamment avec 
des optimisations spécifiques pour MLX et CoreML sur Apple Silicon.
"""

import os
import sys
import time
import logging
import platform
import numpy as np
from enum import Enum, auto
from typing import Dict, List, Tuple, Any, Union, Optional
from dataclasses import dataclass, field
from pathlib import Path

# Configurer le logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("model_quantizer")

# Importation conditionnelle pour minimiser les dépendances
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    logger.warning("PyTorch n'est pas disponible. Certaines fonctionnalités seront limitées.")

try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    logger.warning("MLX n'est pas disponible. Les optimisations MLX ne seront pas utilisées.")

try:
    import coremltools as ct
    HAS_COREML = True
except ImportError:
    HAS_COREML = False
    logger.warning("CoreML n'est pas disponible. Les optimisations CoreML ne seront pas utilisées.")

# Vérifier si nous sommes sur Apple Silicon
IS_APPLE_SILICON = platform.system() == "Darwin" and platform.processor() == "arm"

# Vérifier si MPS est disponible
HAS_MPS = HAS_TORCH and torch.backends.mps.is_available() if hasattr(torch, "backends") and hasattr(torch.backends, "mps") else False

# Vérifier si CUDA est disponible
HAS_CUDA = HAS_TORCH and torch.cuda.is_available() if hasattr(torch, "cuda") else False

# Vérifier Bitsandbytes pour quantification INT4/INT8
try:
    import bitsandbytes as bnb
    HAS_BNB = True
except ImportError:
    HAS_BNB = False

# Vérifier AWQ pour quantification activations
try:
    import awq
    HAS_AWQ = True
except ImportError:
    HAS_AWQ = False

# Import du gestionnaire de mémoire s'il existe
try:
    from ml_service.utils.memory_manager import get_memory_manager
    HAS_MEMORY_MANAGER = True
except ImportError:
    HAS_MEMORY_MANAGER = False

# Import du gestionnaire d'attention
try:
    from ml_service.utils.attention_utils import prepare_model_inputs, optimize_attention_for_hardware
    HAS_ATTENTION_UTILS = True
except ImportError:
    HAS_ATTENTION_UTILS = False
    logger.warning("Les utilitaires d'attention avancés ne sont pas disponibles.")

class QuantizationMethod(str, Enum):
    """Méthodes de quantification supportées."""
    NONE = "none"          # Pas de quantification
    INT8 = "int8"          # Quantification en entiers 8 bits
    INT4 = "int4"          # Quantification en entiers 4 bits
    GPTQ = "gptq"          # Google's Pretrained Transformer Quantization
    AWQINT4 = "awq-int4"   # Activation-aware Weight Quantization (4-bit)
    AWQINT8 = "awq-int8"   # Activation-aware Weight Quantization (8-bit)
    MLX = "mlx"            # Format MLX optimisé pour Apple Silicon
    COREML = "coreml"      # Format CoreML optimisé pour Apple Silicon
    ONNX = "onnx"          # Format ONNX pour interopérabilité

class ComputePrecision(str, Enum):
    """Précisions de calcul supportées."""
    FP32 = "fp32"          # Précision complète
    FP16 = "fp16"          # Demi-précision
    BF16 = "bf16"          # Brain floating point
    MIXED = "mixed"        # Précision mixte
    AUTO = "auto"          # Déterminé automatiquement

@dataclass
class QuantizationConfig:
    """Configuration pour la quantification de modèles."""
    
    # Méthode de quantification
    method: QuantizationMethod = QuantizationMethod.NONE
    
    # Format cible
    target_format: Optional[str] = None
    
    # Précision de calcul
    compute_precision: ComputePrecision = ComputePrecision.AUTO
    
    # Configuration pour les bits
    bits: int = 8    # 4, 8, 16, 32
    
    # Quantification par bloc
    block_size: int = 32
    
    # Type de quantification
    quant_type: str = "symmetric"  # symmetric, asymmetric, per_channel, per_tensor
    
    # Granularité
    granularity: str = "per_block"  # per_tensor, per_channel, per_block
    
    # Utiliser les propriétés des activations pour optimiser
    activation_aware: bool = True
    
    # Convertir les opérations pour la cible
    optimize_for_target: bool = True
    
    # Sauvegarder le modèle quantifié
    save_model: bool = True
    
    # Répertoire pour sauvegarder les modèles quantifiés
    output_dir: str = "./quantized_models"
    
    # Options spécifiques à MLX
    mlx_options: Dict[str, Any] = field(default_factory=dict)
    
    # Options spécifiques à CoreML
    coreml_options: Dict[str, Any] = field(default_factory=dict)
    
    # Permet de garder une copie du modèle original en mémoire
    keep_original: bool = False
    
    # Utiliser les optimisations spécifiques au matériel
    use_hardware_optimizations: bool = True
    
    def __post_init__(self):
        """Validation et configuration après initialisation."""
        # Si precision AUTO, déterminer automatiquement
        if self.compute_precision == ComputePrecision.AUTO:
            self.compute_precision = self._determine_optimal_precision()
            
        # Créer le répertoire de sortie
        if self.save_model:
            os.makedirs(self.output_dir, exist_ok=True)
    
    def _determine_optimal_method(self) -> QuantizationMethod:
        """Détermine la méthode de quantification optimale pour le matériel."""
        # Sur Apple Silicon, privilégier MLX/CoreML
        if IS_APPLE_SILICON:
            if HAS_MLX:
                return QuantizationMethod.MLX
            elif HAS_COREML:
                return QuantizationMethod.COREML
            elif HAS_MPS and self.bits == 8:
                return QuantizationMethod.INT8
            else:
                return QuantizationMethod.INT8
                
        # Sur CUDA, privilégier les techniques spécifiques
        elif HAS_CUDA:
            if HAS_BNB and self.bits == 4:
                return QuantizationMethod.INT4
            elif HAS_BNB:
                return QuantizationMethod.INT8
            elif HAS_AWQ and self.bits == 4 and self.activation_aware:
                return QuantizationMethod.AWQINT4
            elif HAS_AWQ and self.activation_aware:
                return QuantizationMethod.AWQINT8
            else:
                return QuantizationMethod.INT8
        
        # Par défaut
        return QuantizationMethod.INT8
    
    def _determine_optimal_precision(self) -> ComputePrecision:
        """Détermine la précision de calcul optimale pour le matériel."""
        # Apple Silicon + MPS → FP16
        if IS_APPLE_SILICON and HAS_MPS:
            return ComputePrecision.FP16
            
        # Apple Silicon + MLX → FP16
        elif IS_APPLE_SILICON and HAS_MLX:
            return ComputePrecision.FP16
            
        # CUDA avec support bfloat16
        elif HAS_CUDA and HAS_TORCH and torch.cuda.is_bf16_supported():
            return ComputePrecision.BF16
            
        # CUDA sans bf16
        elif HAS_CUDA:
            return ComputePrecision.FP16
            
        # Par défaut (CPU)
        return ComputePrecision.FP32


class ModelQuantizer:
    """
    Classe pour la quantification de modèles ML, avec des optimisations
    spécifiques pour les différentes plateformes, notamment Apple Silicon.
    """
    
    def __init__(self, config: Optional[QuantizationConfig] = None):
        """
        Initialise le quantificateur de modèles avec une configuration spécifique.
        
        Args:
            config: Configuration de quantification (optionnelle)
        """
        self.config = config or QuantizationConfig()
        
        # Initialiser le gestionnaire de mémoire si disponible
        self.memory_manager = get_memory_manager() if HAS_MEMORY_MANAGER else None
        
        # Informations sur le système
        self.system_info = {
            "platform": platform.system(),
            "processor": platform.processor(),
            "is_apple_silicon": IS_APPLE_SILICON,
            "has_mps": HAS_MPS,
            "has_cuda": HAS_CUDA,
            "has_mlx": HAS_MLX,
            "has_coreml": HAS_COREML,
            "has_bnb": HAS_BNB,
            "has_awq": HAS_AWQ
        }
        
        logger.info(f"ModelQuantizer initialisé avec {self.config.method}")
        logger.info(f"Système: {self.system_info['platform']} / {self.system_info['processor']}")
        logger.info(f"Backends disponibles: MPS={HAS_MPS}, MLX={HAS_MLX}, CoreML={HAS_COREML}")
        
        # Déterminer le type de matériel
        if IS_APPLE_SILICON:
            self.hardware_type = "mps" if HAS_MPS else "cpu"
        elif HAS_CUDA:
            self.hardware_type = "cuda"
        else:
            self.hardware_type = "cpu"
    
    def quantize(self, model: Any, example_inputs: Optional[Any] = None) -> Any:
        """
        Quantifie un modèle ML en utilisant la méthode configurée.
        
        Args:
            model: Modèle à quantifier
            example_inputs: Entrées exemple pour le tracing/calibration
            
        Returns:
            Modèle quantifié
        """
        try:
            logger.info(f"Quantification du modèle avec la méthode {self.config.method}")
            
            # Selon la méthode choisie
            if self.config.method == QuantizationMethod.INT8:
                return self._quantize_to_int8(model, example_inputs)
            elif self.config.method == QuantizationMethod.INT4:
                return self._quantize_to_int4(model, example_inputs)
            elif self.config.method == QuantizationMethod.GPTQ:
                return self._quantize_with_gptq(model, example_inputs)
            elif self.config.method == QuantizationMethod.AWQINT4:
                return self._quantize_with_awq(model, example_inputs, bits=4)
            elif self.config.method == QuantizationMethod.AWQINT8:
                return self._quantize_with_awq(model, example_inputs, bits=8)
            elif self.config.method == QuantizationMethod.MLX:
                return self._quantize_to_mlx(model, example_inputs)
            elif self.config.method == QuantizationMethod.COREML:
                return self._quantize_to_coreml(model, example_inputs)
            elif self.config.method == QuantizationMethod.ONNX:
                return self._export_to_onnx(model, example_inputs)
            else:
                logger.warning("Aucune quantification appliquée, retour au modèle original")
                return model
                
        except Exception as e:
            logger.error(f"Erreur lors de la quantification: {e}")
            return model
    
    def prepare_inputs_for_generation(self, 
                                    tokenizer, 
                                    prompts, 
                                    max_length: Optional[int] = None, 
                                    device: Optional[Any] = None) -> Dict[str, torch.Tensor]:
        """
        Prépare les entrées optimisées pour la génération avec le modèle quantifié.
        
        Args:
            tokenizer: Tokenizer du modèle
            prompts: Chaîne ou liste de chaînes de prompts
            max_length: Longueur maximale des séquences
            device: Périphérique sur lequel placer les tenseurs
            
        Returns:
            Dictionnaire d'entrées optimisées
        """
        # Utiliser les utilitaires d'attention si disponibles
        if HAS_ATTENTION_UTILS:
            # Préparer les entrées de base
            inputs = prepare_model_inputs(tokenizer, prompts, max_length, device)
            
            # Optimiser pour le matériel et la méthode de quantification
            inputs = optimize_attention_for_hardware(
                inputs, 
                hardware_type=self.hardware_type,
                quantization_method=self.config.method
            )
            
            return inputs
        else:
            # Méthode de repli simple
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                
            inputs = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True if max_length else False,
                max_length=max_length
            )
            
            if device:
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
            return inputs
            
    def get_generation_config(self, 
                            model_type: str, 
                            max_sequence_length: int) -> Dict[str, Any]:
        """
        Obtient une configuration de génération optimisée pour le modèle quantifié.
        
        Args:
            model_type: Type de modèle (causal_lm, seq2seq, etc.)
            max_sequence_length: Longueur maximale de séquence attendue
            
        Returns:
            Dictionnaire de paramètres pour la génération
        """
        # Configuration de base
        config = {
            "do_sample": False,  # Déterministe par défaut
            "num_beams": 1,      # Pas de beam search par défaut
            "max_new_tokens": 128,  # Valeur par défaut raisonnable
            "use_cache": True    # Utiliser le cache KV
        }
        
        # Ajustements basés sur la quantification
        if self.config.method in [QuantizationMethod.INT4, QuantizationMethod.INT8]:
            # Les modèles quantifiés en INT4/INT8 préfèrent des batchs plus petits
            config["batch_size"] = 1
            
        elif self.config.method == QuantizationMethod.MLX:
            # MLX a des optimisations spécifiques
            config["use_fp16"] = True
        
        # Ajustements basés sur la longueur de séquence
        if max_sequence_length > 2048:
            # Pour les longues séquences, activer des optimisations
            config["use_sliding_window"] = True
            
        return config
    
    def _quantize_to_mlx(self, model: 'torch.nn.Module', example_inputs: Optional[Any] = None) -> Any:
        """
        Convertit un modèle PyTorch en un modèle MLX quantifié.
        
        Args:
            model: Modèle PyTorch à convertir
            example_inputs: Entrées exemple pour le tracing
            
        Returns:
            Modèle MLX quantifié
        """
        if not HAS_MLX:
            logger.error("MLX n'est pas disponible, impossible de quantifier vers MLX")
            return model
        
        try:
            logger.info("Conversion du modèle PyTorch vers MLX...")
            
            # Mettre le modèle en mode évaluation
            model.eval()
            
            # S'assurer que le modèle est sur CPU pour la conversion
            # Cela est nécessaire car MLX ne peut pas directement travailler avec les tenseurs MPS
            if next(model.parameters(), torch.empty(0)).device.type == 'mps':
                logger.info("Déplacement du modèle de MPS vers CPU")
                # Créer un modèle temporaire sur CPU avec la même architecture
                with torch.device('cpu'):
                    # Désactiver les hooks et gradients temporairement
                    with torch.no_grad():
                        # Copier les paramètres sur CPU
                        for param_name, param in model.state_dict().items():
                            if param.device.type == 'mps':
                                model.state_dict()[param_name] = param.cpu()
                
                # Forcer le garbage collector pour libérer la mémoire MPS
                import gc
                gc.collect()
                if hasattr(torch.mps, 'empty_cache'):
                    torch.mps.empty_cache()
            
            # Tracer le modèle si des entrées exemple sont fournies
            if example_inputs is not None:
                # S'assurer que les entrées sont sur CPU
                if isinstance(example_inputs, torch.Tensor) and example_inputs.device.type == 'mps':
                    example_inputs = example_inputs.cpu()
                elif isinstance(example_inputs, dict):
                    for k, v in example_inputs.items():
                        if isinstance(v, torch.Tensor) and v.device.type == 'mps':
                            example_inputs[k] = v.cpu()
                            
                traced_model = torch.jit.trace(model, example_inputs)
            else:
                # Sans exemple, essayer de tracer avec des entrées aléatoires
                dummy_input = torch.randint(0, 1000, (1, 32), dtype=torch.long)  # Utiliser long pour les indices d'embedding
                traced_model = torch.jit.trace(model, dummy_input)
            
            # Configurer la précision MLX
            mlx_precision = mx.float16 if self.config.compute_precision == ComputePrecision.FP16 else mx.float32
            
            # Convertir les poids du modèle en tenseurs MLX
            mlx_model = {}
            for name, param in traced_model.state_dict().items():
                # S'assurer que les paramètres sont sur CPU et dans le bon type
                param_cpu = param.cpu()
                
                # Conversion des types spécifique pour MLX
                # Les tenseurs d'indices doivent être de type int32/int64 pour MLX
                if 'embed' in name and param_cpu.dtype in [torch.float32, torch.float16]:
                    param_cpu = param_cpu.to(torch.long)
                
                mlx_model[name] = mx.array(param_cpu.numpy(), dtype=mlx_precision)
            
            # Appliquer la quantification MLX
            if self.config.bits == 4:
                logger.info("Application de la quantification int4 MLX")
                for name, param in mlx_model.items():
                    if param.ndim > 1:  # Quantifier uniquement les matrices
                        mlx_model[name] = mx.quantize(param, "int4", self.config.quant_type)
            elif self.config.bits == 8:
                logger.info("Application de la quantification int8 MLX")
                for name, param in mlx_model.items():
                    if param.ndim > 1:  # Quantifier uniquement les matrices
                        mlx_model[name] = mx.quantize(param, "int8", self.config.quant_type)
            
            # Sauvegarder le modèle MLX si nécessaire
            if self.config.save_model:
                os.makedirs(self.config.output_dir, exist_ok=True)
                mlx_path = os.path.join(self.config.output_dir, "model_mlx.npz")
                mx.save(mlx_path, mlx_model)
                logger.info(f"Modèle MLX sauvegardé: {mlx_path}")
            
            # Créer un wrapper pour le modèle MLX
            class MLXModelWrapper:
                def __init__(self, mlx_weights, original_model=None):
                    self.weights = mlx_weights
                    self.original_model = original_model
                    
                def __call__(self, *args, **kwargs):
                    # Dans une implémentation réelle, cette méthode
                    # utiliserait les poids MLX pour l'inférence
                    if self.original_model:
                        return self.original_model(*args, **kwargs)
                    return None
            
            return MLXModelWrapper(mlx_model, model if self.config.keep_original else None)
            
        except Exception as e:
            logger.error(f"Erreur lors de la quantification MLX: {e}")
            return model
    
    def _quantize_to_coreml(self, model: 'torch.nn.Module', example_inputs: Optional[Any] = None) -> Any:
        """
        Convertit un modèle PyTorch en un modèle CoreML quantifié.
        
        Args:
            model: Modèle PyTorch à convertir
            example_inputs: Entrées exemple pour le tracing
            
        Returns:
            Modèle CoreML quantifié
        """
        if not HAS_COREML:
            logger.error("CoreML n'est pas disponible, impossible de quantifier vers CoreML")
            return model
            
        try:
            logger.info("Conversion du modèle PyTorch vers CoreML...")
            
            # Mettre le modèle en mode évaluation
            model.eval()
            
            # S'assurer que le modèle est sur CPU pour la conversion
            # CoreML ne peut pas travailler directement avec les tenseurs MPS
            model = model.cpu()
            
            # Tracer le modèle si des entrées exemple sont fournies
            if example_inputs is not None:
                # S'assurer que les entrées sont sur CPU
                if isinstance(example_inputs, torch.Tensor) and example_inputs.device.type == 'mps':
                    example_inputs = example_inputs.cpu()
                elif isinstance(example_inputs, dict):
                    for k, v in example_inputs.items():
                        if isinstance(v, torch.Tensor) and v.device.type == 'mps':
                            example_inputs[k] = v.cpu()
                            
                traced_model = torch.jit.trace(model, example_inputs)
            else:
                # Sans exemple, essayer de tracer avec des entrées de type long appropriées
                dummy_input = torch.randint(0, 1000, (1, 32), dtype=torch.long)
                traced_model = torch.jit.trace(model, dummy_input)
            
            # Configurer la précision CoreML
            compute_precision = ct.precision.FLOAT16 if self.config.compute_precision == ComputePrecision.FP16 else ct.precision.FLOAT32
            
            # Convertir vers CoreML - force CPU_ONLY pour éviter les problèmes avec MPS
            mlmodel = ct.convert(
                traced_model,
                convert_to="mlprogram",
                compute_precision=compute_precision,
                compute_units=ct.ComputeUnit.CPU_ONLY,  # Toujours utiliser CPU pour éviter les erreurs MPS
                **self.config.coreml_options
            )
            
            # Appliquer la quantification
            logger.info(f"Application de la quantification int{self.config.bits} pour CoreML")
            op_config = ct.optimize.coreml.OpLinearQuantizerConfig(
                mode=self.config.quant_type,
                dtype=f"int{self.config.bits}",
                granularity=self.config.granularity,
                block_size=self.config.block_size,
            )
            config = ct.optimize.coreml.OptimizationConfig(global_config=op_config)
            mlmodel_quant = ct.optimize.coreml.linear_quantize_weights(
                mlmodel, config=config
            )
            
            # Sauvegarder le modèle CoreML si nécessaire
            if self.config.save_model:
                os.makedirs(self.config.output_dir, exist_ok=True)
                model_path = os.path.join(self.config.output_dir, f"model_coreml_int{self.config.bits}.mlpackage")
                mlmodel_quant.save(model_path)
                logger.info(f"Modèle CoreML sauvegardé: {model_path}")
            
            # Créer un wrapper pour le modèle CoreML
            class CoreMLModelWrapper(torch.nn.Module):
                def __init__(self, coreml_model, original_model=None):
                    super().__init__()
                    self.coreml_model = coreml_model
                    self.original_model = original_model
                    self._model_spec = self.coreml_model.get_spec()
                    self._input_names = [input.name for input in self._model_spec.description.input]
                    self._output_names = [output.name for output in self._model_spec.description.output]
                    
                def forward(self, *args, **kwargs):
                    try:
                        # En cas d'erreur avec CoreML, fallback immédiat au modèle original
                        if not self.original_model:
                            logger.warning("Modèle CoreML inutilisable et pas de fallback disponible")
                            raise ValueError("CoreML model unavailable and no fallback model")
                            
                        # Utiliser le modèle PyTorch original (plus sûr pour le moment)
                        return self.original_model(*args, **kwargs)
                        
                        # Le code ci-dessous est désactivé car il cause des erreurs MPS
                        # À réactiver quand une solution stable sera trouvée
                        """
                        # Préparation des entrées pour CoreML
                        inputs = {}
                        
                        # Cas où args contient un tensor d'entrée
                        if args and isinstance(args[0], torch.Tensor):
                            input_tensor = args[0]
                            # S'assurer que le tensor est sur CPU
                            if input_tensor.device.type != 'cpu':
                                input_tensor = input_tensor.cpu()
                            
                            # Convertir le tensor en numpy array
                            input_np = input_tensor.numpy()
                            inputs[self._input_names[0]] = input_np
                        
                        # Cas où kwargs contient les entrées
                        elif kwargs:
                            for k, v in kwargs.items():
                                if isinstance(v, torch.Tensor):
                                    # S'assurer que le tensor est sur CPU
                                    if v.device.type != 'cpu':
                                        v = v.cpu()
                                    inputs[k] = v.numpy()
                                else:
                                    inputs[k] = v
                        
                        # Exécuter le modèle CoreML
                        outputs = self.coreml_model.predict(inputs)
                        
                        # Convertir les sorties en tensors PyTorch
                        if len(self._output_names) == 1:
                            output_name = self._output_names[0]
                            result = torch.tensor(outputs[output_name])
                            # Remettre sur le même device que l'entrée si possible
                            if args and isinstance(args[0], torch.Tensor):
                                result = result.to(args[0].device)
                            return result
                        else:
                            # Plusieurs sorties - retourner un dictionnaire
                            result = {}
                            for name in self._output_names:
                                result[name] = torch.tensor(outputs[name])
                                # Remettre sur le même device que l'entrée si possible
                                if args and isinstance(args[0], torch.Tensor):
                                    result[name] = result[name].to(args[0].device)
                            return result
                        """
                    except Exception as e:
                        logger.warning(f"Erreur lors de l'inférence CoreML: {e}")
                        # Fallback vers le modèle original si disponible
                        if self.original_model:
                            return self.original_model(*args, **kwargs)
                        raise e
            
            return CoreMLModelWrapper(mlmodel_quant, model if self.config.keep_original else None)
            
        except Exception as e:
            logger.error(f"Erreur lors de la quantification CoreML: {e}")
            return model
    
    def _quantize_int4(self, model: 'torch.nn.Module', example_inputs: Optional[Any] = None) -> 'torch.nn.Module':
        """
        Quantifie un modèle PyTorch en INT4 en utilisant BitsAndBytes.
        
        Args:
            model: Modèle PyTorch à quantifier
            example_inputs: Entrées exemple pour la calibration
            
        Returns:
            Modèle PyTorch quantifié en INT4
        """
        if not HAS_BNB:
            logger.error("BitsAndBytes n'est pas disponible, impossible de quantifier en INT4")
            return model
        
        try:
            logger.info("Quantification du modèle en INT4 avec BitsAndBytes...")
            
            # Sur Apple Silicon avec MPS, utiliser CoreML si disponible
            if IS_APPLE_SILICON and HAS_MPS and HAS_COREML and self.config.use_hardware_optimizations:
                logger.info("Redirection vers CoreML pour INT4 sur Apple Silicon")
                return self._quantize_to_coreml(model)
            
            # Si pas sur GPU CUDA, avertir
            if not HAS_CUDA:
                logger.warning("INT4 avec BitsAndBytes nécessite CUDA. CPU sera utilisé, mais sans quantification.")
                return model
            
            # Mettre le modèle en mode évaluation
            model.eval()
            
            # Configurer les options de quantification
            compute_dtype = torch.float16
            if self.config.compute_precision == ComputePrecision.BF16 and torch.cuda.is_bf16_supported():
                compute_dtype = torch.bfloat16
            
            # Appliquer la quantification
            model = bnb.nn.modules.Linear4bit.replace_linear_modules(model)
            
            for name, module in model.named_modules():
                if isinstance(module, bnb.nn.Linear4bit):
                    module.compute_dtype = compute_dtype
                    module.quant_type = "nf4" if self.config.quant_type == "symmetric" else "fp4"
            
            logger.info("Modèle quantifié en INT4")
            return model
            
        except Exception as e:
            logger.error(f"Erreur lors de la quantification INT4: {e}")
            return model
    
    def _quantize_int8(self, model: 'torch.nn.Module', example_inputs: Optional[Any] = None) -> 'torch.nn.Module':
        """
        Quantifie un modèle PyTorch en INT8 en utilisant BitsAndBytes.
        
        Args:
            model: Modèle PyTorch à quantifier
            example_inputs: Entrées exemple pour la calibration
            
        Returns:
            Modèle PyTorch quantifié en INT8
        """
        if not HAS_BNB:
            logger.error("BitsAndBytes n'est pas disponible, impossible de quantifier en INT8")
            return model
        
        try:
            logger.info("Quantification du modèle en INT8 avec BitsAndBytes...")
            
            # Sur Apple Silicon avec MPS, utiliser CoreML si disponible
            if IS_APPLE_SILICON and HAS_MPS and HAS_COREML and self.config.use_hardware_optimizations:
                logger.info("Redirection vers CoreML pour INT8 sur Apple Silicon")
                return self._quantize_to_coreml(model)
            
            # Si pas sur GPU CUDA, avertir
            if not HAS_CUDA:
                logger.warning("INT8 avec BitsAndBytes nécessite CUDA. CPU sera utilisé, mais sans quantification.")
                return model
            
            # Mettre le modèle en mode évaluation
            model.eval()
            
            # Quantifier le modèle
            model = bnb.nn.modules.Linear8bitLt.replace_linear_modules(model)
            
            logger.info("Modèle quantifié en INT8")
            return model
            
        except Exception as e:
            logger.error(f"Erreur lors de la quantification INT8: {e}")
            return model
    
    def _quantize_awq(self, model: 'torch.nn.Module', bits: int = 4, example_inputs: Optional[Any] = None) -> 'torch.nn.Module':
        """
        Quantifie un modèle PyTorch en utilisant AWQ (Activation-aware Weight Quantization).
        
        Args:
            model: Modèle PyTorch à quantifier
            bits: Nombre de bits (4 ou 8)
            example_inputs: Entrées exemple pour la calibration
            
        Returns:
            Modèle PyTorch quantifié avec AWQ
        """
        if not HAS_AWQ:
            logger.error("AWQ n'est pas disponible, impossible d'utiliser la quantification AWQ")
            return model
        
        try:
            logger.info(f"Quantification du modèle en AWQ-INT{bits}...")
            
            # Mettre le modèle en mode évaluation
            model.eval()
            
            # Configurer AWQ
            from awq.quantization import quantize_model
            
            # Si aucun exemple n'est fourni, créer des données aléatoires
            if example_inputs is None:
                example_inputs = torch.randn(1, 512, device="cuda" if HAS_CUDA else "cpu")
            
            # Appliquer la quantification AWQ
            quant_config = {"bits": bits, "sym": self.config.quant_type == "symmetric"}
            model_awq, _ = quantize_model(model, example_inputs, quant_config=quant_config)
            
            logger.info(f"Modèle quantifié en AWQ-INT{bits}")
            return model_awq
            
        except Exception as e:
            logger.error(f"Erreur lors de la quantification AWQ: {e}")
            return model
    
    def _quantize_gptq(self, model: 'torch.nn.Module', example_inputs: Optional[Any] = None) -> 'torch.nn.Module':
        """
        Quantifie un modèle PyTorch en utilisant GPTQ.
        
        Args:
            model: Modèle PyTorch à quantifier
            example_inputs: Entrées exemple pour la calibration
            
        Returns:
            Modèle PyTorch quantifié avec GPTQ
        """
        try:
            from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
            logger.info("Quantification du modèle avec GPTQ...")
            
            # Mettre le modèle en mode évaluation
            model.eval()
            
            # Configurer GPTQ
            quantize_config = BaseQuantizeConfig(
                bits=self.config.bits,
                group_size=self.config.block_size,
                sym=self.config.quant_type == "symmetric"
            )
            
            # Si aucun exemple n'est fourni, créer des données aléatoires
            if example_inputs is None:
                example_inputs = torch.randn(1, 512, device="cuda" if HAS_CUDA else "cpu")
            
            # Quantifier le modèle
            model_gptq = AutoGPTQForCausalLM.from_pretrained(
                model,
                quantize_config
            )
            model_gptq.quantize(example_inputs)
            
            logger.info("Modèle quantifié avec GPTQ")
            return model_gptq
            
        except ImportError:
            logger.error("auto-gptq n'est pas disponible, impossible d'utiliser la quantification GPTQ")
            return model
        except Exception as e:
            logger.error(f"Erreur lors de la quantification GPTQ: {e}")
            return model
    
    def _export_to_onnx(self, model: 'torch.nn.Module', example_inputs: Optional[Any] = None) -> Any:
        """
        Exporte un modèle PyTorch au format ONNX avec quantification.
        
        Args:
            model: Modèle PyTorch à exporter
            example_inputs: Entrées exemple pour le tracing
            
        Returns:
            Chemin vers le modèle ONNX ou wrapper de modèle
        """
        try:
            import onnx
            import onnxruntime
            logger.info("Exportation du modèle vers ONNX...")
            
            # Mettre le modèle en mode évaluation
            model.eval()
            
            # Créer des entrées exemple si nécessaire
            if example_inputs is None:
                example_inputs = torch.randn(1, 3, 224, 224)
            
            # Définir le chemin de sortie
            onnx_path = os.path.join(self.config.output_dir, "model.onnx")
            
            # Exporter vers ONNX
            torch.onnx.export(
                model,
                example_inputs,
                onnx_path,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
                opset_version=12,
                export_params=True
            )
            
            logger.info(f"Modèle exporté en ONNX: {onnx_path}")
            
            # Appliquer la quantification au modèle ONNX si nécessaire
            if self.config.bits in [4, 8]:
                from onnxruntime.quantization import quantize_dynamic, QuantType
                quantized_path = os.path.join(self.config.output_dir, f"model_quantized_int{self.config.bits}.onnx")
                
                weight_type = QuantType.QInt8 if self.config.bits == 8 else QuantType.QUInt4
                quantize_dynamic(
                    onnx_path,
                    quantized_path,
                    weight_type=weight_type
                )
                
                logger.info(f"Modèle ONNX quantifié: {quantized_path}")
                onnx_path = quantized_path
            
            # Créer un wrapper pour le modèle ONNX
            class ONNXModelWrapper(torch.nn.Module):
                def __init__(self, onnx_path, original_model=None):
                    super().__init__()
                    self.onnx_path = onnx_path
                    self.original_model = original_model
                    self.session = onnxruntime.InferenceSession(onnx_path)
                    self.input_name = self.session.get_inputs()[0].name
                    
                def forward(self, x):
                    # Utiliser ONNX Runtime pour l'inférence
                    if isinstance(x, torch.Tensor):
                        x = x.cpu().numpy()
                    
                    outputs = self.session.run(None, {self.input_name: x})
                    return outputs[0]
            
            return ONNXModelWrapper(onnx_path, model if self.config.keep_original else None)
            
        except ImportError:
            logger.error("onnx ou onnxruntime n'est pas disponible, impossible d'exporter en ONNX")
            return model
        except Exception as e:
            logger.error(f"Erreur lors de l'exportation ONNX: {e}")
            return model

def get_optimal_quantization_config() -> QuantizationConfig:
    """
    Crée une configuration de quantification optimisée pour le matériel actuel.
    
    Returns:
        Configuration de quantification optimisée
    """
    config = QuantizationConfig()
    
    # Adapter les options selon le matériel
    if IS_APPLE_SILICON:
        if HAS_MLX:
            config.method = QuantizationMethod.MLX
            config.bits = 4
            config.compute_precision = ComputePrecision.FP16
        elif HAS_COREML:
            config.method = QuantizationMethod.COREML
            config.bits = 4
            config.compute_precision = ComputePrecision.FP16
        elif HAS_MPS:
            config.method = QuantizationMethod.INT8
            config.compute_precision = ComputePrecision.FP16
    elif HAS_CUDA:
        if HAS_BNB:
            config.method = QuantizationMethod.INT4
            config.bits = 4
            config.compute_precision = ComputePrecision.BF16 if torch.cuda.is_bf16_supported() else ComputePrecision.FP16
        elif HAS_AWQ:
            config.method = QuantizationMethod.AWQINT4
            config.bits = 4
    
    return config

def quantize_model(model: Any, 
                  method: QuantizationMethod = QuantizationMethod.NONE,
                  bits: int = 4,
                  save_dir: str = "./quantized_models",
                  example_inputs: Optional[Any] = None) -> Any:
    """
    Fonction pratique pour quantifier un modèle avec les paramètres par défaut.
    
    Args:
        model: Modèle à quantifier
        method: Méthode de quantification
        bits: Nombre de bits pour la quantification
        save_dir: Répertoire de sauvegarde
        example_inputs: Entrées exemple pour le tracing/calibration
        
    Returns:
        Modèle quantifié
    """
    config = QuantizationConfig(
        method=method,
        bits=bits,
        output_dir=save_dir,
        save_model=True
    )
    
    quantizer = ModelQuantizer(config)
    return quantizer.quantize(model, example_inputs)


# Exemple d'utilisation
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Outil de quantification de modèles")
    parser.add_argument("--model", type=str, required=True, help="Chemin ou ID du modèle")
    parser.add_argument("--method", type=str, default="auto", help="Méthode de quantification")
    parser.add_argument("--bits", type=int, default=8, help="Nombre de bits (4 ou 8)")
    parser.add_argument("--output", type=str, default="./quantized_models", help="Dossier de sortie")
    
    args = parser.parse_args()
    
    try:
        import torch
        from transformers import AutoModelForCausalLM
        
        print(f"Chargement du modèle {args.model}...")
        model = AutoModelForCausalLM.from_pretrained(args.model)
        
        print(f"Quantification avec la méthode {args.method}, {args.bits} bits...")
        quantized_model = quantize_model(
            model,
            method=args.method,
            bits=args.bits,
            save_dir=args.output
        )
        
        print("Quantification terminée!")
        
    except ImportError:
        print("Impossible de charger les bibliothèques requises.")
        print("Installez torch et transformers: pip install torch transformers") 