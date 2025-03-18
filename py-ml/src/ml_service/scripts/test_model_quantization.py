#!/usr/bin/env python3
"""
Script de test pour la quantification de modèles optimisée pour Apple Silicon.

Ce script démontre l'utilisation du ModelQuantizer pour optimiser des modèles
pour différents backends, avec un focus sur MLX et CoreML pour Apple Silicon.
"""

import os
import sys
import time
import gc
import json
import argparse
import logging
import torch
import platform
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any
import importlib.util

# Ajouter le répertoire parent au path pour pouvoir importer les modules
script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(script_dir, "../../"))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# Import des modules de ML Service
from ml_service.models.model_loader import ModelLoader, QuantizationType
HAS_MODEL_QUANTIZER = False
try:
    from ml_service.utils.model_quantizer import (
        ModelQuantizer, QuantizationConfig, QuantizationMethod
    )
    HAS_MODEL_QUANTIZER = True
except ImportError:
    logging.warning("ModelQuantizer non disponible")

# Configurer le logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("test_model_quantization")

# Vérifier si nous sommes sur Apple Silicon
IS_APPLE_SILICON = platform.system() == "Darwin" and platform.processor() == "arm"
if not IS_APPLE_SILICON:
    logger.warning("Ce script est optimisé pour Apple Silicon. Les performances seront limitées sur d'autres plateformes.")

def benchmark_inference(model, tokenizer, prompt: str, num_runs: int = 5, max_tokens: int = 100) -> Dict[str, float]:
    """
    Benchmark d'inférence pour un modèle.
    
    Args:
        model: Modèle à tester
        tokenizer: Tokenizer associé
        prompt: Prompt de test
        num_runs: Nombre d'exécutions
        max_tokens: Nombre maximal de tokens à générer
        
    Returns:
        Dictionnaire des métriques de performance
    """
    logger.info(f"Benchmark d'inférence pour {type(model).__name__}")
    logger.info(f"Prompt: {prompt}")
    
    # Essayer d'obtenir le quantizer si disponible dans le modèle
    quantizer = getattr(model, 'quantizer', None)
    
    # Préparation des entrées optimisées pour le modèle
    try:
        # Si le modèle a un quantizer, utiliser ses fonctions spécifiques
        if quantizer is not None and hasattr(quantizer, 'prepare_inputs_for_generation'):
            # Détecter l'appareil du modèle
            if hasattr(model, 'device'):
                model_device = model.device
            elif hasattr(model, 'parameters'):
                try:
                    model_device = next(model.parameters()).device
                except StopIteration:
                    model_device = 'cpu'
            else:
                model_device = 'cpu'
                
            # Préparer les entrées avec le quantizer
            inputs = quantizer.prepare_inputs_for_generation(
                tokenizer, 
                prompt, 
                max_length=None,  # Auto-detect
                device=model_device
            )
            
            input_ids = inputs['input_ids']
            attention_mask = inputs['attention_mask']
                
        else:
            # Méthode standard si pas de quantizer
            # Préparer le tokenizer et configurer les tokens spéciaux si nécessaires
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                logger.info(f"Pad token configuré à: {tokenizer.pad_token}")
            
            # Préparer les entrées avec attention_mask
            inputs = tokenizer(prompt, return_tensors="pt", padding=True)
            input_ids = inputs.input_ids
            attention_mask = inputs.attention_mask
            
            # Détecter l'appareil du modèle
            if hasattr(model, 'device'):
                model_device = model.device
            elif hasattr(model, 'parameters'):
                try:
                    model_device = next(model.parameters()).device
                except StopIteration:
                    model_device = 'cpu'
            else:
                model_device = 'cpu'
                
            # S'assurer que les entrées sont sur le même appareil que le modèle
            logger.info(f"Mise des entrées sur le dispositif {model_device}")
            input_ids = input_ids.to(model_device)
            attention_mask = attention_mask.to(model_device)
    
    except Exception as e:
        logger.warning(f"Erreur lors de la préparation optimisée des entrées: {e}")
        # Fallback simple
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        attention_mask = None
    
    # Tester si nous pouvons importer les utilitaires d'attention avancés
    try:
        from ml_service.utils.attention_utils import get_optimal_attention_config
        model_type = "causal_lm"  # Pour les modèles standards comme Phi-2
        seq_length = input_ids.shape[1] + max_tokens  # Estimation
        attention_config = get_optimal_attention_config(model_type, seq_length)
        logger.info(f"Configuration d'attention optimale: {attention_config}")
    except ImportError:
        attention_config = {}
        logger.info("Utilitaires d'attention avancés non disponibles")
    
    # Premier appel pour "réchauffer" le modèle (compilation/optimisation)
    try:
        # Construire les paramètres de génération
        generation_kwargs = {
            "max_new_tokens": 10,
            "num_beams": 1,
            "do_sample": False,
            "pad_token_id": tokenizer.pad_token_id
        }
        
        # Ajouter attention_mask si disponible
        if attention_mask is not None:
            generation_kwargs["attention_mask"] = attention_mask
            
        # Ajouter les configurations d'attention avancées si disponibles
        if "use_sliding_window" in attention_config and attention_config["use_sliding_window"]:
            # Des frameworks comme transformers supportent certaines options d'attention avancées
            if hasattr(model, "config"):
                model.config.use_cache = True
                
        outputs = model.generate(input_ids, **generation_kwargs)
        warmup_tokens = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info(f"Warmup généré: {warmup_tokens}")
    except Exception as e:
        logger.error(f"Erreur lors du warmup: {e}")
        return {"error": str(e)}
    
    # Benchmark
    times = []
    tokens_generated = []
    
    for i in range(num_runs):
        start_time = time.time()
        
        try:
            # Construire les paramètres de génération
            generation_kwargs = {
                "max_new_tokens": max_tokens,
                "num_beams": 1,
                "do_sample": False,
                "pad_token_id": tokenizer.pad_token_id
            }
            
            # Ajouter attention_mask si disponible
            if attention_mask is not None:
                generation_kwargs["attention_mask"] = attention_mask
                
            # Générer la réponse    
            outputs = model.generate(input_ids, **generation_kwargs)
            
            generation = tokenizer.decode(outputs[0], skip_special_tokens=True)
            num_tokens = len(outputs[0]) - len(input_ids[0])
            
            elapsed = time.time() - start_time
            times.append(elapsed)
            tokens_generated.append(num_tokens)
            
            logger.info(f"Run {i+1}/{num_runs}: {elapsed:.4f}s, {num_tokens} tokens, {num_tokens/elapsed:.2f} tokens/s")
            
            if i == 0:
                logger.info(f"Génération exemple: {generation[:100]}...")
                
        except Exception as e:
            logger.error(f"Erreur lors de l'inférence (run {i+1}): {e}")
            return {"error": str(e)}
    
    # Calculer les métriques
    avg_time = sum(times) / len(times)
    avg_tokens = sum(tokens_generated) / len(tokens_generated)
    tokens_per_second = avg_tokens / avg_time
    
    # Résultats
    results = {
        "avg_time": avg_time,
        "avg_tokens": avg_tokens,
        "tokens_per_second": tokens_per_second,
        "times": times,
        "tokens_generated": tokens_generated
    }
    
    logger.info(f"Résultats: {avg_time:.4f}s en moyenne, {tokens_per_second:.2f} tokens/s")
    
    return results

def compare_methods(model_id, available_methods, prompt, max_tokens=100):
    """Compare différentes méthodes de quantification sur le même modèle."""
    results = {}
    
    for method in available_methods:
        try:
            logger.info(f"\n{'='*50}\nTest de la méthode: {method}\n{'='*50}")
            
            # Charger le modèle avec la méthode spécifiée
            if method == "mlx" and HAS_MODEL_QUANTIZER:
                # Cas spécial pour MLX qui utilise ModelQuantizer
                loader = ModelLoader()
                model, tokenizer = loader.load_model(
                    model_id_or_path=model_id,
                    quantization=QuantizationType.NONE,
                    trust_remote_code=True,
                    lazy_loading=False  # Désactiver lazy loading pour MLX
                )
                
                config = QuantizationConfig(
                    method=QuantizationMethod.MLX,
                    bits=4
                )
                quantizer = ModelQuantizer(config)
                model = quantizer.quantize(model)
            else:
                # Utiliser le ModelLoader standard
                method_map = {
                    "none": QuantizationType.NONE,
                    "int8": QuantizationType.INT8,
                    "int4": QuantizationType.INT4,
                    "gptq": QuantizationType.GPTQ,
                    "ggml": QuantizationType.GGML,
                    "coreml": QuantizationType.COREML
                }
                
                if method in method_map:
                    loader = ModelLoader()
                    model, tokenizer = loader.load_model(
                        model_id_or_path=model_id,
                        quantization=method_map[method],
                        trust_remote_code=True,
                        lazy_loading=True,
                        lazy_loading_threshold=1.0  # Seuil bas pour tester
                    )
                else:
                    logger.warning(f"Méthode {method} non prise en charge, ignorée")
                    continue
            
            # Exécuter le benchmark
            logger.info(f"Benchmark d'inférence pour la méthode {method}")
            start_time = time.time()
            bench_results = benchmark_inference(model, tokenizer, prompt, max_tokens=max_tokens)
            total_time = time.time() - start_time
            
            # Stocker les résultats
            results[method] = {
                "avg_time_per_run": bench_results.get("avg_time", 0),
                "tokens_per_second": bench_results.get("tokens_per_second", 0),
                "total_time": total_time,
                "max_tokens": max_tokens
            }
            
            # Libérer la mémoire
            del model
            del tokenizer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
        except Exception as e:
            logger.error(f"Erreur lors du test de la méthode {method}: {str(e)}")
            results[method] = {"error": str(e)}
    
    # Afficher les résultats
    logger.info("\n\n=== RÉSULTATS DE LA COMPARAISON ===")
    logger.info(f"{'Méthode':<10} | {'Tokens/s':>10} | {'Temps moyen (s)':>15}")
    logger.info("-" * 45)
    
    for method, result in results.items():
        if "error" in result:
            logger.info(f"{method:<10} | {'ERREUR':>10} | {result['error']}")
        else:
            logger.info(f"{method:<10} | {result['tokens_per_second']:>10.2f} | {result['avg_time_per_run']:>15.4f}")
    
    # Sauvegarder les résultats
    with open("benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info("Résultats sauvegardés dans benchmark_results.json")
    
    return results

def detect_available_methods():
    """Détecte les méthodes de quantification disponibles."""
    methods = ["none"]  # Toujours disponible
    
    # Vérifier les bibliothèques disponibles
    if torch.cuda.is_available() or hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        methods.extend(["int8", "int4"])
    
    if importlib.util.find_spec("auto_gptq") is not None:
        methods.append("gptq")
    
    if importlib.util.find_spec("llama_cpp") is not None:
        methods.append("ggml")
    
    if platform.system() == "Darwin" and platform.processor() == "arm":
        if importlib.util.find_spec("coremltools") is not None:
            methods.append("coreml")
        if importlib.util.find_spec("mlx") is not None:
            methods.append("mlx")
    
    return methods

def init_logger():
    """Configure le logger pour le script."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger("test_model_quantization")

def log_system_info():
    """Journalise les informations système."""
    logger.info(f"Système: {platform.system()} {platform.release()}")
    logger.info(f"Processeur: {platform.processor()}")
    logger.info(f"Python: {platform.python_version()}")
    
    # Informations sur PyTorch
    logger.info(f"PyTorch: {torch.__version__}")
    logger.info(f"CUDA disponible: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Vérifier MPS (Metal Performance Shaders) pour macOS
    if hasattr(torch.backends, 'mps') and hasattr(torch.backends.mps, 'is_available'):
        logger.info(f"MPS disponible: {torch.backends.mps.is_available()}")
    
    # Vérifier les bibliothèques spécifiques
    for lib_name in ["transformers", "mlx", "coremltools", "bitsandbytes", "auto_gptq", "llama_cpp"]:
        spec = importlib.util.find_spec(lib_name)
        logger.info(f"{lib_name}: {'Disponible' if spec is not None else 'Non disponible'}")

def parse_args():
    parser = argparse.ArgumentParser(description="Test de quantification de modèles ML")
    parser.add_argument("--model", type=str, default="facebook/opt-125m", help="ID du modèle à tester")
    parser.add_argument("--method", type=str, choices=["none", "int8", "int4", "gptq", "ggml", "coreml", "mlx"], 
                      default="none", help="Méthode de quantification à utiliser")
    parser.add_argument("--bits", type=int, default=4, help="Nombre de bits pour la quantification (4 ou 8)")
    parser.add_argument("--prompt", type=str, 
                      default="Explique-moi comment fonctionne l'architecture MLX d'Apple en 5 points:", 
                      help="Prompt à utiliser pour le test")
    parser.add_argument("--compare", action="store_true", help="Comparer toutes les méthodes disponibles")
    parser.add_argument("--max-tokens", type=int, default=200, help="Nombre maximal de tokens à générer")
    parser.add_argument("--lazy-loading", action="store_true", help="Activer le chargement paresseux (lazy loading)")
    
    return parser.parse_args()

def main():
    args = parse_args()

    # Journaliser les informations système
    log_system_info()
    
    # Détecter les méthodes de quantification disponibles
    available_methods = detect_available_methods()
    
    if args.compare:
        logger.info("Comparaison des méthodes de quantification disponibles")
        compare_methods(args.model, available_methods, args.prompt, args.max_tokens)
    else:
        if args.method not in available_methods:
            logger.error(f"Méthode {args.method} non disponible. Méthodes disponibles: {', '.join(available_methods)}")
            return
        
        # Configuration de lazy loading pour le ModelLoader
        lazy_loading = args.lazy_loading
        
        # Loguer les options sélectionnées
        logger.info(f"Chargement du modèle {args.model} avec quantification {args.method} ({args.bits} bits)")
        if lazy_loading:
            logger.info("Chargement paresseux (lazy loading) activé")
        
        try:
            # Convertir les noms de méthodes en énumérations
            method_map = {
                "none": QuantizationType.NONE,
                "int8": QuantizationType.INT8,
                "int4": QuantizationType.INT4,
                "gptq": QuantizationType.GPTQ,
                "ggml": QuantizationType.GGML,
                "coreml": QuantizationType.COREML,
                "mlx": "mlx"  # Géré spécialement ci-dessous
            }
            
            # Utiliser le ModelQuantizer si disponible
            if HAS_MODEL_QUANTIZER and args.method in ["none", "int4", "int8", "coreml", "mlx"]:
                # Conversion de la méthode en QuantizationMethod
                method_to_quant_method = {
                    "none": QuantizationMethod.NONE,
                    "int4": QuantizationMethod.INT4,
                    "int8": QuantizationMethod.INT8,
                    "coreml": QuantizationMethod.COREML,
                    "mlx": QuantizationMethod.MLX
                }
                
                # On charge le modèle avec le ModelLoader
                logger.info(f"Quantification du modèle avec QuantizationMethod.{method_to_quant_method[args.method].name}")
                
                # Options spécifiques au chargement
                model_kwargs = {
                    "lazy_loading": lazy_loading,
                    "lazy_loading_threshold": 1.0  # Seuil bas pour tester avec des petits modèles aussi
                }
                
                # Charger le modèle en utilisant le ModelLoader
                loader = ModelLoader()
                model, tokenizer = loader.load_model(
                    model_id_or_path=args.model,
                    quantization=method_map[args.method] if args.method != "mlx" else QuantizationType.NONE,
                    trust_remote_code=True,
                    **model_kwargs
                )
                
                # Configurer la quantification si nécessaire
                if args.method != "none" and args.method == "mlx":
                    config = QuantizationConfig(
                        method=method_to_quant_method[args.method],
                        bits=args.bits
                    )
                    quantizer = ModelQuantizer(config)
                    model = quantizer.quantize(model)
                    
                logger.info(f"Modèle quantifié avec la méthode {args.method}")
                
            else:
                # Méthode standard avec le ModelLoader
                loader = ModelLoader()
                model, tokenizer = loader.load_model(
                    model_id_or_path=args.model,
                    quantization=method_map[args.method],
                    trust_remote_code=True,
                    lazy_loading=lazy_loading,
                    lazy_loading_threshold=1.0  # Seuil bas pour tester avec des petits modèles aussi
                )
            
            # Évaluer les performances d'inférence
            logger.info(f"Benchmark d'inférence pour {type(model).__name__}")
            benchmark_inference(model, tokenizer, args.prompt, max_tokens=args.max_tokens)
                
        except Exception as e:
            logger.error(f"Erreur lors du test: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())

if __name__ == "__main__":
    # Initialisation du logger
    logger = init_logger()
    main()