#!/usr/bin/env python3
"""
Module d'adaptation CoreML pour les grands modèles de langage.
Optimisé pour Apple Silicon (M1, M2, M3, M4).

Ce module permet de convertir des modèles Hugging Face en modèles CoreML
pour une inférence optimisée sur les appareils Apple Silicon.
"""

import os
import logging
import torch
import platform
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass
import json
import shutil
import tempfile

# Importations conditionnelles
try:
    import coremltools as ct
    HAS_COREML = True
except ImportError:
    HAS_COREML = False

try:
    from optimum.exporters.coreml import CoreMLConfig, CoreMLQuantizer
    HAS_OPTIMUM_COREML = True
except ImportError:
    HAS_OPTIMUM_COREML = False

try:
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

from ..config import settings

# Configuration du logging
logger = logging.getLogger("ml_service.coreml_adapter")

@dataclass
class CoreMLAdapterConfig:
    """Configuration pour l'adaptation CoreML."""
    # Configurations générales
    compute_units: str = "ALL"  # "CPU_ONLY", "CPU_AND_GPU", "ALL" (inclut Neural Engine)
    quantize_weights: bool = True  # Quantifier les poids pour réduire la taille
    quantize_to: str = "fp16"  # "fp16", "int8", "int4"
    
    # Configurations avancées
    use_neural_engine: bool = True  # Utiliser le Neural Engine quand disponible
    use_mps: bool = True  # Utiliser MPS quand disponible
    batch_size: int = 1  # Taille du batch pour la conversion
    sequence_length: int = 512  # Longueur de séquence pour la conversion
    
    # Configurations de cache
    cache_dir: Optional[str] = None  # Répertoire de cache pour les modèles CoreML
    
    # KV Cache
    enable_kv_cache: bool = True  # Activer le cache KV pour l'inférence
    kv_cache_max_seq_length: int = 1024  # Longueur maximale pour le cache KV
    
    # Optimisation spécifique pour les modèles
    optimize_for_generation: bool = True  # Optimiser pour la génération de texte
    disable_attention_scaling: bool = False  # Désactiver le scaling d'attention
    
    # Parallélisation
    split_model: bool = False  # Diviser le modèle en plusieurs parties
    num_attention_shards: int = 1  # Nombre de shards pour l'attention


class CoreMLAdapter:
    """
    Adaptateur pour convertir et exécuter des modèles de langage avec CoreML.
    Optimisé pour Apple Silicon (M1, M2, M3, M4).
    """
    
    def __init__(
        self,
        config: Optional[CoreMLAdapterConfig] = None,
    ):
        """
        Initialise l'adaptateur CoreML.
        
        Args:
            config: Configuration pour CoreML
        """
        # Vérifier que nous sommes sur macOS avec Apple Silicon
        self.is_apple_silicon = platform.processor() == 'arm' and platform.system() == 'Darwin'
        if not self.is_apple_silicon:
            logger.warning("CoreML est optimisé pour Apple Silicon. Performances réduites sur cette plateforme.")
        
        # Vérifier les dépendances
        if not HAS_COREML:
            raise ImportError("La bibliothèque 'coremltools' n'est pas installée. Installez-la avec 'pip install coremltools'")
        
        if not HAS_OPTIMUM_COREML:
            raise ImportError("La bibliothèque 'optimum' n'est pas installée. Installez-la avec 'pip install optimum[coreml]'")
        
        if not HAS_TRANSFORMERS:
            raise ImportError("La bibliothèque 'transformers' n'est pas installée. Installez-la avec 'pip install transformers'")
        
        self.config = config or CoreMLAdapterConfig()
        self.cache_dir = self.config.cache_dir or str(settings.MODEL_PATH / "coreml_models")
        
        # Créer le répertoire de cache
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Détection de caractéristiques avancées
        self.has_mps = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
        self.mac_model, self.mac_chip = self._get_mac_model_info()
        
        # Optimisation automatique de configuration basée sur le matériel
        self._optimize_for_hardware()
        
        logger.info(f"CoreML Adapter initialisé pour {self.mac_model} avec puce {self.mac_chip}")
    
    def _get_mac_model_info(self):
        """
        Détecte le modèle de Mac et la puce.
        
        Returns:
            Tuple (modèle, puce)
        """
        try:
            # Obtenir le modèle du Mac
            model_result = subprocess.run(
                ["sysctl", "-n", "hw.model"],
                capture_output=True,
                text=True
            )
            mac_model = model_result.stdout.strip()
            
            # Déterminer la puce (M1, M2, M3, M4)
            chip_info = ""
            if "Apple M1" in platform.version():
                chip_info = "M1"
            elif "Apple M2" in platform.version():
                chip_info = "M2"
            elif "Apple M3" in platform.version():
                chip_info = "M3"
            elif "Apple M4" in platform.version():
                chip_info = "M4"
            
            # Si la puce n'est pas détectée via platform.version, essayer avec sysctl
            if not chip_info:
                cpu_result = subprocess.run(
                    ["sysctl", "-n", "machdep.cpu.brand_string"],
                    capture_output=True,
                    text=True
                )
                cpu_info = cpu_result.stdout.strip()
                
                if "Apple M1" in cpu_info:
                    chip_info = "M1"
                elif "Apple M2" in cpu_info:
                    chip_info = "M2"
                elif "Apple M3" in cpu_info:
                    chip_info = "M3"
                elif "Apple M4" in cpu_info:
                    chip_info = "M4"
            
            return mac_model, chip_info
        
        except Exception as e:
            logger.warning(f"Impossible de détecter le modèle Mac: {e}")
            return "Unknown Mac", "Unknown Chip"
    
    def _optimize_for_hardware(self):
        """Optimise la configuration en fonction du matériel détecté."""
        if not self.mac_chip:
            logger.warning("Puce Apple non détectée, utilisation des paramètres par défaut")
            return
        
        # Optimisations pour M4
        if self.mac_chip == "M4":
            logger.info("Optimisation pour Apple M4")
            self.config.compute_units = "ALL"
            self.config.quantize_to = "fp16"  # M4 gère bien le fp16
            self.config.use_neural_engine = True
            self.config.num_attention_shards = 2
        
        # Optimisations pour M3
        elif self.mac_chip == "M3":
            logger.info("Optimisation pour Apple M3")
            self.config.compute_units = "ALL"
            self.config.quantize_to = "fp16" 
            self.config.use_neural_engine = True
            self.config.num_attention_shards = 2
        
        # Optimisations pour M2
        elif self.mac_chip == "M2":
            logger.info("Optimisation pour Apple M2")
            self.config.compute_units = "ALL"
            self.config.quantize_to = "fp16"
            self.config.use_neural_engine = True
            self.config.num_attention_shards = 1
        
        # Optimisations pour M1
        elif self.mac_chip == "M1":
            logger.info("Optimisation pour Apple M1")
            self.config.compute_units = "ALL"
            self.config.quantize_to = "fp16"
            self.config.use_neural_engine = True
            self.config.num_attention_shards = 1
    
    def get_coreml_model_path(self, model_id: str) -> str:
        """
        Détermine le chemin du modèle CoreML pour un modèle Hugging Face.
        
        Args:
            model_id: ID du modèle Hugging Face
            
        Returns:
            Chemin du modèle CoreML
        """
        # Créer un identifiant sécurisé pour le chemin
        safe_model_id = model_id.replace("/", "_").replace(":", "_")
        
        # Ajouter des informations de configuration à l'identifiant
        config_suffix = f"_{self.config.quantize_to}"
        if self.config.enable_kv_cache:
            config_suffix += "_kvcache"
        if self.config.split_model:
            config_suffix += f"_split{self.config.num_attention_shards}"
        
        # Ajouter des informations sur la puce
        if self.mac_chip:
            config_suffix += f"_{self.mac_chip}"
        
        coreml_model_path = os.path.join(self.cache_dir, f"{safe_model_id}{config_suffix}")
        
        return coreml_model_path
    
    def is_model_converted(self, model_id: str) -> bool:
        """
        Vérifie si un modèle a déjà été converti en CoreML.
        
        Args:
            model_id: ID du modèle Hugging Face
            
        Returns:
            True si le modèle est déjà converti, False sinon
        """
        coreml_model_path = self.get_coreml_model_path(model_id)
        
        # Vérifier si le répertoire existe
        if os.path.exists(coreml_model_path):
            # Vérifier si le modèle est complet (au moins 1 fichier .mlpackage)
            mlpackages = list(Path(coreml_model_path).glob("*.mlpackage"))
            metadata_file = os.path.join(coreml_model_path, "model_metadata.json")
            
            return len(mlpackages) > 0 and os.path.exists(metadata_file)
        
        return False
    
    def convert_model(self, model_id: str, use_auth_token: Optional[str] = None) -> str:
        """
        Convertit un modèle Hugging Face en modèle CoreML.
        
        Args:
            model_id: ID du modèle Hugging Face
            use_auth_token: Token d'authentification pour modèles privés
            
        Returns:
            Chemin vers le modèle CoreML converti
        """
        # Vérifier si le modèle est déjà converti
        if self.is_model_converted(model_id):
            logger.info(f"Modèle {model_id} déjà converti en CoreML")
            return self.get_coreml_model_path(model_id)
        
        logger.info(f"Conversion du modèle {model_id} en CoreML")
        
        # Créer le répertoire de sortie
        output_path = self.get_coreml_model_path(model_id)
        os.makedirs(output_path, exist_ok=True)
        
        # Configuration CoreML
        compute_units = getattr(ct.ComputeUnit, self.config.compute_units)
        
        # Déterminer les optimisations spécifiques au modèle
        model_config = AutoConfig.from_pretrained(model_id, use_auth_token=use_auth_token)
        model_type = model_config.model_type
        
        # Créer la configuration CoreML
        coreml_config = CoreMLConfig(
            quantize=self.config.quantize_weights,
            quantize_output_type=self.config.quantize_to,
            compute_units=self.config.compute_units,
            batch_size=self.config.batch_size,
            sequence_length=self.config.sequence_length,
            kv_cache=self.config.enable_kv_cache,
            kv_cache_seq_len=self.config.kv_cache_max_seq_length,
        )
        
        # Convertir le modèle
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Charger le modèle
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                use_auth_token=use_auth_token
            )
            
            tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                use_auth_token=use_auth_token
            )
            
            # Exporter vers CoreML
            from optimum.exporters.coreml import export_coreml
            
            try:
                export_coreml(
                    model=model,
                    config=coreml_config,
                    task="text-generation",
                    export_path=output_path,
                    atol=1e-4,
                )
                
                # Sauvegarder les métadonnées
                metadata = {
                    "model_id": model_id,
                    "model_type": model_type,
                    "quantized": self.config.quantize_weights,
                    "quantized_to": self.config.quantize_to,
                    "compute_units": self.config.compute_units,
                    "kv_cache": self.config.enable_kv_cache,
                    "mac_chip": self.mac_chip,
                    "creation_date": str(datetime.datetime.now()),
                }
                
                with open(os.path.join(output_path, "model_metadata.json"), "w") as f:
                    json.dump(metadata, f, indent=2)
                
                # Sauvegarder également le tokenizer
                if not os.path.exists(os.path.join(output_path, "tokenizer")):
                    tokenizer.save_pretrained(os.path.join(output_path, "tokenizer"))
                
                logger.info(f"Modèle {model_id} converti avec succès en CoreML dans {output_path}")
                
                return output_path
                
            except Exception as e:
                logger.error(f"Erreur lors de la conversion du modèle {model_id} en CoreML: {e}")
                shutil.rmtree(output_path, ignore_errors=True)
                raise
    
    def load_model(self, model_id_or_path: str, use_auth_token: Optional[str] = None):
        """
        Charge un modèle CoreML.
        
        Args:
            model_id_or_path: ID du modèle Hugging Face ou chemin vers un modèle CoreML
            use_auth_token: Token d'authentification pour modèles privés
            
        Returns:
            Modèle CoreML chargé et son tokenizer
        """
        # Déterminer le chemin du modèle
        if os.path.exists(model_id_or_path) and os.path.isdir(model_id_or_path):
            model_path = model_id_or_path
        else:
            # Vérifier si le modèle est déjà converti
            if not self.is_model_converted(model_id_or_path):
                # Convertir le modèle
                model_path = self.convert_model(model_id_or_path, use_auth_token=use_auth_token)
            else:
                model_path = self.get_coreml_model_path(model_id_or_path)
        
        logger.info(f"Chargement du modèle CoreML depuis {model_path}")
        
        # Charger les métadonnées
        metadata_path = os.path.join(model_path, "model_metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            
            model_id = metadata.get("model_id", model_id_or_path)
        else:
            model_id = model_id_or_path
        
        # Charger le tokenizer
        tokenizer_path = os.path.join(model_path, "tokenizer")
        if os.path.exists(tokenizer_path):
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=use_auth_token)
        
        # Charger le modèle CoreML avec optimum-coreml
        from optimum.coreml import CoreMLModelForCausalLM
        
        model = CoreMLModelForCausalLM.from_pretrained(
            model_path,
            compute_units=self.config.compute_units,
            use_auth_token=use_auth_token
        )
        
        return model, tokenizer
    
    def optimize_for_generation(self, model, sampling_config=None):
        """
        Optimise un modèle CoreML pour la génération de texte.
        
        Args:
            model: Modèle CoreML
            sampling_config: Configuration de sampling (optionnel)
            
        Returns:
            Modèle optimisé
        """
        # Vérifier que nous avons un modèle CoreML
        if not hasattr(model, "config") or not hasattr(model, "generation_config"):
            logger.warning("Le modèle fourni n'est pas un modèle CoreML compatible")
            return model
        
        # Configurer la génération
        from transformers import GenerationConfig
        
        if sampling_config is None:
            sampling_config = {}
        
        # Configuration par défaut
        default_config = {
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 50,
            "repetition_penalty": 1.1,
            "max_new_tokens": 1024
        }
        
        # Fusionner avec la configuration fournie
        config = {**default_config, **sampling_config}
        
        # Appliquer la configuration
        generation_config = GenerationConfig(**config)
        model.generation_config = generation_config
        
        # Activer le KV cache si disponible
        if hasattr(model, "use_cache") and self.config.enable_kv_cache:
            model.use_cache = True
        
        return model
    
    def generate_text(self, model, tokenizer, prompt, sampling_config=None, **kwargs):
        """
        Génère du texte à partir d'un modèle CoreML.
        
        Args:
            model: Modèle CoreML
            tokenizer: Tokenizer
            prompt: Prompt pour la génération
            sampling_config: Configuration de sampling (optionnel)
            **kwargs: Arguments supplémentaires pour la génération
            
        Returns:
            Texte généré
        """
        # Optimiser le modèle pour la génération
        if self.config.optimize_for_generation:
            model = self.optimize_for_generation(model, sampling_config)
        
        # Préparer les entrées
        inputs = tokenizer(prompt, return_tensors="pt")
        
        # Générer
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            **kwargs
        )
        
        # Décoder la sortie
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return generated_text
    
    def benchmark_inference(self, model_id: str, prompt: str, num_runs: int = 5, **kwargs):
        """
        Réalise un benchmark d'inférence sur un modèle CoreML.
        
        Args:
            model_id: ID du modèle Hugging Face
            prompt: Prompt pour la génération
            num_runs: Nombre d'exécutions pour le benchmark
            **kwargs: Arguments supplémentaires pour la génération
            
        Returns:
            Dict avec les résultats du benchmark
        """
        import time
        
        # Charger le modèle
        model, tokenizer = self.load_model(model_id)
        
        # Mesurer le temps de génération
        generation_times = []
        tokens_generated = []
        
        for _ in range(num_runs):
            # Générer
            start_time = time.time()
            output = self.generate_text(model, tokenizer, prompt, **kwargs)
            end_time = time.time()
            
            # Calculer les statistiques
            generation_time = end_time - start_time
            output_tokens = len(tokenizer.encode(output)) - len(tokenizer.encode(prompt))
            
            generation_times.append(generation_time)
            tokens_generated.append(output_tokens)
        
        # Calculer les moyennes
        avg_generation_time = sum(generation_times) / len(generation_times)
        avg_tokens_generated = sum(tokens_generated) / len(tokens_generated)
        tokens_per_second = avg_tokens_generated / avg_generation_time
        
        # Résultats
        results = {
            "model_id": model_id,
            "prompt_length": len(tokenizer.encode(prompt)),
            "avg_generation_time": avg_generation_time,
            "avg_tokens_generated": avg_tokens_generated,
            "tokens_per_second": tokens_per_second,
            "mac_model": self.mac_model,
            "mac_chip": self.mac_chip,
            "compute_units": self.config.compute_units,
            "quantization": self.config.quantize_to if self.config.quantize_weights else "none",
        }
        
        return results


# Exemple d'utilisation:
if __name__ == "__main__":
    # Créer l'adaptateur CoreML
    adapter = CoreMLAdapter(
        config=CoreMLAdapterConfig(
            compute_units="ALL",
            quantize_weights=True,
            quantize_to="fp16",
            enable_kv_cache=True
        )
    )
    
    # Convertir un modèle
    model_id = "google/gemma-3-2b-it"
    model_path = adapter.convert_model(model_id)
    
    # Charger le modèle
    model, tokenizer = adapter.load_model(model_id)
    
    # Générer du texte
    prompt = "Explique-moi comment fonctionne l'architecture Apple Silicon en 3 phrases simples."
    generated_text = adapter.generate_text(
        model,
        tokenizer,
        prompt,
        max_new_tokens=100
    )
    
    print(f"Prompt: {prompt}")
    print(f"Generated: {generated_text}")
    
    # Réaliser un benchmark
    benchmark_results = adapter.benchmark_inference(
        model_id,
        prompt,
        num_runs=3,
        max_new_tokens=100
    )
    
    print(f"Benchmark results: {benchmark_results}") 