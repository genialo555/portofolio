#!/usr/bin/env python3
"""
Script d'orchestration pour exécuter un pipeline optimisé pour Apple Silicon.
Intègre DSPy, QLoRA Vision et optimisations CoreML.

Utilisation:
    python run_optimized_pipeline.py --task [vision|rag|chat] --model [model_id]
"""

import os
import sys
import argparse
import logging
import torch
import platform
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
import json
import time

# Ajouter le chemin du projet au PYTHONPATH
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.append(project_root)

from ml_service.utils.memory_manager import get_memory_manager
from ml_service.utils.coreml_adapter import CoreMLAdapter, CoreMLAdapterConfig
from ml_service.agents.dspy_optimizer import DSPyOptimizer, DSPyConfig
from ml_service.agents.vision_qlora import VisionQLoRAAdapter, QLoRAConfig
from ml_service.config import settings

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(project_root, "logs/optimized_pipeline.log"))
    ]
)
logger = logging.getLogger("ml_service.optimized_pipeline")

# Créer le répertoire de logs s'il n'existe pas
os.makedirs(os.path.join(project_root, "logs"), exist_ok=True)


class OptimizedPipeline:
    """
    Pipeline optimisé qui intègre DSPy, QLoRA et CoreML pour une
    performance maximale sur Apple Silicon (M1, M2, M3, M4).
    """
    
    def __init__(self, task: str = "chat", model_id: str = None):
        """
        Initialise le pipeline optimisé.
        
        Args:
            task: Type de tâche (vision, rag, chat)
            model_id: ID du modèle à utiliser
        """
        self.task = task
        self.model_id = model_id or self._get_default_model()
        
        # Détection pour Apple Silicon
        self.is_apple_silicon = platform.processor() == 'arm' and platform.system() == 'Darwin'
        self.has_mps = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
        
        # Détection du modèle Mac et puce
        self.mac_model, self.mac_chip = self._get_mac_model_info()
        
        # Initialiser le memory manager
        self.memory_manager = get_memory_manager()
        
        # Initialiser les composants selon la tâche
        self.coreml_adapter = None
        self.dspy_optimizer = None
        self.vision_adapter = None
        
        # Modèles et tokenizers
        self.model = None
        self.tokenizer = None
        
        logger.info(f"Pipeline optimisé initialisé pour {self.task} avec {self.model_id}")
        logger.info(f"Système: {self.mac_model} avec puce {self.mac_chip}")
    
    def _get_default_model(self):
        """Retourne le modèle par défaut selon la tâche."""
        defaults = {
            "vision": "google/gemma-3-12b-it",
            "rag": "meta-llama/Llama-3-8B-Instruct",
            "chat": "google/gemma-3-2b-it"
        }
        
        return defaults.get(self.task, "google/gemma-3-2b-it")
    
    def _get_mac_model_info(self):
        """
        Détecte le modèle de Mac et la puce.
        
        Returns:
            Tuple (modèle, puce)
        """
        import subprocess
        
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
    
    def setup(self):
        """
        Configure le pipeline selon la tâche.
        
        Returns:
            Self pour chaînage
        """
        if self.task == "vision":
            self._setup_vision()
        elif self.task == "rag":
            self._setup_rag()
        else:  # "chat" par défaut
            self._setup_chat()
        
        return self
    
    def _setup_vision(self):
        """Configure le pipeline pour la vision."""
        logger.info("Configuration du pipeline pour la vision")
        
        # Configurer QLoRA pour Vision
        qlora_config = QLoRAConfig(
            r=16,
            lora_alpha=32,
            batch_size=1,
            vision_tower="openai/clip-vit-large-patch14"
        )
        
        self.vision_adapter = VisionQLoRAAdapter(
            base_model_id=self.model_id,
            qlora_config=qlora_config
        )
        
        # Préparer le modèle pour l'entraînement ou l'inférence
        self.model, self.tokenizer = self.vision_adapter.prepare_model_for_training()
    
    def _setup_rag(self):
        """Configure le pipeline pour RAG."""
        logger.info("Configuration du pipeline pour RAG")
        
        # Configurer DSPy
        dspy_config = DSPyConfig(
            primary_model_id=self.model_id,
            secondary_models=[
                {"id": "google/gemma-3-2b-it", "name": "gemma_small"}
            ],
            use_colbert=True
        )
        
        self.dspy_optimizer = DSPyOptimizer(config=dspy_config)
        
        # Charger les modèles
        self.lm_dict = self.dspy_optimizer.load_models()
        
        # Préparer le retriever
        # Nous utiliserons un retriever fictif pour cet exemple
        # Dans un cas réel, vous devriez configurer un retriever avec vos documents
        import dspy
        documents = [
            dspy.Document("Document 1", "Contenu du document 1"),
            dspy.Document("Document 2", "Contenu du document 2"),
        ]
        
        self.retriever = self.dspy_optimizer.setup_retriever(documents=documents)
    
    def _setup_chat(self):
        """Configure le pipeline pour le chat."""
        logger.info("Configuration du pipeline pour le chat")
        
        # Si nous sommes sur Apple Silicon, utiliser CoreML
        if self.is_apple_silicon:
            logger.info("Utilisation de CoreML pour le chat sur Apple Silicon")
            
            # Configurer CoreML
            coreml_config = CoreMLAdapterConfig(
                compute_units="ALL",
                quantize_weights=True,
                quantize_to="fp16",
                enable_kv_cache=True
            )
            
            self.coreml_adapter = CoreMLAdapter(config=coreml_config)
            
            # Charger le modèle CoreML
            try:
                self.model, self.tokenizer = self.coreml_adapter.load_model(self.model_id)
                logger.info(f"Modèle {self.model_id} chargé avec CoreML")
                return
            except Exception as e:
                logger.warning(f"Échec du chargement avec CoreML: {e}")
                logger.info("Repli sur le chargement standard")
        
        # Si CoreML échoue ou si nous ne sommes pas sur Apple Silicon
        # Utiliser le memory manager pour charger le modèle de manière optimisée
        logger.info("Chargement du modèle avec le memory manager")
        
        load_options = self.memory_manager.optimize_model_loading(
            self.model_id,
            quantization="int4" if not self.is_apple_silicon else "mps"
        )
        
        # Importer transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            **load_options
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            use_fast=True
        )
        
        logger.info(f"Modèle {self.model_id} chargé avec le memory manager")
    
    def run_chat(self, prompt, max_new_tokens=100, temperature=0.7):
        """
        Exécute une tâche de chat.
        
        Args:
            prompt: Prompt pour le modèle
            max_new_tokens: Nombre maximum de tokens à générer
            temperature: Température pour la génération
            
        Returns:
            Texte généré
        """
        if not self.model or not self.tokenizer:
            logger.error("Modèle ou tokenizer non initialisé. Appelez setup() d'abord.")
            return None
        
        logger.info(f"Génération de texte avec prompt: {prompt[:50]}...")
        
        # Si CoreML est disponible
        if self.coreml_adapter and hasattr(self.model, "generate"):
            return self.coreml_adapter.generate_text(
                self.model,
                self.tokenizer,
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature
            )
        
        # Sinon, utiliser l'API standard de transformers
        inputs = self.tokenizer(prompt, return_tensors="pt")
        
        # Déplacer sur le bon device
        if self.is_apple_silicon and self.has_mps:
            inputs = {k: v.to("mps") for k, v in inputs.items()}
        elif torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        
        # Générer
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=0.95,
                top_k=50,
            )
        
        # Décoder
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return generated_text
    
    def run_rag(self, query):
        """
        Exécute une tâche de RAG.
        
        Args:
            query: Requête pour la recherche
            
        Returns:
            Réponse générée
        """
        if not self.dspy_optimizer:
            logger.error("DSPy Optimizer non initialisé. Appelez setup() d'abord.")
            return None
        
        logger.info(f"Exécution de RAG avec requête: {query}")
        
        # Créer un module RAG
        import dspy
        
        class RAG(dspy.Module):
            def __init__(self, retriever):
                super().__init__()
                self.retriever = retriever
                self.generate = dspy.ChainOfThought("context, query -> answer")
            
            def forward(self, query):
                # Récupérer les documents pertinents
                docs = self.retriever(query)
                context = "\n".join([d.text for d in docs])
                
                # Générer une réponse basée sur les documents
                answer = self.generate(context=context, query=query).answer
                
                return dspy.Prediction(answer=answer, documents=docs)
        
        # Créer un module RAG avec notre retriever
        rag_module = RAG(self.retriever)
        
        # Exécuter l'inférence
        result = self.dspy_optimizer.run_inference(
            rag_module,
            {"query": query}
        )
        
        return result.answer if hasattr(result, "answer") else str(result)
    
    def run_vision(self, image_path, prompt=None):
        """
        Exécute une tâche de vision.
        
        Args:
            image_path: Chemin vers l'image
            prompt: Prompt textuel (optionnel)
            
        Returns:
            Texte généré à partir de l'image
        """
        if not self.vision_adapter or not self.model or not self.tokenizer:
            logger.error("Vision Adapter non initialisé. Appelez setup() d'abord.")
            return None
        
        logger.info(f"Génération à partir de l'image: {image_path}")
        
        # Générer à partir de l'image
        return self.vision_adapter.generate_from_image(
            self.model,
            self.tokenizer,
            image_path,
            prompt=prompt
        )
    
    def run(self, input_data):
        """
        Exécute le pipeline approprié selon la tâche configurée.
        
        Args:
            input_data: Données d'entrée (varie selon la tâche)
            
        Returns:
            Résultat du pipeline
        """
        if self.task == "vision":
            if isinstance(input_data, dict):
                return self.run_vision(input_data.get("image"), input_data.get("prompt"))
            else:
                return self.run_vision(input_data)
        elif self.task == "rag":
            return self.run_rag(input_data)
        else:  # "chat" par défaut
            return self.run_chat(input_data)
    
    def benchmark(self, input_data, num_runs=3):
        """
        Exécute un benchmark du pipeline.
        
        Args:
            input_data: Données d'entrée pour le benchmark
            num_runs: Nombre d'exécutions
            
        Returns:
            Résultats du benchmark
        """
        logger.info(f"Exécution d'un benchmark pour la tâche {self.task}")
        
        run_times = []
        outputs = []
        
        for i in range(num_runs):
            logger.info(f"Run {i+1}/{num_runs}")
            
            # Mesurer le temps d'exécution
            start_time = time.time()
            output = self.run(input_data)
            end_time = time.time()
            
            run_time = end_time - start_time
            run_times.append(run_time)
            outputs.append(output)
            
            logger.info(f"Temps d'exécution: {run_time:.2f} secondes")
        
        # Calculer les statistiques
        avg_time = sum(run_times) / len(run_times)
        min_time = min(run_times)
        max_time = max(run_times)
        
        results = {
            "task": self.task,
            "model_id": self.model_id,
            "mac_model": self.mac_model,
            "mac_chip": self.mac_chip,
            "num_runs": num_runs,
            "avg_time": avg_time,
            "min_time": min_time,
            "max_time": max_time,
            "times": run_times,
            "outputs": outputs
        }
        
        return results


def parse_args():
    """Parse les arguments de ligne de commande."""
    parser = argparse.ArgumentParser(description="Exécuter un pipeline optimisé pour Apple Silicon")
    
    parser.add_argument(
        "--task",
        type=str,
        choices=["vision", "rag", "chat"],
        default="chat",
        help="Type de tâche à exécuter"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        help="ID du modèle à utiliser"
    )
    
    parser.add_argument(
        "--input",
        type=str,
        help="Entrée pour le pipeline (prompt, requête ou chemin vers une image)"
    )
    
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Exécuter un benchmark"
    )
    
    parser.add_argument(
        "--num_runs",
        type=int,
        default=3,
        help="Nombre d'exécutions pour le benchmark"
    )
    
    return parser.parse_args()


def main():
    """Fonction principale."""
    args = parse_args()
    
    # Vérifier si nous avons une entrée
    if not args.input:
        # Utiliser des entrées par défaut selon la tâche
        if args.task == "vision":
            args.input = "path/to/image.jpg"  # Remplacer par un chemin réel
        elif args.task == "rag":
            args.input = "Quels sont les avantages des puces Apple Silicon?"
        else:
            args.input = "Explique-moi comment fonctionne l'architecture Apple Silicon en 3 phrases simples."
    
    # Initialiser le pipeline
    pipeline = OptimizedPipeline(task=args.task, model_id=args.model)
    
    # Configurer le pipeline
    pipeline.setup()
    
    # Exécuter le pipeline
    if args.benchmark:
        logger.info("Exécution du benchmark")
        results = pipeline.benchmark(args.input, num_runs=args.num_runs)
        
        # Afficher les résultats
        print("\n=== Résultats du benchmark ===")
        print(f"Tâche: {results['task']}")
        print(f"Modèle: {results['model_id']}")
        print(f"Système: {results['mac_model']} avec puce {results['mac_chip']}")
        print(f"Nombre d'exécutions: {results['num_runs']}")
        print(f"Temps moyen: {results['avg_time']:.2f} secondes")
        print(f"Temps minimum: {results['min_time']:.2f} secondes")
        print(f"Temps maximum: {results['max_time']:.2f} secondes")
        
        # Sauvegarder les résultats
        results_path = os.path.join(project_root, "logs/benchmark_results.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"Résultats sauvegardés dans {results_path}")
    else:
        logger.info("Exécution du pipeline")
        output = pipeline.run(args.input)
        
        # Afficher le résultat
        print("\n=== Résultat ===")
        print(output)


if __name__ == "__main__":
    main() 