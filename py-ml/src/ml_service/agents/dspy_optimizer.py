#!/usr/bin/env python3
"""
Module d'optimisation DSPy pour les grands modèles de langage.
Permet d'optimiser les prompts et d'orchestrer plusieurs modèles ensemble.
Optimisé pour Apple Silicon (M1, M2, M3, M4).

DSPy est un framework qui permet de composer des modèles de langage
et d'optimiser automatiquement leurs prompts pour améliorer les performances.
"""

import os
import logging
import torch
import platform
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass
import json

# Importation conditionnelle de DSPy
try:
    import dspy
    from dspy.teleprompt import BootstrapFewShot
    from dspy.retrieve import ColBERTv2
    HAS_DSPY = True
except ImportError:
    HAS_DSPY = False

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

from ..utils.memory_manager import get_memory_manager
from ..config import settings

# Configuration du logging
logger = logging.getLogger("ml_service.dspy_optimizer")

@dataclass
class DSPyConfig:
    """Configuration pour l'optimisation DSPy."""
    # Configuration des modèles
    primary_model_id: str = "meta-llama/Llama-3-8B-Instruct"  # Modèle principal
    secondary_models: List[Dict[str, str]] = None  # Modèles secondaires
    
    # Configuration de DSPy
    cache_path: str = "dspy_cache"  # Chemin pour le cache DSPy
    metric_class: str = "Accuracy"  # Métrique d'évaluation
    bootstrap_examples: int = 3  # Nombre d'exemples pour le bootstrap
    max_bootstrapped_demos: int = 5  # Nombre max de démonstrations
    
    # Configuration spécifique à Apple Silicon
    use_mps: bool = True  # Utiliser MPS quand disponible
    use_coreml: bool = True  # Utiliser CoreML quand disponible
    optimize_for_m_series: bool = True  # Optimisations spécifiques pour les puces M
    
    # Configuration avancée
    use_colbert: bool = False  # Utiliser ColBERT pour la recherche
    colbert_url: Optional[str] = None  # URL pour ColBERT
    use_ensemble: bool = True  # Utiliser un ensemble de modèles
    ensemble_strategy: str = "voting"  # Stratégie d'ensemble (voting, averaging)
    
    # Configuration des modules
    modules: List[str] = None  # Modules DSPy à utiliser


class DSPyOptimizer:
    """
    Optimiseur DSPy pour les grands modèles de langage.
    Permet d'optimiser les prompts et d'orchestrer plusieurs modèles ensemble.
    """
    
    def __init__(
        self,
        config: Optional[DSPyConfig] = None,
        cache_dir: Optional[str] = None,
    ):
        """
        Initialise l'optimiseur DSPy.
        
        Args:
            config: Configuration DSPy
            cache_dir: Répertoire de cache pour les modèles
        """
        if not HAS_DSPY:
            raise ImportError("La bibliothèque 'dspy' n'est pas installée. Installez-la avec 'pip install dspy-ai'")
        
        self.config = config or DSPyConfig()
        self.cache_dir = cache_dir or str(settings.MODEL_PATH)
        
        # Détection pour Apple Silicon
        self.is_apple_silicon = platform.processor() == 'arm' and platform.system() == 'Darwin'
        self.has_mps = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
        
        # Configurer le Memory Manager
        self.memory_manager = get_memory_manager(cache_dir=self.cache_dir)
        
        # Initialiser les modèles et modules
        self.lm_dict = {}  # Dict des modèles de langage
        self.primary_lm = None  # Modèle principal
        self.retriever = None  # Retriever pour RAG
        
        # Configurer DSPy
        self._configure_dspy()
        
        logger.info(f"DSPy Optimizer initialisé avec le modèle principal {self.config.primary_model_id}")
    
    def _configure_dspy(self):
        """Configure les paramètres de base de DSPy."""
        # Configurer le cache
        dspy.settings.configure(cache_path=self.config.cache_path)
        
        # Configurer le traçage DSPy
        dspy.settings.configure(trace=True)
        
        # Optimisations pour Apple Silicon
        if self.is_apple_silicon and self.config.optimize_for_m_series:
            logger.info("Application des optimisations pour Apple Silicon")
            
            # Activer la fusion des opérations pour torch
            if hasattr(torch, "_dynamo"):
                torch._dynamo.config.cache_size_limit = 128
            
            # Utiliser la parallélisation optimisée
            torch.set_num_threads(8)  # Ajuster selon le nombre de cœurs
    
    def load_models(self):
        """
        Charge les modèles de langage pour DSPy.
        
        Returns:
            Dict des modèles de langage
        """
        # Charger le modèle principal
        primary_model_config = self.memory_manager.optimize_model_loading(
            self.config.primary_model_id,
            quantization="coreml" if self.is_apple_silicon and self.config.use_coreml else "int4"
        )
        
        # Créer le LM dans DSPy
        self.primary_lm = self._create_dspy_lm(
            model_id=self.config.primary_model_id,
            model_config=primary_model_config,
            name="primary"
        )
        
        # Définir le LM par défaut
        dspy.settings.configure(lm=self.primary_lm)
        
        # Charger les modèles secondaires si spécifiés
        if self.config.secondary_models:
            for model_info in self.config.secondary_models:
                model_id = model_info["id"]
                model_name = model_info.get("name", model_id.split("/")[-1])
                
                # Optimiser le chargement du modèle
                model_config = self.memory_manager.optimize_model_loading(
                    model_id,
                    quantization="coreml" if self.is_apple_silicon and self.config.use_coreml else "int4"
                )
                
                # Créer le LM dans DSPy
                self.lm_dict[model_name] = self._create_dspy_lm(
                    model_id=model_id,
                    model_config=model_config,
                    name=model_name
                )
        
        # Ajouter le modèle principal au dictionnaire
        self.lm_dict["primary"] = self.primary_lm
        
        return self.lm_dict
    
    def _create_dspy_lm(self, model_id, model_config, name):
        """
        Crée un modèle de langage DSPy.
        
        Args:
            model_id: ID du modèle
            model_config: Configuration du modèle
            name: Nom du modèle
            
        Returns:
            Modèle de langage DSPy
        """
        logger.info(f"Création du modèle DSPy {name} avec {model_id}")
        
        # Vérifier si nous utilisons l'API HF
        use_hf_api = model_config.get("use_hf_api", False)
        
        if use_hf_api:
            # Utiliser l'API HF si demandé
            return dspy.HFClientLM(
                model=model_id,
                name=name,
                max_tokens=512,
                api_key=model_config.get("api_key", None)
            )
        else:
            # Configurer le modèle local
            return dspy.HFLocalLM(
                model=model_id,
                name=name,
                max_tokens=512,
                device_map="auto",
                model_kwargs=model_config,
                tokenizer_kwargs={"use_fast": True}
            )
    
    def setup_retriever(self, documents=None, url=None):
        """
        Configure un retriever pour la recherche d'information.
        
        Args:
            documents: Liste de documents pour l'indexation
            url: URL du service de recherche (optionnel)
            
        Returns:
            Retriever DSPy
        """
        if self.config.use_colbert and (self.config.colbert_url or url):
            # Utiliser ColBERT pour la recherche
            self.retriever = ColBERTv2(url=url or self.config.colbert_url)
            logger.info(f"Retriever ColBERT configuré avec {url or self.config.colbert_url}")
        elif documents:
            # Créer un retriever basé sur les documents
            self.retriever = dspy.VectorDBRetriever(documents)
            logger.info(f"Retriever vectoriel configuré avec {len(documents)} documents")
        else:
            logger.warning("Pas de documents ou d'URL fournis pour le retriever")
            return None
        
        return self.retriever
    
    def create_module(self, module_type, **kwargs):
        """
        Crée un module DSPy.
        
        Args:
            module_type: Type de module
            **kwargs: Arguments pour le module
            
        Returns:
            Module DSPy
        """
        if not hasattr(dspy, module_type):
            raise ValueError(f"Type de module DSPy inconnu: {module_type}")
        
        # Créer le module DSPy
        module_class = getattr(dspy, module_type)
        return module_class(**kwargs)
    
    def create_multi_model_module(self):
        """
        Crée un module qui utilise plusieurs modèles pour le raisonnement.
        
        Returns:
            Module MultiModelReasoner
        """
        if len(self.lm_dict) <= 1:
            logger.warning("Un seul modèle disponible, utilisation d'un module standard.")
            return self.create_module("ChainOfThought", questions=["input"], answer="output")
        
        # Définition d'une classe pour le raisonnement multi-modèles
        class MultiModelReasoner(dspy.Module):
            def __init__(self, models_dict, ensemble_strategy="voting"):
                super().__init__()
                self.models = models_dict
                self.ensemble_strategy = ensemble_strategy
                self.signatures = {
                    name: dspy.Signature(
                        input_fields=["question"],
                        output_fields=["answer"]
                    ) for name in models_dict.keys()
                }
                
            def forward(self, question):
                # Interroger chaque modèle
                responses = {}
                for name, model in self.models.items():
                    with dspy.settings.context(lm=model):
                        responses[name] = dspy.Predict(self.signatures[name])(question=question).answer
                        
                # Fusionner les réponses selon la stratégie
                if self.ensemble_strategy == "voting":
                    # Utiliser le modèle principal comme arbitre
                    with dspy.settings.context(lm=self.models.get("primary")):
                        fusion = dspy.ChainOfThought("question, model_responses -> final_answer")
                        result = fusion(
                            question=question,
                            model_responses=str(responses)
                        )
                    return result.final_answer
                else:
                    # Retourner toutes les réponses (stratégie par défaut)
                    return responses
        
        return MultiModelReasoner(self.lm_dict, ensemble_strategy=self.config.ensemble_strategy)
    
    def optimize_prompts(self, train_data, eval_data, metric=None):
        """
        Optimise les prompts pour un module DSPy.
        
        Args:
            train_data: Données d'entraînement
            eval_data: Données d'évaluation
            metric: Métrique d'évaluation (optionnel)
            
        Returns:
            Module optimisé
        """
        # Créer un module
        if self.config.use_ensemble:
            module = self.create_multi_model_module()
        else:
            # Utiliser un module simple
            module = self.create_module("ChainOfThought", 
                                        questions=["question"], 
                                        answer="answer")
        
        # Créer une métrique si non fournie
        if metric is None:
            if not hasattr(dspy.evaluate, self.config.metric_class):
                logger.warning(f"Métrique {self.config.metric_class} non trouvée, utilisation de Accuracy")
                metric_class = dspy.evaluate.Accuracy
            else:
                metric_class = getattr(dspy.evaluate, self.config.metric_class)
            
            metric = lambda example, pred: metric_class().score(example, pred)
        
        # Configurer l'optimiseur de prompt
        teleprompter = BootstrapFewShot(
            metric=metric,
            max_bootstrapped_demos=self.config.max_bootstrapped_demos,
            num_candidate_programs=self.config.bootstrap_examples
        )
        
        # Optimiser les prompts
        compiled_module = teleprompter.compile(
            module=module,
            trainset=train_data,
            valset=eval_data
        )
        
        logger.info("Prompts optimisés avec succès")
        
        return compiled_module
    
    def run_inference(self, module, inputs, lm_name=None):
        """
        Exécute l'inférence avec un module DSPy.
        
        Args:
            module: Module DSPy
            inputs: Entrées pour le module
            lm_name: Nom du modèle à utiliser (optionnel)
            
        Returns:
            Résultats de l'inférence
        """
        # Sélectionner le modèle à utiliser
        if lm_name and lm_name in self.lm_dict:
            with dspy.settings.context(lm=self.lm_dict[lm_name]):
                return module(**inputs)
        else:
            # Utiliser le modèle par défaut
            return module(**inputs)
    
    def save_optimized_module(self, module, path=None):
        """
        Sauvegarde un module optimisé.
        
        Args:
            module: Module optimisé
            path: Chemin de sauvegarde (optionnel)
            
        Returns:
            Chemin de sauvegarde
        """
        # Créer le chemin de sauvegarde si non fourni
        if path is None:
            path = os.path.join(
                settings.MODEL_PATH, 
                "dspy_modules", 
                f"{module.__class__.__name__.lower()}.json"
            )
        
        # Créer le répertoire si nécessaire
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Sérialiser le module
        state = {
            "type": module.__class__.__name__,
            "config": vars(self.config),
            "demonstrations": module.demos if hasattr(module, "demos") else [],
            "prompt_template": module.prompt_template if hasattr(module, "prompt_template") else None
        }
        
        # Sauvegarder l'état
        with open(path, "w") as f:
            json.dump(state, f, indent=2)
        
        logger.info(f"Module optimisé sauvegardé dans {path}")
        
        return path
    
    def load_optimized_module(self, path):
        """
        Charge un module optimisé.
        
        Args:
            path: Chemin du module
            
        Returns:
            Module optimisé
        """
        # Charger l'état
        with open(path, "r") as f:
            state = json.load(f)
        
        # Récupérer le type de module
        module_type = state["type"]
        if not hasattr(dspy, module_type):
            raise ValueError(f"Type de module DSPy inconnu: {module_type}")
        
        # Créer le module
        module_class = getattr(dspy, module_type)
        module = module_class()
        
        # Restaurer les démonstrations si présentes
        if "demonstrations" in state and hasattr(module, "demos"):
            module.demos = state["demonstrations"]
        
        # Restaurer le template de prompt si présent
        if "prompt_template" in state and hasattr(module, "prompt_template"):
            module.prompt_template = state["prompt_template"]
        
        logger.info(f"Module optimisé chargé depuis {path}")
        
        return module


# Exemple d'utilisation
if __name__ == "__main__":
    # Configuration pour utiliser Gemma et Llama
    config = DSPyConfig(
        primary_model_id="google/gemma-3-12b-it",
        secondary_models=[
            {"id": "meta-llama/Llama-3-8B-Instruct", "name": "llama3"}
        ],
        use_ensemble=True
    )
    
    # Créer l'optimiseur DSPy
    dspy_optimizer = DSPyOptimizer(config=config)
    
    # Charger les modèles
    models = dspy_optimizer.load_models()
    
    # Créer quelques données d'exemple
    train_data = [
        dspy.Example(
            question="Quelle est la capitale de la France ?",
            answer="La capitale de la France est Paris."
        ),
        dspy.Example(
            question="Qu'est-ce que l'intelligence artificielle ?",
            answer="L'intelligence artificielle est un domaine de l'informatique qui se concentre sur la création de machines capables d'effectuer des tâches qui nécessiteraient normalement l'intelligence humaine."
        )
    ]
    
    eval_data = [
        dspy.Example(
            question="Quelle est la capitale de l'Allemagne ?",
            answer="La capitale de l'Allemagne est Berlin."
        )
    ]
    
    # Optimiser les prompts
    optimized_module = dspy_optimizer.optimize_prompts(train_data, eval_data)
    
    # Exécuter l'inférence
    result = dspy_optimizer.run_inference(
        optimized_module,
        {"question": "Quelle est la capitale de l'Espagne ?"}
    )
    
    print(f"Résultat: {result}")
    
    # Sauvegarder le module optimisé
    dspy_optimizer.save_optimized_module(optimized_module) 