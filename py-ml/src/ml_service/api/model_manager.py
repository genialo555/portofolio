import os
import logging
import torch
import time
from typing import Dict, Any, List, Optional
from pathlib import Path
from functools import lru_cache
from fastapi import Depends

from ml_service.models.teacher_model import TeacherModel, TeacherConfig
from ml_service.models.image_teacher import ImageTeacherModel
from ml_service.config import settings

# Configuration du logging
logger = logging.getLogger("ml_api.model_manager")

class ModelManager:
    """
    Gestionnaire de modèles ML.
    
    Cette classe est responsable de:
    - Charger et gérer les modèles ML
    - Gérer le cycle de vie des modèles (chargement, déchargement)
    - Optimiser l'utilisation des ressources (GPU, mémoire)
    - Fournir une interface unifiée pour tous les modèles
    """
    
    def __init__(self, model_cache_dir: str = None):
        """
        Initialise le gestionnaire de modèles.
        
        Args:
            model_cache_dir (str, optional): Répertoire de cache pour les modèles. Defaults to None.
        """
        self.model_cache_dir = model_cache_dir or str(settings.MODEL_PATH)
        self.models = {}  # Modèles chargés en mémoire
        
        # Liste des modèles supportés
        self.supported_models = {
            "text": ["teacher", "qwen25", "mixtral8x7b", "llama3"],
            "image": ["sdxl", "dalle3", "midjourney"]
        }
        
        # Vérification de la disponibilité du GPU
        self.gpu_available = torch.cuda.is_available()
        if self.gpu_available:
            logger.info(f"GPU disponible: {torch.cuda.get_device_name(0)}")
            self.device = torch.device("cuda")
        else:
            logger.warning("Aucun GPU disponible, utilisation du CPU")
            self.device = torch.device("cpu")
            
        # Métriques
        self.metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_processing_time": 0,
            "models": {}
        }
        
        # Préchargement des modèles de base
        try:
            self._preload_essential_models()
        except Exception as e:
            logger.error(f"Erreur lors du préchargement des modèles: {str(e)}", exc_info=True)
    
    def _preload_essential_models(self):
        """Précharge les modèles essentiels."""
        logger.info("Préchargement des modèles essentiels...")
        # Préchargement du TeacherModel
        self.load_model("teacher")
    
    def list_available_models(self) -> List[Dict[str, Any]]:
        """
        Liste tous les modèles disponibles.
        
        Returns:
            List[Dict[str, Any]]: Liste des modèles disponibles
        """
        available_models = []
        
        # Modèles de texte
        for model_name in self.supported_models["text"]:
            model_info = {
                "name": model_name,
                "type": "text",
                "version": "1.0.0",
                "description": f"Modèle de texte {model_name}",
                "status": "loaded" if model_name in self.models else "available",
                "capabilities": ["evaluate", "synthesize"]
            }
            available_models.append(model_info)
        
        # Modèles d'image
        for model_name in self.supported_models["image"]:
            model_info = {
                "name": model_name,
                "type": "image",
                "version": "1.0.0",
                "description": f"Modèle d'image {model_name}",
                "status": "loaded" if model_name in self.models else "available",
                "capabilities": ["generate"]
            }
            available_models.append(model_info)
        
        return available_models
    
    def load_model(self, model_name: str) -> Any:
        """
        Charge un modèle en mémoire.
        
        Args:
            model_name (str): Nom du modèle à charger
            
        Returns:
            Any: Instance du modèle chargé
            
        Raises:
            ValueError: Si le modèle n'est pas supporté
        """
        # Vérification si le modèle est déjà chargé
        if model_name in self.models:
            logger.info(f"Modèle {model_name} déjà chargé")
            return self.models[model_name]
        
        # Vérification si le modèle est supporté
        if model_name not in self.supported_models["text"] and model_name not in self.supported_models["image"]:
            raise ValueError(f"Modèle {model_name} non supporté")
        
        logger.info(f"Chargement du modèle {model_name}...")
        start_time = time.time()
        
        try:
            # Chargement du modèle en fonction de son type
            if model_name in self.supported_models["text"]:
                if model_name == "teacher":
                    # Chargement du TeacherModel
                    config = TeacherConfig()
                    model = TeacherModel(config)
                else:
                    # Simulation pour les autres modèles de texte
                    model = TeacherModel(TeacherConfig(model_name=model_name))
            
            elif model_name in self.supported_models["image"]:
                # Chargement de l'ImageTeacherModel
                model = ImageTeacherModel(model_name=model_name)
            
            # Enregistrement du modèle
            self.models[model_name] = model
            
            # Enregistrement des métriques
            loading_time = time.time() - start_time
            if model_name not in self.metrics["models"]:
                self.metrics["models"][model_name] = {
                    "total_requests": 0,
                    "successful_requests": 0,
                    "failed_requests": 0,
                    "total_processing_time": 0,
                    "loading_time": loading_time
                }
            else:
                self.metrics["models"][model_name]["loading_time"] = loading_time
            
            logger.info(f"Modèle {model_name} chargé en {loading_time:.2f}s")
            return model
        
        except Exception as e:
            logger.error(f"Erreur lors du chargement du modèle {model_name}: {str(e)}", exc_info=True)
            raise
    
    def unload_model(self, model_name: str) -> bool:
        """
        Décharge un modèle de la mémoire.
        
        Args:
            model_name (str): Nom du modèle à décharger
            
        Returns:
            bool: True si le modèle a été déchargé, False sinon
        """
        if model_name not in self.models:
            logger.warning(f"Modèle {model_name} non chargé")
            return False
        
        logger.info(f"Déchargement du modèle {model_name}...")
        try:
            # Suppression du modèle
            del self.models[model_name]
            
            # Nettoyage du cache CUDA si nécessaire
            if self.gpu_available:
                torch.cuda.empty_cache()
            
            logger.info(f"Modèle {model_name} déchargé")
            return True
        
        except Exception as e:
            logger.error(f"Erreur lors du déchargement du modèle {model_name}: {str(e)}", exc_info=True)
            return False
    
    def get_metrics(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Récupère les métriques d'un modèle ou de tous les modèles.
        
        Args:
            model_name (str, optional): Nom du modèle. Defaults to None.
            
        Returns:
            Dict[str, Any]: Métriques du modèle ou de tous les modèles
        """
        if model_name:
            if model_name not in self.metrics["models"]:
                return {}
            return self.metrics["models"][model_name]
        
        return self.metrics
    
    def update_metrics(self, model_name: str, success: bool, processing_time: float):
        """
        Met à jour les métriques d'un modèle.
        
        Args:
            model_name (str): Nom du modèle
            success (bool): Indique si la requête a réussi
            processing_time (float): Temps de traitement en secondes
        """
        # Mise à jour des métriques globales
        self.metrics["total_requests"] += 1
        if success:
            self.metrics["successful_requests"] += 1
        else:
            self.metrics["failed_requests"] += 1
        self.metrics["total_processing_time"] += processing_time
        
        # Mise à jour des métriques du modèle
        if model_name not in self.metrics["models"]:
            self.metrics["models"][model_name] = {
                "total_requests": 0,
                "successful_requests": 0,
                "failed_requests": 0,
                "total_processing_time": 0
            }
        
        self.metrics["models"][model_name]["total_requests"] += 1
        if success:
            self.metrics["models"][model_name]["successful_requests"] += 1
        else:
            self.metrics["models"][model_name]["failed_requests"] += 1
        self.metrics["models"][model_name]["total_processing_time"] += processing_time
    
    def evaluate_response(self, response: str, context: Optional[Dict[str, str]] = None, model: str = "teacher") -> Dict[str, Any]:
        """
        Évalue une réponse avec un modèle de texte.
        
        Args:
            response (str): Réponse à évaluer
            context (Dict[str, str], optional): Contexte additionnel. Defaults to None.
            model (str, optional): Nom du modèle à utiliser. Defaults to "teacher".
            
        Returns:
            Dict[str, Any]: Résultats de l'évaluation
        """
        start_time = time.time()
        success = False
        
        try:
            # Chargement du modèle si nécessaire
            if model not in self.models:
                self.load_model(model)
            
            # Vérification que le modèle est un modèle de texte
            if model not in self.supported_models["text"]:
                raise ValueError(f"Le modèle {model} n'est pas un modèle de texte")
            
            # Évaluation de la réponse
            model_instance = self.models[model]
            evaluation = model_instance.evaluate_response(response, context)
            
            # Mise à jour des métriques
            processing_time = time.time() - start_time
            success = True
            self.update_metrics(model, success, processing_time)
            
            return evaluation
        
        except Exception as e:
            # Mise à jour des métriques
            processing_time = time.time() - start_time
            self.update_metrics(model, success, processing_time)
            
            logger.error(f"Erreur lors de l'évaluation avec le modèle {model}: {str(e)}", exc_info=True)
            raise
    
    def synthesize_debate(self, perspective_a: str, perspective_b: str, history: Optional[List[str]] = None, model: str = "teacher") -> Dict[str, Any]:
        """
        Synthétise un débat avec un modèle de texte.
        
        Args:
            perspective_a (str): Première perspective
            perspective_b (str): Deuxième perspective
            history (List[str], optional): Historique de la conversation. Defaults to None.
            model (str, optional): Nom du modèle à utiliser. Defaults to "teacher".
            
        Returns:
            Dict[str, Any]: Résultats de la synthèse
        """
        start_time = time.time()
        success = False
        
        try:
            # Chargement du modèle si nécessaire
            if model not in self.models:
                self.load_model(model)
            
            # Vérification que le modèle est un modèle de texte
            if model not in self.supported_models["text"]:
                raise ValueError(f"Le modèle {model} n'est pas un modèle de texte")
            
            # Synthèse du débat
            model_instance = self.models[model]
            synthesis = model_instance.synthesize(perspective_a, perspective_b, history or [])
            
            # Mise à jour des métriques
            processing_time = time.time() - start_time
            success = True
            self.update_metrics(model, success, processing_time)
            
            return synthesis
        
        except Exception as e:
            # Mise à jour des métriques
            processing_time = time.time() - start_time
            self.update_metrics(model, success, processing_time)
            
            logger.error(f"Erreur lors de la synthèse avec le modèle {model}: {str(e)}", exc_info=True)
            raise
    
    def generate_image(self, prompt: str, negative_prompt: Optional[str] = None, config: Optional[Dict[str, Any]] = None, model: str = "sdxl") -> Dict[str, Any]:
        """
        Génère une image avec un modèle d'image.
        
        Args:
            prompt (str): Prompt pour la génération d'image
            negative_prompt (str, optional): Prompt négatif. Defaults to None.
            config (Dict[str, Any], optional): Configuration pour la génération. Defaults to None.
            model (str, optional): Nom du modèle à utiliser. Defaults to "sdxl".
            
        Returns:
            Dict[str, Any]: URL de l'image générée et métadonnées
        """
        start_time = time.time()
        success = False
        
        try:
            # Chargement du modèle si nécessaire
            if model not in self.models:
                self.load_model(model)
            
            # Vérification que le modèle est un modèle d'image
            if model not in self.supported_models["image"]:
                raise ValueError(f"Le modèle {model} n'est pas un modèle d'image")
            
            # Génération de l'image
            model_instance = self.models[model]
            image_result = model_instance.generate(prompt, negative_prompt, config)
            
            # Mise à jour des métriques
            processing_time = time.time() - start_time
            success = True
            self.update_metrics(model, success, processing_time)
            
            return image_result
        
        except Exception as e:
            # Mise à jour des métriques
            processing_time = time.time() - start_time
            self.update_metrics(model, success, processing_time)
            
            logger.error(f"Erreur lors de la génération d'image avec le modèle {model}: {str(e)}", exc_info=True)
            raise


@lru_cache()
def get_model_manager() -> ModelManager:
    """
    Obtient l'instance unique du gestionnaire de modèles.
    
    Cette fonction est utilisée comme une dépendance dans FastAPI.
    
    Returns:
        ModelManager: Instance unique du gestionnaire de modèles
    """
    return ModelManager() 