from fastapi import APIRouter, HTTPException, Depends, Body
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field, validator
import logging
import time
from pathlib import Path
import os

# Importation des modèles et du gestionnaire de modèles
from ml_service.models.teacher_model import TeacherModel, TeacherConfig
from ml_service.models.image_teacher import ImageTeacherModel
from ml_service.api.model_manager import ModelManager, get_model_manager

# Configuration du logging
logger = logging.getLogger("ml_api.models")

# Création du router
router = APIRouter()

# Modèles de données pour les requêtes et réponses API
class EvaluateRequest(BaseModel):
    """Modèle de données pour la requête d'évaluation."""
    response: str = Field(..., description="Réponse à évaluer")
    context: Optional[Dict[str, str]] = Field(None, description="Contexte additionnel pour l'évaluation")
    model: Optional[str] = Field("teacher", description="Modèle à utiliser pour l'évaluation")

    @validator('model')
    def validate_model(cls, v):
        valid_models = ["teacher", "qwen25", "mixtral8x7b", "llama3"]
        if v not in valid_models:
            raise ValueError(f"Le modèle doit être l'un des suivants: {', '.join(valid_models)}")
        return v

class SynthesizeRequest(BaseModel):
    """Modèle de données pour la requête de synthèse."""
    perspective_a: str = Field(..., description="Première perspective à synthétiser")
    perspective_b: str = Field(..., description="Deuxième perspective à synthétiser")
    history: Optional[List[str]] = Field([], description="Historique de la conversation")
    model: Optional[str] = Field("teacher", description="Modèle à utiliser pour la synthèse")

    @validator('model')
    def validate_model(cls, v):
        valid_models = ["teacher", "qwen25", "mixtral8x7b", "llama3"]
        if v not in valid_models:
            raise ValueError(f"Le modèle doit être l'un des suivants: {', '.join(valid_models)}")
        return v

class GenerateImageRequest(BaseModel):
    """Modèle de données pour la requête de génération d'image."""
    prompt: str = Field(..., min_length=3, max_length=1000, description="Prompt pour la génération d'image")
    negative_prompt: Optional[str] = Field(None, description="Prompt négatif pour la génération d'image")
    config: Optional[Dict[str, Any]] = Field(None, description="Configuration pour la génération d'image")
    model: Optional[str] = Field("sdxl", description="Modèle à utiliser pour la génération d'image")

    @validator('model')
    def validate_model(cls, v):
        valid_models = ["sdxl", "dalle3", "midjourney"]
        if v not in valid_models:
            raise ValueError(f"Le modèle doit être l'un des suivants: {', '.join(valid_models)}")
        return v

class ModelInfo(BaseModel):
    """Modèle de données pour les informations sur un modèle."""
    name: str
    type: str
    version: str
    description: str
    status: str
    capabilities: List[str]

# Endpoints API
@router.get("/list")
async def list_models(model_manager: ModelManager = Depends(get_model_manager)) -> Dict[str, List[ModelInfo]]:
    """
    Liste tous les modèles disponibles.
    
    Returns:
        Dict[str, List[ModelInfo]]: Liste des modèles disponibles par catégorie
    """
    available_models = model_manager.list_available_models()
    return {
        "models": available_models
    }

@router.post("/teacher/evaluate")
async def evaluate_response(
    request: EvaluateRequest,
    model_manager: ModelManager = Depends(get_model_manager)
) -> Dict[str, Any]:
    """
    Évalue une réponse avec le TeacherModel.
    
    Args:
        request (EvaluateRequest): Requête contenant la réponse à évaluer
        
    Returns:
        Dict[str, Any]: Résultats de l'évaluation
    """
    start_time = time.time()
    try:
        evaluation = model_manager.evaluate_response(
            response=request.response,
            context=request.context,
            model=request.model
        )
        
        processing_time = time.time() - start_time
        return {
            "analysis": evaluation,
            "metadata": {
                "model_used": request.model,
                "processing_time": processing_time,
                "version": "1.0.0"
            }
        }
    except Exception as e:
        logger.error(f"Erreur lors de l'évaluation: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Erreur lors de l'évaluation: {str(e)}")

@router.post("/teacher/synthesize")
async def synthesize_debate(
    request: SynthesizeRequest,
    model_manager: ModelManager = Depends(get_model_manager)
) -> Dict[str, Any]:
    """
    Synthétise un débat avec le TeacherModel.
    
    Args:
        request (SynthesizeRequest): Requête contenant les perspectives à synthétiser
        
    Returns:
        Dict[str, Any]: Résultats de la synthèse
    """
    start_time = time.time()
    try:
        synthesis = model_manager.synthesize_debate(
            perspective_a=request.perspective_a,
            perspective_b=request.perspective_b,
            history=request.history,
            model=request.model
        )
        
        processing_time = time.time() - start_time
        return {
            "synthesis": synthesis,
            "metadata": {
                "model_used": request.model,
                "processing_time": processing_time,
                "version": "1.0.0"
            }
        }
    except Exception as e:
        logger.error(f"Erreur lors de la synthèse: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Erreur lors de la synthèse: {str(e)}")

@router.post("/image/generate")
async def generate_image(
    request: GenerateImageRequest,
    model_manager: ModelManager = Depends(get_model_manager)
) -> Dict[str, Any]:
    """
    Génère une image avec l'ImageTeacherModel.
    
    Args:
        request (GenerateImageRequest): Requête contenant le prompt pour la génération d'image
        
    Returns:
        Dict[str, Any]: URL de l'image générée et métadonnées
    """
    start_time = time.time()
    try:
        result = model_manager.generate_image(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            config=request.config,
            model=request.model
        )
        
        processing_time = time.time() - start_time
        return {
            "image_url": result["image_url"],
            "quality_score": result.get("quality_score", 0.0),
            "metadata": {
                "model_used": request.model,
                "processing_time": processing_time,
                "version": "1.0.0"
            }
        }
    except Exception as e:
        logger.error(f"Erreur lors de la génération d'image: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Erreur lors de la génération d'image: {str(e)}") 