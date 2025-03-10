"""
Routes pour le générateur hybride RAG-KAG.

Ce module expose les fonctionnalités du générateur hybride via une API REST.
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import logging
import time

from ...hybrid.hybrid_generator import HybridGenerator, get_hybrid_generator
from ...hybrid.fusion import KnowledgeFusion, get_knowledge_fusion
from ...rag.retriever import Retriever, get_retriever
from ...kag.knowledge_base import KnowledgeBase, get_knowledge_base
from ..model_manager import ModelManager, get_model_manager

logger = logging.getLogger("ml_api.routes.hybrid")

router = APIRouter(tags=["Hybrid"])

# Modèles de données Pydantic

class GenerateHybridRequest(BaseModel):
    """Modèle de données pour une requête de génération hybride."""
    query: str = Field(..., description="Requête utilisateur pour la génération", min_length=3)
    rag_namespace: Optional[str] = Field("default", description="Espace de noms pour les documents RAG")
    kag_namespace: Optional[str] = Field("default", description="Espace de noms pour les connaissances KAG")
    rag_top_k: Optional[int] = Field(5, description="Nombre de documents à récupérer (RAG)")
    kag_top_k: Optional[int] = Field(5, description="Nombre d'entités à récupérer (KAG)")
    model: Optional[str] = Field("teacher", description="Modèle à utiliser pour la génération")
    max_tokens: Optional[int] = Field(1024, description="Nombre maximum de tokens à générer")
    temperature: Optional[float] = Field(0.7, description="Température pour la génération", ge=0.0, le=1.0)
    rag_weight: Optional[float] = Field(0.5, description="Poids relatif de RAG (0-1)", ge=0.0, le=1.0)
    include_context: Optional[bool] = Field(False, description="Inclure le contexte dans la réponse")
    fusion_strategy: Optional[str] = Field("weighted", description="Stratégie de fusion ('weighted', 'interleave', 'separated')")


# Routes de l'API Hybride

@router.post("/generate")
async def generate_hybrid(
    request: GenerateHybridRequest,
    generator: HybridGenerator = Depends(get_hybrid_generator)
) -> Dict[str, Any]:
    """
    Génère une réponse en utilisant l'approche hybride RAG-KAG.
    
    Args:
        request: Requête utilisateur
        generator: Générateur hybride
    
    Returns:
        Réponse générée avec métadonnées
    """
    start_time = time.time()
    
    try:
        # Générer la réponse hybride
        response = generator.generate(
            query=request.query,
            rag_namespace=request.rag_namespace,
            kag_namespace=request.kag_namespace,
            rag_top_k=request.rag_top_k,
            kag_top_k=request.kag_top_k,
            model=request.model,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            rag_weight=request.rag_weight,
            include_context=request.include_context
        )
        
        process_time = time.time() - start_time
        
        # Ajouter les métadonnées
        result = {
            **response,
            "total_processing_time": process_time
        }
        
        return result
    
    except Exception as e:
        logger.error(f"Erreur lors de la génération hybride: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur lors de la génération hybride: {str(e)}")


@router.get("/metrics")
async def get_hybrid_metrics(
    generator: HybridGenerator = Depends(get_hybrid_generator),
    fusion: KnowledgeFusion = Depends(get_knowledge_fusion)
) -> Dict[str, Any]:
    """
    Récupère les métriques du générateur hybride.
    
    Args:
        generator: Générateur hybride
        fusion: Fusionneur de connaissances
    
    Returns:
        Métriques du générateur hybride
    """
    try:
        # Récupérer les métriques du générateur
        generator_metrics = generator.get_metrics()
        
        # Récupérer les métriques du fusionneur
        fusion_metrics = fusion.get_metrics()
        
        return {
            "status": "success",
            "metrics": {
                "generator": generator_metrics,
                "fusion": fusion_metrics
            }
        }
    
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des métriques: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur lors de la récupération des métriques: {str(e)}")


@router.post("/fuse")
async def fuse_knowledge(
    request: GenerateHybridRequest,
    fusion: KnowledgeFusion = Depends(get_knowledge_fusion),
    retriever: Retriever = Depends(get_retriever),
    knowledge_base: KnowledgeBase = Depends(get_knowledge_base)
) -> Dict[str, Any]:
    """
    Fusionne les connaissances issues de RAG et KAG sans générer de réponse.
    Utile pour le débogage et l'exploration des données.
    
    Args:
        request: Requête utilisateur
        fusion: Fusionneur de connaissances
        retriever: Récupérateur de documents
        knowledge_base: Base de connaissances
    
    Returns:
        Contexte fusionné
    """
    start_time = time.time()
    
    try:
        # Récupérer les documents pertinents (RAG)
        rag_results = retriever.retrieve(
            query=request.query,
            namespace=request.rag_namespace,
            top_k=request.rag_top_k
        )
        
        # Récupérer les connaissances pertinentes (KAG)
        kag_results = knowledge_base.get_knowledge_for_query(
            query=request.query,
            limit=request.kag_top_k
        )
        
        # Fusionner les connaissances
        fused_context = fusion.fuse_knowledge(
            query=request.query,
            rag_results=rag_results,
            kag_results=kag_results,
            rag_weight=request.rag_weight,
            strategy=request.fusion_strategy
        )
        
        process_time = time.time() - start_time
        
        return {
            "status": "success",
            "query": request.query,
            "fusion_strategy": request.fusion_strategy,
            "rag_weight": request.rag_weight,
            "kag_weight": 1.0 - request.rag_weight,
            "fused_context": fused_context,
            "rag_results_count": len(rag_results),
            "kag_facts_count": len(kag_results.get("facts", [])),
            "processing_time": process_time
        }
    
    except Exception as e:
        logger.error(f"Erreur lors de la fusion des connaissances: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur lors de la fusion des connaissances: {str(e)}") 