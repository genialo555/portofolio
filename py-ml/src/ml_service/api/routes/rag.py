from fastapi import APIRouter, HTTPException, Depends, Body
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field, validator
import logging
import time
import json
from pathlib import Path

from ml_service.api.model_manager import ModelManager, get_model_manager
from ml_service.rag.document_store import DocumentStore, get_document_store
from ml_service.rag.retriever import Retriever, get_retriever

# Configuration du logging
logger = logging.getLogger("ml_api.rag")

# Création du router
router = APIRouter()

# Modèles de données pour les requêtes et réponses API
class Document(BaseModel):
    """Modèle de données pour un document."""
    content: str = Field(..., description="Contenu du document")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Métadonnées du document")
    id: Optional[str] = Field(None, description="Identifiant unique du document")

class StoreDocumentsRequest(BaseModel):
    """Modèle de données pour la requête de stockage de documents."""
    documents: List[Document] = Field(..., description="Documents à stocker")
    namespace: Optional[str] = Field("default", description="Espace de noms pour les documents")

class RetrieveRequest(BaseModel):
    """Modèle de données pour la requête de récupération de documents."""
    query: str = Field(..., min_length=3, description="Requête pour la récupération de documents")
    namespace: Optional[str] = Field("default", description="Espace de noms pour les documents")
    top_k: Optional[int] = Field(5, description="Nombre de documents à récupérer")
    filters: Optional[Dict[str, Any]] = Field(None, description="Filtres pour la récupération de documents")

class GenerateRagRequest(BaseModel):
    """Modèle de données pour la requête de génération avec RAG."""
    query: str = Field(..., min_length=3, description="Requête pour la génération")
    namespace: Optional[str] = Field("default", description="Espace de noms pour les documents")
    top_k: Optional[int] = Field(5, description="Nombre de documents à récupérer")
    filters: Optional[Dict[str, Any]] = Field(None, description="Filtres pour la récupération de documents")
    model: Optional[str] = Field("teacher", description="Modèle à utiliser pour la génération")
    max_tokens: Optional[int] = Field(1024, description="Nombre maximum de tokens à générer")
    temperature: Optional[float] = Field(0.7, description="Température pour la génération")
    
    @validator('model')
    def validate_model(cls, v):
        valid_models = ["teacher", "qwen25", "mixtral8x7b", "llama3"]
        if v not in valid_models:
            raise ValueError(f"Le modèle doit être l'un des suivants: {', '.join(valid_models)}")
        return v

# Endpoints API
@router.post("/documents/store")
async def store_documents(
    request: StoreDocumentsRequest,
    document_store: DocumentStore = Depends(get_document_store)
) -> Dict[str, Any]:
    """
    Stocke des documents dans le store documentaire.
    
    Args:
        request (StoreDocumentsRequest): Documents à stocker
        
    Returns:
        Dict[str, Any]: Résultat du stockage
    """
    start_time = time.time()
    try:
        # Conversion des documents Pydantic en dictionnaires
        documents = [doc.dict() for doc in request.documents]
        
        # Stockage des documents
        result = document_store.add_documents(documents, namespace=request.namespace)
        
        processing_time = time.time() - start_time
        return {
            "success": True,
            "document_count": len(documents),
            "namespace": request.namespace,
            "processing_time": processing_time
        }
    except Exception as e:
        logger.error(f"Erreur lors du stockage des documents: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Erreur lors du stockage des documents: {str(e)}")

@router.post("/retrieve")
async def retrieve_documents(
    request: RetrieveRequest,
    retriever: Retriever = Depends(get_retriever)
) -> Dict[str, Any]:
    """
    Récupère les documents pertinents pour une requête.
    
    Args:
        request (RetrieveRequest): Requête pour la récupération de documents
        
    Returns:
        Dict[str, Any]: Documents récupérés
    """
    start_time = time.time()
    try:
        # Récupération des documents
        documents = retriever.retrieve(
            query=request.query,
            namespace=request.namespace,
            top_k=request.top_k,
            filters=request.filters
        )
        
        processing_time = time.time() - start_time
        return {
            "query": request.query,
            "documents": documents,
            "document_count": len(documents),
            "processing_time": processing_time
        }
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des documents: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Erreur lors de la récupération des documents: {str(e)}")

@router.post("/generate")
async def generate_with_rag(
    request: GenerateRagRequest,
    retriever: Retriever = Depends(get_retriever),
    model_manager: ModelManager = Depends(get_model_manager)
) -> Dict[str, Any]:
    """
    Génère une réponse basée sur les documents récupérés.
    
    Args:
        request (GenerateRagRequest): Requête pour la génération avec RAG
        
    Returns:
        Dict[str, Any]: Réponse générée et métadonnées
    """
    start_time = time.time()
    retrieval_time = 0
    generation_time = 0
    
    try:
        # Récupération des documents
        retrieval_start = time.time()
        documents = retriever.retrieve(
            query=request.query,
            namespace=request.namespace,
            top_k=request.top_k,
            filters=request.filters
        )
        retrieval_time = time.time() - retrieval_start
        
        # Préparation du contexte à partir des documents
        context = "\n\n".join([f"Document {i+1}:\n{doc['content']}" for i, doc in enumerate(documents)])
        
        # Génération de la réponse
        generation_start = time.time()
        response = model_manager.rag_generate(
            query=request.query,
            context_documents=documents,
            model=request.model
        )
        generation_time = time.time() - generation_start
        
        processing_time = time.time() - start_time
        return {
            "query": request.query,
            "response": response,
            "documents": documents,
            "document_count": len(documents),
            "metadata": {
                "model_used": request.model,
                "retrieval_time": retrieval_time,
                "generation_time": generation_time,
                "total_processing_time": processing_time,
                "version": "1.0.0"
            }
        }
    except Exception as e:
        logger.error(f"Erreur lors de la génération avec RAG: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Erreur lors de la génération avec RAG: {str(e)}")


@router.get("/namespaces")
async def list_namespaces(
    document_store: DocumentStore = Depends(get_document_store)
) -> Dict[str, Any]:
    """
    Liste tous les espaces de noms disponibles.
    
    Returns:
        Dict[str, Any]: Liste des espaces de noms
    """
    try:
        namespaces = document_store.list_namespaces()
        
        return {
            "namespaces": namespaces,
            "count": len(namespaces)
        }
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des espaces de noms: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Erreur lors de la récupération des espaces de noms: {str(e)}")


@router.delete("/documents/{namespace}")
async def delete_documents(
    namespace: str,
    document_store: DocumentStore = Depends(get_document_store)
) -> Dict[str, Any]:
    """
    Supprime tous les documents d'un espace de noms.
    
    Args:
        namespace (str): Espace de noms à supprimer
        
    Returns:
        Dict[str, Any]: Résultat de la suppression
    """
    try:
        result = document_store.delete_documents(namespace=namespace)
        
        return {
            "success": True,
            "namespace": namespace,
            "deleted_count": result.get("deleted_count", 0)
        }
    except Exception as e:
        logger.error(f"Erreur lors de la suppression des documents: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Erreur lors de la suppression des documents: {str(e)}") 