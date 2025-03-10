"""
Routes pour le Knowledge Augmented Generation (KAG).

Ce module expose les fonctionnalités du système KAG via une API REST.
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import logging
import time

from ...kag.knowledge_base import KnowledgeBase, get_knowledge_base
from ...kag.knowledge_graph import KnowledgeGraph, get_knowledge_graph
from ...kag.extractor import KnowledgeExtractor, get_knowledge_extractor
from ..model_manager import ModelManager, get_model_manager

logger = logging.getLogger("ml_api.routes.kag")

router = APIRouter(tags=["KAG"])

# Modèles de données Pydantic

class Triplet(BaseModel):
    """Modèle de données pour un triplet (sujet, prédicat, objet)."""
    subject: str = Field(..., description="Sujet du triplet")
    predicate: str = Field(..., description="Prédicat du triplet")
    object: str = Field(..., description="Objet du triplet")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Métadonnées du triplet")


class AddKnowledgeRequest(BaseModel):
    """Modèle de données pour une requête d'ajout de connaissance."""
    triplets: List[Triplet] = Field(..., description="Liste des triplets à ajouter", min_items=1)
    namespace: Optional[str] = Field("default", description="Espace de noms pour les triplets")


class ExtractKnowledgeRequest(BaseModel):
    """Modèle de données pour une requête d'extraction de connaissances."""
    text: str = Field(..., description="Texte à analyser", min_length=3)
    source: Optional[str] = Field(None, description="Source du texte")
    confidence_threshold: Optional[float] = Field(0.5, description="Seuil de confiance pour les triplets", ge=0.0, le=1.0)
    store_results: Optional[bool] = Field(True, description="Stocker les résultats dans la base de connaissances")


class QueryKnowledgeRequest(BaseModel):
    """Modèle de données pour une requête de connaissances."""
    query: str = Field(..., description="Requête en langage naturel", min_length=3)
    namespace: Optional[str] = Field("default", description="Espace de noms pour la recherche")
    limit: Optional[int] = Field(10, description="Nombre maximum de résultats", ge=1, le=100)


class VerifyStatementsRequest(BaseModel):
    """Modèle de données pour une requête de vérification d'affirmations."""
    statements: List[Triplet] = Field(..., description="Affirmations à vérifier", min_items=1)
    namespace: Optional[str] = Field("default", description="Espace de noms pour la vérification")


class GenerateKagRequest(BaseModel):
    """Modèle de données pour une requête de génération augmentée par connaissances."""
    query: str = Field(..., description="Requête utilisateur pour la génération", min_length=3)
    namespace: Optional[str] = Field("default", description="Espace de noms pour les connaissances")
    model: Optional[str] = Field("teacher", description="Modèle à utiliser pour la génération")
    max_tokens: Optional[int] = Field(1024, description="Nombre maximum de tokens à générer")
    temperature: Optional[float] = Field(0.7, description="Température pour la génération", ge=0.0, le=1.0)
    include_knowledge_in_response: Optional[bool] = Field(False, description="Inclure les connaissances utilisées dans la réponse")


class CreateEntityRequest(BaseModel):
    """Modèle de données pour une requête de création d'entité."""
    name: str = Field(..., description="Nom de l'entité")
    entity_type: str = Field(..., description="Type de l'entité")
    properties: Optional[Dict[str, Any]] = Field(None, description="Propriétés de l'entité")


# Routes de l'API KAG

@router.post("/knowledge/add")
async def add_knowledge(
    request: AddKnowledgeRequest,
    knowledge_base: KnowledgeBase = Depends(get_knowledge_base)
) -> Dict[str, Any]:
    """
    Ajoute des connaissances à la base sous forme de triplets.
    
    Args:
        request: Requête contenant les triplets à ajouter
        knowledge_base: Base de connaissances
    
    Returns:
        État de l'opération et identifiants des triplets ajoutés
    """
    start_time = time.time()
    
    try:
        added_triplets = []
        
        for triplet in request.triplets:
            metadata = triplet.metadata or {}
            metadata["added_at"] = time.time()
            metadata["source"] = metadata.get("source", "api")
            
            subject_id, object_id = knowledge_base.add_knowledge(
                subject=triplet.subject,
                predicate=triplet.predicate,
                obj=triplet.object,
                metadata=metadata
            )
            
            added_triplets.append({
                "subject_id": subject_id,
                "object_id": object_id,
                "subject": triplet.subject,
                "predicate": triplet.predicate,
                "object": triplet.object
            })
        
        # Sauvegarder la base de connaissances
        knowledge_base.save()
        
        process_time = time.time() - start_time
        
        return {
            "status": "success",
            "message": f"{len(added_triplets)} triplets ajoutés",
            "triplets": added_triplets,
            "processing_time": process_time
        }
    
    except Exception as e:
        logger.error(f"Erreur lors de l'ajout de connaissances: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur lors de l'ajout de connaissances: {str(e)}")


@router.post("/knowledge/extract")
async def extract_knowledge(
    request: ExtractKnowledgeRequest,
    extractor: KnowledgeExtractor = Depends(get_knowledge_extractor)
) -> Dict[str, Any]:
    """
    Extrait des connaissances à partir d'un texte.
    
    Args:
        request: Requête contenant le texte à analyser
        extractor: Extracteur de connaissances
    
    Returns:
        Triplets extraits du texte
    """
    start_time = time.time()
    
    try:
        if request.store_results:
            # Extraire et stocker les triplets
            stored_count = extractor.extract_and_store(
                text=request.text,
                source=request.source,
                confidence_threshold=request.confidence_threshold
            )
            
            process_time = time.time() - start_time
            
            return {
                "status": "success",
                "message": f"{stored_count} triplets extraits et stockés",
                "stored_count": stored_count,
                "processing_time": process_time
            }
        else:
            # Extraire les triplets sans les stocker
            triplets = extractor.extract_from_text(
                text=request.text,
                source=request.source,
                confidence_threshold=request.confidence_threshold
            )
            
            process_time = time.time() - start_time
            
            return {
                "status": "success",
                "message": f"{len(triplets)} triplets extraits",
                "triplets": triplets,
                "processing_time": process_time
            }
    
    except Exception as e:
        logger.error(f"Erreur lors de l'extraction de connaissances: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur lors de l'extraction de connaissances: {str(e)}")


@router.post("/knowledge/query")
async def query_knowledge(
    request: QueryKnowledgeRequest,
    knowledge_base: KnowledgeBase = Depends(get_knowledge_base)
) -> Dict[str, Any]:
    """
    Interroge la base de connaissances avec une requête en langage naturel.
    
    Args:
        request: Requête en langage naturel
        knowledge_base: Base de connaissances
    
    Returns:
        Résultats de la requête
    """
    start_time = time.time()
    
    try:
        # Interroger la base de connaissances
        results = knowledge_base.query_knowledge(
            query=request.query,
            limit=request.limit
        )
        
        process_time = time.time() - start_time
        
        return {
            "status": "success",
            "query": request.query,
            "results": results,
            "result_count": len(results),
            "processing_time": process_time
        }
    
    except Exception as e:
        logger.error(f"Erreur lors de la requête de connaissances: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur lors de la requête de connaissances: {str(e)}")


@router.post("/statements/verify")
async def verify_statements(
    request: VerifyStatementsRequest,
    knowledge_base: KnowledgeBase = Depends(get_knowledge_base)
) -> Dict[str, Any]:
    """
    Vérifie la validité d'un ensemble d'affirmations en les comparant à la base de connaissances.
    
    Args:
        request: Affirmations à vérifier
        knowledge_base: Base de connaissances
    
    Returns:
        Résultats de la vérification
    """
    start_time = time.time()
    
    try:
        # Convertir les triplets en format attendu
        statements = []
        for triplet in request.statements:
            statements.append({
                "subject": triplet.subject,
                "predicate": triplet.predicate,
                "object": triplet.object
            })
        
        # Vérifier les affirmations
        verification_results = knowledge_base.verify_statements(statements)
        
        process_time = time.time() - start_time
        
        return {
            "status": "success",
            "verification_results": verification_results,
            "statement_count": len(request.statements),
            "processing_time": process_time
        }
    
    except Exception as e:
        logger.error(f"Erreur lors de la vérification des affirmations: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur lors de la vérification des affirmations: {str(e)}")


@router.post("/generate")
async def generate_with_kag(
    request: GenerateKagRequest,
    knowledge_base: KnowledgeBase = Depends(get_knowledge_base),
    model_manager: ModelManager = Depends(get_model_manager)
) -> Dict[str, Any]:
    """
    Génère une réponse augmentée par des connaissances.
    
    Args:
        request: Requête utilisateur
        knowledge_base: Base de connaissances
        model_manager: Gestionnaire de modèles
    
    Returns:
        Réponse générée avec les connaissances utilisées
    """
    start_time = time.time()
    
    try:
        # Récupérer les connaissances pertinentes
        knowledge = knowledge_base.get_knowledge_for_query(
            query=request.query,
            limit=5  # Récupérer les 5 entités les plus pertinentes
        )
        
        # Formater les connaissances pour le modèle
        knowledge_text = knowledge_base.format_knowledge_for_llm(knowledge, format_type="text")
        
        # Construire le prompt
        prompt = f"""Utilisez les connaissances suivantes pour répondre à la question :

{knowledge_text}

Question: {request.query}

Réponse:"""
        
        # Générer la réponse avec le modèle
        generation_result = model_manager.evaluate_response(
            response=prompt,
            model=request.model
        )
        
        # Récupérer la réponse générée
        response = generation_result.get("analysis", {}).get("synthesis", "")
        
        process_time = time.time() - start_time
        
        result = {
            "status": "success",
            "query": request.query,
            "response": response,
            "model": request.model,
            "processing_time": process_time
        }
        
        # Inclure les connaissances utilisées si demandé
        if request.include_knowledge_in_response:
            result["knowledge_used"] = knowledge
        
        return result
    
    except Exception as e:
        logger.error(f"Erreur lors de la génération avec KAG: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur lors de la génération avec KAG: {str(e)}")


@router.post("/entities/create")
async def create_entity(
    request: CreateEntityRequest,
    knowledge_base: KnowledgeBase = Depends(get_knowledge_base)
) -> Dict[str, Any]:
    """
    Crée une nouvelle entité dans la base de connaissances.
    
    Args:
        request: Informations sur l'entité à créer
        knowledge_base: Base de connaissances
    
    Returns:
        État de l'opération et identifiant de l'entité créée
    """
    start_time = time.time()
    
    try:
        # Créer l'entité
        entity_id = knowledge_base.add_entity(
            name=request.name,
            entity_type=request.entity_type,
            properties=request.properties
        )
        
        # Sauvegarder la base de connaissances
        knowledge_base.save()
        
        process_time = time.time() - start_time
        
        return {
            "status": "success",
            "message": f"Entité '{request.name}' créée",
            "entity_id": entity_id,
            "entity": {
                "name": request.name,
                "type": request.entity_type,
                "properties": request.properties or {}
            },
            "processing_time": process_time
        }
    
    except Exception as e:
        logger.error(f"Erreur lors de la création de l'entité: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur lors de la création de l'entité: {str(e)}")


@router.get("/metrics")
async def get_kag_metrics(
    knowledge_base: KnowledgeBase = Depends(get_knowledge_base),
    extractor: KnowledgeExtractor = Depends(get_knowledge_extractor)
) -> Dict[str, Any]:
    """
    Récupère les métriques du système KAG.
    
    Args:
        knowledge_base: Base de connaissances
        extractor: Extracteur de connaissances
    
    Returns:
        Métriques du système KAG
    """
    try:
        # Récupérer les métriques de la base de connaissances
        kb_metrics = knowledge_base.get_metrics()
        
        # Récupérer les métriques de l'extracteur
        extractor_metrics = extractor.get_metrics()
        
        return {
            "status": "success",
            "metrics": {
                "knowledge_base": kb_metrics,
                "extractor": extractor_metrics
            }
        }
    
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des métriques: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur lors de la récupération des métriques: {str(e)}")


@router.get("/namespaces")
async def list_namespaces(
    knowledge_graph: KnowledgeGraph = Depends(get_knowledge_graph)
) -> Dict[str, Any]:
    """
    Liste tous les espaces de noms disponibles dans le graphe de connaissances.
    
    Args:
        knowledge_graph: Graphe de connaissances
    
    Returns:
        Liste des espaces de noms
    """
    try:
        return {
            "status": "success",
            "namespaces": ["default"]  # Pour l'instant, un seul espace de noms
        }
    
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des espaces de noms: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur lors de la récupération des espaces de noms: {str(e)}")


@router.delete("/clear")
async def clear_knowledge_graph(
    knowledge_graph: KnowledgeGraph = Depends(get_knowledge_graph)
) -> Dict[str, Any]:
    """
    Efface tout le contenu du graphe de connaissances.
    
    Args:
        knowledge_graph: Graphe de connaissances
    
    Returns:
        État de l'opération
    """
    try:
        # Effacer le graphe
        knowledge_graph.clear()
        
        # Sauvegarder le graphe
        knowledge_graph.save()
        
        return {
            "status": "success",
            "message": "Graphe de connaissances effacé"
        }
    
    except Exception as e:
        logger.error(f"Erreur lors de l'effacement du graphe: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur lors de l'effacement du graphe: {str(e)}") 