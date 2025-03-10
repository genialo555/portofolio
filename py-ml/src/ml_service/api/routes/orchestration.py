"""
API pour le système d'orchestration d'agents.

Ce module fournit des endpoints pour traiter les requêtes entrantes
via le système d'orchestration d'agents (analyse + coordination).
"""

import logging
import time
import uuid
from typing import Dict, List, Optional, Any, Union
from fastapi import APIRouter, Depends, Path, HTTPException, Body, Query
from pydantic import BaseModel, Field

from ...agents.agent_manager import AgentManager, get_agent_manager
from ...agents.analyzer_agent import RequestType, create_request_analyzer_agent, RequestAnalyzerAgent
from ...agents.coordinator_agent import create_coordinator_agent, CoordinatorAgent, RouteConfig
from ...agents.specialized_agents import (
    create_content_generator_agent, 
    create_sales_agent,
    create_technical_support_agent
)
from ...agents.agent_factory import create_assistant_agent

router = APIRouter(tags=["Orchestration"])

logger = logging.getLogger("ml_api.routes.orchestration")

# Modèles Pydantic pour les requêtes/réponses
class OrchestratorConfig(BaseModel):
    """Configuration pour initialiser le système d'orchestration."""
    analyzer_model: str = Field("r1-teacher", description="ID du modèle à utiliser pour l'analyseur")
    coordinator_model: str = Field("phi-4-distilled", description="ID du modèle à utiliser pour le coordinateur")
    content_generator_model: str = Field("r1-teacher", description="ID du modèle pour la génération de contenu")
    sales_model: str = Field("phi-4-distilled", description="ID du modèle pour l'agent commercial")
    tech_support_model: str = Field("phi-4-distilled", description="ID du modèle pour le support technique")
    general_model: str = Field("phi-4-distilled", description="ID du modèle pour l'agent général")
    route_config: List[Dict[str, Any]] = Field([], description="Configuration des routes")


class IncomingRequest(BaseModel):
    """Requête entrante à traiter par le système d'orchestration."""
    text: str = Field(..., description="Texte de la requête")
    metadata: Dict[str, Any] = Field({}, description="Métadonnées associées à la requête")
    source: str = Field("api", description="Source de la requête (api, email, chat, etc.)")


class OrchestrationResponse(BaseModel):
    """Réponse du système d'orchestration."""
    request_id: str = Field(..., description="Identifiant unique de la requête")
    response: str = Field(..., description="Réponse générée")
    source_agent: str = Field(..., description="ID de l'agent ayant généré la réponse")
    request_type: str = Field(..., description="Type de requête détecté")
    processing_time: float = Field(..., description="Temps de traitement total (secondes)")
    confidence: float = Field(..., description="Niveau de confiance dans l'analyse")
    details: Dict[str, Any] = Field({}, description="Détails supplémentaires sur le traitement")


# Variables globales pour stocker les IDs des agents du système d'orchestration
ORCHESTRATOR_IDS = {
    "analyzer": None,
    "coordinator": None,
    "content_generator": None,
    "sales_agent": None,
    "tech_support": None,
    "general_assistant": None,
    "initialized": False
}


@router.post("/initialize", response_model=Dict[str, Any])
async def initialize_orchestration_system(
    config: OrchestratorConfig = Body(...),
    agent_manager: AgentManager = Depends(get_agent_manager)
) -> Dict[str, Any]:
    """
    Initialise le système d'orchestration avec les agents nécessaires.
    
    Args:
        config: Configuration pour le système d'orchestration
        agent_manager: Gestionnaire d'agents
    
    Returns:
        Informations sur le système initialisé
    """
    global ORCHESTRATOR_IDS
    
    start_time = time.time()
    
    try:
        # 1. Créer l'agent d'analyse
        analyzer = create_request_analyzer_agent(
            name="Request Analyzer",
            model_id=config.analyzer_model
        )
        analyzer_id, _ = agent_manager.create_agent(RequestAnalyzerAgent, analyzer.config)
        ORCHESTRATOR_IDS["analyzer"] = analyzer_id
        
        # 2. Créer les agents spécialisés
        content_agent = create_content_generator_agent(
            name="Content Generator",
            model_id=config.content_generator_model
        )
        content_id, _ = agent_manager.create_agent(type(content_agent), content_agent.config)
        ORCHESTRATOR_IDS["content_generator"] = content_id
        
        sales_agent = create_sales_agent(
            name="Sales Assistant",
            model_id=config.sales_model
        )
        sales_id, _ = agent_manager.create_agent(type(sales_agent), sales_agent.config)
        ORCHESTRATOR_IDS["sales_agent"] = sales_id
        
        tech_agent = create_technical_support_agent(
            name="Technical Support",
            model_id=config.tech_support_model
        )
        tech_id, _ = agent_manager.create_agent(type(tech_agent), tech_agent.config)
        ORCHESTRATOR_IDS["tech_support"] = tech_id
        
        general_agent = create_assistant_agent(
            name="General Assistant",
            model_id=config.general_model
        )
        general_id, _ = agent_manager.create_agent(type(general_agent), general_agent.config)
        ORCHESTRATOR_IDS["general_assistant"] = general_id
        
        # 3. Créer l'agent coordinateur avec toutes les routes
        # Routes par défaut
        default_routes = [
            {"request_type": RequestType.CONTENT_GENERATION, "agent_id": content_id, "priority": 1},
            {"request_type": RequestType.SALES_INQUIRY, "agent_id": sales_id, "priority": 1},
            {"request_type": RequestType.TECHNICAL_SUPPORT, "agent_id": tech_id, "priority": 1},
            {"request_type": RequestType.GENERAL_INQUIRY, "agent_id": general_id, "priority": 1},
            {"request_type": RequestType.FEEDBACK, "agent_id": general_id, "priority": 1},
            {"request_type": RequestType.COMPLAINT, "agent_id": sales_id, "priority": 1}
        ]
        
        # Fusionner avec les routes personnalisées
        routes = default_routes
        if config.route_config:
            # Remplacer les routes par défaut si une route personnalisée existe pour le même type
            route_types = {r["request_type"] for r in config.route_config}
            routes = [r for r in default_routes if r["request_type"] not in route_types]
            routes.extend(config.route_config)
        
        coordinator = create_coordinator_agent(
            name="Request Coordinator",
            analyzer_agent_id=analyzer_id,
            model_id=config.coordinator_model,
            default_agent_id=general_id,
            routes=routes
        )
        coordinator_id, _ = agent_manager.create_agent(CoordinatorAgent, coordinator.config)
        ORCHESTRATOR_IDS["coordinator"] = coordinator_id
        
        # Marquer comme initialisé
        ORCHESTRATOR_IDS["initialized"] = True
        
        return {
            "status": "success",
            "message": "Système d'orchestration initialisé avec succès",
            "agent_ids": ORCHESTRATOR_IDS,
            "route_count": len(routes),
            "initialization_time": time.time() - start_time
        }
        
    except Exception as e:
        logger.error(f"Erreur lors de l'initialisation du système d'orchestration: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de l'initialisation du système d'orchestration: {str(e)}"
        )


@router.post("/process", response_model=OrchestrationResponse)
async def process_request(
    request: IncomingRequest,
    agent_manager: AgentManager = Depends(get_agent_manager)
) -> OrchestrationResponse:
    """
    Traite une requête via le système d'orchestration.
    
    Args:
        request: Requête à traiter
        agent_manager: Gestionnaire d'agents
    
    Returns:
        Résultat du traitement
    """
    # Vérifier que le système est initialisé
    if not ORCHESTRATOR_IDS["initialized"]:
        raise HTTPException(
            status_code=400,
            detail="Le système d'orchestration n'a pas été initialisé. Appelez /orchestration/initialize d'abord."
        )
    
    request_id = str(uuid.uuid4())
    start_time = time.time()
    
    try:
        # Récupérer l'agent coordinateur
        coordinator = agent_manager.get_agent(ORCHESTRATOR_IDS["coordinator"])
        if not coordinator or not isinstance(coordinator, CoordinatorAgent):
            raise ValueError("Agent coordinateur non trouvé ou type incorrect")
        
        # Prétraiter la requête avec les métadonnées
        request_text = request.text
        if request.metadata:
            metadata_str = ", ".join([f"{k}: {v}" for k, v in request.metadata.items()])
            request_text = f"{request_text}\n[Métadonnées: {metadata_str}]"
        
        if request.source != "api":
            request_text = f"[Source: {request.source}] {request_text}"
        
        # Traiter la requête avec le coordinateur
        response, process_info = coordinator.process_request(request_text)
        
        # Créer la réponse
        analysis = process_info.get("analysis", {})
        return OrchestrationResponse(
            request_id=request_id,
            response=response,
            source_agent=process_info.get("response_agent_id", "unknown"),
            request_type=analysis.get("type", RequestType.UNDEFINED),
            processing_time=process_info.get("processing_time", time.time() - start_time),
            confidence=analysis.get("confidence", 0.0),
            details={
                "analyzer_time": process_info.get("analyzer_time", 0),
                "routing_time": process_info.get("routing_time", 0),
                "agent_time": process_info.get("agent_time", 0),
                "route_info": process_info.get("route_info", {}),
                "urgency": analysis.get("urgency", 3),
                "key_info": analysis.get("key_info", {})
            }
        )
        
    except Exception as e:
        logger.error(f"Erreur lors du traitement de la requête: {str(e)}")
        # Retourner une réponse d'erreur formatée
        return OrchestrationResponse(
            request_id=request_id,
            response=f"Désolé, une erreur s'est produite lors du traitement de votre demande: {str(e)}",
            source_agent="error",
            request_type=RequestType.UNDEFINED,
            processing_time=time.time() - start_time,
            confidence=0.0,
            details={"error": str(e)}
        )


@router.get("/status", response_model=Dict[str, Any])
async def get_orchestration_status(
    agent_manager: AgentManager = Depends(get_agent_manager)
) -> Dict[str, Any]:
    """
    Récupère le statut actuel du système d'orchestration.
    
    Args:
        agent_manager: Gestionnaire d'agents
    
    Returns:
        Informations sur le statut du système
    """
    if not ORCHESTRATOR_IDS["initialized"]:
        return {
            "status": "not_initialized",
            "message": "Le système d'orchestration n'a pas été initialisé",
            "agent_ids": {}
        }
    
    try:
        # Récupérer les statistiques des agents
        status = {
            "status": "operational",
            "agent_ids": ORCHESTRATOR_IDS,
            "agents": {}
        }
        
        # Récupérer l'état de chaque agent
        for role, agent_id in ORCHESTRATOR_IDS.items():
            if role == "initialized":
                continue
                
            agent = agent_manager.get_agent(agent_id)
            if agent:
                agent_info = agent_manager.get_agent_info(agent_id)
                status["agents"][role] = {
                    "id": agent_id,
                    "type": agent.__class__.__name__,
                    "model": agent.config.model_id,
                    "metrics": agent.get_metrics()
                }
                
                if role == "coordinator" and isinstance(agent, CoordinatorAgent):
                    # Ajouter les statistiques de routage
                    status["routing_stats"] = agent.routing_stats
        
        return status
        
    except Exception as e:
        logger.error(f"Erreur lors de la récupération du statut: {str(e)}")
        return {
            "status": "error",
            "message": f"Erreur lors de la récupération du statut: {str(e)}",
            "agent_ids": ORCHESTRATOR_IDS
        }


@router.post("/reset", response_model=Dict[str, Any])
async def reset_orchestration_system(
    agent_manager: AgentManager = Depends(get_agent_manager)
) -> Dict[str, Any]:
    """
    Réinitialise le système d'orchestration en supprimant tous les agents.
    
    Args:
        agent_manager: Gestionnaire d'agents
    
    Returns:
        Statut de la réinitialisation
    """
    global ORCHESTRATOR_IDS
    
    if not ORCHESTRATOR_IDS["initialized"]:
        return {
            "status": "warning",
            "message": "Le système n'était pas initialisé, aucune action nécessaire"
        }
    
    try:
        # Supprimer tous les agents
        for role, agent_id in ORCHESTRATOR_IDS.items():
            if role != "initialized" and agent_id:
                agent_manager.delete_agent(agent_id)
        
        # Réinitialiser les IDs
        ORCHESTRATOR_IDS = {
            "analyzer": None,
            "coordinator": None,
            "content_generator": None,
            "sales_agent": None,
            "tech_support": None,
            "general_assistant": None,
            "initialized": False
        }
        
        return {
            "status": "success",
            "message": "Système d'orchestration réinitialisé avec succès"
        }
        
    except Exception as e:
        logger.error(f"Erreur lors de la réinitialisation du système: {str(e)}")
        return {
            "status": "error",
            "message": f"Erreur lors de la réinitialisation du système: {str(e)}"
        } 