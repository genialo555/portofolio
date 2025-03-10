"""
Routes pour les agents.

Ce module expose les fonctionnalités des agents via une API REST.
"""

from fastapi import APIRouter, Depends, HTTPException, Query, Path
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional, Union
import logging
import time
import uuid

from ...agents.agent_manager import AgentManager, get_agent_manager
from ...agents.agent import AgentConfig, Message, Role
from ...agents.agent_factory import create_agent
from ...agents.tools import ToolManager, get_tool_manager

logger = logging.getLogger("ml_api.routes.agents")

router = APIRouter(tags=["Agents"])

# Modèles de données Pydantic

class CreateAgentRequest(BaseModel):
    """Modèle de données pour une requête de création d'agent."""
    agent_type: str = Field(..., description="Type d'agent ('teacher', 'assistant', 'rag', 'hybrid')")
    name: str = Field(..., description="Nom de l'agent")
    instruction: Optional[str] = Field(None, description="Instruction système pour l'agent")
    model_id: Optional[str] = Field(None, description="Identifiant du modèle à utiliser")


class MessageRequest(BaseModel):
    """Modèle de données pour une requête d'envoi de message."""
    content: str = Field(..., description="Contenu du message")
    system_instruction: Optional[str] = Field(None, description="Instruction système pour cette génération")
    options: Optional[Dict[str, Any]] = Field(None, description="Options supplémentaires pour la génération")


class ToolCallRequest(BaseModel):
    """Modèle de données pour une requête d'appel d'outil."""
    tool_name: str = Field(..., description="Nom de l'outil à appeler")
    args: Dict[str, Any] = Field(..., description="Arguments pour l'appel de l'outil")


# Routes de l'API Agents

@router.post("/create")
async def create_new_agent(
    request: CreateAgentRequest,
    agent_manager: AgentManager = Depends(get_agent_manager)
) -> Dict[str, Any]:
    """
    Crée un nouvel agent.
    
    Args:
        request: Requête de création d'agent
        agent_manager: Gestionnaire d'agents
    
    Returns:
        Informations sur l'agent créé
    """
    start_time = time.time()
    
    try:
        # Valider le type d'agent
        if request.agent_type not in ["teacher", "assistant", "rag", "hybrid"]:
            raise HTTPException(
                status_code=400,
                detail=f"Type d'agent non supporté: {request.agent_type}"
            )
        
        # Créer l'agent
        agent = create_agent(
            agent_type=request.agent_type,
            name=request.name,
            instruction=request.instruction,
            model_id=request.model_id
        )
        
        # Enregistrer l'agent dans le gestionnaire
        agent_id = str(uuid.uuid4())
        agent_manager.agents[agent_id] = agent
        
        # Mettre à jour les métriques
        agent_manager.metrics["agents_created"] += 1
        
        process_time = time.time() - start_time
        
        return {
            "status": "success",
            "message": f"Agent '{request.name}' créé avec succès",
            "agent_id": agent_id,
            "agent_type": request.agent_type,
            "name": request.name,
            "model_id": agent.config.model_id,
            "processing_time": process_time
        }
    
    except Exception as e:
        logger.error(f"Erreur lors de la création de l'agent: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la création de l'agent: {str(e)}"
        )


@router.get("/list")
async def list_agents(
    agent_manager: AgentManager = Depends(get_agent_manager)
) -> Dict[str, Any]:
    """
    Liste tous les agents actifs.
    
    Args:
        agent_manager: Gestionnaire d'agents
    
    Returns:
        Liste des agents actifs
    """
    try:
        agents = agent_manager.list_agents()
        
        return {
            "status": "success",
            "agents": agents,
            "count": len(agents)
        }
    
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des agents: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la récupération des agents: {str(e)}"
        )


@router.get("/{agent_id}")
async def get_agent_info(
    agent_id: str = Path(..., description="Identifiant de l'agent"),
    agent_manager: AgentManager = Depends(get_agent_manager)
) -> Dict[str, Any]:
    """
    Récupère les informations sur un agent.
    
    Args:
        agent_id: Identifiant de l'agent
        agent_manager: Gestionnaire d'agents
    
    Returns:
        Informations sur l'agent
    """
    # Vérifier si l'agent existe
    agent = agent_manager.get_agent(agent_id)
    if not agent:
        raise HTTPException(
            status_code=404,
            detail=f"Agent non trouvé: {agent_id}"
        )
    
    try:
        agent_info = agent_manager.get_agent_info(agent_id)
        
        return {
            "status": "success",
            "agent": agent_info
        }
    
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des informations de l'agent: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la récupération des informations de l'agent: {str(e)}"
        )


@router.delete("/{agent_id}")
async def delete_agent(
    agent_id: str = Path(..., description="Identifiant de l'agent"),
    agent_manager: AgentManager = Depends(get_agent_manager)
) -> Dict[str, Any]:
    """
    Supprime un agent.
    
    Args:
        agent_id: Identifiant de l'agent
        agent_manager: Gestionnaire d'agents
    
    Returns:
        État de l'opération
    """
    # Vérifier si l'agent existe
    agent = agent_manager.get_agent(agent_id)
    if not agent:
        raise HTTPException(
            status_code=404,
            detail=f"Agent non trouvé: {agent_id}"
        )
    
    try:
        # Supprimer l'agent
        agent_manager.delete_agent(agent_id)
        
        return {
            "status": "success",
            "message": f"Agent {agent_id} supprimé avec succès"
        }
    
    except Exception as e:
        logger.error(f"Erreur lors de la suppression de l'agent: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la suppression de l'agent: {str(e)}"
        )


@router.post("/{agent_id}/message")
async def send_message(
    request: MessageRequest,
    agent_id: str = Path(..., description="Identifiant de l'agent"),
    agent_manager: AgentManager = Depends(get_agent_manager)
) -> Dict[str, Any]:
    """
    Envoie un message à un agent et récupère sa réponse.
    
    Args:
        request: Requête contenant le message
        agent_id: Identifiant de l'agent
        agent_manager: Gestionnaire d'agents
    
    Returns:
        Réponse de l'agent
    """
    # Vérifier si l'agent existe
    agent = agent_manager.get_agent(agent_id)
    if not agent:
        raise HTTPException(
            status_code=404,
            detail=f"Agent non trouvé: {agent_id}"
        )
    
    start_time = time.time()
    
    try:
        # Extraire les options de génération spécifiques au type d'agent
        options = request.options or {}
        
        # Générer la réponse avec les options appropriées
        if agent.__class__.__name__ == "RAGAgent":
            response, metadata = agent.generate_response(
                user_input=request.content,
                system_instruction=request.system_instruction,
                rag_namespace=options.get("namespace", "default"),
                top_k=options.get("top_k", 5)
            )
        
        elif agent.__class__.__name__ == "HybridAgent":
            response, metadata = agent.generate_response(
                user_input=request.content,
                system_instruction=request.system_instruction,
                rag_weight=options.get("rag_weight", 0.5)
            )
        
        else:
            # Pour les agents de base (Teacher, Assistant)
            response, metadata = agent.generate_response(
                user_input=request.content,
                system_instruction=request.system_instruction
            )
        
        process_time = time.time() - start_time
        
        return {
            "status": "success",
            "message": request.content,
            "response": response,
            "agent_id": agent_id,
            "agent_name": agent.config.name,
            "agent_type": agent.__class__.__name__,
            "metadata": metadata,
            "processing_time": process_time
        }
    
    except Exception as e:
        logger.error(f"Erreur lors de l'envoi du message à l'agent: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de l'envoi du message à l'agent: {str(e)}"
        )


@router.post("/{agent_id}/history/clear")
async def clear_agent_history(
    agent_id: str = Path(..., description="Identifiant de l'agent"),
    agent_manager: AgentManager = Depends(get_agent_manager)
) -> Dict[str, Any]:
    """
    Efface l'historique des messages d'un agent.
    
    Args:
        agent_id: Identifiant de l'agent
        agent_manager: Gestionnaire d'agents
    
    Returns:
        État de l'opération
    """
    # Vérifier si l'agent existe
    agent = agent_manager.get_agent(agent_id)
    if not agent:
        raise HTTPException(
            status_code=404,
            detail=f"Agent non trouvé: {agent_id}"
        )
    
    try:
        # Effacer l'historique
        agent.clear_history()
        
        return {
            "status": "success",
            "message": f"Historique de l'agent {agent_id} effacé avec succès"
        }
    
    except Exception as e:
        logger.error(f"Erreur lors de l'effacement de l'historique de l'agent: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de l'effacement de l'historique de l'agent: {str(e)}"
        )


@router.get("/{agent_id}/history")
async def get_agent_history(
    agent_id: str = Path(..., description="Identifiant de l'agent"),
    agent_manager: AgentManager = Depends(get_agent_manager)
) -> Dict[str, Any]:
    """
    Récupère l'historique des messages d'un agent.
    
    Args:
        agent_id: Identifiant de l'agent
        agent_manager: Gestionnaire d'agents
    
    Returns:
        Historique des messages
    """
    # Vérifier si l'agent existe
    agent = agent_manager.get_agent(agent_id)
    if not agent:
        raise HTTPException(
            status_code=404,
            detail=f"Agent non trouvé: {agent_id}"
        )
    
    try:
        # Récupérer et formater l'historique
        history = agent.get_history()
        formatted_history = [
            {
                "role": msg.role.value,
                "content": msg.content,
                "id": msg.id,
                "timestamp": msg.timestamp,
                "metadata": msg.metadata
            }
            for msg in history
        ]
        
        return {
            "status": "success",
            "agent_id": agent_id,
            "history": formatted_history,
            "messages_count": len(formatted_history)
        }
    
    except Exception as e:
        logger.error(f"Erreur lors de la récupération de l'historique de l'agent: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la récupération de l'historique de l'agent: {str(e)}"
        )


@router.post("/tools/call")
async def call_tool(
    request: ToolCallRequest,
    tool_manager: ToolManager = Depends(get_tool_manager)
) -> Dict[str, Any]:
    """
    Appelle un outil.
    
    Args:
        request: Requête d'appel d'outil
        tool_manager: Gestionnaire d'outils
    
    Returns:
        Résultat de l'appel
    """
    start_time = time.time()
    
    try:
        # Exécuter l'outil
        result = tool_manager.execute_tool(
            tool_name=request.tool_name,
            args=request.args
        )
        
        process_time = time.time() - start_time
        
        return {
            "status": "success",
            "tool_name": request.tool_name,
            "args": request.args,
            "result": result,
            "processing_time": process_time
        }
    
    except Exception as e:
        logger.error(f"Erreur lors de l'appel de l'outil: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de l'appel de l'outil: {str(e)}"
        )


@router.get("/tools/list")
async def list_tools(
    tool_manager: ToolManager = Depends(get_tool_manager)
) -> Dict[str, Any]:
    """
    Liste tous les outils disponibles.
    
    Args:
        tool_manager: Gestionnaire d'outils
    
    Returns:
        Liste des outils disponibles
    """
    try:
        tools = tool_manager.list_tools()
        
        return {
            "status": "success",
            "tools": tools,
            "count": len(tools)
        }
    
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des outils: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la récupération des outils: {str(e)}"
        )


@router.get("/metrics")
async def get_agents_metrics(
    agent_manager: AgentManager = Depends(get_agent_manager),
    tool_manager: ToolManager = Depends(get_tool_manager)
) -> Dict[str, Any]:
    """
    Récupère les métriques des agents et des outils.
    
    Args:
        agent_manager: Gestionnaire d'agents
        tool_manager: Gestionnaire d'outils
    
    Returns:
        Métriques des agents et des outils
    """
    try:
        agent_metrics = agent_manager.get_metrics()
        tool_metrics = tool_manager.get_metrics()
        
        return {
            "status": "success",
            "metrics": {
                "agents": agent_metrics,
                "tools": tool_metrics
            }
        }
    
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des métriques: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la récupération des métriques: {str(e)}"
        ) 