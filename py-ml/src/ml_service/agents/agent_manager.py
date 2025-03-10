"""
Module du gestionnaire d'agents.

Ce module fournit la classe AgentManager qui permet de créer, gérer
et suivre les différentes instances d'agents.
"""

import time
import logging
import uuid
from typing import Dict, List, Any, Optional, Union, Set, Tuple
from functools import lru_cache

from .agent import Agent, AgentConfig

logger = logging.getLogger("ml_api.agents.manager")

class AgentManager:
    """
    Gestionnaire d'agents.
    
    Cette classe permet de créer, gérer et suivre les différentes instances
    d'agents, en veillant à leur cycle de vie et en évitant les fuites de mémoire.
    """
    
    def __init__(self):
        """Initialise le gestionnaire d'agents."""
        # Dictionnaire des agents actifs
        self.agents: Dict[str, Agent] = {}
        
        # Métriques
        self.metrics = {
            "agents_created": 0,
            "agents_deleted": 0,
            "total_conversations": 0,
            "total_messages": 0,
        }
        
        logger.info("AgentManager initialisé")
    
    def create_agent(self, agent_class: type, config: AgentConfig, agent_id: Optional[str] = None) -> Tuple[str, Agent]:
        """
        Crée une nouvelle instance d'agent.
        
        Args:
            agent_class: Classe de l'agent à créer
            config: Configuration de l'agent
            agent_id: Identifiant de l'agent (généré automatiquement si None)
        
        Returns:
            Tuple (identifiant de l'agent, instance de l'agent)
        """
        # Générer un identifiant si non fourni
        if agent_id is None:
            agent_id = str(uuid.uuid4())
        
        # Créer l'agent
        agent = agent_class(config)
        
        # Ajouter l'agent aux agents actifs
        self.agents[agent_id] = agent
        
        # Mettre à jour les métriques
        self.metrics["agents_created"] += 1
        
        logger.info(f"Agent '{config.name}' créé avec l'ID {agent_id}")
        
        return agent_id, agent
    
    def get_agent(self, agent_id: str) -> Optional[Agent]:
        """
        Récupère un agent par son identifiant.
        
        Args:
            agent_id: Identifiant de l'agent
        
        Returns:
            Agent ou None si non trouvé
        """
        return self.agents.get(agent_id)
    
    def delete_agent(self, agent_id: str) -> bool:
        """
        Supprime un agent.
        
        Args:
            agent_id: Identifiant de l'agent
        
        Returns:
            True si l'agent a été supprimé, False sinon
        """
        if agent_id in self.agents:
            # Récupérer l'agent
            agent = self.agents[agent_id]
            
            # Mettre à jour les métriques
            self.metrics["agents_deleted"] += 1
            self.metrics["total_conversations"] += 1
            self.metrics["total_messages"] += len(agent.get_history())
            
            # Supprimer l'agent
            del self.agents[agent_id]
            
            logger.info(f"Agent {agent_id} supprimé")
            return True
        
        logger.warn(f"Agent {agent_id} non trouvé, impossible de le supprimer")
        return False
    
    def list_agents(self) -> List[Dict[str, Any]]:
        """
        Liste tous les agents actifs.
        
        Returns:
            Liste des agents avec leurs informations
        """
        result = []
        for agent_id, agent in self.agents.items():
            result.append({
                "id": agent_id,
                "name": agent.config.name,
                "model_id": agent.config.model_id,
                "messages_count": len(agent.get_history()),
                "metrics": agent.get_metrics()
            })
        return result
    
    def get_agent_info(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """
        Récupère les informations sur un agent.
        
        Args:
            agent_id: Identifiant de l'agent
        
        Returns:
            Informations sur l'agent ou None si non trouvé
        """
        agent = self.get_agent(agent_id)
        if agent:
            return {
                "id": agent_id,
                "name": agent.config.name,
                "model_id": agent.config.model_id,
                "messages_count": len(agent.get_history()),
                "metrics": agent.get_metrics(),
                "config": agent.config.to_dict()
            }
        return None
    
    def clear_inactive_agents(self, max_inactive_time: float = 3600) -> int:
        """
        Supprime les agents inactifs depuis un certain temps.
        
        Args:
            max_inactive_time: Temps maximum d'inactivité en secondes
        
        Returns:
            Nombre d'agents supprimés
        """
        current_time = time.time()
        inactive_agents = []
        
        # Identifier les agents inactifs
        for agent_id, agent in self.agents.items():
            # Vérifier si l'agent a des messages
            if agent.history:
                # Récupérer le timestamp du dernier message
                last_message_time = max(msg.timestamp for msg in agent.history)
                
                # Vérifier si l'agent est inactif
                if current_time - last_message_time > max_inactive_time:
                    inactive_agents.append(agent_id)
        
        # Supprimer les agents inactifs
        for agent_id in inactive_agents:
            self.delete_agent(agent_id)
        
        if inactive_agents:
            logger.info(f"{len(inactive_agents)} agents inactifs supprimés")
        
        return len(inactive_agents)
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Récupère les métriques du gestionnaire d'agents.
        
        Returns:
            Métriques du gestionnaire
        """
        metrics = dict(self.metrics)
        
        # Ajouter des métriques supplémentaires
        metrics["active_agents"] = len(self.agents)
        
        # Agréger les métriques des agents
        total_generation_time = 0
        total_generations = 0
        total_tokens = 0
        
        for agent in self.agents.values():
            agent_metrics = agent.get_metrics()
            total_generation_time += agent_metrics.get("total_generation_time", 0)
            total_generations += agent_metrics.get("generations_count", 0)
            total_tokens += agent_metrics.get("tokens_generated", 0)
        
        metrics["total_generation_time"] = total_generation_time
        metrics["total_generations"] = total_generations
        metrics["total_tokens"] = total_tokens
        
        if total_generations > 0:
            metrics["avg_generation_time"] = total_generation_time / total_generations
            metrics["avg_tokens_per_generation"] = total_tokens / total_generations
        
        return metrics


@lru_cache()
def get_agent_manager() -> AgentManager:
    """
    Fonction pour obtenir une instance singleton du gestionnaire d'agents.
    
    Returns:
        Instance du gestionnaire d'agents
    """
    return AgentManager() 