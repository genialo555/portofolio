"""
Module d'agent coordinateur.

Ce module fournit un agent coordinateur qui joue le rôle de dispatcher
pour router les requêtes vers les agents spécialisés appropriés.
"""

import logging
import time
from typing import Dict, List, Any, Optional, Tuple, Union, Callable

from .agent import Agent, AgentConfig, Message, Role
from .agent_manager import AgentManager, get_agent_manager
from .analyzer_agent import RequestType, RequestAnalyzerAgent
from ..models.model_loader import ModelLoader
from ..config import settings

logger = logging.getLogger("ml_api.agents.coordinator")

class RouteConfig:
    """Configuration pour une route d'acheminement de requête."""
    def __init__(
        self, 
        request_type: str, 
        agent_id: str, 
        priority: int = 1,
        filter_func: Optional[Callable[[Dict[str, Any]], bool]] = None
    ):
        """
        Initialise une configuration de route.
        
        Args:
            request_type: Type de requête (voir RequestType)
            agent_id: ID de l'agent de destination
            priority: Priorité (1 = plus haute, valeurs plus élevées = priorité plus basse)
            filter_func: Fonction de filtrage optionnelle pour des règles personnalisées
        """
        self.request_type = request_type
        self.agent_id = agent_id
        self.priority = priority
        self.filter_func = filter_func
        self.usage_count = 0
        self.last_used = 0.0
    
    def matches(self, analysis: Dict[str, Any]) -> bool:
        """
        Détermine si cette route correspond à l'analyse de la requête.
        
        Args:
            analysis: Analyse de la requête
            
        Returns:
            True si la route correspond, False sinon
        """
        if analysis.get("type") != self.request_type:
            return False
        
        # Si une fonction de filtrage est définie, l'utiliser comme condition supplémentaire
        if self.filter_func is not None:
            return self.filter_func(analysis)
        
        return True
    
    def use(self) -> None:
        """Marque cette route comme utilisée."""
        self.usage_count += 1
        self.last_used = time.time()


class CoordinatorAgent(Agent):
    """
    Agent coordinateur qui route les requêtes vers les agents spécialisés appropriés.
    """
    
    def __init__(self, 
                config: AgentConfig, 
                analyzer_agent_id: str,
                model_loader: Optional[ModelLoader] = None):
        """
        Initialise l'agent coordinateur.
        
        Args:
            config: Configuration de l'agent
            analyzer_agent_id: ID de l'agent analyseur à utiliser
            model_loader: Chargeur de modèle (optionnel)
        """
        super().__init__(config, model_loader)
        self.analyzer_agent_id = analyzer_agent_id
        self.routes: List[RouteConfig] = []
        self.agent_manager = get_agent_manager()
        self.default_agent_id = None
        self.routing_stats = {
            "total_routed": 0,
            "successful_routes": 0,
            "failed_routes": 0,
            "route_usage": {}
        }
    
    def add_route(self, route: RouteConfig) -> None:
        """
        Ajoute une nouvelle route.
        
        Args:
            route: Configuration de la route à ajouter
        """
        self.routes.append(route)
        # Trier les routes par priorité
        self.routes.sort(key=lambda r: r.priority)
        self.routing_stats["route_usage"][route.request_type] = 0
        logger.info(f"Route ajoutée pour {route.request_type} vers l'agent {route.agent_id}")
    
    def set_default_agent(self, agent_id: str) -> None:
        """
        Définit l'agent par défaut à utiliser si aucune route ne correspond.
        
        Args:
            agent_id: ID de l'agent par défaut
        """
        self.default_agent_id = agent_id
        logger.info(f"Agent par défaut défini: {agent_id}")
    
    def process_request(self, request_text: str) -> Tuple[str, Dict[str, Any]]:
        """
        Traite une requête en l'analysant puis en la routant vers l'agent approprié.
        
        Args:
            request_text: Texte de la requête à traiter
            
        Returns:
            Tuple (réponse, informations de traitement)
        """
        # Incrémenter le compteur de routage
        self.routing_stats["total_routed"] += 1
        
        start_time = time.time()
        process_info = {
            "request_text": request_text,
            "processing_time": 0,
            "analyzer_time": 0,
            "routing_time": 0,
            "agent_time": 0,
            "route_info": None,
            "response_agent_id": None
        }
        
        try:
            # 1. Obtenir l'agent d'analyse
            analyzer = self.agent_manager.get_agent(self.analyzer_agent_id)
            if not analyzer or not isinstance(analyzer, RequestAnalyzerAgent):
                raise ValueError(f"Agent d'analyse {self.analyzer_agent_id} non trouvé ou de type incorrect")
            
            # 2. Analyser la requête
            analyzer_start = time.time()
            analysis = analyzer.analyze_request(request_text)
            process_info["analyzer_time"] = time.time() - analyzer_start
            process_info["analysis"] = analysis
            
            # 3. Trouver la route appropriée
            routing_start = time.time()
            target_agent_id, route_info = self._find_route(analysis)
            process_info["routing_time"] = time.time() - routing_start
            process_info["route_info"] = route_info
            process_info["response_agent_id"] = target_agent_id
            
            # 4. Obtenir l'agent cible
            target_agent = self.agent_manager.get_agent(target_agent_id)
            if not target_agent:
                raise ValueError(f"Agent cible {target_agent_id} non trouvé")
            
            # 5. Traiter la requête avec l'agent cible
            agent_start = time.time()
            target_agent.add_user_message(request_text)
            
            # Adapter les instructions système en fonction de l'analyse
            system_instruction = self._build_system_instruction(analysis)
            
            response, response_info = target_agent.generate_response(system_instruction=system_instruction)
            process_info["agent_time"] = time.time() - agent_start
            process_info["agent_response_info"] = response_info
            
            # Marquer comme succès
            self.routing_stats["successful_routes"] += 1
            
            # Mettre à jour les statistiques de la route
            request_type = analysis.get("type", RequestType.UNDEFINED)
            if request_type in self.routing_stats["route_usage"]:
                self.routing_stats["route_usage"][request_type] += 1
            
            return response, process_info
            
        except Exception as e:
            logger.error(f"Erreur lors du traitement de la requête: {str(e)}")
            
            # Marquer comme échec
            self.routing_stats["failed_routes"] += 1
            
            process_info["error"] = str(e)
            return f"Désolé, je n'ai pas pu traiter votre demande: {str(e)}", process_info
        finally:
            process_info["processing_time"] = time.time() - start_time
    
    def _find_route(self, analysis: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """
        Trouve la route appropriée pour l'analyse donnée.
        
        Args:
            analysis: Analyse de la requête
            
        Returns:
            Tuple (ID de l'agent cible, informations sur la route)
        """
        request_type = analysis.get("type", RequestType.UNDEFINED)
        confidence = analysis.get("confidence", 0.0)
        
        route_info = {
            "request_type": request_type,
            "confidence": confidence,
            "matched_route": None,
            "using_default": False
        }
        
        # Parcourir les routes par ordre de priorité
        for route in self.routes:
            if route.matches(analysis):
                route.use()
                route_info["matched_route"] = {
                    "type": route.request_type,
                    "agent_id": route.agent_id,
                    "priority": route.priority,
                    "usage_count": route.usage_count
                }
                return route.agent_id, route_info
        
        # Si aucune route ne correspond, utiliser l'agent par défaut
        route_info["using_default"] = True
        if self.default_agent_id:
            return self.default_agent_id, route_info
        
        # Si pas d'agent par défaut, lever une exception
        raise ValueError(f"Aucune route trouvée pour le type de requête {request_type} et pas d'agent par défaut")
    
    def _build_system_instruction(self, analysis: Dict[str, Any]) -> str:
        """
        Construit une instruction système adaptée à l'analyse de la requête.
        
        Args:
            analysis: Analyse de la requête
            
        Returns:
            Instruction système adaptée
        """
        request_type = analysis.get("type", RequestType.UNDEFINED)
        key_info = analysis.get("key_info", {})
        urgency = analysis.get("urgency", 3)
        
        # Instructions par type de requête
        type_instructions = {
            RequestType.CONTENT_GENERATION: "Créez un contenu de qualité qui répond aux besoins spécifiques.",
            RequestType.SALES_INQUIRY: "Répondez en tant que conseiller commercial en tenant compte des besoins du client.",
            RequestType.TECHNICAL_SUPPORT: "Fournissez une assistance technique précise et étape par étape.",
            RequestType.GENERAL_INQUIRY: "Fournissez des informations claires et directes.",
            RequestType.FEEDBACK: "Remerciez pour le retour et répondez de manière constructive.",
            RequestType.COMPLAINT: "Adressez la plainte avec empathie et proposez des solutions concrètes.",
            RequestType.UNDEFINED: "Répondez de manière générale et demandez des précisions si nécessaire."
        }
        
        # Instruction de base selon le type
        base_instruction = type_instructions.get(request_type, type_instructions[RequestType.UNDEFINED])
        
        # Ajouter des modifications selon l'urgence
        urgency_addition = ""
        if urgency >= 4:
            urgency_addition = " Cette demande est prioritaire, répondez avec précision et rapidité."
        elif urgency <= 2:
            urgency_addition = " Prenez le temps d'expliquer en détail."
        
        # Ajouter des éléments basés sur les informations clés
        key_info_str = ""
        if key_info:
            key_info_str = " Tenez compte des éléments suivants: "
            for key, value in key_info.items():
                if isinstance(value, str):
                    key_info_str += f"{key}: {value}; "
        
        # Combiner tous les éléments
        return f"{base_instruction}{urgency_addition}{key_info_str}"
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Récupère les métriques d'utilisation de l'agent coordinateur.
        
        Returns:
            Métriques d'utilisation
        """
        metrics = super().get_metrics()
        
        # Ajouter les statistiques de routage
        metrics["routing_stats"] = self.routing_stats
        
        # Ajouter les informations sur les routes
        metrics["routes"] = []
        for route in self.routes:
            metrics["routes"].append({
                "request_type": route.request_type,
                "agent_id": route.agent_id,
                "priority": route.priority,
                "usage_count": route.usage_count,
                "last_used": route.last_used
            })
        
        return metrics


def create_coordinator_agent(
    name: str, 
    analyzer_agent_id: str,
    instruction: Optional[str] = None,
    model_id: Optional[str] = None,
    default_agent_id: Optional[str] = None,
    routes: Optional[List[Dict[str, Any]]] = None
) -> CoordinatorAgent:
    """
    Crée un agent coordinateur avec sa configuration complète.
    
    Args:
        name: Nom de l'agent
        analyzer_agent_id: ID de l'agent analyseur
        instruction: Instruction système (optionnelle)
        model_id: Identifiant du modèle (optionnel)
        default_agent_id: ID de l'agent par défaut (optionnel)
        routes: Liste de configurations de routes (optionnelle)
    
    Returns:
        Instance de l'agent coordinateur
    """
    # Configurer l'agent
    config = AgentConfig(
        name=name,
        description="Agent coordinateur pour le routage des requêtes",
        instruction=instruction or "Vous êtes un coordinateur qui route les requêtes vers les agents spécialisés.",
        model_id=model_id or "phi-4-distilled",  # Modèle léger par défaut
        temperature=0.3,
        max_tokens=512
    )
    
    # Créer l'agent coordinateur
    agent = CoordinatorAgent(config, analyzer_agent_id)
    
    # Définir l'agent par défaut si fourni
    if default_agent_id:
        agent.set_default_agent(default_agent_id)
    
    # Ajouter les routes si fournies
    if routes:
        for route_config in routes:
            route = RouteConfig(
                request_type=route_config["request_type"],
                agent_id=route_config["agent_id"],
                priority=route_config.get("priority", 1),
                filter_func=route_config.get("filter_func")
            )
            agent.add_route(route)
    
    logger.info(f"Agent coordinateur '{name}' créé avec {len(routes or [])} routes")
    
    return agent 