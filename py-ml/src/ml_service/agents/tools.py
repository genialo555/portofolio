"""
Module d'outils pour les agents.

Ce module fournit un système d'outils (tools) qui peuvent être utilisés par les agents
pour effectuer des actions spécifiques, comme la recherche d'informations,
le calcul, l'accès à des API externes, etc.
"""

import time
import logging
import json
import inspect
from typing import Dict, List, Any, Optional, Callable, Union, Tuple, TypeVar, Generic
from dataclasses import dataclass, field
from functools import lru_cache

from ..rag.retriever import Retriever, get_retriever
from ..hybrid.hybrid_generator import HybridGenerator, get_hybrid_generator
from ..kag.knowledge_base import KnowledgeBase, get_knowledge_base
from ..config import settings

logger = logging.getLogger("ml_api.agents.tools")

T = TypeVar('T')

@dataclass
class Tool:
    """
    Représentation d'un outil (tool) qui peut être utilisé par un agent.
    
    Attributes:
        name: Nom de l'outil
        description: Description de l'outil
        function: Fonction à exécuter
        args_schema: Schéma des arguments (JSON Schema)
        required_args: Liste des arguments requis
    """
    name: str
    description: str
    function: Callable
    args_schema: Dict[str, Any] = field(default_factory=dict)
    required_args: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Initialisation après la création de l'instance."""
        # Si le schéma d'arguments n'est pas défini, le générer à partir de la fonction
        if not self.args_schema:
            self.args_schema = self._generate_args_schema()
        
        # Si la liste des arguments requis n'est pas définie, l'extraire du schéma
        if not self.required_args:
            self.required_args = self._extract_required_args()
    
    def _generate_args_schema(self) -> Dict[str, Any]:
        """
        Génère un schéma d'arguments à partir de la fonction.
        
        Returns:
            Schéma d'arguments au format JSON Schema
        """
        schema = {
            "type": "object",
            "properties": {},
            "required": []
        }
        
        # Extraire les informations sur les paramètres de la fonction
        signature = inspect.signature(self.function)
        
        for param_name, param in signature.parameters.items():
            # Ignorer les paramètres spéciaux comme *args et **kwargs
            if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                continue
            
            # Déterminer le type du paramètre
            param_type = param.annotation
            if param_type is inspect.Parameter.empty:
                param_type = str
            
            # Déterminer si le paramètre est requis
            is_required = param.default is inspect.Parameter.empty
            
            # Ajouter le paramètre au schéma
            if param_type is str:
                schema["properties"][param_name] = {"type": "string"}
            elif param_type is int:
                schema["properties"][param_name] = {"type": "integer"}
            elif param_type is float:
                schema["properties"][param_name] = {"type": "number"}
            elif param_type is bool:
                schema["properties"][param_name] = {"type": "boolean"}
            elif param_type is list or param_type is List:
                schema["properties"][param_name] = {"type": "array", "items": {"type": "string"}}
            elif param_type is dict or param_type is Dict:
                schema["properties"][param_name] = {"type": "object"}
            else:
                schema["properties"][param_name] = {"type": "string"}
            
            # Ajouter à la liste des paramètres requis si nécessaire
            if is_required:
                schema["required"].append(param_name)
        
        return schema
    
    def _extract_required_args(self) -> List[str]:
        """
        Extrait la liste des arguments requis du schéma.
        
        Returns:
            Liste des arguments requis
        """
        return self.args_schema.get("required", [])
    
    def validate_args(self, args: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Valide les arguments fournis par rapport au schéma.
        
        Args:
            args: Arguments à valider
        
        Returns:
            Tuple (validité, message d'erreur)
        """
        # Vérifier que tous les arguments requis sont présents
        for arg_name in self.required_args:
            if arg_name not in args:
                return False, f"Argument requis manquant: {arg_name}"
        
        # Vérifier le type des arguments (validation simplifiée)
        for arg_name, arg_value in args.items():
            if arg_name in self.args_schema["properties"]:
                expected_type = self.args_schema["properties"][arg_name]["type"]
                
                if expected_type == "string" and not isinstance(arg_value, str):
                    return False, f"L'argument {arg_name} devrait être une chaîne de caractères"
                
                elif expected_type == "integer" and not isinstance(arg_value, int):
                    return False, f"L'argument {arg_name} devrait être un entier"
                
                elif expected_type == "number" and not isinstance(arg_value, (int, float)):
                    return False, f"L'argument {arg_name} devrait être un nombre"
                
                elif expected_type == "boolean" and not isinstance(arg_value, bool):
                    return False, f"L'argument {arg_name} devrait être un booléen"
                
                elif expected_type == "array" and not isinstance(arg_value, list):
                    return False, f"L'argument {arg_name} devrait être une liste"
                
                elif expected_type == "object" and not isinstance(arg_value, dict):
                    return False, f"L'argument {arg_name} devrait être un objet"
        
        return True, ""
    
    def execute(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """
        Exécute l'outil avec les arguments fournis.
        
        Args:
            args: Arguments pour l'exécution
        
        Returns:
            Résultat de l'exécution
        """
        start_time = time.time()
        
        try:
            # Valider les arguments
            is_valid, error_message = self.validate_args(args)
            if not is_valid:
                return {
                    "status": "error",
                    "error": error_message,
                    "execution_time": time.time() - start_time
                }
            
            # Exécuter la fonction
            result = self.function(**args)
            
            # Structurer le résultat
            execution_time = time.time() - start_time
            
            return {
                "status": "success",
                "result": result,
                "execution_time": execution_time
            }
            
        except Exception as e:
            logger.error(f"Erreur lors de l'exécution de l'outil {self.name}: {str(e)}")
            
            return {
                "status": "error",
                "error": str(e),
                "execution_time": time.time() - start_time
            }
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convertit l'outil en dictionnaire.
        
        Returns:
            Dictionnaire représentant l'outil
        """
        return {
            "name": self.name,
            "description": self.description,
            "args_schema": self.args_schema,
            "required_args": self.required_args
        }


class ToolManager:
    """
    Gestionnaire d'outils.
    
    Cette classe permet de gérer un ensemble d'outils qui peuvent être utilisés
    par les agents pour effectuer des actions spécifiques.
    """
    
    def __init__(self):
        """Initialise le gestionnaire d'outils."""
        self.tools: Dict[str, Tool] = {}
        
        # Métriques
        self.metrics = {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "total_execution_time": 0,
        }
        
        # Charger les outils par défaut
        self._register_default_tools()
        
        logger.info(f"ToolManager initialisé avec {len(self.tools)} outils")
    
    def _register_default_tools(self) -> None:
        """Enregistre les outils par défaut."""
        # Outils RAG
        self.register_tool(
            Tool(
                name="search_documents",
                description="Recherche des documents pertinents par rapport à une requête",
                function=self._search_documents,
                args_schema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "namespace": {"type": "string"},
                        "top_k": {"type": "integer", "minimum": 1, "maximum": 20}
                    },
                    "required": ["query"]
                }
            )
        )
        
        # Outils KAG
        self.register_tool(
            Tool(
                name="query_knowledge",
                description="Interroge la base de connaissances avec une requête",
                function=self._query_knowledge,
                args_schema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "limit": {"type": "integer", "minimum": 1, "maximum": 20}
                    },
                    "required": ["query"]
                }
            )
        )
        
        # Outils Hybrides
        self.register_tool(
            Tool(
                name="generate_hybrid_response",
                description="Génère une réponse en utilisant l'approche hybride RAG-KAG",
                function=self._generate_hybrid_response,
                args_schema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "rag_weight": {"type": "number", "minimum": 0, "maximum": 1},
                        "model": {"type": "string"}
                    },
                    "required": ["query"]
                }
            )
        )
        
        # Outils d'extraction d'informations
        self.register_tool(
            Tool(
                name="extract_information",
                description="Extrait des informations structurées d'un texte",
                function=self._extract_information,
                args_schema={
                    "type": "object",
                    "properties": {
                        "text": {"type": "string"},
                        "extraction_type": {"type": "string", "enum": ["entities", "summary", "keywords"]}
                    },
                    "required": ["text", "extraction_type"]
                }
            )
        )
    
    def register_tool(self, tool: Tool) -> None:
        """
        Enregistre un outil dans le gestionnaire.
        
        Args:
            tool: Outil à enregistrer
        """
        self.tools[tool.name] = tool
        logger.info(f"Outil '{tool.name}' enregistré")
    
    def get_tool(self, tool_name: str) -> Optional[Tool]:
        """
        Récupère un outil par son nom.
        
        Args:
            tool_name: Nom de l'outil
        
        Returns:
            Outil ou None si non trouvé
        """
        return self.tools.get(tool_name)
    
    def list_tools(self) -> List[Dict[str, Any]]:
        """
        Liste tous les outils disponibles.
        
        Returns:
            Liste des outils avec leurs informations
        """
        return [tool.to_dict() for tool in self.tools.values()]
    
    def execute_tool(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """
        Exécute un outil avec les arguments fournis.
        
        Args:
            tool_name: Nom de l'outil
            args: Arguments pour l'exécution
        
        Returns:
            Résultat de l'exécution
        """
        start_time = time.time()
        self.metrics["total_executions"] += 1
        
        # Vérifier si l'outil existe
        tool = self.get_tool(tool_name)
        if not tool:
            self.metrics["failed_executions"] += 1
            return {
                "status": "error",
                "error": f"Outil non trouvé: {tool_name}",
                "execution_time": time.time() - start_time
            }
        
        # Exécuter l'outil
        result = tool.execute(args)
        
        # Mettre à jour les métriques
        execution_time = time.time() - start_time
        self.metrics["total_execution_time"] += execution_time
        
        if result.get("status") == "success":
            self.metrics["successful_executions"] += 1
        else:
            self.metrics["failed_executions"] += 1
        
        return result
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Récupère les métriques du gestionnaire d'outils.
        
        Returns:
            Métriques du gestionnaire
        """
        metrics = dict(self.metrics)
        
        # Ajouter des métriques dérivées
        if self.metrics["total_executions"] > 0:
            metrics["success_rate"] = self.metrics["successful_executions"] / self.metrics["total_executions"]
            metrics["avg_execution_time"] = self.metrics["total_execution_time"] / self.metrics["total_executions"]
        
        metrics["available_tools"] = len(self.tools)
        
        return metrics
    
    # Implémentations des fonctions d'outils
    
    def _search_documents(self, query: str, namespace: str = "default", top_k: int = 5) -> Dict[str, Any]:
        """
        Recherche des documents pertinents.
        
        Args:
            query: Requête de recherche
            namespace: Espace de noms pour la recherche
            top_k: Nombre de documents à récupérer
        
        Returns:
            Résultats de la recherche
        """
        retriever = get_retriever()
        
        try:
            results = retriever.retrieve(
                query=query,
                namespace=namespace,
                top_k=top_k
            )
            
            return {
                "query": query,
                "namespace": namespace,
                "results": results,
                "result_count": len(results)
            }
            
        except Exception as e:
            logger.error(f"Erreur lors de la recherche de documents: {str(e)}")
            return {
                "error": f"Erreur lors de la recherche: {str(e)}",
                "query": query,
                "namespace": namespace,
                "results": []
            }
    
    def _query_knowledge(self, query: str, limit: int = 5) -> Dict[str, Any]:
        """
        Interroge la base de connaissances.
        
        Args:
            query: Requête pour la base de connaissances
            limit: Nombre maximum de résultats
        
        Returns:
            Résultats de la requête
        """
        knowledge_base = get_knowledge_base()
        
        try:
            knowledge = knowledge_base.get_knowledge_for_query(
                query=query,
                limit=limit
            )
            
            return {
                "query": query,
                "knowledge": knowledge
            }
            
        except Exception as e:
            logger.error(f"Erreur lors de l'interrogation de la base de connaissances: {str(e)}")
            return {
                "error": f"Erreur lors de l'interrogation: {str(e)}",
                "query": query,
                "knowledge": {"entities": [], "facts": []}
            }
    
    def _generate_hybrid_response(self, query: str, rag_weight: float = 0.5, model: str = "teacher") -> Dict[str, Any]:
        """
        Génère une réponse en utilisant l'approche hybride RAG-KAG.
        
        Args:
            query: Requête utilisateur
            rag_weight: Poids relatif de RAG (0-1)
            model: Modèle à utiliser
        
        Returns:
            Réponse générée
        """
        generator = get_hybrid_generator()
        
        try:
            response = generator.generate(
                query=query,
                rag_weight=rag_weight,
                model=model,
                include_context=False
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Erreur lors de la génération hybride: {str(e)}")
            return {
                "error": f"Erreur lors de la génération: {str(e)}",
                "query": query,
                "response": "Je n'ai pas pu générer de réponse en raison d'une erreur."
            }
    
    def _extract_information(self, text: str, extraction_type: str) -> Dict[str, Any]:
        """
        Extrait des informations structurées d'un texte.
        
        Args:
            text: Texte à analyser
            extraction_type: Type d'extraction ('entities', 'summary', 'keywords')
        
        Returns:
            Informations extraites
        """
        # Pour l'instant, une implémentation simplifiée
        result = {
            "text": text[:100] + "..." if len(text) > 100 else text,
            "extraction_type": extraction_type,
        }
        
        if extraction_type == "entities":
            # Simuler l'extraction d'entités
            result["entities"] = [
                {"text": "Entity1", "type": "PERSON", "start": 0, "end": 0},
                {"text": "Entity2", "type": "ORGANIZATION", "start": 0, "end": 0}
            ]
            
        elif extraction_type == "summary":
            # Simuler la génération d'un résumé
            result["summary"] = "Résumé automatique du texte fourni."
            
        elif extraction_type == "keywords":
            # Simuler l'extraction de mots-clés
            result["keywords"] = ["mot-clé1", "mot-clé2", "mot-clé3"]
            
        else:
            result["error"] = f"Type d'extraction non supporté: {extraction_type}"
        
        return result


@lru_cache()
def get_tool_manager() -> ToolManager:
    """
    Fonction pour obtenir une instance singleton du gestionnaire d'outils.
    
    Returns:
        Instance du gestionnaire d'outils
    """
    return ToolManager() 