"""
Module de la classe de base Agent.

Ce module fournit la classe Agent, qui est la base de tous les agents conversationnels,
avec les structures de données et méthodes nécessaires pour générer des réponses.
"""

import time
import logging
import uuid
import json
from enum import Enum
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
from dataclasses import dataclass, field, asdict
from abc import ABC, abstractmethod

from ..models.model_loader import ModelLoader, get_model_loader
from ..config import settings

logger = logging.getLogger("ml_api.agents.agent")

class Role(str, Enum):
    """Rôles possibles dans une conversation."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    FUNCTION = "function"


@dataclass
class Message:
    """
    Représentation d'un message dans une conversation.
    
    Attributes:
        role: Rôle de l'émetteur du message
        content: Contenu du message
        id: Identifiant unique du message
        timestamp: Horodatage du message
        metadata: Métadonnées supplémentaires
    """
    role: Role
    content: str
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convertit le message en dictionnaire."""
        return {
            "role": self.role,
            "content": self.content,
            "id": self.id,
            "timestamp": self.timestamp,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        """Crée un message à partir d'un dictionnaire."""
        return cls(
            role=data["role"],
            content=data["content"],
            id=data.get("id", str(uuid.uuid4())),
            timestamp=data.get("timestamp", time.time()),
            metadata=data.get("metadata", {})
        )
    
    @classmethod
    def system(cls, content: str) -> 'Message':
        """Crée un message système."""
        return cls(role=Role.SYSTEM, content=content)
    
    @classmethod
    def user(cls, content: str) -> 'Message':
        """Crée un message utilisateur."""
        return cls(role=Role.USER, content=content)
    
    @classmethod
    def assistant(cls, content: str) -> 'Message':
        """Crée un message assistant."""
        return cls(role=Role.ASSISTANT, content=content)
    
    @classmethod
    def function(cls, content: str) -> 'Message':
        """Crée un message fonction."""
        return cls(role=Role.FUNCTION, content=content)


@dataclass
class AgentConfig:
    """
    Configuration d'un agent.
    
    Attributes:
        name: Nom de l'agent
        description: Description de l'agent
        instruction: Instruction système pour l'agent
        model_id: Identifiant du modèle à utiliser
        temperature: Température pour la génération
        max_tokens: Nombre maximum de tokens à générer
        top_p: Valeur de top-p pour la génération
        stop_sequences: Séquences qui arrêtent la génération
        repetition_penalty: Pénalité pour les répétitions
    """
    name: str
    description: str = "Un agent conversationnel basé sur des LLMs"
    instruction: str = "Vous êtes un assistant IA utile et concis. Répondez en français."
    model_id: str = "default"
    temperature: float = 0.7
    max_tokens: int = 1024
    top_p: float = 0.9
    stop_sequences: List[str] = field(default_factory=list)
    repetition_penalty: float = 1.1

    def to_dict(self) -> Dict[str, Any]:
        """Convertit la configuration en dictionnaire."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentConfig':
        """Crée une configuration à partir d'un dictionnaire."""
        return cls(**data)


class Agent:
    """
    Classe de base pour tous les agents.
    
    Cette classe fournit les fonctionnalités de base pour les agents,
    comme la gestion des messages et la génération de réponses.
    """
    
    def __init__(self, 
                config: AgentConfig,
                model_loader: Optional[ModelLoader] = None):
        """
        Initialise un agent.
        
        Args:
            config: Configuration de l'agent
            model_loader: Chargeur de modèles à utiliser
        """
        self.config = config
        self.model_loader = model_loader or get_model_loader()
        self.history: List[Message] = []
        
        # Métriques
        self.metrics = {
            "total_generation_time": 0,
            "generations_count": 0,
            "tokens_generated": 0,
            "successful_generations": 0,
            "failed_generations": 0,
        }
        
        # Ajouter le message système initial si une instruction est définie
        if config.instruction:
            self.add_message(Message.system(config.instruction))
        
        logger.info(f"Agent '{config.name}' initialisé avec le modèle {config.model_id}")
    
    def add_message(self, message: Message) -> None:
        """
        Ajoute un message à l'historique.
        
        Args:
            message: Message à ajouter
        """
        self.history.append(message)
    
    def add_user_message(self, content: str) -> Message:
        """
        Ajoute un message utilisateur à l'historique.
        
        Args:
            content: Contenu du message
        
        Returns:
            Message créé
        """
        message = Message.user(content)
        self.add_message(message)
        return message
    
    def add_system_message(self, content: str) -> Message:
        """
        Ajoute un message système à l'historique.
        
        Args:
            content: Contenu du message
        
        Returns:
            Message créé
        """
        message = Message.system(content)
        self.add_message(message)
        return message
    
    def add_assistant_message(self, content: str) -> Message:
        """
        Ajoute un message assistant à l'historique.
        
        Args:
            content: Contenu du message
        
        Returns:
            Message créé
        """
        message = Message.assistant(content)
        self.add_message(message)
        return message
    
    def clear_history(self) -> None:
        """Efface l'historique des messages."""
        # Conserver uniquement les messages système
        self.history = [msg for msg in self.history if msg.role == Role.SYSTEM]
    
    def get_history(self) -> List[Message]:
        """
        Récupère l'historique des messages.
        
        Returns:
            Liste des messages
        """
        return self.history
    
    def load_history(self, messages: List[Union[Message, Dict[str, Any]]]) -> None:
        """
        Charge un historique de messages.
        
        Args:
            messages: Liste de messages ou de dictionnaires représentant des messages
        """
        self.history = []
        for msg in messages:
            if isinstance(msg, dict):
                msg = Message.from_dict(msg)
            self.add_message(msg)
    
    def generate_response(self, 
                          user_input: Optional[str] = None,
                          system_instruction: Optional[str] = None) -> Tuple[str, Dict[str, Any]]:
        """
        Génère une réponse de l'agent.
        
        Args:
            user_input: Entrée utilisateur (optionnelle)
            system_instruction: Instruction système à appliquer pour cette génération
        
        Returns:
            Tuple (réponse générée, métadonnées)
        """
        start_time = time.time()
        self.metrics["generations_count"] += 1
        
        # Ajouter l'entrée utilisateur si fournie
        if user_input:
            self.add_user_message(user_input)
        
        # Ajouter une instruction système temporaire si fournie
        temp_system_msg = None
        if system_instruction:
            temp_system_msg = Message.system(system_instruction)
            self.history.insert(0, temp_system_msg)
        
        try:
            # Formater l'historique pour le modèle
            formatted_history = self._format_history_for_model()
            
            # Générer la réponse en utilisant le modèle
            response_text, generation_info = self._generate(formatted_history)
            
            # Ajouter la réponse à l'historique
            response_message = self.add_assistant_message(response_text)
            
            # Mettre à jour les métriques
            generation_time = time.time() - start_time
            self.metrics["total_generation_time"] += generation_time
            self.metrics["successful_generations"] += 1
            
            if "tokens" in generation_info:
                self.metrics["tokens_generated"] += generation_info["tokens"]
            
            # Créer les métadonnées de génération
            metadata = {
                "generation_time": generation_time,
                "model_id": self.config.model_id,
                "temperature": self.config.temperature,
                **generation_info
            }
            
            logger.info(f"Agent '{self.config.name}' a généré une réponse en {generation_time:.2f}s")
            
            return response_text, metadata
        
        except Exception as e:
            logger.error(f"Erreur lors de la génération de réponse: {str(e)}")
            self.metrics["failed_generations"] += 1
            
            # Retourner un message d'erreur
            error_message = f"Désolé, je n'ai pas pu générer de réponse. Erreur: {str(e)}"
            return error_message, {"error": str(e), "generation_time": time.time() - start_time}
        
        finally:
            # Supprimer l'instruction système temporaire si elle a été ajoutée
            if temp_system_msg and temp_system_msg in self.history:
                self.history.remove(temp_system_msg)
    
    def _format_history_for_model(self) -> List[Dict[str, str]]:
        """
        Formate l'historique des messages pour le modèle.
        
        Returns:
            Historique formaté pour le modèle
        """
        # Format par défaut (peut être surchargé par les sous-classes)
        formatted_messages = []
        for msg in self.history:
            formatted_messages.append({
                "role": msg.role.value,
                "content": msg.content
            })
        return formatted_messages
    
    def _generate(self, formatted_history: List[Dict[str, str]]) -> Tuple[str, Dict[str, Any]]:
        """
        Génère une réponse en utilisant le modèle.
        
        Args:
            formatted_history: Historique formaté pour le modèle
        
        Returns:
            Tuple (réponse générée, informations de génération)
        """
        # Méthode à implémenter par les sous-classes
        return "Fonctionnalité non implémentée.", {"method": "base"}
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Récupère les métriques de l'agent.
        
        Returns:
            Métriques de l'agent
        """
        # Calculer des métriques dérivées
        metrics = dict(self.metrics)
        metrics["avg_generation_time"] = self.metrics["total_generation_time"] / max(1, self.metrics["generations_count"])
        metrics["avg_tokens_per_generation"] = self.metrics["tokens_generated"] / max(1, self.metrics["generations_count"])
        metrics["success_rate"] = self.metrics["successful_generations"] / max(1, self.metrics["generations_count"])
        
        return metrics 