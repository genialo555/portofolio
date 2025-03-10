"""
Module d'agents spécialisés.

Ce module fournit des agents conversationnels spécialisés pour différentes tâches
spécifiques comme le service commercial, la génération de contenu, etc.
"""

import logging
import time
from typing import Dict, List, Any, Optional, Union, Tuple

from .agent import Agent, AgentConfig, Message, Role
from ..models.model_loader import ModelLoader
from ..config import settings

logger = logging.getLogger("ml_api.agents.specialized")

class ContentGeneratorAgent(Agent):
    """
    Agent spécialisé dans la génération de contenu (articles, descriptions, etc.).
    """
    
    def _generate(self, formatted_history: List[Dict[str, str]]) -> Tuple[str, Dict[str, Any]]:
        """
        Génère une réponse en tant que créateur de contenu.
        
        Args:
            formatted_history: Historique formaté
        
        Returns:
            Tuple (réponse générée, informations de génération)
        """
        prompt = self._build_prompt(formatted_history)
        
        try:
            # Simuler la génération (à remplacer par l'appel réel au modèle)
            response_text = "Voici le contenu généré en tant que spécialiste de la création de contenu."
            
            return response_text, {
                "method": "content_generator",
                "model": self.config.model_id,
                "tokens": len(response_text.split())
            }
            
        except Exception as e:
            logger.error(f"Erreur lors de la génération avec ContentGeneratorAgent: {str(e)}")
            return f"Je n'ai pas pu générer le contenu demandé: {str(e)}", {"error": str(e)}
    
    def _build_prompt(self, formatted_history: List[Dict[str, str]]) -> str:
        """
        Construit le prompt spécialisé pour la génération de contenu.
        
        Args:
            formatted_history: Historique formaté
        
        Returns:
            Prompt pour le modèle
        """
        system_prompt = """Vous êtes un expert en création de contenu. Vous savez rédiger:
- Des articles de blog engageants
- Des descriptions de produits persuasives
- Des posts pour réseaux sociaux viraux
- Des newsletters informatives
- Des scripts vidéo captivants

Votre contenu est toujours adapté à l'audience cible et optimisé pour les moteurs de recherche.
"""
        
        # Remplacer le message système par défaut
        for i, msg in enumerate(formatted_history):
            if msg["role"] == "system":
                formatted_history[i]["content"] = system_prompt
                break
        else:
            # Si aucun message système n'a été trouvé, en ajouter un
            formatted_history.insert(0, {"role": "system", "content": system_prompt})
        
        # Convertir l'historique formaté en prompt
        prompt = ""
        for msg in formatted_history:
            if msg["role"] == "system":
                prompt += f"Instructions: {msg['content']}\n\n"
            elif msg["role"] == "user":
                prompt += f"Demande: {msg['content']}\n\n"
            elif msg["role"] == "assistant":
                prompt += f"Réponse précédente: {msg['content']}\n\n"
        
        prompt += "Nouveau contenu: "
        
        return prompt


class SalesAgent(Agent):
    """
    Agent spécialisé dans l'assistance commerciale et le service client.
    """
    
    def _generate(self, formatted_history: List[Dict[str, str]]) -> Tuple[str, Dict[str, Any]]:
        """
        Génère une réponse en tant qu'assistant commercial.
        
        Args:
            formatted_history: Historique formaté
        
        Returns:
            Tuple (réponse générée, informations de génération)
        """
        prompt = self._build_prompt(formatted_history)
        
        try:
            # Simuler la génération (à remplacer par l'appel réel au modèle)
            response_text = "En tant qu'assistant commercial, je vous propose la solution suivante..."
            
            return response_text, {
                "method": "sales_agent",
                "model": self.config.model_id,
                "tokens": len(response_text.split())
            }
            
        except Exception as e:
            logger.error(f"Erreur lors de la génération avec SalesAgent: {str(e)}")
            return f"Je n'ai pas pu traiter votre demande commerciale: {str(e)}", {"error": str(e)}
    
    def _build_prompt(self, formatted_history: List[Dict[str, str]]) -> str:
        """
        Construit le prompt spécialisé pour l'assistance commerciale.
        
        Args:
            formatted_history: Historique formaté
        
        Returns:
            Prompt pour le modèle
        """
        system_prompt = """Vous êtes un assistant commercial expert. Votre rôle est d'aider:
- Répondre aux questions sur les produits et services
- Résoudre les problèmes clients
- Fournir des informations sur les prix et la disponibilité
- Conseiller sur les meilleures options d'achat
- Gérer les demandes de retours et remboursements

Votre ton est toujours professionnel, empathique et orienté solution.
"""
        
        # Remplacer le message système par défaut
        for i, msg in enumerate(formatted_history):
            if msg["role"] == "system":
                formatted_history[i]["content"] = system_prompt
                break
        else:
            # Si aucun message système n'a été trouvé, en ajouter un
            formatted_history.insert(0, {"role": "system", "content": system_prompt})
        
        # Convertir l'historique formaté en prompt
        prompt = ""
        for msg in formatted_history:
            if msg["role"] == "system":
                prompt += f"Instructions: {msg['content']}\n\n"
            elif msg["role"] == "user":
                prompt += f"Client: {msg['content']}\n\n"
            elif msg["role"] == "assistant":
                prompt += f"Assistant commercial: {msg['content']}\n\n"
        
        prompt += "Réponse: "
        
        return prompt


class TechnicalSupportAgent(Agent):
    """
    Agent spécialisé dans le support technique et la résolution de problèmes.
    """
    
    def _generate(self, formatted_history: List[Dict[str, str]]) -> Tuple[str, Dict[str, Any]]:
        """
        Génère une réponse en tant que support technique.
        
        Args:
            formatted_history: Historique formaté
        
        Returns:
            Tuple (réponse générée, informations de génération)
        """
        prompt = self._build_prompt(formatted_history)
        
        try:
            # Simuler la génération (à remplacer par l'appel réel au modèle)
            response_text = "En tant que technicien, voici comment résoudre votre problème technique..."
            
            return response_text, {
                "method": "technical_support",
                "model": self.config.model_id,
                "tokens": len(response_text.split())
            }
            
        except Exception as e:
            logger.error(f"Erreur lors de la génération avec TechnicalSupportAgent: {str(e)}")
            return f"Je n'ai pas pu résoudre votre problème technique: {str(e)}", {"error": str(e)}
    
    def _build_prompt(self, formatted_history: List[Dict[str, str]]) -> str:
        """
        Construit le prompt spécialisé pour le support technique.
        
        Args:
            formatted_history: Historique formaté
        
        Returns:
            Prompt pour le modèle
        """
        system_prompt = """Vous êtes un expert en support technique. Vos responsabilités incluent:
- Diagnostiquer les problèmes techniques
- Fournir des instructions étape par étape pour résoudre les problèmes
- Expliquer les concepts techniques de manière accessible
- Conseiller sur les meilleures pratiques
- Orienter vers les ressources adéquates

Votre approche est méthodique, patiente et pédagogique.
"""
        
        # Remplacer le message système par défaut
        for i, msg in enumerate(formatted_history):
            if msg["role"] == "system":
                formatted_history[i]["content"] = system_prompt
                break
        else:
            # Si aucun message système n'a été trouvé, en ajouter un
            formatted_history.insert(0, {"role": "system", "content": system_prompt})
        
        # Convertir l'historique formaté en prompt
        prompt = ""
        for msg in formatted_history:
            if msg["role"] == "system":
                prompt += f"Instructions: {msg['content']}\n\n"
            elif msg["role"] == "user":
                prompt += f"Problème: {msg['content']}\n\n"
            elif msg["role"] == "assistant":
                prompt += f"Solution précédente: {msg['content']}\n\n"
        
        prompt += "Solution technique: "
        
        return prompt


def create_content_generator_agent(name: str, instruction: Optional[str] = None,
                                  model_id: Optional[str] = None) -> Agent:
    """
    Crée un agent spécialisé dans la génération de contenu.
    
    Args:
        name: Nom de l'agent
        instruction: Instruction système (optionnelle)
        model_id: Identifiant du modèle (optionnel)
    
    Returns:
        Instance de l'agent générateur de contenu
    """
    # Configurer l'agent
    config = AgentConfig(
        name=name,
        description="Agent spécialisé dans la génération de contenu",
        instruction=instruction or "Vous êtes un expert en création de contenu engageant et optimisé.",
        model_id=model_id or "r1-teacher",  # Utiliser le modèle puissant par défaut
        temperature=0.7,  # Température moyenne pour encourager la créativité
        max_tokens=2048   # Tokens suffisants pour du contenu substantiel
    )
    
    # Créer l'agent
    agent = ContentGeneratorAgent(config)
    
    logger.info(f"Agent générateur de contenu '{name}' créé")
    
    return agent


def create_sales_agent(name: str, instruction: Optional[str] = None,
                      model_id: Optional[str] = None) -> Agent:
    """
    Crée un agent spécialisé dans l'assistance commerciale.
    
    Args:
        name: Nom de l'agent
        instruction: Instruction système (optionnelle)
        model_id: Identifiant du modèle (optionnel)
    
    Returns:
        Instance de l'agent commercial
    """
    # Configurer l'agent
    config = AgentConfig(
        name=name,
        description="Agent spécialisé dans l'assistance commerciale",
        instruction=instruction or "Vous êtes un assistant commercial professionnel et orienté solution.",
        model_id=model_id or "phi-4-distilled",  # Modèle équilibré par défaut
        temperature=0.4,  # Température plus basse pour des réponses plus cohérentes
        max_tokens=1536
    )
    
    # Créer l'agent
    agent = SalesAgent(config)
    
    logger.info(f"Agent commercial '{name}' créé")
    
    return agent


def create_technical_support_agent(name: str, instruction: Optional[str] = None,
                                 model_id: Optional[str] = None) -> Agent:
    """
    Crée un agent spécialisé dans le support technique.
    
    Args:
        name: Nom de l'agent
        instruction: Instruction système (optionnelle)
        model_id: Identifiant du modèle (optionnel)
    
    Returns:
        Instance de l'agent de support technique
    """
    # Configurer l'agent
    config = AgentConfig(
        name=name,
        description="Agent spécialisé dans le support technique",
        instruction=instruction or "Vous êtes un expert technique qui résout les problèmes de manière méthodique.",
        model_id=model_id or "phi-4-distilled",  # Modèle équilibré par défaut
        temperature=0.3,  # Température basse pour des réponses précises
        max_tokens=1536
    )
    
    # Créer l'agent
    agent = TechnicalSupportAgent(config)
    
    logger.info(f"Agent de support technique '{name}' créé")
    
    return agent 