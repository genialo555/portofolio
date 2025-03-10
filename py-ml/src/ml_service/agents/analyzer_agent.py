"""
Module d'agent analyseur de requêtes.

Ce module fournit un agent spécialisé dans l'analyse et la classification des requêtes
entrantes afin de les router vers les agents spécialisés appropriés.
"""

import logging
import json
from typing import Dict, List, Any, Optional, Tuple, Union

from .agent import Agent, AgentConfig, Message, Role
from ..models.model_loader import ModelLoader
from ..config import settings

logger = logging.getLogger("ml_api.agents.analyzer")

class RequestType:
    """Types de requêtes reconnus par l'agent analyseur."""
    CONTENT_GENERATION = "content_generation"
    SALES_INQUIRY = "sales_inquiry"
    TECHNICAL_SUPPORT = "technical_support"
    GENERAL_INQUIRY = "general_inquiry"
    FEEDBACK = "feedback"
    COMPLAINT = "complaint"
    UNDEFINED = "undefined"


class RequestAnalyzerAgent(Agent):
    """
    Agent spécialisé dans l'analyse et la classification des requêtes entrantes.
    Utilise un modèle plus imposant pour l'analyse fine des intentions et du contexte.
    """
    
    def __init__(self, config: AgentConfig, model_loader: Optional[ModelLoader] = None):
        """
        Initialise l'agent d'analyse avec sa configuration.
        
        Args:
            config: Configuration de l'agent
            model_loader: Chargeur de modèle (optionnel)
        """
        super().__init__(config, model_loader)
        # Initialiser le compteur de requêtes par type
        self.request_counters = {
            RequestType.CONTENT_GENERATION: 0,
            RequestType.SALES_INQUIRY: 0,
            RequestType.TECHNICAL_SUPPORT: 0,
            RequestType.GENERAL_INQUIRY: 0,
            RequestType.FEEDBACK: 0,
            RequestType.COMPLAINT: 0,
            RequestType.UNDEFINED: 0
        }
    
    def analyze_request(self, request_text: str) -> Dict[str, Any]:
        """
        Analyse une requête et détermine son type, son urgence et les informations clés.
        
        Args:
            request_text: Texte de la requête à analyser
        
        Returns:
            Dictionnaire contenant l'analyse de la requête
        """
        # Ajouter un message utilisateur avec la requête à analyser
        self.add_user_message(request_text)
        
        # Générer l'analyse avec le modèle
        system_instruction = """Analysez cette requête ou ce message et retournez un JSON avec:
1. type: Le type de la requête parmi ["content_generation", "sales_inquiry", "technical_support", "general_inquiry", "feedback", "complaint"]
2. urgency: Le niveau d'urgence entre 1 (faible) et 5 (critique)
3. key_info: Un dictionnaire contenant les informations clés extraites
4. next_action: L'action recommandée pour traiter cette requête
5. confidence: Votre niveau de confiance dans cette analyse (0.0 à 1.0)
"""
        
        response, info = self.generate_response(system_instruction=system_instruction)
        
        # Parser la réponse JSON (avec gestion d'erreur)
        try:
            # Extraire le JSON de la réponse (au cas où il y aurait du texte avant/après)
            json_response = self._extract_json(response)
            analysis = json.loads(json_response)
            
            # Valider et normaliser l'analyse
            analysis = self._validate_analysis(analysis)
            
            # Mettre à jour les compteurs
            request_type = analysis.get("type", RequestType.UNDEFINED)
            if request_type in self.request_counters:
                self.request_counters[request_type] += 1
            
            return analysis
            
        except Exception as e:
            logger.error(f"Erreur lors de l'analyse de la requête: {str(e)}")
            # Retourner une analyse par défaut en cas d'erreur
            return {
                "type": RequestType.UNDEFINED,
                "urgency": 3,
                "key_info": {"error": str(e)},
                "next_action": "manual_review",
                "confidence": 0.0,
                "error": str(e)
            }
    
    def _extract_json(self, text: str) -> str:
        """
        Extrait la partie JSON d'une réponse textuelle.
        
        Args:
            text: Texte contenant potentiellement du JSON
            
        Returns:
            Chaîne JSON extraite
        """
        # Chercher le JSON entre accolades
        start_idx = text.find('{')
        if start_idx == -1:
            raise ValueError("Aucun JSON trouvé dans la réponse")
        
        # Parcourir le texte pour trouver l'accolade fermante correspondante
        open_braces = 0
        for i in range(start_idx, len(text)):
            if text[i] == '{':
                open_braces += 1
            elif text[i] == '}':
                open_braces -= 1
                if open_braces == 0:
                    return text[start_idx:i+1]
        
        raise ValueError("JSON incomplet ou mal formé")
    
    def _validate_analysis(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Valide et normalise l'analyse de requête.
        
        Args:
            analysis: Analyse brute
            
        Returns:
            Analyse validée et normalisée
        """
        # Assurez-vous que toutes les clés requises sont présentes
        valid_analysis = {
            "type": analysis.get("type", RequestType.UNDEFINED),
            "urgency": min(max(analysis.get("urgency", 3), 1), 5),  # Entre 1 et 5
            "key_info": analysis.get("key_info", {}),
            "next_action": analysis.get("next_action", "review"),
            "confidence": min(max(analysis.get("confidence", 0.5), 0.0), 1.0)  # Entre 0 et 1
        }
        
        # Validation du type de requête
        if valid_analysis["type"] not in self.request_counters:
            valid_analysis["type"] = RequestType.UNDEFINED
        
        return valid_analysis
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Récupère les métriques d'utilisation de l'agent analyseur.
        
        Returns:
            Métriques d'utilisation
        """
        metrics = super().get_metrics()
        
        # Ajouter les compteurs de requêtes par type
        metrics["request_counters"] = self.request_counters
        
        # Ajouter le nombre total de requêtes analysées
        metrics["total_requests"] = sum(self.request_counters.values())
        
        return metrics


def create_request_analyzer_agent(name: str, instruction: Optional[str] = None,
                                 model_id: Optional[str] = None) -> RequestAnalyzerAgent:
    """
    Crée un agent d'analyse de requêtes.
    
    Args:
        name: Nom de l'agent
        instruction: Instruction système (optionnelle)
        model_id: Identifiant du modèle (optionnel)
    
    Returns:
        Instance de l'agent d'analyse
    """
    # Configurer l'agent avec un modèle puissant pour l'analyse
    config = AgentConfig(
        name=name,
        description="Agent d'analyse et de classification des requêtes entrantes",
        instruction=instruction or "Vous êtes un expert en analyse de requêtes. Votre rôle est de classifier précisément les intentions des utilisateurs.",
        model_id=model_id or "r1-teacher",  # Utiliser le modèle le plus puissant par défaut
        temperature=0.2,  # Température basse pour des analyses cohérentes
        max_tokens=1024
    )
    
    # Créer l'agent
    agent = RequestAnalyzerAgent(config)
    
    logger.info(f"Agent d'analyse de requêtes '{name}' créé")
    
    return agent 