"""
Module de factory pour les agents.

Ce module fournit des fonctions pour créer différents types d'agents
avec des configurations pré-définies.
"""

import logging
from typing import Dict, List, Any, Optional, Union, Tuple, Type

from .agent import Agent, AgentConfig, Message, Role
from ..models.model_loader import ModelLoader, get_model_loader, create_r1_teacher_model_loader, create_phi4_distilled_model_loader
from ..config import settings

logger = logging.getLogger("ml_api.agents.factory")

# Définir ici les implémentations spécifiques des agents
class TeacherAgent(Agent):
    """
    Agent utilisant le modèle R1 quantifié comme teacher.
    """
    
    def _generate(self, formatted_history: List[Dict[str, str]]) -> Tuple[str, Dict[str, Any]]:
        """
        Génère une réponse en utilisant le modèle teacher.
        
        Args:
            formatted_history: Historique formaté
        
        Returns:
            Tuple (réponse générée, informations de génération)
        """
        # Construire le prompt
        prompt = self._build_prompt(formatted_history)
        
        try:
            # Simuler la génération avec le modèle R1 (à remplacer par l'appel réel)
            # Dans une implémentation réelle, on utiliserait:
            # model, tokenizer = self.model_loader.load_model(self.config.model_id)
            # inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            # response = model.generate(...)
            
            # Simuler une réponse pour le développement
            response_text = "Voici une réponse générée par le TeacherAgent utilisant le modèle R1."
            
            return response_text, {
                "method": "teacher_model",
                "model": self.config.model_id,
                "tokens": len(response_text.split())
            }
            
        except Exception as e:
            logger.error(f"Erreur lors de la génération avec le TeacherAgent: {str(e)}")
            return f"Désolé, je n'ai pas pu générer de réponse: {str(e)}", {"error": str(e)}
    
    def _build_prompt(self, formatted_history: List[Dict[str, str]]) -> str:
        """
        Construit le prompt pour le modèle.
        
        Args:
            formatted_history: Historique formaté
        
        Returns:
            Prompt pour le modèle
        """
        # Format spécifique pour le modèle R1
        system_prompt = "Vous êtes un enseignant patient et pédagogue. Répondez de manière claire et instructive."
        
        # Extraire les messages système s'ils existent
        for msg in formatted_history:
            if msg["role"] == "system":
                system_prompt = msg["content"]
                break
        
        # Construire le prompt avec le format approprié
        prompt = f"<s>[INST] {system_prompt} [/INST]\n\n"
        
        # Ajouter les messages de l'historique (sauf système)
        for msg in formatted_history:
            if msg["role"] == "system":
                continue
            elif msg["role"] == "user":
                prompt += f"[INST] {msg['content']} [/INST]\n"
            elif msg["role"] == "assistant":
                prompt += f"{msg['content']}\n"
        
        return prompt


class AssistantAgent(Agent):
    """
    Agent utilisant le modèle Phi-4 distillé comme assistant.
    """
    
    def _generate(self, formatted_history: List[Dict[str, str]]) -> Tuple[str, Dict[str, Any]]:
        """
        Génère une réponse en utilisant le modèle assistant.
        
        Args:
            formatted_history: Historique formaté
        
        Returns:
            Tuple (réponse générée, informations de génération)
        """
        # Construire le prompt
        prompt = self._build_prompt(formatted_history)
        
        try:
            # Simuler la génération avec le modèle Phi-4 (à remplacer par l'appel réel)
            
            # Simuler une réponse pour le développement
            response_text = "Voici une réponse générée par l'AssistantAgent utilisant le modèle Phi-4 distillé."
            
            return response_text, {
                "method": "assistant_model",
                "model": self.config.model_id,
                "tokens": len(response_text.split())
            }
            
        except Exception as e:
            logger.error(f"Erreur lors de la génération avec l'AssistantAgent: {str(e)}")
            return f"Désolé, je n'ai pas pu générer de réponse: {str(e)}", {"error": str(e)}
    
    def _build_prompt(self, formatted_history: List[Dict[str, str]]) -> str:
        """
        Construit le prompt pour le modèle.
        
        Args:
            formatted_history: Historique formaté
        
        Returns:
            Prompt pour le modèle
        """
        # Format spécifique pour le modèle Phi-4
        system_prompt = "Vous êtes un assistant AI utile, précis et concis. Répondez en français."
        
        # Extraire les messages système s'ils existent
        for msg in formatted_history:
            if msg["role"] == "system":
                system_prompt = msg["content"]
                break
        
        # Construire le prompt avec le format approprié
        prompt = f"<|system|>\n{system_prompt}\n<|end|>\n"
        
        # Ajouter les messages de l'historique (sauf système)
        for msg in formatted_history:
            if msg["role"] == "system":
                continue
            elif msg["role"] == "user":
                prompt += f"<|user|>\n{msg['content']}\n<|end|>\n"
            elif msg["role"] == "assistant":
                prompt += f"<|assistant|>\n{msg['content']}\n<|end|>\n"
        
        # Ajouter le marqueur pour la réponse de l'assistant
        prompt += "<|assistant|>\n"
        
        return prompt


class RAGAgent(Agent):
    """
    Agent augmenté par RAG.
    """
    
    def __init__(self, config: AgentConfig, model_loader: Optional[ModelLoader] = None):
        """
        Initialise un agent RAG.
        
        Args:
            config: Configuration de l'agent
            model_loader: Chargeur de modèles
        """
        super().__init__(config, model_loader)
        
        # Importer ici pour éviter les imports circulaires
        from ..rag.retriever import get_retriever
        self.retriever = get_retriever()
    
    def generate_response(self, 
                          user_input: Optional[str] = None,
                          system_instruction: Optional[str] = None,
                          rag_namespace: str = "default",
                          top_k: int = 5) -> Tuple[str, Dict[str, Any]]:
        """
        Génère une réponse augmentée par RAG.
        
        Args:
            user_input: Entrée utilisateur
            system_instruction: Instruction système
            rag_namespace: Espace de noms pour la recherche
            top_k: Nombre de documents à récupérer
        
        Returns:
            Tuple (réponse générée, métadonnées)
        """
        # Si pas d'entrée utilisateur, utiliser la méthode de base
        if not user_input:
            return super().generate_response(user_input, system_instruction)
        
        start_time = time.time()
        self.metrics["generations_count"] += 1
        
        try:
            # Rechercher des documents pertinents
            documents = self.retriever.retrieve(
                query=user_input,
                namespace=rag_namespace,
                top_k=top_k
            )
            
            # Construire le contexte à partir des documents
            context = self._build_context_from_documents(documents)
            
            # Construire une instruction système augmentée
            augmented_instruction = system_instruction or self.config.instruction
            if augmented_instruction and context:
                augmented_instruction = f"{augmented_instruction}\n\nContexte pour répondre à la question:\n{context}"
            
            # Générer la réponse avec le contexte augmenté
            response, metadata = super().generate_response(user_input, augmented_instruction)
            
            # Ajouter des informations sur les documents utilisés
            metadata["rag_info"] = {
                "namespace": rag_namespace,
                "documents_count": len(documents),
                "documents": [
                    {
                        "score": doc.get("score", 0),
                        "source": doc.get("metadata", {}).get("source", "inconnu"),
                        "content_preview": doc.get("content", "")[:100] + "..." if len(doc.get("content", "")) > 100 else doc.get("content", "")
                    }
                    for doc in documents
                ]
            }
            
            return response, metadata
            
        except Exception as e:
            logger.error(f"Erreur lors de la génération avec RAG: {str(e)}")
            self.metrics["failed_generations"] += 1
            
            # Fallback sur la méthode de base en cas d'erreur
            return super().generate_response(user_input, system_instruction)
    
    def _build_context_from_documents(self, documents: List[Dict[str, Any]]) -> str:
        """
        Construit un contexte à partir des documents récupérés.
        
        Args:
            documents: Documents récupérés
        
        Returns:
            Contexte formaté
        """
        if not documents:
            return ""
        
        context = "Voici des informations pertinentes pour répondre à la question :\n\n"
        
        for i, doc in enumerate(documents, 1):
            content = doc.get("content", "")
            source = doc.get("metadata", {}).get("source", "Source inconnue")
            
            context += f"[Document {i}] ({source}): {content}\n\n"
        
        return context


class HybridAgent(Agent):
    """
    Agent utilisant l'approche hybride RAG-KAG.
    """
    
    def __init__(self, config: AgentConfig, model_loader: Optional[ModelLoader] = None):
        """
        Initialise un agent hybride.
        
        Args:
            config: Configuration de l'agent
            model_loader: Chargeur de modèles
        """
        super().__init__(config, model_loader)
        
        # Importer ici pour éviter les imports circulaires
        from ..hybrid.hybrid_generator import get_hybrid_generator
        self.hybrid_generator = get_hybrid_generator()
    
    def generate_response(self, 
                          user_input: Optional[str] = None,
                          system_instruction: Optional[str] = None,
                          rag_weight: float = 0.5) -> Tuple[str, Dict[str, Any]]:
        """
        Génère une réponse en utilisant l'approche hybride.
        
        Args:
            user_input: Entrée utilisateur
            system_instruction: Instruction système
            rag_weight: Poids relatif de RAG (0-1)
        
        Returns:
            Tuple (réponse générée, métadonnées)
        """
        # Si pas d'entrée utilisateur, utiliser la méthode de base
        if not user_input:
            return super().generate_response(user_input, system_instruction)
        
        # Ajouter le message utilisateur à l'historique
        self.add_user_message(user_input)
        
        try:
            # Générer la réponse avec l'approche hybride
            result = self.hybrid_generator.generate(
                query=user_input,
                rag_weight=rag_weight,
                model=self.config.model_id,
                include_context=True
            )
            
            # Extraire la réponse
            response = result.get("response", "Je n'ai pas réussi à générer une réponse.")
            
            # Ajouter la réponse à l'historique
            self.add_assistant_message(response)
            
            # Mettre à jour les métriques
            self.metrics["successful_generations"] += 1
            
            if "processing_time" in result:
                self.metrics["total_generation_time"] += result["processing_time"]
            
            # Réponse est déjà formatée, donc on la retourne directement
            return response, result
            
        except Exception as e:
            logger.error(f"Erreur lors de la génération hybride: {str(e)}")
            self.metrics["failed_generations"] += 1
            
            # Fallback sur la méthode de base en cas d'erreur
            return super().generate_response(None, system_instruction)


def create_agent(agent_type: str, name: str, 
                instruction: Optional[str] = None,
                model_id: Optional[str] = None) -> Agent:
    """
    Crée un agent du type spécifié.
    
    Args:
        agent_type: Type d'agent ('teacher', 'assistant', 'rag', 'hybrid')
        name: Nom de l'agent
        instruction: Instruction système (optionnelle)
        model_id: Identifiant du modèle (optionnel)
    
    Returns:
        Instance de l'agent créé
    """
    if agent_type == "teacher":
        return create_teacher_agent(name, instruction, model_id)
    elif agent_type == "assistant":
        return create_assistant_agent(name, instruction, model_id)
    elif agent_type == "rag":
        return create_rag_agent(name, instruction, model_id)
    elif agent_type == "hybrid":
        return create_hybrid_agent(name, instruction, model_id)
    else:
        raise ValueError(f"Type d'agent non supporté: {agent_type}")


def create_teacher_agent(name: str, instruction: Optional[str] = None,
                       model_id: Optional[str] = None) -> Agent:
    """
    Crée un agent teacher.
    
    Args:
        name: Nom de l'agent
        instruction: Instruction système (optionnelle)
        model_id: Identifiant du modèle (optionnel)
    
    Returns:
        Instance de l'agent teacher
    """
    # Configurer l'agent
    config = AgentConfig(
        name=name,
        description="Agent enseignant basé sur le modèle R1 quantifié",
        instruction=instruction or "Vous êtes un enseignant patient et pédagogue. Répondez de manière claire et instructive.",
        model_id=model_id or "r1-teacher",
        temperature=0.2,  # Température basse pour des réponses plus déterministes
        max_tokens=1024
    )
    
    # Créer l'agent avec le modèle loader approprié
    model_loader = create_r1_teacher_model_loader()
    agent = TeacherAgent(config, model_loader)
    
    logger.info(f"Agent teacher '{name}' créé")
    
    return agent


def create_assistant_agent(name: str, instruction: Optional[str] = None,
                         model_id: Optional[str] = None) -> Agent:
    """
    Crée un agent assistant.
    
    Args:
        name: Nom de l'agent
        instruction: Instruction système (optionnelle)
        model_id: Identifiant du modèle (optionnel)
    
    Returns:
        Instance de l'agent assistant
    """
    # Configurer l'agent
    config = AgentConfig(
        name=name,
        description="Agent assistant basé sur le modèle Phi-4 distillé",
        instruction=instruction or "Vous êtes un assistant AI utile, précis et concis. Répondez en français.",
        model_id=model_id or "phi-4-distilled",
        temperature=0.7,  # Température moyenne pour un bon équilibre
        max_tokens=1024
    )
    
    # Créer l'agent avec le modèle loader approprié
    model_loader = create_phi4_distilled_model_loader()
    agent = AssistantAgent(config, model_loader)
    
    logger.info(f"Agent assistant '{name}' créé")
    
    return agent


def create_rag_agent(name: str, instruction: Optional[str] = None,
                   model_id: Optional[str] = None) -> Agent:
    """
    Crée un agent RAG.
    
    Args:
        name: Nom de l'agent
        instruction: Instruction système (optionnelle)
        model_id: Identifiant du modèle (optionnel)
    
    Returns:
        Instance de l'agent RAG
    """
    # Configurer l'agent
    config = AgentConfig(
        name=name,
        description="Agent augmenté par RAG",
        instruction=instruction or "Vous êtes un assistant AI qui utilise une base de documents pour répondre précisément aux questions. Basez vos réponses sur les informations fournies.",
        model_id=model_id or "phi-4-distilled",  # Par défaut, utiliser le modèle plus léger
        temperature=0.3,  # Température basse pour rester fidèle aux documents
        max_tokens=1536
    )
    
    # Créer l'agent
    agent = RAGAgent(config)
    
    logger.info(f"Agent RAG '{name}' créé")
    
    return agent


def create_hybrid_agent(name: str, instruction: Optional[str] = None,
                      model_id: Optional[str] = None) -> Agent:
    """
    Crée un agent hybride RAG-KAG.
    
    Args:
        name: Nom de l'agent
        instruction: Instruction système (optionnelle)
        model_id: Identifiant du modèle (optionnel)
    
    Returns:
        Instance de l'agent hybride
    """
    # Configurer l'agent
    config = AgentConfig(
        name=name,
        description="Agent hybride RAG-KAG",
        instruction=instruction or "Vous êtes un assistant IA avancé qui combine des documents et des connaissances structurées pour répondre de manière complète et précise.",
        model_id=model_id or "r1-teacher",  # Par défaut, utiliser le modèle plus puissant
        temperature=0.4,
        max_tokens=2048
    )
    
    # Créer l'agent
    agent = HybridAgent(config)
    
    logger.info(f"Agent hybride '{name}' créé")
    
    return agent 