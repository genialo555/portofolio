"""
Module du générateur hybride RAG-KAG.

Ce module fournit la classe HybridGenerator qui combine les approches
RAG et KAG pour enrichir les réponses générées avec des connaissances.
"""

import time
import logging
import json
from typing import Dict, List, Any, Optional, Tuple
from functools import lru_cache

from ..rag.retriever import Retriever, get_retriever
from ..rag.document_store import DocumentStore, get_document_store
from ..kag.knowledge_base import KnowledgeBase, get_knowledge_base
from ..config import settings
from ..api.model_manager import ModelManager, get_model_manager
from .fusion import KnowledgeFusion, get_knowledge_fusion

logger = logging.getLogger("ml_api.hybrid.generator")

class HybridGenerator:
    """
    Classe qui combine RAG et KAG pour la génération de réponses.
    
    Cette classe intègre:
    - La récupération de documents pertinents (RAG)
    - L'extraction de connaissances structurées (KAG)
    - La fusion des deux approches
    - La génération de réponses enrichies
    """
    
    def __init__(self, 
                retriever: Optional[Retriever] = None,
                knowledge_base: Optional[KnowledgeBase] = None,
                model_manager: Optional[ModelManager] = None,
                fusion: Optional[KnowledgeFusion] = None):
        """
        Initialise le générateur hybride.
        
        Args:
            retriever: Récupérateur de documents (RAG)
            knowledge_base: Base de connaissances (KAG)
            model_manager: Gestionnaire de modèles
            fusion: Mécanisme de fusion des connaissances
        """
        self.retriever = retriever or get_retriever()
        self.knowledge_base = knowledge_base or get_knowledge_base()
        self.model_manager = model_manager or get_model_manager()
        self.fusion = fusion or get_knowledge_fusion()
        
        # Métriques
        self.metrics = {
            "total_generation_time": 0,
            "generation_count": 0,
            "successful_generations": 0,
            "failed_generations": 0,
            "avg_document_count": 0,
            "avg_fact_count": 0,
        }
        
        logger.info("HybridGenerator initialisé")
    
    def generate(self, 
                query: str, 
                rag_namespace: str = "default",
                kag_namespace: str = "default",
                rag_top_k: int = 5,
                kag_top_k: int = 5,
                model: str = "teacher",
                max_tokens: int = 1024,
                temperature: float = 0.7,
                rag_weight: float = 0.5,  # Poids relatif de RAG (0-1)
                include_context: bool = False) -> Dict[str, Any]:
        """
        Génère une réponse en combinant RAG et KAG.
        
        Args:
            query: Requête utilisateur
            rag_namespace: Espace de noms pour les documents RAG
            kag_namespace: Espace de noms pour les connaissances KAG
            rag_top_k: Nombre de documents à récupérer
            kag_top_k: Nombre d'entités à récupérer
            model: Modèle à utiliser pour la génération
            max_tokens: Nombre maximum de tokens à générer
            temperature: Température pour la génération
            rag_weight: Poids relatif de RAG (0-1)
            include_context: Inclure le contexte dans la réponse
        
        Returns:
            Réponse générée et métadonnées
        """
        start_time = time.time()
        self.metrics["generation_count"] += 1
        
        try:
            # 1. Récupérer les documents pertinents (RAG)
            rag_results = self.retriever.retrieve(
                query=query,
                namespace=rag_namespace,
                top_k=rag_top_k
            )
            
            # 2. Récupérer les connaissances pertinentes (KAG)
            kag_results = self.knowledge_base.get_knowledge_for_query(
                query=query,
                limit=kag_top_k
            )
            
            # 3. Fusionner les connaissances RAG et KAG
            fused_context = self.fusion.fuse_knowledge(
                query=query,
                rag_results=rag_results,
                kag_results=kag_results,
                rag_weight=rag_weight
            )
            
            # 4. Formater le contexte pour le modèle
            formatted_context = self._format_context_for_model(fused_context)
            
            # 5. Construire le prompt
            prompt = self._build_prompt(query, formatted_context)
            
            # 6. Générer la réponse avec le modèle
            generation_result = self.model_manager.evaluate_response(
                response=prompt,
                model=model
            )
            
            # 7. Extraire la réponse générée
            response = generation_result.get("analysis", {}).get("synthesis", "")
            
            # Mettre à jour les métriques
            generation_time = time.time() - start_time
            self.metrics["total_generation_time"] += generation_time
            self.metrics["successful_generations"] += 1
            self.metrics["avg_document_count"] += (len(rag_results) / max(1, self.metrics["generation_count"]))
            self.metrics["avg_fact_count"] += (len(kag_results.get("facts", [])) / max(1, self.metrics["generation_count"]))
            
            # Construire le résultat
            result = {
                "query": query,
                "response": response,
                "model": model,
                "processing_time": generation_time
            }
            
            # Inclure le contexte si demandé
            if include_context:
                result["context"] = {
                    "rag": rag_results,
                    "kag": kag_results,
                    "fused": fused_context
                }
            
            logger.info(f"Génération hybride réussie en {generation_time:.3f}s: {query[:50]}...")
            
            return result
        
        except Exception as e:
            logger.error(f"Erreur lors de la génération hybride: {str(e)}")
            self.metrics["failed_generations"] += 1
            
            # Retourner un résultat d'erreur
            return {
                "query": query,
                "error": str(e),
                "model": model,
                "processing_time": time.time() - start_time
            }
    
    def _format_context_for_model(self, fused_context: Dict[str, Any]) -> str:
        """
        Formate le contexte fusionné pour le modèle.
        
        Args:
            fused_context: Contexte fusionné
        
        Returns:
            Contexte formaté pour le modèle
        """
        formatted_text = "Informations de contexte:\n\n"
        
        # Ajouter les documents
        if "documents" in fused_context and fused_context["documents"]:
            formatted_text += "Documents pertinents:\n"
            for i, doc in enumerate(fused_context["documents"], 1):
                content = doc.get("content", "")
                source = doc.get("metadata", {}).get("source", "inconnu")
                formatted_text += f"{i}. [{source}] {content[:300]}{'...' if len(content) > 300 else ''}\n\n"
        
        # Ajouter les faits
        if "facts" in fused_context and fused_context["facts"]:
            formatted_text += "Faits pertinents:\n"
            for i, fact in enumerate(fused_context["facts"], 1):
                subject = fact.get("subject", "")
                predicate = fact.get("predicate", "")
                obj = fact.get("object", "")
                formatted_text += f"{i}. {subject} {predicate} {obj}\n"
            
            formatted_text += "\n"
        
        # Ajouter les entités
        if "entities" in fused_context and fused_context["entities"]:
            formatted_text += "Entités pertinentes: "
            entities = [f"{entity.get('label', '')} ({entity.get('type', 'entité')})" 
                       for entity in fused_context["entities"]]
            formatted_text += ", ".join(entities)
            formatted_text += "\n\n"
        
        return formatted_text
    
    def _build_prompt(self, query: str, context: str) -> str:
        """
        Construit le prompt pour le modèle.
        
        Args:
            query: Requête utilisateur
            context: Contexte formaté
        
        Returns:
            Prompt complet pour le modèle
        """
        return f"""Vous êtes un assistant IA expert. Utilisez les informations de contexte pour répondre à la question de manière précise et informative. 
Si les informations ne sont pas suffisantes, répondez au mieux de vos connaissances, mais indiquez clairement ce qui provient du contexte et ce qui est une supposition.

{context}

Question: {query}

Réponse:"""
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Récupère les métriques du générateur hybride.
        
        Returns:
            Métriques du générateur
        """
        metrics = {
            "avg_generation_time": self.metrics["total_generation_time"] / max(1, self.metrics["generation_count"]),
            "success_rate": self.metrics["successful_generations"] / max(1, self.metrics["generation_count"]),
            "generation_count": self.metrics["generation_count"],
            "avg_document_count": self.metrics["avg_document_count"],
            "avg_fact_count": self.metrics["avg_fact_count"]
        }
        
        # Ajouter les métriques des composants
        try:
            fusion_metrics = self.fusion.get_metrics()
            metrics["fusion"] = fusion_metrics
        except:
            pass
        
        return metrics


@lru_cache()
def get_hybrid_generator() -> HybridGenerator:
    """
    Fonction pour obtenir une instance singleton du générateur hybride.
    
    Returns:
        Instance du générateur hybride
    """
    return HybridGenerator() 