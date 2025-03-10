"""
Module de fusion des connaissances.

Ce module fournit la classe KnowledgeFusion qui permet de combiner
intelligemment les connaissances provenant de RAG et KAG.
"""

import time
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Set
from functools import lru_cache

from ..config import settings
from ..rag.vectorizer import Vectorizer, get_vectorizer

logger = logging.getLogger("ml_api.hybrid.fusion")

class KnowledgeFusion:
    """
    Classe qui fusionne les connaissances issues de RAG et KAG.
    
    Cette classe implémente plusieurs stratégies pour combiner 
    les documents récupérés par RAG et les connaissances structurées 
    extraites par KAG.
    """
    
    def __init__(self, vectorizer: Optional[Vectorizer] = None):
        """
        Initialise le fusionneur de connaissances.
        
        Args:
            vectorizer: Vectoriseur pour les calculs de similarité
        """
        self.vectorizer = vectorizer or get_vectorizer()
        
        # Métriques
        self.metrics = {
            "total_fusion_time": 0,
            "fusion_count": 0,
            "avg_document_score": 0,
            "avg_fact_score": 0,
        }
        
        logger.info("KnowledgeFusion initialisé")
    
    def fuse_knowledge(self, 
                      query: str,
                      rag_results: List[Dict[str, Any]],
                      kag_results: Dict[str, Any],
                      rag_weight: float = 0.5,
                      strategy: str = "weighted") -> Dict[str, Any]:
        """
        Fusionne les connaissances issues de RAG et KAG.
        
        Args:
            query: Requête utilisateur
            rag_results: Résultats de la récupération RAG
            kag_results: Résultats de la récupération KAG
            rag_weight: Poids relatif de RAG (0-1)
            strategy: Stratégie de fusion ('weighted', 'interleave', 'separated')
        
        Returns:
            Contexte fusionné
        """
        start_time = time.time()
        self.metrics["fusion_count"] += 1
        
        # Valider les entrées
        if not isinstance(rag_results, list):
            rag_results = []
        
        if not isinstance(kag_results, dict):
            kag_results = {"entities": [], "facts": [], "query": query}
        
        # Normaliser les poids
        kag_weight = 1.0 - rag_weight
        
        # Choisir la stratégie de fusion
        if strategy == "interleave":
            fused_context = self._interleave_fusion(query, rag_results, kag_results)
        elif strategy == "separated":
            fused_context = self._separated_fusion(query, rag_results, kag_results)
        else:  # Par défaut: weighted
            fused_context = self._weighted_fusion(query, rag_results, kag_results, rag_weight, kag_weight)
        
        # Mettre à jour les métriques
        fusion_time = time.time() - start_time
        self.metrics["total_fusion_time"] += fusion_time
        
        # Calculer les scores moyens
        if rag_results:
            avg_doc_score = sum(doc.get("score", 0) for doc in rag_results) / len(rag_results)
            self.metrics["avg_document_score"] = ((self.metrics["avg_document_score"] * (self.metrics["fusion_count"] - 1)) + avg_doc_score) / self.metrics["fusion_count"]
        
        facts = kag_results.get("facts", [])
        if facts:
            avg_fact_score = sum(fact.get("confidence", 0) for fact in facts) / len(facts)
            self.metrics["avg_fact_score"] = ((self.metrics["avg_fact_score"] * (self.metrics["fusion_count"] - 1)) + avg_fact_score) / self.metrics["fusion_count"]
        
        logger.debug(f"Fusion des connaissances terminée en {fusion_time:.3f}s (stratégie: {strategy})")
        
        return fused_context
    
    def _weighted_fusion(self, 
                        query: str,
                        rag_results: List[Dict[str, Any]],
                        kag_results: Dict[str, Any],
                        rag_weight: float,
                        kag_weight: float) -> Dict[str, Any]:
        """
        Fusionne les connaissances en utilisant une approche pondérée.
        
        Args:
            query: Requête utilisateur
            rag_results: Résultats de la récupération RAG
            kag_results: Résultats de la récupération KAG
            rag_weight: Poids de RAG
            kag_weight: Poids de KAG
        
        Returns:
            Contexte fusionné
        """
        # 1. Extraire et normaliser les scores des documents
        documents = []
        for doc in rag_results:
            normalized_score = doc.get("score", 0.5) * rag_weight
            documents.append({
                **doc,
                "score": normalized_score
            })
        
        # 2. Extraire et normaliser les scores des faits
        facts = []
        for fact in kag_results.get("facts", []):
            normalized_score = fact.get("confidence", 0.5) * kag_weight
            facts.append({
                **fact,
                "confidence": normalized_score
            })
        
        # 3. Extraire les entités
        entities = kag_results.get("entities", [])
        
        # 4. Trier les documents et les faits par score
        documents.sort(key=lambda x: x.get("score", 0), reverse=True)
        facts.sort(key=lambda x: x.get("confidence", 0), reverse=True)
        
        # 5. Construire le contexte fusionné
        return {
            "query": query,
            "documents": documents,
            "facts": facts,
            "entities": entities,
            "rag_weight": rag_weight,
            "kag_weight": kag_weight,
            "fusion_strategy": "weighted"
        }
    
    def _interleave_fusion(self,
                          query: str,
                          rag_results: List[Dict[str, Any]],
                          kag_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fusionne les connaissances en alternant documents et faits.
        
        Args:
            query: Requête utilisateur
            rag_results: Résultats de la récupération RAG
            kag_results: Résultats de la récupération KAG
        
        Returns:
            Contexte fusionné
        """
        # 1. Extraire les documents et les faits
        documents = rag_results
        facts = kag_results.get("facts", [])
        entities = kag_results.get("entities", [])
        
        # 2. Trier les documents et les faits par score
        documents.sort(key=lambda x: x.get("score", 0), reverse=True)
        facts.sort(key=lambda x: x.get("confidence", 0), reverse=True)
        
        # 3. Interleaver les documents et les faits
        interleaved_context = []
        max_items = max(len(documents), len(facts))
        
        for i in range(max_items):
            if i < len(documents):
                interleaved_context.append({"type": "document", "content": documents[i]})
            if i < len(facts):
                interleaved_context.append({"type": "fact", "content": facts[i]})
        
        # 4. Construire le contexte fusionné
        return {
            "query": query,
            "documents": documents,
            "facts": facts,
            "entities": entities,
            "interleaved": interleaved_context,
            "fusion_strategy": "interleave"
        }
    
    def _separated_fusion(self,
                         query: str,
                         rag_results: List[Dict[str, Any]],
                         kag_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fusionne les connaissances en séparant documents et faits.
        
        Args:
            query: Requête utilisateur
            rag_results: Résultats de la récupération RAG
            kag_results: Résultats de la récupération KAG
        
        Returns:
            Contexte fusionné
        """
        # Simplement regrouper les deux sources sans modification
        return {
            "query": query,
            "documents": rag_results,
            "facts": kag_results.get("facts", []),
            "entities": kag_results.get("entities", []),
            "fusion_strategy": "separated"
        }
    
    def rerank_by_relevance(self, 
                           query: str,
                           documents: List[Dict[str, Any]],
                           facts: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Reclasse les documents et les faits en fonction de leur pertinence.
        
        Args:
            query: Requête utilisateur
            documents: Liste de documents
            facts: Liste de faits
        
        Returns:
            Tuple (documents reclassés, faits reclassés)
        """
        # Vectoriser la requête
        query_embedding = self.vectorizer.embed_query(query)
        
        # Reclasser les documents
        reranked_documents = []
        for doc in documents:
            content = doc.get("content", "")
            try:
                doc_embedding = self.vectorizer.embed_document(content)
                similarity = self._cosine_similarity(query_embedding, doc_embedding)
                reranked_documents.append({
                    **doc,
                    "relevance_score": float(similarity)
                })
            except Exception as e:
                logger.warning(f"Erreur lors du calcul de similarité pour un document: {str(e)}")
                reranked_documents.append(doc)
        
        # Trier par pertinence
        reranked_documents.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
        
        # Reclasser les faits
        reranked_facts = []
        for fact in facts:
            subject = fact.get("subject", "")
            predicate = fact.get("predicate", "")
            obj = fact.get("object", "")
            fact_text = f"{subject} {predicate} {obj}"
            
            try:
                fact_embedding = self.vectorizer.embed_document(fact_text)
                similarity = self._cosine_similarity(query_embedding, fact_embedding)
                reranked_facts.append({
                    **fact,
                    "relevance_score": float(similarity)
                })
            except Exception as e:
                logger.warning(f"Erreur lors du calcul de similarité pour un fait: {str(e)}")
                reranked_facts.append(fact)
        
        # Trier par pertinence
        reranked_facts.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
        
        return reranked_documents, reranked_facts
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Calcule la similarité cosinus entre deux vecteurs.
        
        Args:
            a: Premier vecteur
            b: Deuxième vecteur
        
        Returns:
            Similarité cosinus entre les vecteurs
        """
        if a is None or b is None:
            return 0.0
        
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return float(np.dot(a, b) / (norm_a * norm_b))
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Récupère les métriques du fusionneur.
        
        Returns:
            Métriques du fusionneur
        """
        return {
            "avg_fusion_time": self.metrics["total_fusion_time"] / max(1, self.metrics["fusion_count"]),
            "fusion_count": self.metrics["fusion_count"],
            "avg_document_score": self.metrics["avg_document_score"],
            "avg_fact_score": self.metrics["avg_fact_score"]
        }


@lru_cache()
def get_knowledge_fusion() -> KnowledgeFusion:
    """
    Fonction pour obtenir une instance singleton du fusionneur de connaissances.
    
    Returns:
        Instance du fusionneur de connaissances
    """
    return KnowledgeFusion() 