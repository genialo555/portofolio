import os
import logging
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
from functools import lru_cache
import time

from ml_service.rag.document_store import DocumentStore, get_document_store
from ml_service.rag.vectorizer import Vectorizer, get_vectorizer

# Configuration du logging
logger = logging.getLogger("ml_api.rag.retriever")

class Retriever:
    """
    Récupérateur de documents pour RAG.
    
    Cette classe est responsable de:
    - Récupérer les documents pertinents pour une requête
    - Calculer les scores de similarité
    - Filtrer les documents en fonction de métadonnées
    """
    
    def __init__(self, document_store: Optional[DocumentStore] = None, vectorizer: Optional[Vectorizer] = None):
        """
        Initialise le récupérateur de documents.
        
        Args:
            document_store (DocumentStore, optional): Stockage de documents. Defaults to None.
            vectorizer (Vectorizer, optional): Vectoriseur. Defaults to None.
        """
        self.document_store = document_store or get_document_store()
        self.vectorizer = vectorizer or get_vectorizer()
    
    def retrieve(self, query: str, namespace: str = "default", top_k: int = 5, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Récupère les documents pertinents pour une requête.
        
        Args:
            query (str): Requête
            namespace (str, optional): Espace de noms. Defaults to "default".
            top_k (int, optional): Nombre de documents à récupérer. Defaults to 5.
            filters (Dict[str, Any], optional): Filtres pour les métadonnées. Defaults to None.
            
        Returns:
            List[Dict[str, Any]]: Liste des documents pertinents
        """
        start_time = time.time()
        logger.info(f"Récupération des documents pour la requête: '{query}' dans l'espace de noms: {namespace}")
        
        # Vérification si l'espace de noms existe
        if namespace not in self.document_store.documents:
            logger.warning(f"Espace de noms {namespace} non trouvé")
            return []
        
        # Récupération de tous les documents de l'espace de noms
        all_docs = self.document_store.get_documents(namespace=namespace, limit=1000)
        
        # Application des filtres si nécessaire
        if filters:
            all_docs = self._apply_filters(all_docs, filters)
            logger.info(f"Après filtrage: {len(all_docs)} documents")
        
        # Si aucun document, retourner une liste vide
        if not all_docs:
            logger.info("Aucun document trouvé")
            return []
        
        # Vectorisation de la requête
        query_embedding = self.vectorizer.embed_query(query)
        
        # Calcul des scores de similarité
        scored_docs = self._calculate_similarities(query_embedding, all_docs, namespace)
        
        # Tri par score et sélection des top_k documents
        top_docs = sorted(scored_docs, key=lambda x: x[1], reverse=True)[:top_k]
        
        # Conversion en dictionnaires avec score de similarité
        result_docs = []
        for doc, score in top_docs:
            doc_dict = doc.copy()
            doc_dict["score"] = float(score)  # Conversion pour la sérialisation JSON
            result_docs.append(doc_dict)
        
        processing_time = time.time() - start_time
        logger.info(f"Récupération terminée en {processing_time:.3f}s, {len(result_docs)} documents trouvés")
        
        return result_docs
    
    def _apply_filters(self, documents: List[Dict[str, Any]], filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Applique des filtres sur les métadonnées des documents.
        
        Args:
            documents (List[Dict[str, Any]]): Liste de documents
            filters (Dict[str, Any]): Filtres pour les métadonnées
            
        Returns:
            List[Dict[str, Any]]: Liste des documents filtrés
        """
        filtered_docs = []
        
        for doc in documents:
            # Vérifie si le document a des métadonnées
            if "metadata" not in doc:
                continue
            
            # Vérifie si le document correspond aux filtres
            match = True
            for key, value in filters.items():
                # Gestion des filtres imbriqués
                if "." in key:
                    parts = key.split(".")
                    current = doc["metadata"]
                    for part in parts[:-1]:
                        if part not in current:
                            match = False
                            break
                        current = current[part]
                    
                    if not match or parts[-1] not in current or current[parts[-1]] != value:
                        match = False
                # Filtres simples
                elif key not in doc["metadata"] or doc["metadata"][key] != value:
                    match = False
            
            if match:
                filtered_docs.append(doc)
        
        return filtered_docs
    
    def _calculate_similarities(self, query_embedding: np.ndarray, documents: List[Dict[str, Any]], namespace: str) -> List[Tuple[Dict[str, Any], float]]:
        """
        Calcule les scores de similarité entre une requête et des documents.
        
        Args:
            query_embedding (np.ndarray): Embedding de la requête
            documents (List[Dict[str, Any]]): Liste de documents
            namespace (str): Espace de noms
            
        Returns:
            List[Tuple[Dict[str, Any], float]]: Liste de tuples (document, score)
        """
        scored_docs = []
        docs_to_vectorize = []
        
        # Séparation des documents avec et sans embeddings
        for doc in documents:
            doc_id = doc.get("id")
            
            # Vérification si l'embedding existe
            embedding = self.document_store.get_embedding(doc_id, namespace)
            
            if embedding is not None:
                # Calcul de la similarité cosinus
                similarity = self._cosine_similarity(query_embedding, embedding)
                scored_docs.append((doc, similarity))
            else:
                # Ajout à la liste des documents à vectoriser
                docs_to_vectorize.append(doc)
        
        # Vectorisation des documents sans embeddings
        if docs_to_vectorize:
            logger.info(f"Vectorisation de {len(docs_to_vectorize)} documents")
            
            for doc in docs_to_vectorize:
                doc_id = doc.get("id")
                content = doc.get("content", "")
                
                # Vectorisation du document
                doc_embedding = self.vectorizer.embed_document(content)
                
                # Sauvegarde de l'embedding
                self.document_store.save_embedding(doc_id, doc_embedding, namespace)
                
                # Calcul de la similarité cosinus
                similarity = self._cosine_similarity(query_embedding, doc_embedding)
                scored_docs.append((doc, similarity))
        
        return scored_docs
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Calcule la similarité cosinus entre deux vecteurs.
        
        Args:
            a (np.ndarray): Premier vecteur
            b (np.ndarray): Deuxième vecteur
            
        Returns:
            float: Similarité cosinus
        """
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


@lru_cache()
def get_retriever() -> Retriever:
    """
    Obtient l'instance unique du récupérateur de documents.
    
    Cette fonction est utilisée comme une dépendance dans FastAPI.
    
    Returns:
        Retriever: Instance unique du récupérateur de documents
    """
    return Retriever() 