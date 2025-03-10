import os
import logging
import numpy as np
import pickle
from typing import Dict, Any, List, Optional, Union
from functools import lru_cache
import time
import torch
from pathlib import Path

# Configuration du logging
logger = logging.getLogger("ml_api.rag.vectorizer")

class Vectorizer:
    """
    Vectoriseur de texte pour RAG.
    
    Cette classe est responsable de:
    - Transformer les textes en vecteurs (embeddings)
    - Utiliser un modèle d'embedding efficace
    - Gérer le cache d'embeddings
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", cache_dir: Optional[str] = None):
        """
        Initialise le vectoriseur.
        
        Args:
            model_name (str, optional): Nom du modèle d'embedding. Defaults to "all-MiniLM-L6-v2".
            cache_dir (str, optional): Répertoire de cache. Defaults to None.
        """
        self.model_name = model_name
        
        # Configuration du cache
        from ml_service.config import settings
        
        self.cache_dir = cache_dir or str(settings.DATA_PATH / "rag" / "vectorizer_cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Métriques
        self.metrics = {
            "embed_query_count": 0,
            "embed_document_count": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "total_embedding_time": 0
        }
        
        # Chargement du modèle
        self._load_model()
    
    def _load_model(self):
        """Charge le modèle d'embedding."""
        logger.info(f"Chargement du modèle d'embedding: {self.model_name}")
        start_time = time.time()
        
        try:
            # Utilisation de sentence-transformers s'il est disponible
            try:
                from sentence_transformers import SentenceTransformer
                
                # Vérification du GPU
                device = "cuda" if torch.cuda.is_available() else "cpu"
                self.model = SentenceTransformer(self.model_name, device=device)
                self.embedding_type = "sentence_transformers"
                
                logger.info(f"Modèle d'embedding chargé avec sentence-transformers sur {device}")
            
            # Fallback sur un modèle plus simple
            except ImportError:
                logger.warning("sentence-transformers non disponible, utilisation d'un modèle de fallback")
                from sklearn.feature_extraction.text import TfidfVectorizer
                
                self.model = TfidfVectorizer(max_features=768)
                self.embedding_type = "tfidf"
                
                # Initialisation du vectoriseur avec un corpus vide
                self.model.fit([""])
                
                logger.info("Modèle TF-IDF initialisé comme fallback")
            
            loading_time = time.time() - start_time
            logger.info(f"Modèle d'embedding chargé en {loading_time:.2f}s")
        
        except Exception as e:
            logger.error(f"Erreur lors du chargement du modèle d'embedding: {str(e)}", exc_info=True)
            raise
    
    def embed_query(self, query: str) -> np.ndarray:
        """
        Transforme une requête en vecteur.
        
        Args:
            query (str): Requête à vectoriser
            
        Returns:
            np.ndarray: Vecteur de la requête
        """
        start_time = time.time()
        self.metrics["embed_query_count"] += 1
        
        # Vérification du cache
        cache_key = f"query_{hash(query)}"
        cache_path = Path(self.cache_dir) / f"{cache_key}.pkl"
        
        if cache_path.exists():
            try:
                with open(cache_path, "rb") as f:
                    embedding = pickle.load(f)
                
                self.metrics["cache_hits"] += 1
                return embedding
            except Exception as e:
                logger.warning(f"Erreur lors du chargement du cache pour la requête: {str(e)}")
        
        self.metrics["cache_misses"] += 1
        
        # Vectorisation de la requête
        try:
            if self.embedding_type == "sentence_transformers":
                embedding = self.model.encode(query, normalize_embeddings=True)
            else:
                # Utilisation du TF-IDF
                embedding = self.model.transform([query]).toarray()[0]
                # Normalisation L2
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding = embedding / norm
            
            # Sauvegarde dans le cache
            with open(cache_path, "wb") as f:
                pickle.dump(embedding, f)
            
            embedding_time = time.time() - start_time
            self.metrics["total_embedding_time"] += embedding_time
            
            return embedding
        
        except Exception as e:
            logger.error(f"Erreur lors de la vectorisation de la requête: {str(e)}", exc_info=True)
            raise
    
    def embed_document(self, document: str) -> np.ndarray:
        """
        Transforme un document en vecteur.
        
        Args:
            document (str): Document à vectoriser
            
        Returns:
            np.ndarray: Vecteur du document
        """
        start_time = time.time()
        self.metrics["embed_document_count"] += 1
        
        # Vérification du cache
        cache_key = f"doc_{hash(document)}"
        cache_path = Path(self.cache_dir) / f"{cache_key}.pkl"
        
        if cache_path.exists():
            try:
                with open(cache_path, "rb") as f:
                    embedding = pickle.load(f)
                
                self.metrics["cache_hits"] += 1
                return embedding
            except Exception as e:
                logger.warning(f"Erreur lors du chargement du cache pour le document: {str(e)}")
        
        self.metrics["cache_misses"] += 1
        
        # Vectorisation du document
        try:
            if self.embedding_type == "sentence_transformers":
                embedding = self.model.encode(document, normalize_embeddings=True)
            else:
                # Utilisation du TF-IDF
                embedding = self.model.transform([document]).toarray()[0]
                # Normalisation L2
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding = embedding / norm
            
            # Sauvegarde dans le cache
            with open(cache_path, "wb") as f:
                pickle.dump(embedding, f)
            
            embedding_time = time.time() - start_time
            self.metrics["total_embedding_time"] += embedding_time
            
            return embedding
        
        except Exception as e:
            logger.error(f"Erreur lors de la vectorisation du document: {str(e)}", exc_info=True)
            raise
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Récupère les métriques du vectoriseur.
        
        Returns:
            Dict[str, Any]: Métriques du vectoriseur
        """
        return self.metrics


@lru_cache()
def get_vectorizer() -> Vectorizer:
    """
    Obtient l'instance unique du vectoriseur.
    
    Cette fonction est utilisée comme une dépendance dans FastAPI.
    
    Returns:
        Vectorizer: Instance unique du vectoriseur
    """
    return Vectorizer() 