import os
import logging
import uuid
import json
from typing import Dict, Any, List, Optional
from pathlib import Path
import numpy as np
from functools import lru_cache
from datetime import datetime

# Configuration du logging
logger = logging.getLogger("ml_api.rag.document_store")

class DocumentStore:
    """
    Stockage des documents pour RAG.
    
    Cette classe est responsable de:
    - Stocker les documents et leurs embeddings
    - Gérer les espaces de noms
    - Fournir des méthodes pour ajouter, récupérer et supprimer des documents
    """
    
    def __init__(self, data_dir: str = None):
        """
        Initialise le stockage de documents.
        
        Args:
            data_dir (str, optional): Répertoire de données. Defaults to None.
        """
        from ml_service.config import settings
        
        self.data_dir = data_dir or str(settings.DATA_PATH / "rag")
        self.documents_dir = Path(self.data_dir) / "documents"
        self.embeddings_dir = Path(self.data_dir) / "embeddings"
        
        # Création des répertoires si nécessaires
        os.makedirs(self.documents_dir, exist_ok=True)
        os.makedirs(self.embeddings_dir, exist_ok=True)
        
        # Dictionnaire interne pour stocker les documents en mémoire
        self.documents = {}
        self.embeddings = {}
        
        # Chargement des documents existants
        self._load_documents()
    
    def _load_documents(self):
        """Charge les documents existants à partir du disque."""
        logger.info("Chargement des documents existants...")
        
        # Parcours des espaces de noms
        for namespace_dir in self.documents_dir.iterdir():
            if not namespace_dir.is_dir():
                continue
            
            namespace = namespace_dir.name
            logger.info(f"Chargement des documents pour l'espace de noms: {namespace}")
            
            # Initialisation du dictionnaire pour cet espace de noms
            self.documents[namespace] = {}
            self.embeddings[namespace] = {}
            
            # Chargement des documents
            for doc_file in namespace_dir.glob("*.json"):
                try:
                    with open(doc_file, "r", encoding="utf-8") as f:
                        document = json.load(f)
                    
                    doc_id = document.get("id") or doc_file.stem
                    self.documents[namespace][doc_id] = document
                    
                    # Chargement des embeddings si disponibles
                    embedding_file = self.embeddings_dir / namespace / f"{doc_id}.npy"
                    if embedding_file.exists():
                        try:
                            embedding = np.load(str(embedding_file))
                            self.embeddings[namespace][doc_id] = embedding
                        except Exception as e:
                            logger.warning(f"Erreur lors du chargement de l'embedding pour {doc_id}: {str(e)}")
                    
                except Exception as e:
                    logger.warning(f"Erreur lors du chargement du document {doc_file}: {str(e)}")
            
            logger.info(f"Chargés {len(self.documents[namespace])} documents pour l'espace de noms {namespace}")
    
    def add_documents(self, documents: List[Dict[str, Any]], namespace: str = "default") -> Dict[str, Any]:
        """
        Ajoute des documents au stockage.
        
        Args:
            documents (List[Dict[str, Any]]): Liste de documents à ajouter
            namespace (str, optional): Espace de noms. Defaults to "default".
            
        Returns:
            Dict[str, Any]: Résultat de l'opération
        """
        # Initialisation de l'espace de noms s'il n'existe pas
        if namespace not in self.documents:
            self.documents[namespace] = {}
            self.embeddings[namespace] = {}
        
        # Création des répertoires si nécessaires
        namespace_dir = self.documents_dir / namespace
        embeddings_dir = self.embeddings_dir / namespace
        os.makedirs(namespace_dir, exist_ok=True)
        os.makedirs(embeddings_dir, exist_ok=True)
        
        document_ids = []
        
        # Ajout des documents
        for document in documents:
            # Génération d'un ID s'il n'en a pas
            doc_id = document.get("id") or str(uuid.uuid4())
            document["id"] = doc_id
            
            # Ajout de métadonnées
            if "metadata" not in document:
                document["metadata"] = {}
            
            document["metadata"]["added_at"] = datetime.now().isoformat()
            
            # Stockage du document
            self.documents[namespace][doc_id] = document
            document_ids.append(doc_id)
            
            # Sauvegarde du document sur disque
            with open(namespace_dir / f"{doc_id}.json", "w", encoding="utf-8") as f:
                json.dump(document, f, ensure_ascii=False, indent=2)
        
        return {
            "document_ids": document_ids,
            "document_count": len(document_ids),
            "namespace": namespace
        }
    
    def get_document(self, doc_id: str, namespace: str = "default") -> Optional[Dict[str, Any]]:
        """
        Récupère un document par son ID.
        
        Args:
            doc_id (str): ID du document
            namespace (str, optional): Espace de noms. Defaults to "default".
            
        Returns:
            Optional[Dict[str, Any]]: Document ou None s'il n'existe pas
        """
        if namespace not in self.documents or doc_id not in self.documents[namespace]:
            return None
        
        return self.documents[namespace][doc_id]
    
    def get_documents(self, namespace: str = "default", limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """
        Récupère tous les documents d'un espace de noms.
        
        Args:
            namespace (str, optional): Espace de noms. Defaults to "default".
            limit (int, optional): Nombre maximum de documents à récupérer. Defaults to 100.
            offset (int, optional): Offset pour la pagination. Defaults to 0.
            
        Returns:
            List[Dict[str, Any]]: Liste des documents
        """
        if namespace not in self.documents:
            return []
        
        docs = list(self.documents[namespace].values())
        return docs[offset:offset+limit]
    
    def delete_document(self, doc_id: str, namespace: str = "default") -> bool:
        """
        Supprime un document par son ID.
        
        Args:
            doc_id (str): ID du document
            namespace (str, optional): Espace de noms. Defaults to "default".
            
        Returns:
            bool: True si le document a été supprimé, False sinon
        """
        if namespace not in self.documents or doc_id not in self.documents[namespace]:
            return False
        
        # Suppression du document en mémoire
        del self.documents[namespace][doc_id]
        
        # Suppression de l'embedding s'il existe
        if namespace in self.embeddings and doc_id in self.embeddings[namespace]:
            del self.embeddings[namespace][doc_id]
        
        # Suppression du document sur disque
        doc_file = self.documents_dir / namespace / f"{doc_id}.json"
        if doc_file.exists():
            doc_file.unlink()
        
        # Suppression de l'embedding sur disque
        embedding_file = self.embeddings_dir / namespace / f"{doc_id}.npy"
        if embedding_file.exists():
            embedding_file.unlink()
        
        return True
    
    def delete_documents(self, namespace: str = "default") -> Dict[str, Any]:
        """
        Supprime tous les documents d'un espace de noms.
        
        Args:
            namespace (str, optional): Espace de noms. Defaults to "default".
            
        Returns:
            Dict[str, Any]: Résultat de l'opération
        """
        if namespace not in self.documents:
            return {"deleted_count": 0, "namespace": namespace}
        
        deleted_count = len(self.documents[namespace])
        
        # Suppression des documents en mémoire
        self.documents[namespace] = {}
        
        # Suppression des embeddings en mémoire
        if namespace in self.embeddings:
            self.embeddings[namespace] = {}
        
        # Suppression des fichiers sur disque
        namespace_dir = self.documents_dir / namespace
        embeddings_dir = self.embeddings_dir / namespace
        
        if namespace_dir.exists():
            for doc_file in namespace_dir.glob("*.json"):
                doc_file.unlink()
        
        if embeddings_dir.exists():
            for embedding_file in embeddings_dir.glob("*.npy"):
                embedding_file.unlink()
        
        return {"deleted_count": deleted_count, "namespace": namespace}
    
    def list_namespaces(self) -> List[str]:
        """
        Liste tous les espaces de noms.
        
        Returns:
            List[str]: Liste des espaces de noms
        """
        return list(self.documents.keys())
    
    def count_documents(self, namespace: str = "default") -> int:
        """
        Compte le nombre de documents dans un espace de noms.
        
        Args:
            namespace (str, optional): Espace de noms. Defaults to "default".
            
        Returns:
            int: Nombre de documents
        """
        if namespace not in self.documents:
            return 0
        
        return len(self.documents[namespace])
    
    def save_embedding(self, doc_id: str, embedding: np.ndarray, namespace: str = "default") -> bool:
        """
        Sauvegarde l'embedding d'un document.
        
        Args:
            doc_id (str): ID du document
            embedding (np.ndarray): Embedding du document
            namespace (str, optional): Espace de noms. Defaults to "default".
            
        Returns:
            bool: True si l'embedding a été sauvegardé, False sinon
        """
        if namespace not in self.documents or doc_id not in self.documents[namespace]:
            logger.warning(f"Document {doc_id} non trouvé dans l'espace de noms {namespace}")
            return False
        
        # Initialisation de l'espace de noms s'il n'existe pas
        if namespace not in self.embeddings:
            self.embeddings[namespace] = {}
        
        # Stockage de l'embedding en mémoire
        self.embeddings[namespace][doc_id] = embedding
        
        # Sauvegarde de l'embedding sur disque
        embeddings_dir = self.embeddings_dir / namespace
        os.makedirs(embeddings_dir, exist_ok=True)
        
        np.save(str(embeddings_dir / f"{doc_id}.npy"), embedding)
        
        return True
    
    def get_embedding(self, doc_id: str, namespace: str = "default") -> Optional[np.ndarray]:
        """
        Récupère l'embedding d'un document.
        
        Args:
            doc_id (str): ID du document
            namespace (str, optional): Espace de noms. Defaults to "default".
            
        Returns:
            Optional[np.ndarray]: Embedding du document ou None s'il n'existe pas
        """
        if namespace not in self.embeddings or doc_id not in self.embeddings[namespace]:
            return None
        
        return self.embeddings[namespace][doc_id]
    
    def get_embeddings(self, namespace: str = "default") -> Dict[str, np.ndarray]:
        """
        Récupère tous les embeddings d'un espace de noms.
        
        Args:
            namespace (str, optional): Espace de noms. Defaults to "default".
            
        Returns:
            Dict[str, np.ndarray]: Dictionnaire des embeddings
        """
        if namespace not in self.embeddings:
            return {}
        
        return self.embeddings[namespace]
    
    def get_document_by_metadata(self, metadata_key: str, metadata_value: Any, namespace: str = "default") -> List[Dict[str, Any]]:
        """
        Récupère les documents par une valeur de métadonnée.
        
        Args:
            metadata_key (str): Clé de métadonnée
            metadata_value (Any): Valeur de métadonnée
            namespace (str, optional): Espace de noms. Defaults to "default".
            
        Returns:
            List[Dict[str, Any]]: Liste des documents correspondants
        """
        if namespace not in self.documents:
            return []
        
        matching_docs = []
        
        for doc in self.documents[namespace].values():
            if "metadata" in doc and metadata_key in doc["metadata"] and doc["metadata"][metadata_key] == metadata_value:
                matching_docs.append(doc)
        
        return matching_docs


@lru_cache()
def get_document_store() -> DocumentStore:
    """
    Obtient l'instance unique du stockage de documents.
    
    Cette fonction est utilisée comme une dépendance dans FastAPI.
    
    Returns:
        DocumentStore: Instance unique du stockage de documents
    """
    return DocumentStore() 