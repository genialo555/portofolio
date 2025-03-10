"""
Configuration globale du service ML.

Ce module fournit une configuration centralisée pour tous les composants du service.
"""

import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Dict, Any, Optional, List

# Obtenir le chemin absolu du répertoire racine du projet
ROOT_DIR = Path(__file__).parent.parent.parent.absolute()


class Settings(BaseSettings):
    """
    Configuration globale du service ML.
    
    Cette classe utilise pydantic_settings pour charger les variables d'environnement
    et fournir des valeurs par défaut pour tous les paramètres.
    """
    
    # Chemins des répertoires
    ROOT_PATH: Path = ROOT_DIR
    DATA_PATH: Path = ROOT_DIR / "data"
    MODEL_PATH: Path = ROOT_DIR / "model_cache"
    
    # Configuration de l'API
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    DEBUG: bool = True
    ENVIRONMENT: str = "development"  # development, staging, production
    
    # Configuration des modèles
    DEFAULT_MODEL: str = "teacher"
    MODEL_CACHE_ENABLED: bool = True
    MAX_CONCURRENT_REQUESTS: int = 10
    
    # Configuration RAG
    RAG_ENABLED: bool = True
    RAG_EMBEDDINGS_CACHE_DIR: Path = DATA_PATH / "rag" / "embeddings"
    RAG_DOCUMENTS_DIR: Path = DATA_PATH / "rag" / "documents"
    
    # Configuration KAG
    KAG_ENABLED: bool = True
    KAG_DATA_DIR: Path = DATA_PATH / "kag"
    
    # Configuration Redis
    REDIS_URL: Optional[str] = None
    REDIS_ENABLED: bool = False
    
    # Configuration du logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    
    # Timeouts et ressources
    REQUEST_TIMEOUT: int = 60  # secondes
    MAX_MEMORY_USAGE: int = 8 * 1024  # MB
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"
    )
    
    def __init__(self, **data: Any):
        super().__init__(**data)
        
        # Créer les répertoires nécessaires s'ils n'existent pas
        self.DATA_PATH.mkdir(exist_ok=True, parents=True)
        self.MODEL_PATH.mkdir(exist_ok=True, parents=True)
        self.RAG_EMBEDDINGS_CACHE_DIR.mkdir(exist_ok=True, parents=True)
        self.RAG_DOCUMENTS_DIR.mkdir(exist_ok=True, parents=True)
        self.KAG_DATA_DIR.mkdir(exist_ok=True, parents=True)
        
        # Configurer Redis si activé
        if self.REDIS_URL:
            self.REDIS_ENABLED = True


# Instance singleton des paramètres
settings = Settings() 