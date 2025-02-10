from pathlib import Path
from pydantic_settings import BaseSettings
from ml_service import ROOT_DIR

class Settings(BaseSettings):
    """Application settings."""
    MODEL_PATH: Path = ROOT_DIR / "models"
    DATA_PATH: Path = ROOT_DIR / "data"
    EXPERIMENT_TRACKING_URI: str = "sqlite:///mlflow.db"
    
    class Config:
        env_file = ".env"

settings = Settings() 