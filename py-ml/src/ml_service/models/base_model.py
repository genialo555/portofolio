from abc import ABC, abstractmethod
from typing import Any, Dict
import numpy as np
from pathlib import Path
import mlflow
from ml_service.config import settings
from ml_service.monitoring.metrics import MetricsTracker

class BaseModel(ABC):
    """Base class for all ML models."""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model: Any = None
        self.metrics = MetricsTracker()
        mlflow.set_tracking_uri(settings.EXPERIMENT_TRACKING_URI)
        
    @abstractmethod
    def train(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Train the model."""
        pass
        
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        pass
        
    def save(self, path: Path = None) -> None:
        """Save model to disk."""
        save_path = path or settings.MODEL_PATH / f"{self.model_name}.pkl"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        mlflow.sklearn.save_model(self.model, save_path) 