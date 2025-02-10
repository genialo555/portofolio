from dataclasses import dataclass
from typing import Dict, Any
import time
import psutil
import logging

@dataclass
class ModelMetrics:
    """Class for tracking model metrics."""
    inference_time: float
    memory_usage: float
    prediction_confidence: float
    model_version: str

class MetricsTracker:
    """Class for tracking model performance metrics."""
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def track_inference(self, start_time: float) -> Dict[str, Any]:
        metrics = {
            'inference_time': time.time() - start_time,
            'memory_usage': psutil.Process().memory_info().rss / 1024 / 1024,  # MB
        }
        self.logger.info(f"Inference metrics: {metrics}")
        return metrics 