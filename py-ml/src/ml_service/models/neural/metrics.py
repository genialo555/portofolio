from dataclasses import dataclass
from typing import Dict, List
import torch
import numpy as np

@dataclass
class ComponentMetrics:
    """Métriques pour chaque composant du système."""
    latency: float
    memory_usage: float
    accuracy: float
    confidence: float

class MetricsTracker:
    """Système de suivi des métriques pour l'agent neural."""
    
    def __init__(self):
        self.metrics_history: Dict[str, List[float]] = {
            'deep_latency': [],
            'fast_latency': [],
            'rag_latency': [],
            'cache_hit_rate': [],
            'fusion_quality': []
        }
        
    def update_component_metrics(self, 
                               component: str, 
                               metrics: ComponentMetrics):
        """Met à jour les métriques d'un composant."""
        for key, value in metrics.__dict__.items():
            metric_key = f"{component}_{key}"
            if metric_key not in self.metrics_history:
                self.metrics_history[metric_key] = []
            self.metrics_history[metric_key].append(value)
            
    def get_component_summary(self, component: str) -> Dict[str, float]:
        """Résume les métriques d'un composant."""
        summary = {}
        for key, values in self.metrics_history.items():
            if key.startswith(component):
                summary[key] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values)
                }
        return summary 