import torch
import torch.nn as nn
from typing import Dict, List
import numpy as np

class AdaptiveWeights(nn.Module):
    """Système d'adaptation dynamique des poids basé sur les performances."""
    
    def __init__(self, num_components: int = 3, learning_rate: float = 0.01):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(num_components) / num_components)
        self.performance_history = []
        self.lr = learning_rate
        
    def update_weights(self, metrics: Dict[str, float]):
        """Ajuste les poids en fonction des performances."""
        # Calcul des scores de performance pour chaque composant
        performance_scores = torch.tensor([
            1.0 / (metrics['deep_latency'] + 1e-6),
            1.0 / (metrics['fast_latency'] + 1e-6),
            metrics['rag_accuracy']
        ])
        
        # Mise à jour des poids avec un gradient simple
        with torch.no_grad():
            self.weights.data += self.lr * (performance_scores - self.weights)
            self.weights.data = torch.softmax(self.weights.data, dim=0)
            
        self.performance_history.append(self.weights.detach().cpu().numpy())
        
    def get_current_weights(self) -> torch.Tensor:
        return torch.softmax(self.weights, dim=0) 