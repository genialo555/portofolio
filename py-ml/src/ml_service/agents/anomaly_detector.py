from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import numpy as np
from sklearn.ensemble import IsolationForest
import torch
import torch.nn as nn

@dataclass
class AnomalyReport:
    has_anomaly: bool
    confidence: float
    details: Dict[str, Any]
    severity: str

class AnomalyDetector:
    """Détecteur d'anomalies avec plusieurs méthodes de détection."""
    
    def __init__(self, detector_type: str):
        self.detector_type = detector_type
        self.isolation_forest = IsolationForest(contamination=0.1)
        self.history: List[Dict[str, Any]] = []
        self.error_patterns: Dict[str, int] = {}
        
        # Modèle neuronal pour la détection d'anomalies
        self.neural_detector = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
    def check(self, data: Dict[str, Any]) -> AnomalyReport:
        """Vérifie les anomalies avec plusieurs méthodes."""
        
        # Conversion des données en features
        features = self.extract_features(data)
        
        # Détection statistique
        statistical_score = self.isolation_forest.predict([features])[0]
        
        # Détection par pattern
        pattern_score = self.check_patterns(data)
        
        # Détection neuronale
        neural_score = self.neural_detector(
            torch.tensor(features, dtype=torch.float32)
        ).item()
        
        # Combinaison des scores
        combined_score = np.mean([
            statistical_score,
            pattern_score,
            neural_score
        ])
        
        return AnomalyReport(
            has_anomaly=combined_score < 0.5,
            confidence=abs(0.5 - combined_score) * 2,
            details=self.generate_report(data, combined_score),
            severity=self.determine_severity(combined_score)
        ) 