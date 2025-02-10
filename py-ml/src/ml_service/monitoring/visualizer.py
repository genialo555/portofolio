import matplotlib.pyplot as plt
from typing import Dict, List
import numpy as np
import seaborn as sns
from IPython.display import clear_output
import pandas as pd

class MetricsVisualizer:
    """Visualiseur de métriques en temps réel."""
    
    def __init__(self, update_interval: int = 10):
        self.update_interval = update_interval
        self.iteration = 0
        
    def plot_metrics(self, metrics_tracker: 'MetricsTracker'):
        """Affiche les graphiques des métriques."""
        self.iteration += 1
        if self.iteration % self.update_interval != 0:
            return
            
        clear_output(wait=True)
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Latences
        latencies = pd.DataFrame({
            'Deep': metrics_tracker.metrics_history['deep_latency'],
            'Fast': metrics_tracker.metrics_history['fast_latency'],
            'RAG': metrics_tracker.metrics_history['rag_latency']
        })
        sns.lineplot(data=latencies, ax=axes[0, 0])
        axes[0, 0].set_title('Latences par composant')
        
        # Taux de hit du cache
        hit_rates = metrics_tracker.metrics_history['cache_hit_rate']
        sns.lineplot(data=hit_rates, ax=axes[0, 1])
        axes[0, 1].set_title('Taux de hit du cache')
        
        # Qualité de fusion
        fusion_quality = metrics_tracker.metrics_history['fusion_quality']
        sns.lineplot(data=fusion_quality, ax=axes[1, 0])
        axes[1, 0].set_title('Qualité de fusion')
        
        # Distribution des poids
        weights_df = pd.DataFrame(metrics_tracker.metrics_history['weights'])
        sns.boxplot(data=weights_df, ax=axes[1, 1])
        axes[1, 1].set_title('Distribution des poids')
        
        plt.tight_layout()
        plt.show() 