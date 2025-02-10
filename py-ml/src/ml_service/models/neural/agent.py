import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
from .processors import DeepProcessor, FastProcessor
from .rag import RAGModel, DocumentStore, RAGGenerator
from transformers import DPRQuestionEncoder, DPRContextEncoder
from .cache import LRUCache
from .metrics import MetricsTracker, ComponentMetrics
import time

class NeuralAgent(nn.Module):
    """Agent neuronal combinant RAG, Deep et Fast processing."""
    
    def __init__(self,
                input_dim: int,
                hidden_dims: List[int],
                embedding_dim: int = 768):
        super().__init__()
        
        # Initialisation des processeurs
        self.deep_processor = DeepProcessor(
            input_dim=input_dim,
            hidden_dims=hidden_dims
        )
        
        self.fast_processor = FastProcessor(
            input_dim=hidden_dims[-1],
            hidden_dim=embedding_dim
        )
        
        # Initialisation du RAG
        self.document_store = DocumentStore(embedding_dim=embedding_dim)
        self.question_encoder = DPRQuestionEncoder.from_pretrained('facebook/dpr-question_encoder-single-nq-base')
        self.context_encoder = DPRContextEncoder.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base')
        
        self.generator = RAGGenerator(
            hidden_size=embedding_dim,
            num_layers=4
        )
        
        self.rag = RAGModel(
            question_encoder=self.question_encoder,
            context_encoder=self.context_encoder,
            document_store=self.document_store,
            generator=self.generator
        )
        
        # Fusion des sorties
        self.fusion_layer = nn.Sequential(
            nn.Linear(embedding_dim * 3, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU()
        )
        
        # Ajout du cache
        self.cache = LRUCache(capacity=1000)
        self.metrics = MetricsTracker()
        
        # Poids auto-adaptatifs pour la fusion
        self.fusion_weights = nn.Parameter(torch.ones(3) / 3)  # Initialement égaux
        
    def process_input(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Traitement parallèle avec Deep et Fast processors."""
        deep_output, deep_attention = self.deep_processor(x)
        fast_output = self.fast_processor(x)
        return deep_output, fast_output
        
    def forward(self, 
                x: torch.Tensor,
                query: str,
                context: Optional[List[str]] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass complet de l'agent.
        1. Traitement parallèle (Deep + Fast)
        2. RAG pour la récupération et génération
        3. Fusion des sorties
        """
        start_time = time.time()
        
        # Vérification du cache
        cache_key = f"{query}_{x.shape}"
        cached_result = self.cache.get(cache_key)
        if cached_result is not None:
            return cached_result
            
        # Traitement normal si pas dans le cache
        deep_start = time.time()
        deep_features, fast_features = self.process_input(x)
        deep_latency = time.time() - deep_start
        
        fast_start = time.time()
        fast_features = self.fast_processor(x)
        fast_latency = time.time() - fast_start
        
        rag_start = time.time()
        rag_output, retrieved_docs = self.rag(query=query, retrieved_docs=None)
        rag_latency = time.time() - rag_start
        
        # Fusion avec poids auto-adaptatifs
        weights = F.softmax(self.fusion_weights, dim=0)
        combined_features = torch.cat([
            weights[0] * deep_features,
            weights[1] * fast_features,
            weights[2] * rag_output
        ], dim=-1)
        
        fused_output = self.fusion_layer(combined_features)
        
        # Mise à jour des métriques
        total_latency = time.time() - start_time
        self.metrics.update_component_metrics('deep', ComponentMetrics(
            latency=deep_latency,
            memory_usage=deep_features.element_size() * deep_features.nelement(),
            accuracy=0.0,  # À calculer selon votre cas d'usage
            confidence=F.softmax(deep_features, dim=-1).max().item()
        ))
        
        # Mise en cache du résultat
        result = {
            'output': fused_output,
            'deep_features': deep_features,
            'fast_features': fast_features,
            'rag_output': rag_output,
            'retrieved_docs': retrieved_docs,
            'metrics': {
                'latencies': {
                    'deep': deep_latency,
                    'fast': fast_latency,
                    'rag': rag_latency,
                    'total': total_latency
                },
                'weights': weights.detach().cpu().numpy()
            }
        }
        self.cache.put(cache_key, result)
        
        return result
    
    def train_step(self,
                  x: torch.Tensor,
                  query: str,
                  target: torch.Tensor,
                  optimizer: torch.optim.Optimizer) -> Dict[str, float]:
        """Étape d'entraînement unique."""
        optimizer.zero_grad()
        
        outputs = self.forward(x, query)
        
        # Calcul des pertes pour chaque composant
        deep_loss = F.mse_loss(outputs['deep_features'], target)
        fast_loss = F.mse_loss(outputs['fast_features'], target)
        rag_loss = F.mse_loss(outputs['rag_output'], target)
        
        # Perte totale avec pondération
        total_loss = (0.4 * deep_loss + 
                     0.3 * fast_loss + 
                     0.3 * rag_loss)
        
        total_loss.backward()
        optimizer.step()
        
        return {
            'total_loss': total_loss.item(),
            'deep_loss': deep_loss.item(),
            'fast_loss': fast_loss.item(),
            'rag_loss': rag_loss.item()
        } 