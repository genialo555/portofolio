import torch
import torch.nn as nn
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import faiss
import numpy as np
from transformers import DPRQuestionEncoder, DPRContextEncoder

@dataclass
class RetrievedDocument:
    """Structure pour stocker les documents récupérés."""
    content: str
    score: float
    embedding: np.ndarray
    metadata: Dict

class DocumentStore:
    """Stockage et indexation des documents avec FAISS."""
    
    def __init__(self, embedding_dim: int):
        self.index = faiss.IndexFlatL2(embedding_dim)
        self.documents: List[Dict] = []
        self.embeddings: List[np.ndarray] = []
        
    def add_documents(self, documents: List[str], embeddings: np.ndarray):
        """Ajoute des documents et leurs embeddings à l'index."""
        self.index.add(embeddings)
        self.embeddings.extend(embeddings)
        self.documents.extend([{"content": doc} for doc in documents])
        
    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[RetrievedDocument]:
        """Recherche les documents les plus pertinents."""
        scores, indices = self.index.search(query_embedding.reshape(1, -1), k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:  # Vérifie si l'index est valide
                doc = self.documents[idx]
                results.append(RetrievedDocument(
                    content=doc["content"],
                    score=float(score),
                    embedding=self.embeddings[idx],
                    metadata=doc
                ))
        return results


class RAGModel(nn.Module):
    """Modèle RAG combinant recherche et génération."""
    
    def __init__(self,
                question_encoder: nn.Module,
                context_encoder: nn.Module,
                document_store: DocumentStore,
                generator: nn.Module):
        super().__init__()
        self.question_encoder = question_encoder
        self.context_encoder = context_encoder
        self.document_store = document_store
        self.generator = generator
        
        # Attention pour combiner les documents récupérés
        self.retrieval_attention = nn.MultiheadAttention(
            embed_dim=768,  # Dimension standard des embeddings
            num_heads=8,
            dropout=0.1
        )
        
    def retrieve(self, query: str, k: int = 5) -> List[RetrievedDocument]:
        """Récupère les documents pertinents."""
        query_embedding = self.question_encoder(query)
        return self.document_store.search(query_embedding, k)
        
    def forward(self, 
                query: str,
                retrieved_docs: Optional[List[RetrievedDocument]] = None
                ) -> Tuple[torch.Tensor, List[RetrievedDocument]]:
        """
        Forward pass du modèle RAG.
        1. Encode la requête
        2. Récupère les documents pertinents
        3. Combine les informations
        4. Génère la réponse
        """
        # Encodage de la requête
        query_embedding = self.question_encoder(query)
        
        # Récupération des documents si non fournis
        if retrieved_docs is None:
            retrieved_docs = self.retrieve(query)
            
        # Encodage des documents récupérés
        doc_embeddings = torch.stack([
            self.context_encoder(doc.content)
            for doc in retrieved_docs
        ])
        
        # Attention sur les documents récupérés
        context_vector, attention_weights = self.retrieval_attention(
            query_embedding.unsqueeze(0),
            doc_embeddings,
            doc_embeddings
        )
        
        # Génération de la réponse
        generator_output = self.generator(
            query_embedding=query_embedding,
            context_vector=context_vector,
            retrieved_docs=[doc.content for doc in retrieved_docs]
        )
        
        return generator_output, retrieved_docs


class RAGGenerator(nn.Module):
    """Générateur pour le modèle RAG."""
    
    def __init__(self, 
                hidden_size: int = 768,
                num_layers: int = 4):
        super().__init__()
        
        self.query_projection = nn.Linear(hidden_size, hidden_size)
        self.context_projection = nn.Linear(hidden_size, hidden_size)
        
        self.decoder = nn.TransformerDecoder(
            decoder_layer=nn.TransformerDecoderLayer(
                d_model=hidden_size,
                nhead=8,
                dim_feedforward=hidden_size * 4,
                dropout=0.1,
                activation='gelu'
            ),
            num_layers=num_layers
        )
        
        self.output_projection = nn.Linear(hidden_size, hidden_size)
        
    def forward(self,
               query_embedding: torch.Tensor,
               context_vector: torch.Tensor,
               retrieved_docs: List[str]) -> torch.Tensor:
        """Génère une réponse basée sur la requête et le contexte."""
        query_hidden = self.query_projection(query_embedding)
        context_hidden = self.context_projection(context_vector)
        
        # Décodage
        decoder_output = self.decoder(
            query_hidden.unsqueeze(0),
            context_hidden
        )
        
        return self.output_projection(decoder_output) 