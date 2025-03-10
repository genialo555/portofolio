"""
Module d'intégration hybride RAG-KAG.

Ce module fournit des utilitaires et des classes pour combiner les approches
RAG (Retrieval Augmented Generation) et KAG (Knowledge Augmented Generation)
afin d'améliorer la qualité des réponses générées.
"""

from .hybrid_generator import HybridGenerator, get_hybrid_generator
from .fusion import KnowledgeFusion, get_knowledge_fusion

__all__ = [
    'HybridGenerator',
    'get_hybrid_generator',
    'KnowledgeFusion',
    'get_knowledge_fusion'
] 