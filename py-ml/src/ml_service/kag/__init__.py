"""
Module KAG (Knowledge Augmented Generation).

Ce module fournit les fonctionnalités nécessaires pour enrichir les générations de texte
en utilisant un graphe de connaissances structuré.
"""

from .knowledge_graph import KnowledgeGraph, get_knowledge_graph
from .knowledge_base import KnowledgeBase, get_knowledge_base
from .extractor import KnowledgeExtractor, get_knowledge_extractor

__all__ = [
    'KnowledgeGraph',
    'get_knowledge_graph',
    'KnowledgeBase',
    'get_knowledge_base', 
    'KnowledgeExtractor',
    'get_knowledge_extractor'
] 