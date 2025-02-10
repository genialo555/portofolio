from typing import Dict, Any, Optional
import time
from collections import OrderedDict
import torch

class LRUCache:
    """Cache LRU (Least Recently Used) pour les documents et embeddings."""
    
    def __init__(self, capacity: int = 1000):
        self.cache = OrderedDict()
        self.capacity = capacity
        self.hits = 0
        self.misses = 0
        
    def get(self, key: str) -> Optional[Any]:
        if key in self.cache:
            self.hits += 1
            value = self.cache.pop(key)
            self.cache[key] = value
            return value
        self.misses += 1
        return None
        
    def put(self, key: str, value: Any):
        if key in self.cache:
            self.cache.pop(key)
        elif len(self.cache) >= self.capacity:
            self.cache.popitem(last=False)
        self.cache[key] = value
        
    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0 

class SmartCache(LRUCache):
    """Cache avec préchargement intelligent."""
    
    def __init__(self, capacity: int = 1000, preload_threshold: float = 0.7):
        super().__init__(capacity)
        self.access_patterns = {}
        self.preload_threshold = preload_threshold
        
    def update_pattern(self, key: str, next_key: str):
        """Met à jour les motifs d'accès."""
        if key not in self.access_patterns:
            self.access_patterns[key] = {}
        if next_key not in self.access_patterns[key]:
            self.access_patterns[key][next_key] = 0
        self.access_patterns[key][next_key] += 1
        
    def preload_likely_next(self, key: str, loader_func):
        """Précharge les éléments susceptibles d'être demandés ensuite."""
        if key not in self.access_patterns:
            return
            
        patterns = self.access_patterns[key]
        total = sum(patterns.values())
        
        for next_key, count in patterns.items():
            probability = count / total
            if probability > self.preload_threshold and next_key not in self.cache:
                value = loader_func(next_key)
                self.put(next_key, value) 