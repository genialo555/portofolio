"""
Module d'extraction de connaissances.

Ce module fournit des outils pour extraire des connaissances structurées à partir de textes
non structurés et les transformer en triplets (sujet, prédicat, objet) pour le graphe
de connaissances.
"""

import os
import re
import time
import logging
import json
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from functools import lru_cache
import numpy as np

from .knowledge_base import KnowledgeBase, get_knowledge_base
from ..config import settings

logger = logging.getLogger("ml_api.kag.extractor")

class KnowledgeExtractor:
    """
    Classe qui extrait des connaissances structurées à partir de textes.
    
    Cette classe utilise une approche basée sur des règles ou des modèles de ML
    pour extraire des triplets (sujet, prédicat, objet) à partir de textes.
    """
    
    def __init__(self, knowledge_base: Optional[KnowledgeBase] = None, 
                 use_llm: bool = False, llm_model_name: str = "teacher"):
        """
        Initialise l'extracteur de connaissances.
        
        Args:
            knowledge_base: Base de connaissances pour stocker les triplets
            use_llm: Utiliser un LLM pour l'extraction (plus précis mais plus lent)
            llm_model_name: Nom du modèle LLM à utiliser si use_llm=True
        """
        self.knowledge_base = knowledge_base or get_knowledge_base()
        self.use_llm = use_llm
        self.llm_model_name = llm_model_name
        
        # Métriques
        self.metrics = {
            "total_extraction_time": 0,
            "extraction_count": 0,
            "triplets_extracted": 0,
            "successful_extractions": 0,
            "failed_extractions": 0,
        }
        
        # Règles d'extraction simples basées sur des patterns
        self.subject_patterns = [
            r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",  # Noms propres
            r"(The\s+[A-Z][a-z]+(?:\s+[a-z]+)*)",  # "The X"
            r"((?:This|That|These|Those)\s+[a-z]+(?:\s+[a-z]+)*)",  # Démonstratives
        ]
        
        self.predicate_patterns = [
            r"(is|are|was|were)\s+(a|an|the)",  # Est un/une
            r"(has|have|had)",  # A/ont
            r"(contains|contain|contained)",  # Contient
            r"(consists of|consist of|consisted of)",  # Consiste en
            r"(involves|involve|involved)",  # Implique
            r"(causes|cause|caused)",  # Cause
            r"(leads to|lead to|led to)",  # Mène à
            r"(results in|result in|resulted in)",  # Résulte en
            r"(depends on|depend on|depended on)",  # Dépend de
            r"(relates to|relate to|related to)",  # Est lié à
        ]
        
        self.object_patterns = [
            r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",  # Noms propres
            r"((?:a|an|the)\s+[a-z]+(?:\s+[a-z]+)*)",  # "a/an/the X"
            r"([a-z]+(?:\s+[a-z]+)*)",  # Mots quelconques
        ]
        
        logger.info(f"KnowledgeExtractor initialisé avec le modèle {llm_model_name}")
    
    def extract_from_text(self, text: str, source: Optional[str] = None,
                         confidence_threshold: float = 0.5) -> List[Dict[str, Any]]:
        """
        Extrait des connaissances à partir d'un texte.
        
        Args:
            text: Texte à analyser
            source: Source du texte (pour la provenance)
            confidence_threshold: Seuil de confiance pour les triplets
        
        Returns:
            Liste des triplets extraits avec métadonnées
        """
        start_time = time.time()
        self.metrics["extraction_count"] += 1
        
        try:
            # Choisir la méthode d'extraction
            if self.use_llm:
                triplets = self._extract_with_llm(text)
            else:
                triplets = self._extract_with_rules(text)
            
            # Filtrer par confiance
            triplets = [t for t in triplets if t.get("confidence", 0) >= confidence_threshold]
            
            # Ajouter la source si fournie
            if source:
                for triplet in triplets:
                    if "metadata" not in triplet:
                        triplet["metadata"] = {}
                    triplet["metadata"]["source"] = source
            
            # Mettre à jour les métriques
            extraction_time = time.time() - start_time
            self.metrics["total_extraction_time"] += extraction_time
            self.metrics["triplets_extracted"] += len(triplets)
            self.metrics["successful_extractions"] += 1
            
            logger.debug(f"Extraction terminée en {extraction_time:.3f}s, {len(triplets)} triplets extraits")
            
            return triplets
        
        except Exception as e:
            logger.error(f"Erreur lors de l'extraction de connaissances: {str(e)}")
            self.metrics["failed_extractions"] += 1
            return []
    
    def extract_and_store(self, text: str, source: Optional[str] = None,
                        confidence_threshold: float = 0.5) -> int:
        """
        Extrait des connaissances et les stocke dans la base de connaissances.
        
        Args:
            text: Texte à analyser
            source: Source du texte (pour la provenance)
            confidence_threshold: Seuil de confiance pour les triplets
        
        Returns:
            Nombre de triplets extraits et stockés
        """
        # Extraire les triplets
        triplets = self.extract_from_text(text, source, confidence_threshold)
        
        # Stocker les triplets dans la base de connaissances
        stored_count = 0
        for triplet in triplets:
            try:
                subject = triplet["subject"]
                predicate = triplet["predicate"]
                obj = triplet["object"]
                metadata = triplet.get("metadata", {})
                
                # Ajouter des métadonnées supplémentaires
                metadata["confidence"] = triplet.get("confidence", 0.5)
                metadata["extracted_at"] = time.time()
                if source:
                    metadata["source"] = source
                
                # Stocker le triplet
                self.knowledge_base.add_knowledge(subject, predicate, obj, metadata=metadata)
                stored_count += 1
                
            except Exception as e:
                logger.error(f"Erreur lors du stockage du triplet: {str(e)}")
        
        logger.info(f"Stockage terminé: {stored_count}/{len(triplets)} triplets stockés")
        
        return stored_count
    
    def _extract_with_rules(self, text: str) -> List[Dict[str, Any]]:
        """
        Extrait des triplets à l'aide de règles basées sur des expressions régulières.
        
        Args:
            text: Texte à analyser
        
        Returns:
            Liste des triplets extraits
        """
        triplets = []
        
        # Diviser le texte en phrases
        sentences = re.split(r'[.!?]', text)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # Extraire les sujets potentiels
            subjects = []
            for pattern in self.subject_patterns:
                matches = re.finditer(pattern, sentence)
                for match in matches:
                    subjects.append({
                        "text": match.group(1),
                        "start": match.start(1),
                        "end": match.end(1)
                    })
            
            # Extraire les prédicats potentiels
            predicates = []
            for pattern in self.predicate_patterns:
                matches = re.finditer(pattern, sentence)
                for match in matches:
                    predicates.append({
                        "text": match.group(0),
                        "start": match.start(0),
                        "end": match.end(0)
                    })
            
            # Extraire les objets potentiels
            objects = []
            for pattern in self.object_patterns:
                matches = re.finditer(pattern, sentence)
                for match in matches:
                    objects.append({
                        "text": match.group(1),
                        "start": match.start(1),
                        "end": match.end(1)
                    })
            
            # Former des triplets en fonction des positions
            for subject in subjects:
                for predicate in predicates:
                    if predicate["start"] > subject["end"]:  # Le prédicat est après le sujet
                        for obj in objects:
                            if obj["start"] > predicate["end"]:  # L'objet est après le prédicat
                                # Calculer un score de confiance simple
                                confidence = 0.5
                                
                                # Augmenter la confiance pour les sujets et objets clairs
                                if re.match(r'^[A-Z]', subject["text"]):
                                    confidence += 0.1
                                if re.match(r'^[A-Z]', obj["text"]):
                                    confidence += 0.1
                                
                                # Vérifier la cohérence de la phrase
                                if predicate["start"] - subject["end"] <= 5 and obj["start"] - predicate["end"] <= 5:
                                    confidence += 0.2
                                
                                triplets.append({
                                    "subject": subject["text"],
                                    "predicate": predicate["text"],
                                    "object": obj["text"],
                                    "confidence": min(1.0, confidence),
                                    "sentence": sentence
                                })
        
        return triplets
    
    def _extract_with_llm(self, text: str) -> List[Dict[str, Any]]:
        """
        Extrait des triplets à l'aide d'un modèle de langage.
        
        Args:
            text: Texte à analyser
        
        Returns:
            Liste des triplets extraits
        """
        # Note: cette méthode nécessiterait l'intégration d'un modèle de LLM
        # Pour l'instant, nous retournons une liste vide et log un avertissement
        logger.warning("Extraction avec LLM non implémentée, utilisation des règles à la place")
        return self._extract_with_rules(text)
    
    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        Extrait uniquement les entités du texte.
        
        Args:
            text: Texte à analyser
        
        Returns:
            Liste des entités extraites
        """
        entities = []
        
        # Extraire les entités nommées
        for pattern in self.subject_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                entity_text = match.group(1)
                
                # Déterminer le type d'entité (simpliste)
                entity_type = "unknown"
                if re.match(r'^[A-Z][a-z]+\s+[A-Z][a-z]+$', entity_text):
                    entity_type = "person"
                elif re.match(r'^The\s+[A-Z][a-z]+(?:\s+of\s+.+)?$', entity_text):
                    entity_type = "organization"
                
                entities.append({
                    "text": entity_text,
                    "type": entity_type,
                    "start": match.start(1),
                    "end": match.end(1)
                })
        
        return entities
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Récupère les métriques de l'extracteur.
        
        Returns:
            Dictionnaire des métriques
        """
        return {
            "avg_extraction_time": self.metrics["total_extraction_time"] / max(1, self.metrics["extraction_count"]),
            "triplets_per_extraction": self.metrics["triplets_extracted"] / max(1, self.metrics["extraction_count"]),
            "success_rate": self.metrics["successful_extractions"] / max(1, self.metrics["extraction_count"]),
            "total_triplets_extracted": self.metrics["triplets_extracted"]
        }


@lru_cache()
def get_knowledge_extractor() -> KnowledgeExtractor:
    """
    Fonction pour obtenir une instance singleton de l'extracteur de connaissances.
    
    Returns:
        Instance de l'extracteur de connaissances
    """
    return KnowledgeExtractor() 