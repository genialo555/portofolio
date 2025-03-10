"""
Module de gestion de la base de connaissances.

Ce module fournit une couche d'abstraction pour interagir avec le graphe de connaissances,
en offrant des fonctionnalités avancées comme la recherche sémantique, les requêtes en langage
naturel et la génération augmentée par connaissances.
"""

import os
import time
import logging
import json
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from functools import lru_cache
import numpy as np

from .knowledge_graph import KnowledgeGraph, get_knowledge_graph
from ..rag.vectorizer import Vectorizer, get_vectorizer
from ..config import settings

logger = logging.getLogger("ml_api.kag.knowledge_base")

class KnowledgeBase:
    """
    Classe qui gère l'accès à la base de connaissances.
    
    Cette classe s'appuie sur un graphe de connaissances et ajoute des fonctionnalités
    comme la recherche sémantique et l'extraction d'informations contextuelles.
    """
    
    def __init__(self, graph_name: str = "default", data_dir: Optional[str] = None,
                 vectorizer: Optional[Vectorizer] = None):
        """
        Initialise la base de connaissances.
        
        Args:
            graph_name: Nom du graphe de connaissances à utiliser
            data_dir: Répertoire pour stocker les données
            vectorizer: Vectoriseur pour les recherches sémantiques
        """
        self.graph_name = graph_name
        self.data_dir = data_dir or os.path.join(str(settings.DATA_PATH), "kag")
        
        # Créer le répertoire de données s'il n'existe pas
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Obtenir le graphe de connaissances
        self.graph = get_knowledge_graph(name=graph_name)
        
        # Obtenir ou créer le vectoriseur
        self.vectorizer = vectorizer or get_vectorizer()
        
        # Métriques
        self.metrics = {
            "total_query_time": 0,
            "queries_count": 0,
            "semantic_searches": 0,
            "successful_queries": 0,
            "failed_queries": 0,
        }
        
        # Caches internes
        self._node_embeddings = {}  # Cache des embeddings des nœuds
        
        logger.info(f"KnowledgeBase '{graph_name}' initialisée avec {len(self.graph)} nœuds")
    
    def add_knowledge(self, subject: str, predicate: str, obj: str,
                     metadata: Optional[Dict[str, Any]] = None) -> Tuple[str, str]:
        """
        Ajoute une connaissance (triplet) à la base.
        
        Args:
            subject: Sujet (entité source)
            predicate: Prédicat (relation)
            obj: Objet (entité cible)
            metadata: Métadonnées supplémentaires
        
        Returns:
            Tuple des identifiants (sujet, objet)
        """
        metadata = metadata or {}
        subject_attrs = {"type": "entity", "label": subject, **metadata.get("subject", {})}
        obj_attrs = {"type": "entity", "label": obj, **metadata.get("object", {})}
        edge_attrs = {"weight": 1.0, **metadata.get("relation", {})}
        
        # Ajouter la date si non présente
        if "created_at" not in edge_attrs:
            edge_attrs["created_at"] = time.time()
        
        # Ajouter le triplet au graphe
        subject_id, obj_id = self.graph.add_triplet(
            subject, predicate, obj,
            subject_attrs=subject_attrs,
            obj_attrs=obj_attrs,
            edge_attrs=edge_attrs
        )
        
        # Générer et stocker les embeddings
        self._update_node_embedding(subject_id)
        self._update_node_embedding(obj_id)
        
        # Sauvegarder le graphe
        self.graph.save()
        
        logger.debug(f"Connaissance ajoutée: {subject} --[{predicate}]--> {obj}")
        
        return subject_id, obj_id
    
    def add_entity(self, name: str, entity_type: str, 
                  properties: Optional[Dict[str, Any]] = None) -> str:
        """
        Ajoute une entité à la base de connaissances.
        
        Args:
            name: Nom de l'entité
            entity_type: Type d'entité
            properties: Propriétés supplémentaires
        
        Returns:
            Identifiant du nœud créé
        """
        properties = properties or {}
        attributes = {
            "type": "entity",
            "entity_type": entity_type,
            "label": name,
            **properties
        }
        
        # Ajouter le nœud au graphe
        node_id = self.graph.add_node(**attributes)
        
        # Générer et stocker l'embedding
        self._update_node_embedding(node_id)
        
        # Sauvegarder le graphe
        self.graph.save()
        
        logger.debug(f"Entité ajoutée: {name} (type: {entity_type})")
        
        return node_id
    
    def add_relation(self, source_id: str, target_id: str, predicate: str,
                    properties: Optional[Dict[str, Any]] = None) -> bool:
        """
        Ajoute une relation entre deux entités existantes.
        
        Args:
            source_id: Identifiant du nœud source
            target_id: Identifiant du nœud cible
            predicate: Type de relation
            properties: Propriétés supplémentaires
        
        Returns:
            True si la relation a été ajoutée, False sinon
        """
        properties = properties or {}
        
        try:
            self.graph.add_edge(source_id, target_id, predicate, **properties)
            self.graph.save()
            logger.debug(f"Relation ajoutée: {source_id} --[{predicate}]--> {target_id}")
            return True
        except Exception as e:
            logger.error(f"Erreur lors de l'ajout de la relation: {str(e)}")
            return False
    
    def search_entities(self, query: str, entity_type: Optional[str] = None, 
                      limit: int = 10) -> List[Dict[str, Any]]:
        """
        Recherche des entités dans la base de connaissances par similarité sémantique.
        
        Args:
            query: Requête de recherche
            entity_type: Type d'entité à rechercher (optionnel)
            limit: Nombre maximum de résultats
        
        Returns:
            Liste des entités correspondantes
        """
        start_time = time.time()
        self.metrics["semantic_searches"] += 1
        
        try:
            # Vectoriser la requête
            query_embedding = self.vectorizer.embed_query(query)
            
            # Filtrer les nœuds par type d'entité si spécifié
            filtered_nodes = []
            for node_id, data in self.graph.graph.nodes(data=True):
                if data.get("type") == "entity" and (entity_type is None or data.get("entity_type") == entity_type):
                    filtered_nodes.append((node_id, data))
            
            # Calculer les similarités
            similarities = []
            for node_id, data in filtered_nodes:
                node_embedding = self._get_node_embedding(node_id)
                if node_embedding is not None:
                    similarity = self._cosine_similarity(query_embedding, node_embedding)
                    similarities.append((node_id, data, similarity))
            
            # Trier par similarité décroissante
            similarities.sort(key=lambda x: x[2], reverse=True)
            
            # Construire les résultats
            results = []
            for node_id, data, similarity in similarities[:limit]:
                results.append({
                    "id": node_id,
                    "score": float(similarity),
                    **data
                })
            
            # Mettre à jour les métriques
            query_time = time.time() - start_time
            self.metrics["total_query_time"] += query_time
            self.metrics["queries_count"] += 1
            self.metrics["successful_queries"] += 1
            
            logger.debug(f"Recherche d'entités pour '{query}' terminée en {query_time:.3f}s, {len(results)} résultats")
            
            return results
        
        except Exception as e:
            logger.error(f"Erreur lors de la recherche d'entités: {str(e)}")
            self.metrics["failed_queries"] += 1
            return []
    
    def get_entity_context(self, entity_id: str, depth: int = 1) -> Dict[str, Any]:
        """
        Récupère le contexte d'une entité (nœud voisin et relations).
        
        Args:
            entity_id: Identifiant de l'entité
            depth: Profondeur du contexte (nombre de sauts)
        
        Returns:
            Contexte de l'entité
        """
        entity = self.graph.get_node(entity_id)
        if not entity:
            return {"entity": None, "context": []}
        
        # Obtenir les relations sortantes et entrantes
        context = self.graph.query("neighbors", node_id=entity_id, direction="both", depth=depth)
        
        return {
            "entity": entity,
            "context": context
        }
    
    def query_knowledge(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Interroge la base de connaissances avec une requête en langage naturel.
        
        Args:
            query: Requête en langage naturel
            limit: Nombre maximum de résultats
        
        Returns:
            Résultats de la requête
        """
        start_time = time.time()
        
        try:
            # Rechercher les entités pertinentes
            entities = self.search_entities(query, limit=limit)
            
            # Récupérer le contexte pour chaque entité
            results = []
            for entity in entities:
                context = self.get_entity_context(entity["id"], depth=1)
                results.append({
                    "entity": entity,
                    "context": context["context"]
                })
            
            # Mettre à jour les métriques
            query_time = time.time() - start_time
            self.metrics["total_query_time"] += query_time
            self.metrics["queries_count"] += 1
            self.metrics["successful_queries"] += 1
            
            logger.debug(f"Requête de connaissance pour '{query}' terminée en {query_time:.3f}s, {len(results)} résultats")
            
            return results
        
        except Exception as e:
            logger.error(f"Erreur lors de la requête de connaissance: {str(e)}")
            self.metrics["failed_queries"] += 1
            return []
    
    def get_knowledge_for_query(self, query: str, limit: int = 5) -> Dict[str, Any]:
        """
        Récupère les connaissances pertinentes pour une requête.
        Cette fonction est utilisée par le système KAG pour enrichir la génération.
        
        Args:
            query: Requête utilisateur
            limit: Nombre maximum d'entités à récupérer
        
        Returns:
            Connaissances pertinentes pour la requête
        """
        # Rechercher les entités pertinentes
        entities = self.search_entities(query, limit=limit)
        
        # Récupérer les relations pour chaque entité
        facts = []
        entities_info = []
        
        for entity in entities:
            entity_id = entity["id"]
            context = self.get_entity_context(entity_id, depth=1)
            
            # Ajouter l'entité
            entities_info.append({
                "id": entity_id,
                "label": entity.get("label", ""),
                "type": entity.get("entity_type", ""),
                "score": entity.get("score", 0.0)
            })
            
            # Extraire les faits des relations
            for item in context["context"]:
                node = item.get("node", {})
                edge = item.get("edge", {})
                
                if edge.get("direction") == "out":
                    fact = {
                        "subject": entity.get("label", ""),
                        "predicate": edge.get("predicate", ""),
                        "object": node.get("label", node.get("value", "")),
                        "confidence": float(entity.get("score", 0.0)) * float(edge.get("weight", 1.0))
                    }
                else:
                    fact = {
                        "subject": node.get("label", node.get("value", "")),
                        "predicate": edge.get("predicate", ""),
                        "object": entity.get("label", ""),
                        "confidence": float(entity.get("score", 0.0)) * float(edge.get("weight", 1.0))
                    }
                
                facts.append(fact)
        
        # Trier les faits par confiance décroissante
        facts.sort(key=lambda x: x["confidence"], reverse=True)
        
        return {
            "entities": entities_info,
            "facts": facts[:limit * 3],  # Limiter le nombre de faits
            "query": query
        }
    
    def format_knowledge_for_llm(self, knowledge: Dict[str, Any], format_type: str = "text") -> str:
        """
        Formate les connaissances pour l'injection dans un LLM.
        
        Args:
            knowledge: Connaissances à formater
            format_type: Type de format (text, json, markdown)
        
        Returns:
            Connaissances formatées
        """
        if format_type == "json":
            return json.dumps(knowledge, ensure_ascii=False, indent=2)
        
        elif format_type == "markdown":
            md = f"# Connaissances pertinentes pour: {knowledge['query']}\n\n"
            
            md += "## Entités\n\n"
            for entity in knowledge["entities"]:
                md += f"- **{entity['label']}** (type: {entity['type']}, confiance: {entity['score']:.2f})\n"
            
            md += "\n## Faits\n\n"
            for fact in knowledge["facts"]:
                md += f"- {fact['subject']} *{fact['predicate']}* {fact['object']} (confiance: {fact['confidence']:.2f})\n"
            
            return md
        
        else:  # format_type == "text"
            text = f"Informations pertinentes pour votre requête '{knowledge['query']}':\n\n"
            
            # Extraire les entités
            entities_str = []
            for entity in knowledge["entities"]:
                entities_str.append(f"{entity['label']} ({entity['type']})")
            
            if entities_str:
                text += "Entités pertinentes: " + ", ".join(entities_str) + "\n\n"
            
            # Extraire les faits
            if knowledge["facts"]:
                text += "Faits pertinents:\n"
                for fact in knowledge["facts"]:
                    text += f"- {fact['subject']} {fact['predicate']} {fact['object']}\n"
            
            return text
    
    def verify_statements(self, statements: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """
        Vérifie la validité des affirmations en les comparant avec la base de connaissances.
        
        Args:
            statements: Liste des affirmations à vérifier sous forme de triplets
                        [{"subject": "...", "predicate": "...", "object": "..."}]
        
        Returns:
            Résultats de la vérification
        """
        results = []
        
        for statement in statements:
            subject = statement.get("subject", "")
            predicate = statement.get("predicate", "")
            obj = statement.get("object", "")
            
            # Rechercher des entités similaires pour le sujet et l'objet
            subject_entities = self.search_entities(subject, limit=3)
            object_entities = self.search_entities(obj, limit=3)
            
            # Vérifier l'existence de relations similaires
            verification = {
                "statement": statement,
                "verified": False,
                "confidence": 0.0,
                "evidence": []
            }
            
            for subj_entity in subject_entities:
                for obj_entity in object_entities:
                    # Chercher des relations entre les entités
                    edges = self.graph.get_edges(
                        source=subj_entity["id"],
                        target=obj_entity["id"]
                    )
                    
                    for edge in edges:
                        edge_predicate = edge.get("predicate", "")
                        # Calculer la similarité entre les prédicats
                        pred_sim = self._string_similarity(predicate, edge_predicate)
                        
                        # Calculer la confiance globale
                        confidence = (
                            subj_entity.get("score", 0.0) * 
                            obj_entity.get("score", 0.0) * 
                            pred_sim * 
                            float(edge.get("weight", 1.0))
                        )
                        
                        verification["evidence"].append({
                            "subject": {
                                "id": subj_entity["id"],
                                "label": subj_entity.get("label", ""),
                                "score": subj_entity.get("score", 0.0)
                            },
                            "predicate": edge_predicate,
                            "object": {
                                "id": obj_entity["id"],
                                "label": obj_entity.get("label", ""),
                                "score": obj_entity.get("score", 0.0)
                            },
                            "predicate_similarity": pred_sim,
                            "confidence": confidence
                        })
                        
                        # Mettre à jour la confiance si supérieure
                        if confidence > verification["confidence"]:
                            verification["confidence"] = confidence
                            verification["verified"] = confidence > 0.5  # Seuil de confiance
            
            # Trier les preuves par confiance décroissante
            verification["evidence"].sort(key=lambda x: x["confidence"], reverse=True)
            
            results.append(verification)
        
        return results
    
    def _update_node_embedding(self, node_id: str) -> Optional[np.ndarray]:
        """
        Génère et stocke l'embedding d'un nœud.
        
        Args:
            node_id: Identifiant du nœud
        
        Returns:
            Embedding du nœud ou None en cas d'erreur
        """
        node = self.graph.get_node(node_id)
        if not node:
            return None
        
        try:
            # Construire le texte représentatif du nœud
            text = ""
            if "label" in node:
                text += node["label"] + " "
            if "entity_type" in node:
                text += node["entity_type"] + " "
            if "value" in node:
                text += str(node["value"]) + " "
            if "description" in node:
                text += node["description"] + " "
            
            text = text.strip()
            if not text:
                text = f"Node_{node_id}"
            
            # Générer l'embedding
            embedding = self.vectorizer.embed_document(text)
            
            # Stocker l'embedding
            self._node_embeddings[node_id] = embedding
            
            return embedding
        
        except Exception as e:
            logger.error(f"Erreur lors de la génération de l'embedding pour le nœud {node_id}: {str(e)}")
            return None
    
    def _get_node_embedding(self, node_id: str) -> Optional[np.ndarray]:
        """
        Récupère l'embedding d'un nœud.
        
        Args:
            node_id: Identifiant du nœud
        
        Returns:
            Embedding du nœud ou None s'il n'existe pas
        """
        # Vérifier si l'embedding est déjà en cache
        if node_id in self._node_embeddings:
            return self._node_embeddings[node_id]
        
        # Sinon, le générer
        return self._update_node_embedding(node_id)
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Calcule la similarité cosinus entre deux vecteurs.
        
        Args:
            a: Premier vecteur
            b: Deuxième vecteur
        
        Returns:
            Similarité cosinus entre les vecteurs
        """
        if a is None or b is None:
            return 0.0
        
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return float(np.dot(a, b) / (norm_a * norm_b))
    
    def _string_similarity(self, a: str, b: str) -> float:
        """
        Calcule la similarité simple entre deux chaînes.
        
        Args:
            a: Première chaîne
            b: Deuxième chaîne
        
        Returns:
            Score de similarité entre 0 et 1
        """
        # Pour une similarité plus sophistiquée, utiliser des embeddings
        # Pour cette version simple, on utilise une comparaison de chaînes
        a_lower = a.lower()
        b_lower = b.lower()
        
        if a_lower == b_lower:
            return 1.0
        elif a_lower in b_lower or b_lower in a_lower:
            return 0.8
        else:
            # Ici on pourrait implémenter une mesure de distance comme Levenshtein
            return 0.0
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Récupère les métriques de la base de connaissances.
        
        Returns:
            Dictionnaire des métriques
        """
        graph_metrics = self.graph.get_metrics()
        
        kb_metrics = {
            "avg_query_time": self.metrics["total_query_time"] / max(1, self.metrics["queries_count"]),
            "semantic_searches": self.metrics["semantic_searches"],
            "success_rate": self.metrics["successful_queries"] / max(1, self.metrics["queries_count"] + self.metrics["failed_queries"]),
            "cached_embeddings": len(self._node_embeddings)
        }
        
        return {
            "knowledge_base": kb_metrics,
            "knowledge_graph": graph_metrics
        }
    
    def save(self) -> bool:
        """
        Sauvegarde l'état de la base de connaissances.
        
        Returns:
            True si la sauvegarde a réussi, False sinon
        """
        return self.graph.save()


@lru_cache()
def get_knowledge_base() -> KnowledgeBase:
    """
    Fonction pour obtenir une instance singleton de la base de connaissances.
    
    Returns:
        Instance de la base de connaissances
    """
    return KnowledgeBase() 