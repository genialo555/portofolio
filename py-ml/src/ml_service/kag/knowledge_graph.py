"""
Module de gestion du graphe de connaissances.

Ce module fournit les outils pour créer, manipuler et interroger un graphe de connaissances.
Le graphe est composé de nœuds (entités) et d'arêtes (relations) qui forment des triplets
(sujet, prédicat, objet).
"""

import os
import json
import logging
import networkx as nx
from typing import Dict, List, Any, Optional, Set, Tuple
from functools import lru_cache
from pathlib import Path
import time
import uuid

from ..config import settings

logger = logging.getLogger("ml_api.kag.knowledge_graph")

class KnowledgeGraph:
    """
    Classe qui gère un graphe de connaissances.
    
    Le graphe utilise NetworkX comme moteur sous-jacent et stocke des triplets sous forme
    de (sujet, prédicat, objet) où le sujet et l'objet sont des nœuds, et le prédicat
    est une relation entre eux.
    """
    
    def __init__(self, data_dir: Optional[str] = None, name: str = "default"):
        """
        Initialise un graphe de connaissances.
        
        Args:
            data_dir: Répertoire pour stocker les données du graphe
            name: Nom du graphe pour différencier plusieurs graphes
        """
        self.name = name
        self.data_dir = data_dir or os.path.join(str(settings.DATA_PATH), "kag")
        self.graph_file = os.path.join(self.data_dir, f"{name}_graph.json")
        
        # Métriques
        self.metrics = {
            "nodes_added": 0,
            "edges_added": 0,
            "nodes_removed": 0,
            "edges_removed": 0,
            "total_query_time": 0,
            "queries_count": 0,
        }
        
        # Créer le répertoire de données s'il n'existe pas
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Initialiser le graphe
        self.graph = nx.DiGraph()
        
        # Charger le graphe s'il existe
        self._load_graph()
        
        logger.info(f"KnowledgeGraph '{name}' initialisé avec {self.graph.number_of_nodes()} nœuds et {self.graph.number_of_edges()} arêtes")
    
    def _load_graph(self) -> None:
        """Charge le graphe à partir du fichier."""
        if os.path.exists(self.graph_file):
            try:
                logger.info(f"Chargement du graphe depuis {self.graph_file}")
                with open(self.graph_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Reconstruire le graphe
                for node_id, node_data in data.get("nodes", {}).items():
                    self.graph.add_node(node_id, **node_data)
                
                for edge in data.get("edges", []):
                    self.graph.add_edge(
                        edge["source"],
                        edge["target"],
                        **{k: v for k, v in edge.items() if k not in ["source", "target"]}
                    )
                    
                logger.info(f"Graphe chargé avec succès: {self.graph.number_of_nodes()} nœuds, {self.graph.number_of_edges()} arêtes")
            except Exception as e:
                logger.error(f"Erreur lors du chargement du graphe: {str(e)}")
                # Initialiser un nouveau graphe en cas d'erreur
                self.graph = nx.DiGraph()
    
    def _save_graph(self) -> None:
        """Sauvegarde le graphe dans un fichier."""
        try:
            # Sérialiser le graphe au format JSON
            data = {
                "nodes": {},
                "edges": []
            }
            
            # Sauvegarder les nœuds
            for node_id, node_data in self.graph.nodes(data=True):
                data["nodes"][node_id] = {k: v for k, v in node_data.items()}
                
            # Sauvegarder les arêtes
            for source, target, edge_data in self.graph.edges(data=True):
                edge_info = {"source": source, "target": target}
                edge_info.update(edge_data)
                data["edges"].append(edge_info)
            
            # Écrire dans le fichier
            with open(self.graph_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
                
            logger.info(f"Graphe sauvegardé dans {self.graph_file}")
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde du graphe: {str(e)}")
    
    def add_node(self, node_id: Optional[str] = None, **attributes) -> str:
        """
        Ajoute un nœud au graphe.
        
        Args:
            node_id: Identifiant du nœud (généré automatiquement si None)
            **attributes: Attributs du nœud
        
        Returns:
            L'identifiant du nœud ajouté
        """
        node_id = node_id or f"node_{uuid.uuid4().hex}"
        
        # Vérifier si le nœud existe déjà
        if not self.graph.has_node(node_id):
            self.graph.add_node(node_id, **attributes)
            self.metrics["nodes_added"] += 1
            logger.debug(f"Nœud ajouté: {node_id}")
        else:
            # Mettre à jour les attributs si le nœud existe
            for key, value in attributes.items():
                self.graph.nodes[node_id][key] = value
            logger.debug(f"Nœud mis à jour: {node_id}")
        
        return node_id
    
    def add_edge(self, source: str, target: str, predicate: str, **attributes) -> Tuple[str, str]:
        """
        Ajoute une arête entre deux nœuds.
        
        Args:
            source: Identifiant du nœud source
            target: Identifiant du nœud cible
            predicate: Type de relation
            **attributes: Attributs supplémentaires de l'arête
        
        Returns:
            Tuple des identifiants (source, target)
        """
        # Vérifier si les nœuds existent
        if not self.graph.has_node(source):
            self.add_node(source)
        
        if not self.graph.has_node(target):
            self.add_node(target)
        
        # Ajouter l'arête avec le prédicat comme attribut
        attributes["predicate"] = predicate
        
        # Ajouter une date de création si non spécifiée
        if "created_at" not in attributes:
            attributes["created_at"] = time.time()
        
        self.graph.add_edge(source, target, **attributes)
        self.metrics["edges_added"] += 1
        
        logger.debug(f"Arête ajoutée: {source} --[{predicate}]--> {target}")
        
        return source, target
    
    def add_triplet(self, subject: str, predicate: str, obj: str, 
                   subject_attrs: Dict[str, Any] = None, 
                   obj_attrs: Dict[str, Any] = None, 
                   edge_attrs: Dict[str, Any] = None) -> Tuple[str, str]:
        """
        Ajoute un triplet (sujet, prédicat, objet) au graphe.
        
        Args:
            subject: Identifiant ou valeur du sujet
            predicate: Type de relation
            obj: Identifiant ou valeur de l'objet
            subject_attrs: Attributs du nœud sujet
            obj_attrs: Attributs du nœud objet
            edge_attrs: Attributs de l'arête
        
        Returns:
            Tuple des identifiants (sujet, objet)
        """
        subject_attrs = subject_attrs or {}
        obj_attrs = obj_attrs or {}
        edge_attrs = edge_attrs or {}
        
        # Générer des identifiants si non fournis
        if not subject.startswith("node_"):
            subject_attrs["value"] = subject
            subject = f"node_{uuid.uuid4().hex}"
        
        if not obj.startswith("node_"):
            obj_attrs["value"] = obj
            obj = f"node_{uuid.uuid4().hex}"
        
        # Ajouter les nœuds et l'arête
        self.add_node(subject, **subject_attrs)
        self.add_node(obj, **obj_attrs)
        self.add_edge(subject, obj, predicate, **edge_attrs)
        
        return subject, obj
    
    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """
        Récupère un nœud par son identifiant.
        
        Args:
            node_id: Identifiant du nœud
        
        Returns:
            Attributs du nœud ou None si le nœud n'existe pas
        """
        if self.graph.has_node(node_id):
            return {**self.graph.nodes[node_id], "id": node_id}
        return None
    
    def get_edges(self, source: Optional[str] = None, target: Optional[str] = None, 
                 predicate: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Récupère les arêtes en fonction des critères.
        
        Args:
            source: Identifiant du nœud source (optionnel)
            target: Identifiant du nœud cible (optionnel)
            predicate: Type de relation (optionnel)
        
        Returns:
            Liste des arêtes correspondant aux critères
        """
        edges = []
        
        for s, t, data in self.graph.edges(data=True):
            if (source is None or s == source) and \
               (target is None or t == target) and \
               (predicate is None or data.get("predicate") == predicate):
                edges.append({
                    "source": s,
                    "target": t,
                    **data
                })
        
        return edges
    
    def query(self, query_type: str, **params) -> List[Dict[str, Any]]:
        """
        Exécute une requête sur le graphe.
        
        Args:
            query_type: Type de requête (neighbors, path, subgraph)
            **params: Paramètres spécifiques à la requête
        
        Returns:
            Résultats de la requête
        """
        start_time = time.time()
        results = []
        
        try:
            if query_type == "neighbors":
                node_id = params.get("node_id")
                direction = params.get("direction", "both")  # in, out, both
                depth = params.get("depth", 1)
                
                if not node_id or not self.graph.has_node(node_id):
                    return []
                
                # Récupérer les voisins en fonction de la direction
                if direction == "out" or direction == "both":
                    for successor in self.graph.successors(node_id):
                        edge_data = self.graph.get_edge_data(node_id, successor)
                        results.append({
                            "node": self.get_node(successor),
                            "edge": {**edge_data, "direction": "out", "source": node_id, "target": successor}
                        })
                
                if direction == "in" or direction == "both":
                    for predecessor in self.graph.predecessors(node_id):
                        edge_data = self.graph.get_edge_data(predecessor, node_id)
                        results.append({
                            "node": self.get_node(predecessor),
                            "edge": {**edge_data, "direction": "in", "source": predecessor, "target": node_id}
                        })
                
            elif query_type == "path":
                source = params.get("source")
                target = params.get("target")
                
                if not source or not target or \
                   not self.graph.has_node(source) or not self.graph.has_node(target):
                    return []
                
                try:
                    path = nx.shortest_path(self.graph, source=source, target=target)
                    
                    # Construire le résultat avec les nœuds et les arêtes du chemin
                    for i in range(len(path) - 1):
                        current = path[i]
                        next_node = path[i + 1]
                        edge_data = self.graph.get_edge_data(current, next_node)
                        
                        results.append({
                            "source_node": self.get_node(current),
                            "target_node": self.get_node(next_node),
                            "edge": {**edge_data, "source": current, "target": next_node}
                        })
                except nx.NetworkXNoPath:
                    # Aucun chemin trouvé
                    pass
                
            elif query_type == "subgraph":
                nodes = params.get("nodes", [])
                include_neighbors = params.get("include_neighbors", False)
                
                if not nodes:
                    return []
                
                subgraph_nodes = set(nodes)
                
                # Ajouter les voisins si demandé
                if include_neighbors:
                    for node in nodes:
                        if self.graph.has_node(node):
                            subgraph_nodes.update(self.graph.successors(node))
                            subgraph_nodes.update(self.graph.predecessors(node))
                
                # Créer le sous-graphe
                subgraph = self.graph.subgraph(subgraph_nodes)
                
                # Transformer le sous-graphe en dictionnaire
                for node in subgraph.nodes():
                    results.append(self.get_node(node))
                
            # Ajouter d'autres types de requêtes selon les besoins
            else:
                logger.warning(f"Type de requête inconnu: {query_type}")
        
        except Exception as e:
            logger.error(f"Erreur lors de l'exécution de la requête {query_type}: {str(e)}")
        
        # Mettre à jour les métriques
        query_time = time.time() - start_time
        self.metrics["total_query_time"] += query_time
        self.metrics["queries_count"] += 1
        
        logger.debug(f"Requête {query_type} exécutée en {query_time:.3f}s, {len(results)} résultats")
        
        return results
    
    def search_nodes(self, query: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Recherche des nœuds en fonction des critères.
        
        Args:
            query: Dictionnaire des critères de recherche
        
        Returns:
            Liste des nœuds correspondant aux critères
        """
        results = []
        
        for node_id, data in self.graph.nodes(data=True):
            match = True
            
            for key, value in query.items():
                if key not in data or data[key] != value:
                    match = False
                    break
            
            if match:
                results.append({**data, "id": node_id})
        
        return results
    
    def delete_node(self, node_id: str) -> bool:
        """
        Supprime un nœud du graphe.
        
        Args:
            node_id: Identifiant du nœud à supprimer
        
        Returns:
            True si le nœud a été supprimé, False sinon
        """
        if self.graph.has_node(node_id):
            self.graph.remove_node(node_id)
            self.metrics["nodes_removed"] += 1
            logger.debug(f"Nœud supprimé: {node_id}")
            return True
        return False
    
    def delete_edge(self, source: str, target: str, predicate: Optional[str] = None) -> bool:
        """
        Supprime une arête entre deux nœuds.
        
        Args:
            source: Identifiant du nœud source
            target: Identifiant du nœud cible
            predicate: Type de relation (optionnel)
        
        Returns:
            True si l'arête a été supprimée, False sinon
        """
        if self.graph.has_edge(source, target):
            edge_data = self.graph.get_edge_data(source, target)
            
            # Vérifier le prédicat si spécifié
            if predicate is not None and edge_data.get("predicate") != predicate:
                return False
            
            self.graph.remove_edge(source, target)
            self.metrics["edges_removed"] += 1
            logger.debug(f"Arête supprimée: {source} --> {target}")
            return True
        return False
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Récupère les métriques du graphe.
        
        Returns:
            Dictionnaire des métriques
        """
        # Ajouter les métriques de base du graphe
        base_metrics = {
            "nodes_count": self.graph.number_of_nodes(),
            "edges_count": self.graph.number_of_edges(),
            "avg_query_time": self.metrics["total_query_time"] / max(1, self.metrics["queries_count"])
        }
        
        return {**self.metrics, **base_metrics}
    
    def export_graph(self, format: str = "json") -> Any:
        """
        Exporte le graphe dans un format spécifique.
        
        Args:
            format: Format d'export (json, cytoscape, networkx)
        
        Returns:
            Graphe exporté dans le format spécifié
        """
        if format == "json":
            # Export au format JSON
            data = {
                "nodes": [],
                "edges": []
            }
            
            for node_id, node_data in self.graph.nodes(data=True):
                data["nodes"].append({
                    "id": node_id,
                    **node_data
                })
            
            for source, target, edge_data in self.graph.edges(data=True):
                data["edges"].append({
                    "source": source,
                    "target": target,
                    **edge_data
                })
            
            return data
        
        elif format == "cytoscape":
            # Export au format Cytoscape
            elements = {
                "nodes": [],
                "edges": []
            }
            
            for node_id, node_data in self.graph.nodes(data=True):
                elements["nodes"].append({
                    "data": {
                        "id": node_id,
                        **node_data
                    }
                })
            
            for source, target, edge_data in self.graph.edges(data=True):
                elements["edges"].append({
                    "data": {
                        "id": f"{source}_{target}",
                        "source": source,
                        "target": target,
                        **edge_data
                    }
                })
            
            return elements
        
        elif format == "networkx":
            # Retourner le graphe NetworkX directement
            return self.graph
        
        else:
            logger.warning(f"Format d'export inconnu: {format}")
            return None
    
    def save(self) -> bool:
        """
        Sauvegarde le graphe sur disque.
        
        Returns:
            True si la sauvegarde a réussi, False sinon
        """
        try:
            self._save_graph()
            return True
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde du graphe: {str(e)}")
            return False
    
    def clear(self) -> None:
        """Efface tout le contenu du graphe."""
        self.graph = nx.DiGraph()
        self.metrics = {
            "nodes_added": 0,
            "edges_added": 0,
            "nodes_removed": 0,
            "edges_removed": 0,
            "total_query_time": 0,
            "queries_count": 0,
        }
        logger.info("Graphe effacé")
    
    def __len__(self) -> int:
        """Retourne le nombre de nœuds dans le graphe."""
        return self.graph.number_of_nodes()


@lru_cache()
def get_knowledge_graph(name: str = "default") -> KnowledgeGraph:
    """
    Fonction pour obtenir une instance singleton du graphe de connaissances.
    
    Args:
        name: Nom du graphe
    
    Returns:
        Instance du graphe de connaissances
    """
    return KnowledgeGraph(name=name) 