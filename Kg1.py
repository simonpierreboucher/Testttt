import json
import uuid
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set, Union, Callable
import logging
from collections import defaultdict, deque
import csv
import networkx as nx
import matplotlib.pyplot as plt
import threading

# Configuration avancée du logger avec gestion des fichiers de log
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Niveau de log plus détaillé

# Création d'un handler pour la console
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(console_formatter)
logger.addHandler(console_handler)

# Création d'un handler pour le fichier de log
file_handler = logging.FileHandler('knowledge_graph.log', encoding='utf-8')
file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)

@dataclass
class Node:
    """
    Représente un nœud dans le graphe de connaissances.
    """
    id: str
    labels: Set[str]
    properties: Dict[str, Any]

@dataclass
class Edge:
    """
    Représente une arête (relation) entre deux nœuds dans le graphe de connaissances.
    """
    id: str
    source: str
    target: str
    type: str
    properties: Dict[str, Any] = field(default_factory=dict)

class KnowledgeGraphError(Exception):
    """Exception de base pour les erreurs du KnowledgeGraph."""
    pass

class NodeNotFoundError(KnowledgeGraphError):
    """Exception levée lorsqu'un nœud n'est pas trouvé."""
    pass

class EdgeNotFoundError(KnowledgeGraphError):
    """Exception levée lorsqu'une arête n'est pas trouvée."""
    pass

class TransactionError(KnowledgeGraphError):
    """Exception levée en cas d'erreur lors d'une transaction."""
    pass

class KnowledgeGraph:
    """
    Classe représentant un graphe de connaissances avec des nœuds et des arêtes.
    """

    def __init__(self):
        """
        Initialise un Knowledge Graph vide avec des index pour une recherche rapide.
        """
        self.nodes: Dict[str, Node] = {}
        self.edges: Dict[str, Edge] = {}
        self.label_index: defaultdict = defaultdict(set)
        self.property_index: defaultdict = defaultdict(lambda: defaultdict(set))
        self.adjacency: defaultdict = defaultdict(set)
        self.lock = threading.Lock()  # Pour la sécurité des threads
        logger.info("Initialisation du KnowledgeGraph vide.")

    def _generate_unique_id(self, prefix: str) -> str:
        """
        Génère un identifiant unique avec un préfixe donné.
        """
        unique_id = f"{prefix}_{uuid.uuid4()}"
        logger.debug(f"Généré ID unique: {unique_id}")
        return unique_id

    def add_node(self, labels: List[str], properties: Dict[str, Any]) -> str:
        """
        Ajoute un nouveau nœud au graphe et retourne son identifiant unique.
        Met à jour les index de labels et de propriétés.
        """
        with self.lock:
            node_id = self._generate_unique_id("node")
            node = Node(id=node_id, labels=set(labels), properties=properties)
            self.nodes[node_id] = node
            logger.info(f"Nœud ajouté: {node_id} avec labels {labels} et propriétés {properties}")

            # Mise à jour de l'index des labels
            for label in labels:
                self.label_index[label].add(node_id)
                logger.debug(f"Indexé le label '{label}' pour le nœud {node_id}.")

            # Mise à jour de l'index des propriétés
            for prop, value in properties.items():
                self.property_index[prop][value].add(node_id)
                logger.debug(f"Indexée la propriété '{prop}: {value}' pour le nœud {node_id}.")

            return node_id

    def add_edge(
        self, 
        source_id: str, 
        target_id: str, 
        rel_type: str, 
        properties: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Ajoute une arête entre deux nœuds existants et retourne son identifiant unique.
        Met à jour l'adjacence pour les deux nœuds.
        """
        with self.lock:
            if source_id not in self.nodes:
                logger.error(f"Le nœud source {source_id} n'existe pas.")
                raise NodeNotFoundError(f"Le nœud source {source_id} n'existe pas.")
            if target_id not in self.nodes:
                logger.error(f"Le nœud cible {target_id} n'existe pas.")
                raise NodeNotFoundError(f"Le nœud cible {target_id} n'existe pas.")
            
            if properties is None:
                properties = {}
            
            edge_id = self._generate_unique_id("edge")
            edge = Edge(id=edge_id, source=source_id, target=target_id, type=rel_type, properties=properties)
            self.edges[edge_id] = edge
            logger.info(f"Arête ajoutée: {edge_id} de {source_id} à {target_id} de type '{rel_type}' avec propriétés {properties}")

            # Mise à jour de l'adjacence
            self.adjacency[source_id].add(edge_id)
            self.adjacency[target_id].add(edge_id)
            logger.debug(f"Mis à jour l'adjacence pour les nœuds {source_id} et {target_id}.")

            return edge_id

    def get_node(self, node_id: str) -> Optional[Node]:
        """
        Récupère un nœud par son identifiant.
        """
        node = self.nodes.get(node_id)
        if node:
            logger.debug(f"Nœud récupéré: {node_id}")
        else:
            logger.warning(f"Tentative de récupération d'un nœud inexistant: {node_id}")
        return node

    def update_node_properties(self, node_id: str, new_properties: Dict[str, Any]) -> None:
        """
        Met à jour les propriétés d'un nœud de manière incrémentale.
        Met à jour les index de propriétés en conséquence.
        """
        with self.lock:
            node = self.get_node(node_id)
            if node is None:
                logger.error(f"Le nœud {node_id} n'existe pas.")
                raise NodeNotFoundError(f"Le nœud {node_id} n'existe pas.")
            
            # Mise à jour des index de propriétés
            for prop, value in new_properties.items():
                # Retirer l'ancien index si la propriété existait
                if prop in node.properties:
                    old_value = node.properties[prop]
                    self.property_index[prop][old_value].discard(node_id)
                    if not self.property_index[prop][old_value]:
                        del self.property_index[prop][old_value]
                    logger.debug(f"Retirée l'ancienne propriété '{prop}: {old_value}' du nœud {node_id}.")
                
                # Ajouter le nouvel index
                self.property_index[prop][value].add(node_id)
                logger.debug(f"Indexée la nouvelle propriété '{prop}: {value}' pour le nœud {node_id}.")

            # Mise à jour des propriétés du nœud
            node.properties.update(new_properties)
            logger.info(f"Propriétés du nœud {node_id} mises à jour avec {new_properties}.")

    def remove_node(self, node_id: str) -> None:
        """
        Supprime un nœud et toutes ses arêtes associées.
        Met à jour les index et l'adjacence en conséquence.
        """
        with self.lock:
            if node_id not in self.nodes:
                logger.error(f"Le nœud {node_id} n'existe pas.")
                raise NodeNotFoundError(f"Le nœud {node_id} n'existe pas.")
            
            # Supprimer les arêtes associées
            related_edges = list(self.adjacency[node_id])
            for edge_id in related_edges:
                self.remove_edge_by_id(edge_id)
            
            # Supprimer les index de labels
            node = self.nodes[node_id]
            for label in node.labels:
                self.label_index[label].discard(node_id)
                if not self.label_index[label]:
                    del self.label_index[label]
                logger.debug(f"Retiré le nœud {node_id} de l'index du label '{label}'.")

            # Supprimer les index de propriétés
            for prop, value in node.properties.items():
                self.property_index[prop][value].discard(node_id)
                if not self.property_index[prop][value]:
                    del self.property_index[prop][value]
                if not self.property_index[prop]:
                    del self.property_index[prop]
                logger.debug(f"Retiré le nœud {node_id} de l'index de la propriété '{prop}: {value}'.")

            # Supprimer le nœud lui-même
            del self.nodes[node_id]
            del self.adjacency[node_id]
            logger.info(f"Nœud {node_id} et ses arêtes associées ont été supprimés.")

    def remove_edge_by_id(self, edge_id: str) -> None:
        """
        Supprime une arête spécifique par son identifiant.
        Met à jour l'adjacence en conséquence.
        """
        with self.lock:
            if edge_id not in self.edges:
                logger.error(f"L'arête {edge_id} n'existe pas.")
                raise EdgeNotFoundError(f"L'arête {edge_id} n'existe pas.")
            
            edge = self.edges[edge_id]
            # Mettre à jour l'adjacence
            self.adjacency[edge.source].discard(edge_id)
            self.adjacency[edge.target].discard(edge_id)
            logger.debug(f"Mis à jour l'adjacence pour les nœuds {edge.source} et {edge.target} en retirant l'arête {edge_id}.")

            # Supprimer l'arête
            del self.edges[edge_id]
            logger.info(f"Arête {edge_id} supprimée.")

    def get_edges_from_node(self, node_id: str) -> List[Edge]:
        """
        Récupère toutes les arêtes sortantes depuis un nœud.
        """
        if node_id not in self.nodes:
            logger.error(f"Le nœud {node_id} n'existe pas.")
            raise NodeNotFoundError(f"Le nœud {node_id} n'existe pas.")
        
        outgoing_edges = [
            self.edges[edge_id] for edge_id in self.adjacency[node_id]
            if self.edges[edge_id].source == node_id
        ]
        logger.debug(f"Récupérées {len(outgoing_edges)} arêtes sortantes depuis le nœud {node_id}.")
        return outgoing_edges

    def get_edges_to_node(self, node_id: str) -> List[Edge]:
        """
        Récupère toutes les arêtes entrantes vers un nœud.
        """
        if node_id not in self.nodes:
            logger.error(f"Le nœud {node_id} n'existe pas.")
            raise NodeNotFoundError(f"Le nœud {node_id} n'existe pas.")
        
        incoming_edges = [
            self.edges[edge_id] for edge_id in self.adjacency[node_id]
            if self.edges[edge_id].target == node_id
        ]
        logger.debug(f"Récupérées {len(incoming_edges)} arêtes entrantes vers le nœud {node_id}.")
        return incoming_edges

    def get_edges_of_node(self, node_id: str) -> List[Edge]:
        """
        Récupère toutes les arêtes connectées à un nœud.
        """
        if node_id not in self.nodes:
            logger.error(f"Le nœud {node_id} n'existe pas.")
            raise NodeNotFoundError(f"Le nœud {node_id} n'existe pas.")
        
        connected_edges = [self.edges[edge_id] for edge_id in self.adjacency[node_id]]
        logger.debug(f"Récupérées {len(connected_edges)} arêtes connectées au nœud {node_id}.")
        return connected_edges

    def remove_edge(self, source_id: str, target_id: str, rel_type: str) -> None:
        """
        Supprime une arête spécifique basée sur la source, la cible et le type.
        """
        with self.lock:
            found_edges = [
                edge_id for edge_id, edge in self.edges.items()
                if edge.source == source_id and edge.target == target_id and edge.type == rel_type
            ]
            if not found_edges:
                logger.error(f"Aucune arête trouvée entre {source_id} et {target_id} de type {rel_type}.")
                raise EdgeNotFoundError(f"Aucune arête trouvée entre {source_id} et {target_id} de type {rel_type}.")
            
            for edge_id in found_edges:
                self.remove_edge_by_id(edge_id)
                logger.info(f"Arête {edge_id} supprimée entre {source_id} et {target_id} de type {rel_type}.")

    def find_nodes_by_label(self, label: str) -> List[Node]:
        """
        Trouve tous les nœuds portant un label donné.
        """
        node_ids = self.label_index.get(label, set())
        nodes = [self.nodes[node_id] for node_id in node_ids]
        logger.info(f"Trouvé {len(nodes)} nœuds avec le label '{label}'.")
        return nodes

    def find_nodes_by_property(self, property_name: str, property_value: Any) -> List[Node]:
        """
        Trouve tous les nœuds ayant une propriété donnée avec une valeur spécifique.
        """
        node_ids = self.property_index.get(property_name, {}).get(property_value, set())
        nodes = [self.nodes[node_id] for node_id in node_ids]
        logger.info(f"Trouvé {len(nodes)} nœuds avec la propriété '{property_name}: {property_value}'.")
        return nodes

    def find_nodes(self, labels: Optional[List[str]] = None, properties: Optional[Dict[str, Any]] = None) -> List[Node]:
        """
        Trouve des nœuds en fonction de labels et/ou de propriétés.
        """
        if not labels and not properties:
            logger.warning("Aucun critère de recherche spécifié. Retour de tous les nœuds.")
            return list(self.nodes.values())
        
        result_ids: Optional[Set[str]] = None

        if labels:
            for label in labels:
                label_ids = self.label_index.get(label, set())
                if result_ids is None:
                    result_ids = label_ids.copy()
                else:
                    result_ids &= label_ids
                logger.debug(f"Filtrage par label '{label}', {len(label_ids)} nœuds trouvés.")

        if properties:
            for prop, value in properties.items():
                prop_ids = self.property_index.get(prop, {}).get(value, set())
                if result_ids is None:
                    result_ids = prop_ids.copy()
                else:
                    result_ids &= prop_ids
                logger.debug(f"Filtrage par propriété '{prop}: {value}', {len(prop_ids)} nœuds trouvés.")

        if result_ids is None:
            result_ids = set()

        nodes = [self.nodes[node_id] for node_id in result_ids]
        logger.info(f"Trouvé {len(nodes)} nœuds avec les critères spécifiés.")
        return nodes

    def update_edge_properties(self, edge_id: str, new_properties: Dict[str, Any]) -> None:
        """
        Met à jour les propriétés d'une arête existante.
        """
        with self.lock:
            if edge_id not in self.edges:
                logger.error(f"L'arête {edge_id} n'existe pas.")
                raise EdgeNotFoundError(f"L'arête {edge_id} n'existe pas.")
            
            edge = self.edges[edge_id]
            edge.properties.update(new_properties)
            logger.info(f"Propriétés de l'arête {edge_id} mises à jour avec {new_properties}.")

    def save_to_json(self, file_path: str) -> None:
        """
        Sauvegarde le graphe complet dans un fichier JSON.
        """
        with self.lock:
            data = {
                "nodes": [
                    {
                        "id": node.id,
                        "labels": list(node.labels),
                        "properties": node.properties
                    }
                    for node in self.nodes.values()
                ],
                "edges": [
                    {
                        "id": edge.id,
                        "source": edge.source,
                        "target": edge.target,
                        "type": edge.type,
                        "properties": edge.properties
                    }
                    for edge in self.edges.values()
                ]
            }
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=4)
                logger.info(f"Graphe sauvegardé dans le fichier {file_path}.")
            except IOError as e:
                logger.error(f"Erreur lors de la sauvegarde du graphe: {e}")
                raise KnowledgeGraphError(f"Erreur lors de la sauvegarde du graphe: {e}")

    def load_from_json(self, file_path: str) -> None:
        """
        Charge un graphe depuis un fichier JSON.
        Réinitialise le graphe actuel.
        """
        with self.lock:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except (IOError, json.JSONDecodeError) as e:
                logger.error(f"Erreur lors du chargement du fichier JSON: {e}")
                raise KnowledgeGraphError(f"Erreur lors du chargement du fichier JSON: {e}")
            
            # Réinitialiser les structures existantes
            self.clear_graph()
            logger.info(f"Chargement du graphe depuis le fichier {file_path}.")

            # Charger les nœuds
            for node_data in data.get("nodes", []):
                node = Node(
                    id=node_data["id"],
                    labels=set(node_data["labels"]),
                    properties=node_data["properties"]
                )
                self.nodes[node.id] = node
                logger.debug(f"Nœud chargé: {node.id}")

                # Mise à jour des index de labels
                for label in node.labels:
                    self.label_index[label].add(node.id)
                    logger.debug(f"Indexé le label '{label}' pour le nœud {node.id}.")

                # Mise à jour des index de propriétés
                for prop, value in node.properties.items():
                    self.property_index[prop][value].add(node.id)
                    logger.debug(f"Indexée la propriété '{prop}: {value}' pour le nœud {node.id}.")

            # Charger les arêtes
            for edge_data in data.get("edges", []):
                edge = Edge(
                    id=edge_data["id"],
                    source=edge_data["source"],
                    target=edge_data["target"],
                    type=edge_data["type"],
                    properties=edge_data.get("properties", {})
                )
                self.edges[edge.id] = edge
                logger.debug(f"Arête chargée: {edge.id}")

                # Mise à jour de l'adjacence
                self.adjacency[edge.source].add(edge.id)
                self.adjacency[edge.target].add(edge.id)
                logger.debug(f"Mis à jour l'adjacence pour les nœuds {edge.source} et {edge.target} avec l'arête {edge.id}.")

            logger.info("Chargement du graphe terminé.")

    def get_all_nodes(self) -> List[Node]:
        """
        Retourne tous les nœuds du graphe.
        """
        logger.debug(f"Récupération de tous les nœuds, total: {len(self.nodes)}.")
        return list(self.nodes.values())

    def get_all_edges(self) -> List[Edge]:
        """
        Retourne toutes les arêtes du graphe.
        """
        logger.debug(f"Récupération de toutes les arêtes, total: {len(self.edges)}.")
        return list(self.edges.values())

    def node_exists(self, node_id: str) -> bool:
        """
        Vérifie si un nœud existe dans le graphe.
        """
        existence = node_id in self.nodes
        logger.debug(f"Existence du nœud {node_id}: {existence}.")
        return existence

    def edge_exists(self, edge_id: str) -> bool:
        """
        Vérifie si une arête spécifique existe dans le graphe.
        """
        existence = edge_id in self.edges
        logger.debug(f"Existence de l'arête {edge_id}: {existence}.")
        return existence

    def merge_properties(self, node_id: str, properties: Dict[str, Any]) -> None:
        """
        Fusionne les propriétés d'un nœud avec un autre dictionnaire.
        Les clés identiques sont écrasées.
        """
        with self.lock:
            self.update_node_properties(node_id, properties)
            logger.info(f"Propriétés du nœud {node_id} fusionnées avec {properties}.")

    def clear_graph(self) -> None:
        """
        Vide complètement le graphe, supprimant tous les nœuds et arêtes.
        """
        with self.lock:
            self.nodes.clear()
            self.edges.clear()
            self.label_index.clear()
            self.property_index.clear()
            self.adjacency.clear()
            logger.info("Le graphe a été vidé complètement.")

    def export_to_csv(self, nodes_file: str, edges_file: str) -> None:
        """
        Exporte les nœuds et les arêtes du graphe dans des fichiers CSV séparés.
        """
        with self.lock:
            try:
                # Exporter les nœuds
                with open(nodes_file, 'w', encoding='utf-8', newline='') as f_nodes:
                    writer = csv.writer(f_nodes)
                    writer.writerow(['id', 'labels', 'properties'])
                    for node in self.nodes.values():
                        writer.writerow([node.id, ",".join(node.labels), json.dumps(node.properties, ensure_ascii=False)])
                logger.info(f"Nœuds exportés dans le fichier {nodes_file}.")

                # Exporter les arêtes
                with open(edges_file, 'w', encoding='utf-8', newline='') as f_edges:
                    writer = csv.writer(f_edges)
                    writer.writerow(['id', 'source', 'target', 'type', 'properties'])
                    for edge in self.edges.values():
                        writer.writerow([edge.id, edge.source, edge.target, edge.type, json.dumps(edge.properties, ensure_ascii=False)])
                logger.info(f"Arêtes exportées dans le fichier {edges_file}.")
            except IOError as e:
                logger.error(f"Erreur lors de l'exportation en CSV: {e}")
                raise KnowledgeGraphError(f"Erreur lors de l'exportation en CSV: {e}")

    def import_from_csv(self, nodes_file: str, edges_file: str) -> None:
        """
        Importe les nœuds et les arêtes du graphe depuis des fichiers CSV séparés.
        """
        with self.lock:
            try:
                # Importer les nœuds
                with open(nodes_file, 'r', encoding='utf-8') as f_nodes:
                    reader = csv.DictReader(f_nodes)
                    for row in reader:
                        node_id = row['id']
                        labels = row['labels'].split(",") if row['labels'] else []
                        properties = json.loads(row['properties']) if row['properties'] else {}
                        node = Node(id=node_id, labels=set(labels), properties=properties)
                        self.nodes[node.id] = node

                        # Mise à jour des index de labels
                        for label in node.labels:
                            self.label_index[label].add(node.id)
                        
                        # Mise à jour des index de propriétés
                        for prop, value in node.properties.items():
                            self.property_index[prop][value].add(node.id)
                logger.info(f"Nœuds importés depuis le fichier {nodes_file}.")

                # Importer les arêtes
                with open(edges_file, 'r', encoding='utf-8') as f_edges:
                    reader = csv.DictReader(f_edges)
                    for row in reader:
                        edge_id = row['id']
                        source = row['source']
                        target = row['target']
                        rel_type = row['type']
                        properties = json.loads(row['properties']) if row['properties'] else {}
                        edge = Edge(id=edge_id, source=source, target=target, type=rel_type, properties=properties)
                        self.edges[edge.id] = edge

                        # Mise à jour de l'adjacence
                        self.adjacency[source].add(edge.id)
                        self.adjacency[target].add(edge.id)
                logger.info(f"Arêtes importées depuis le fichier {edges_file}.")
            except (IOError, csv.Error, json.JSONDecodeError) as e:
                logger.error(f"Erreur lors de l'importation depuis CSV: {e}")
                raise KnowledgeGraphError(f"Erreur lors de l'importation depuis CSV: {e}")

    def find_shortest_path(self, start_node_id: str, end_node_id: str, weighted: bool = False) -> Optional[List[str]]:
        """
        Trouve le chemin le plus court entre deux nœuds en utilisant l'algorithme de BFS ou Dijkstra.
        Retourne la liste des IDs des nœuds dans le chemin, ou None s'il n'y a pas de chemin.
        """
        if start_node_id not in self.nodes or end_node_id not in self.nodes:
            logger.error("Un des nœuds spécifiés n'existe pas.")
            raise NodeNotFoundError("Un des nœuds spécifiés n'existe pas.")
        
        if not weighted:
            # Algorithme BFS pour les graphes non pondérés
            visited = set()
            queue = deque([[start_node_id]])

            while queue:
                path = queue.popleft()
                node_id = path[-1]

                if node_id == end_node_id:
                    logger.info(f"Chemin le plus court trouvé entre {start_node_id} et {end_node_id}: {path}")
                    return path

                if node_id not in visited:
                    visited.add(node_id)
                    for edge_id in self.adjacency[node_id]:
                        edge = self.edges[edge_id]
                        neighbor = edge.target if edge.source == node_id else edge.source
                        if neighbor not in visited:
                            new_path = list(path)
                            new_path.append(neighbor)
                            queue.append(new_path)
            
            logger.info(f"Aucun chemin trouvé entre {start_node_id} et {end_node_id}.")
            return None
        else:
            # Algorithme de Dijkstra pour les graphes pondérés
            G = self._to_networkx_graph(weighted=True)
            try:
                path = nx.dijkstra_path(G, start_node_id, end_node_id, weight='weight')
                logger.info(f"Chemin le plus court pondéré entre {start_node_id} et {end_node_id}: {path}")
                return path
            except nx.NetworkXNoPath:
                logger.info(f"Aucun chemin pondéré trouvé entre {start_node_id} et {end_node_id}.")
                return None
            except nx.NodeNotFound:
                logger.error("Un des nœuds spécifiés n'existe pas dans le graphe NetworkX.")
                raise KnowledgeGraphError("Un des nœuds spécifiés n'existe pas dans le graphe NetworkX.")

    def has_cycle(self) -> bool:
        """
        Vérifie si le graphe contient un cycle en utilisant l'algorithme de détection de cycle en DFS.
        """
        G = self._to_networkx_graph()
        try:
            cycles = list(nx.find_cycle(G, orientation='ignore'))
            if cycles:
                logger.info("Cycle détecté dans le graphe.")
                return True
        except nx.NetworkXNoCycle:
            logger.info("Aucun cycle détecté dans le graphe.")
            return False
        return False

    def traverse_bfs(self, start_node_id: str, visit_func: Callable[[Node], None]) -> None:
        """
        Traverse le graphe en largeur (BFS) à partir d'un nœud de départ.
        Applique la fonction visit_func à chaque nœud visité.
        """
        if start_node_id not in self.nodes:
            logger.error(f"Le nœud de départ {start_node_id} n'existe pas.")
            raise NodeNotFoundError(f"Le nœud de départ {start_node_id} n'existe pas.")
        
        visited = set()
        queue = deque([start_node_id])

        while queue:
            node_id = queue.popleft()
            if node_id not in visited:
                visit_func(self.nodes[node_id])
                visited.add(node_id)
                for edge_id in self.adjacency[node_id]:
                    edge = self.edges[edge_id]
                    neighbor = edge.target if edge.source == node_id else edge.source
                    if neighbor not in visited:
                        queue.append(neighbor)
        logger.info(f"Traversée BFS terminée à partir du nœud {start_node_id}.")

    def traverse_dfs(self, start_node_id: str, visit_func: Callable[[Node], None]) -> None:
        """
        Traverse le graphe en profondeur (DFS) à partir d'un nœud de départ.
        Applique la fonction visit_func à chaque nœud visité.
        """
        if start_node_id not in self.nodes:
            logger.error(f"Le nœud de départ {start_node_id} n'existe pas.")
            raise NodeNotFoundError(f"Le nœud de départ {start_node_id} n'existe pas.")
        
        visited = set()
        stack = [start_node_id]

        while stack:
            node_id = stack.pop()
            if node_id not in visited:
                visit_func(self.nodes[node_id])
                visited.add(node_id)
                for edge_id in self.adjacency[node_id]:
                    edge = self.edges[edge_id]
                    neighbor = edge.target if edge.source == node_id else edge.source
                    if neighbor not in visited:
                        stack.append(neighbor)
        logger.info(f"Traversée DFS terminée à partir du nœud {start_node_id}.")

    def add_label_to_node(self, node_id: str, label: str) -> None:
        """
        Ajoute un label à un nœud existant.
        """
        with self.lock:
            node = self.get_node(node_id)
            if node is None:
                logger.error(f"Le nœud {node_id} n'existe pas.")
                raise NodeNotFoundError(f"Le nœud {node_id} n'existe pas.")
            
            if label not in node.labels:
                node.labels.add(label)
                self.label_index[label].add(node_id)
                logger.info(f"Label '{label}' ajouté au nœud {node_id}.")
            else:
                logger.warning(f"Le nœud {node_id} possède déjà le label '{label}'.")

    def remove_label_from_node(self, node_id: str, label: str) -> None:
        """
        Supprime un label d'un nœud existant.
        """
        with self.lock:
            node = self.get_node(node_id)
            if node is None:
                logger.error(f"Le nœud {node_id} n'existe pas.")
                raise NodeNotFoundError(f"Le nœud {node_id} n'existe pas.")
            
            if label in node.labels:
                node.labels.remove(label)
                self.label_index[label].discard(node_id)
                if not self.label_index[label]:
                    del self.label_index[label]
                logger.info(f"Label '{label}' supprimé du nœud {node_id}.")
            else:
                logger.warning(f"Le nœud {node_id} ne possède pas le label '{label}'.")

    def get_degree_centrality(self) -> Dict[str, int]:
        """
        Calcule la centralité de degré pour chaque nœud du graphe.
        Retourne un dictionnaire avec les IDs des nœuds et leur centralité.
        """
        centrality = {node_id: len(edges) for node_id, edges in self.adjacency.items()}
        logger.info("Calcul de la centralité de degré terminé.")
        return centrality

    def visualize_graph(self, with_labels: bool = True, node_size: int = 300, edge_color: str = 'gray') -> None:
        """
        Visualise le graphe en utilisant NetworkX et Matplotlib.
        """
        G = self._to_networkx_graph()
        pos = nx.spring_layout(G)
        plt.figure(figsize=(12, 8))
        nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color='lightblue')
        nx.draw_networkx_edges(G, pos, edge_color=edge_color)
        if with_labels:
            labels = {node_id: self.nodes[node_id].properties.get('name', node_id) for node_id in G.nodes()}
            nx.draw_networkx_labels(G, pos, labels, font_size=10)
        plt.title("Visualisation du Graphe de Connaissances")
        plt.axis('off')
        plt.show()
        logger.info("Visualisation du graphe terminée.")

    def _to_networkx_graph(self, weighted: bool = False) -> nx.Graph:
        """
        Convertit le KnowledgeGraph en un objet NetworkX Graph.
        """
        G = nx.Graph()
        for node_id, node in self.nodes.items():
            G.add_node(node_id, **node.properties)
        for edge_id, edge in self.edges.items():
            if weighted and 'weight' in edge.properties:
                G.add_edge(edge.source, edge.target, key=edge_id, type=edge.type, weight=edge.properties['weight'], **edge.properties)
            else:
                G.add_edge(edge.source, edge.target, key=edge_id, type=edge.type, **edge.properties)
        logger.debug("Conversion en NetworkX Graph effectuée.")
        return G

    def execute_transaction(self, operations: List[Callable[[], None]]) -> None:
        """
        Exécute une série d'opérations atomiques. Si une opération échoue, annule toutes les opérations précédentes.
        """
        with self.lock:
            # Sauvegarde de l'état actuel
            backup = {
                "nodes": {node_id: node.__dict__.copy() for node_id, node in self.nodes.items()},
                "edges": {edge_id: edge.__dict__.copy() for edge_id, edge in self.edges.items()},
                "label_index": {k: v.copy() for k, v in self.label_index.items()},
                "property_index": {k: {pk: pv.copy() for pk, pv in v.items()} for k, v in self.property_index.items()},
                "adjacency": {k: v.copy() for k, v in self.adjacency.items()}
            }
            logger.debug("Sauvegarde de l'état avant transaction.")

            try:
                for operation in operations:
                    operation()
                logger.info("Transaction exécutée avec succès.")
            except Exception as e:
                # Restauration de l'état précédent en cas d'erreur
                self.nodes = {k: Node(**v) for k, v in backup["nodes"].items()}
                self.edges = {k: Edge(**v) for k, v in backup["edges"].items()}
                self.label_index = defaultdict(set, {k: set(v) for k, v in backup["label_index"].items()})
                self.property_index = defaultdict(lambda: defaultdict(set), {
                    k: defaultdict(set, {pk: set(pv) for pk, pv in v.items()}) 
                    for k, v in backup["property_index"].items()
                })
                self.adjacency = defaultdict(set, {k: set(v) for k, v in backup["adjacency"].items()})
                logger.error(f"Erreur lors de l'exécution de la transaction: {e}. Restauration de l'état précédent.")
                raise TransactionError(f"Erreur lors de l'exécution de la transaction: {e}")

# Exemple d'utilisation avancée du KnowledgeGraph
if __name__ == "__main__":
    kg = KnowledgeGraph()
    
    # Ajout de nœuds
    alice_id = kg.add_node(labels=["Person"], properties={"name": "Alice", "age": 30, "hobby": "chess"})
    bob_id = kg.add_node(labels=["Person"], properties={"name": "Bob", "age": 25, "hobby": "football"})
    charlie_id = kg.add_node(labels=["Person"], properties={"name": "Charlie", "age": 35, "hobby": "guitar"})
    diana_id = kg.add_node(labels=["Person"], properties={"name": "Diana", "age": 28, "hobby": "painting"})
    paris_id = kg.add_node(labels=["City"], properties={"name": "Paris", "country": "France", "population": 2148327})
    openai_id = kg.add_node(labels=["Company"], properties={"name": "OpenAI", "industry": "AI Research"})
    techcorp_id = kg.add_node(labels=["Company"], properties={"name": "TechCorp", "industry": "Software Development"})
    
    # Ajout d'arêtes
    kg.add_edge(alice_id, bob_id, "KNOWS", properties={"since": 2015})
    kg.add_edge(alice_id, charlie_id, "KNOWS", properties={"since": 2018})
    kg.add_edge(bob_id, diana_id, "KNOWS", properties={"since": 2020})
    kg.add_edge(alice_id, paris_id, "LIVES_IN", properties={"since": 2020})
    kg.add_edge(bob_id, openai_id, "WORKS_AT", properties={"role": "Engineer"})
    kg.add_edge(charlie_id, techcorp_id, "WORKS_AT", properties={"role": "Developer"})
    kg.add_edge(openai_id, techcorp_id, "COLLABORATES_WITH", properties={"project": "AI Integration"})
    
    # Requêtes sur les nœuds
    persons = kg.find_nodes_by_label("Person")
    logger.info(f"Liste des personnes dans le graphe: {[person.properties['name'] for person in persons]}")
    
    alice_nodes = kg.find_nodes_by_property("name", "Alice")
    logger.info(f"Recherche du nœud Alice: {[node.id for node in alice_nodes]}")
    
    # Recherche combinée
    chess_players = kg.find_nodes(labels=["Person"], properties={"hobby": "chess"})
    logger.info(f"Liste des joueurs d'échecs: {[node.properties['name'] for node in chess_players]}")
    
    # Mise à jour des propriétés d'un nœud
    kg.update_node_properties(alice_id, {"hobby": "chess", "profession": "Data Scientist"})
    updated_alice = kg.get_node(alice_id)
    logger.info(f"Propriétés mises à jour pour Alice: {updated_alice.properties}")
    
    # Gestion des labels
    kg.add_label_to_node(alice_id, "Expert")
    kg.remove_label_from_node(alice_id, "Person")
    updated_labels_alice = kg.get_node(alice_id).labels
    logger.info(f"Labels actuels pour Alice: {updated_labels_alice}")
    
    # Récupération des arêtes sortantes
    alice_out_edges = kg.get_edges_from_node(alice_id)
    logger.info(f"Arêtes sortantes de Alice: {[edge.type for edge in alice_out_edges]}")
    
    # Mise à jour des propriétés d'une arête
    lives_in_edge_id = next((edge.id for edge in kg.get_edges_from_node(alice_id) if edge.type == "LIVES_IN"), None)
    if lives_in_edge_id:
        kg.update_edge_properties(lives_in_edge_id, {"visited_landmarks": ["Eiffel Tower", "Louvre Museum"]})
        logger.info(f"Propriétés mises à jour pour l'arête LIVES_IN: {kg.edges[lives_in_edge_id].properties}")
    else:
        logger.warning("Arête LIVES_IN non trouvée pour Alice.")
    
    # Traversée BFS
    def print_node(node: Node):
        print(f"Visité: {node.properties.get('name', node.id)}")
    
    logger.info("Début de la traversée BFS à partir de Alice.")
    kg.traverse_bfs(alice_id, print_node)
    
    # Traversée DFS
    logger.info("Début de la traversée DFS à partir de Alice.")
    kg.traverse_dfs(alice_id, print_node)
    
    # Recherche du chemin le plus court
    path = kg.find_shortest_path(alice_id, diana_id)
    if path:
        logger.info(f"Chemin le plus court entre Alice et Diana: {path}")
    else:
        logger.info("Aucun chemin trouvé entre Alice et Diana.")
    
    # Recherche du chemin le plus court pondéré (ajout de poids)
    # Ajout de poids aux arêtes
    for edge in kg.edges.values():
        if edge.type == "KNOWS":
            edge.properties['weight'] = 1
        elif edge.type == "WORKS_AT":
            edge.properties['weight'] = 2
        elif edge.type == "COLLABORATES_WITH":
            edge.properties['weight'] = 3
        elif edge.type == "LIVES_IN":
            edge.properties['weight'] = 1
    
    weighted_path = kg.find_shortest_path(alice_id, diana_id, weighted=True)
    if weighted_path:
        logger.info(f"Chemin le plus court pondéré entre Alice et Diana: {weighted_path}")
    else:
        logger.info("Aucun chemin pondéré trouvé entre Alice et Diana.")
    
    # Calcul de la centralité de degré
    centrality = kg.get_degree_centrality()
    logger.info(f"Centralité de degré: {centrality}")
    
    # Vérification de cycle
    has_cycle = kg.has_cycle()
    logger.info(f"Le graphe contient un cycle: {has_cycle}")
    
    # Visualisation du graphe
    kg.visualize_graph()
    
    # Exportation du graphe dans des fichiers CSV
    kg.export_to_csv("nodes.csv", "edges.csv")
    
    # Sauvegarde du graphe dans un fichier JSON
    kg.save_to_json("graph.json")
    
    # Chargement du graphe depuis un fichier JSON
    kg2 = KnowledgeGraph()
    kg2.load_from_json("graph.json")
    logger.info(f"KG2 loaded nodes: {[node.properties['name'] for node in kg2.get_all_nodes()]}")
    logger.info(f"KG2 loaded edges: {[edge.type for edge in kg2.get_all_edges()]}")
    
    # Importation depuis CSV dans un nouveau graphe
    kg3 = KnowledgeGraph()
    kg3.import_from_csv("nodes.csv", "edges.csv")
    logger.info(f"KG3 imported nodes: {[node.properties['name'] for node in kg3.get_all_nodes()]}")
    logger.info(f"KG3 imported edges: {[edge.type for edge in kg3.get_all_edges()]}")
    
    # Exemple de transaction
    try:
        kg.execute_transaction([
            lambda: kg.add_node(labels=["Person"], properties={"name": "Eve", "age": 22, "hobby": "reading"}),
            lambda: kg.add_edge(alice_id, "non_existent_node", "KNOWS")
        ])
    except TransactionError as te:
        logger.error(f"Transaction échouée: {te}")
    
    # Vérification que la transaction a été annulée
    eve_nodes = kg.find_nodes_by_property("name", "Eve")
    logger.info(f"Recherche du nœud Eve après la transaction: {[node.id for node in eve_nodes]}")
