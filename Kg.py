### Import Standard Libraries
import json
import uuid
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set, Union, Callable
import logging
from collections import defaultdict, deque
import csv
import threading

### Import Third-Party Libraries
import networkx as nx
import matplotlib.pyplot as plt

# Advanced logger configuration with file handling
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # More detailed log level

# Create a console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(console_formatter)
logger.addHandler(console_handler)

# Create a file handler
file_handler = logging.FileHandler('knowledge_graph.log', encoding='utf-8')
file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)

@dataclass
class Node:
    """
    Represents a node within the Knowledge Graph.

    Attributes:
        id (str): A unique identifier for the node.
        labels (Set[str]): A set of labels categorizing the node.
        properties (Dict[str, Any]): A dictionary of properties associated with the node.
    """
    id: str
    labels: Set[str]
    properties: Dict[str, Any]

@dataclass
class Edge:
    """
    Represents an edge (relationship) between two nodes in the Knowledge Graph.

    Attributes:
        id (str): A unique identifier for the edge.
        source (str): The unique identifier of the source node.
        target (str): The unique identifier of the target node.
        type (str): The type of relationship the edge represents.
        properties (Dict[str, Any]): A dictionary of properties associated with the edge.
    """
    id: str
    source: str
    target: str
    type: str
    properties: Dict[str, Any] = field(default_factory=dict)

class KnowledgeGraphError(Exception):
    """Base exception class for KnowledgeGraph-related errors."""
    pass

class NodeNotFoundError(KnowledgeGraphError):
    """Exception raised when a specified node is not found in the graph."""
    pass

class EdgeNotFoundError(KnowledgeGraphError):
    """Exception raised when a specified edge is not found in the graph."""
    pass

class TransactionError(KnowledgeGraphError):
    """Exception raised when a transaction fails."""
    pass

class KnowledgeGraph:
    """
    Represents a Knowledge Graph consisting of nodes and edges, supporting various operations 
    such as addition, removal, querying, and transaction management.

    Attributes:
        nodes (Dict[str, Node]): A dictionary mapping node IDs to Node instances.
        edges (Dict[str, Edge]): A dictionary mapping edge IDs to Edge instances.
        label_index (defaultdict): An index mapping labels to sets of node IDs.
        property_index (defaultdict): An index mapping property names and values to sets of node IDs.
        adjacency (defaultdict): An adjacency list mapping node IDs to sets of edge IDs.
        lock (threading.Lock): A lock to ensure thread-safe operations.
    """

    def __init__(self):
        """
        Initializes an empty Knowledge Graph with indexing structures for efficient querying.
        """
        self.nodes: Dict[str, Node] = {}
        self.edges: Dict[str, Edge] = {}
        self.label_index: defaultdict = defaultdict(set)
        self.property_index: defaultdict = defaultdict(lambda: defaultdict(set))
        self.adjacency: defaultdict = defaultdict(set)
        self.lock = threading.Lock()  # Ensures thread safety
        logger.info("Initialized an empty KnowledgeGraph.")

    def _generate_unique_id(self, prefix: str) -> str:
        """
        Generates a unique identifier with the specified prefix.

        Args:
            prefix (str): The prefix to use for the unique ID.

        Returns:
            str: A unique identifier string.
        """
        unique_id = f"{prefix}_{uuid.uuid4()}"
        logger.debug(f"Generated unique ID: {unique_id}")
        return unique_id

    def add_node(self, labels: List[str], properties: Dict[str, Any]) -> str:
        """
        Adds a new node to the graph and returns its unique identifier.
        Updates label and property indices accordingly.

        Args:
            labels (List[str]): A list of labels to assign to the node.
            properties (Dict[str, Any]): A dictionary of properties for the node.

        Returns:
            str: The unique identifier of the newly added node.
        """
        with self.lock:
            node_id = self._generate_unique_id("node")
            node = Node(id=node_id, labels=set(labels), properties=properties)
            self.nodes[node_id] = node
            logger.info(f"Added node: {node_id} with labels {labels} and properties {properties}")

            # Update label index
            for label in labels:
                self.label_index[label].add(node_id)
                logger.debug(f"Indexed label '{label}' for node {node_id}.")

            # Update property index
            for prop, value in properties.items():
                self.property_index[prop][value].add(node_id)
                logger.debug(f"Indexed property '{prop}: {value}' for node {node_id}.")

            return node_id

    def add_edge(
        self, 
        source_id: str, 
        target_id: str, 
        rel_type: str, 
        properties: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Adds an edge between two existing nodes and returns its unique identifier.
        Updates the adjacency list for both nodes.

        Args:
            source_id (str): The unique identifier of the source node.
            target_id (str): The unique identifier of the target node.
            rel_type (str): The type of relationship the edge represents.
            properties (Optional[Dict[str, Any]]): A dictionary of properties for the edge.

        Returns:
            str: The unique identifier of the newly added edge.

        Raises:
            NodeNotFoundError: If either the source or target node does not exist.
        """
        with self.lock:
            if source_id not in self.nodes:
                logger.error(f"Source node {source_id} does not exist.")
                raise NodeNotFoundError(f"Source node {source_id} does not exist.")
            if target_id not in self.nodes:
                logger.error(f"Target node {target_id} does not exist.")
                raise NodeNotFoundError(f"Target node {target_id} does not exist.")
            
            if properties is None:
                properties = {}
            
            edge_id = self._generate_unique_id("edge")
            edge = Edge(id=edge_id, source=source_id, target=target_id, type=rel_type, properties=properties)
            self.edges[edge_id] = edge
            logger.info(f"Added edge: {edge_id} from {source_id} to {target_id} of type '{rel_type}' with properties {properties}")

            # Update adjacency
            self.adjacency[source_id].add(edge_id)
            self.adjacency[target_id].add(edge_id)
            logger.debug(f"Updated adjacency for nodes {source_id} and {target_id} with edge {edge_id}.")

            return edge_id

    def get_node(self, node_id: str) -> Optional[Node]:
        """
        Retrieves a node by its unique identifier.

        Args:
            node_id (str): The unique identifier of the node.

        Returns:
            Optional[Node]: The Node instance if found, else None.
        """
        node = self.nodes.get(node_id)
        if node:
            logger.debug(f"Retrieved node: {node_id}")
        else:
            logger.warning(f"Attempted to retrieve non-existent node: {node_id}")
        return node

    def update_node_properties(self, node_id: str, new_properties: Dict[str, Any]) -> None:
        """
        Updates the properties of a node incrementally.
        Adjusts property indices accordingly.

        Args:
            node_id (str): The unique identifier of the node to update.
            new_properties (Dict[str, Any]): A dictionary of properties to update.

        Raises:
            NodeNotFoundError: If the node does not exist.
        """
        with self.lock:
            node = self.get_node(node_id)
            if node is None:
                logger.error(f"Node {node_id} does not exist.")
                raise NodeNotFoundError(f"Node {node_id} does not exist.")
            
            # Update property indices
            for prop, value in new_properties.items():
                # Remove old index if property exists
                if prop in node.properties:
                    old_value = node.properties[prop]
                    self.property_index[prop][old_value].discard(node_id)
                    if not self.property_index[prop][old_value]:
                        del self.property_index[prop][old_value]
                    logger.debug(f"Removed old property '{prop}: {old_value}' from node {node_id}.")
                
                # Add new index
                self.property_index[prop][value].add(node_id)
                logger.debug(f"Indexed new property '{prop}: {value}' for node {node_id}.")

            # Update node properties
            node.properties.update(new_properties)
            logger.info(f"Updated properties for node {node_id} with {new_properties}.")

    def remove_node(self, node_id: str) -> None:
        """
        Removes a node and all its associated edges from the graph.
        Updates indices and adjacency lists accordingly.

        Args:
            node_id (str): The unique identifier of the node to remove.

        Raises:
            NodeNotFoundError: If the node does not exist.
        """
        with self.lock:
            if node_id not in self.nodes:
                logger.error(f"Node {node_id} does not exist.")
                raise NodeNotFoundError(f"Node {node_id} does not exist.")
            
            # Remove associated edges
            related_edges = list(self.adjacency[node_id])
            for edge_id in related_edges:
                self.remove_edge_by_id(edge_id)
            
            # Remove label indices
            node = self.nodes[node_id]
            for label in node.labels:
                self.label_index[label].discard(node_id)
                if not self.label_index[label]:
                    del self.label_index[label]
                logger.debug(f"Removed node {node_id} from label index '{label}'.")

            # Remove property indices
            for prop, value in node.properties.items():
                self.property_index[prop][value].discard(node_id)
                if not self.property_index[prop][value]:
                    del self.property_index[prop][value]
                if not self.property_index[prop]:
                    del self.property_index[prop]
                logger.debug(f"Removed node {node_id} from property index '{prop}: {value}'.")

            # Remove the node and its adjacency entry
            del self.nodes[node_id]
            del self.adjacency[node_id]
            logger.info(f"Removed node {node_id} and all its associated edges.")

    def remove_edge_by_id(self, edge_id: str) -> None:
        """
        Removes a specific edge by its unique identifier.
        Updates adjacency lists accordingly.

        Args:
            edge_id (str): The unique identifier of the edge to remove.

        Raises:
            EdgeNotFoundError: If the edge does not exist.
        """
        with self.lock:
            if edge_id not in self.edges:
                logger.error(f"Edge {edge_id} does not exist.")
                raise EdgeNotFoundError(f"Edge {edge_id} does not exist.")
            
            edge = self.edges[edge_id]
            # Update adjacency
            self.adjacency[edge.source].discard(edge_id)
            self.adjacency[edge.target].discard(edge_id)
            logger.debug(f"Updated adjacency for nodes {edge.source} and {edge.target} by removing edge {edge_id}.")

            # Remove the edge
            del self.edges[edge_id]
            logger.info(f"Removed edge {edge_id}.")

    def get_edges_from_node(self, node_id: str) -> List[Edge]:
        """
        Retrieves all outgoing edges from a given node.

        Args:
            node_id (str): The unique identifier of the node.

        Returns:
            List[Edge]: A list of outgoing Edge instances.

        Raises:
            NodeNotFoundError: If the node does not exist.
        """
        if node_id not in self.nodes:
            logger.error(f"Node {node_id} does not exist.")
            raise NodeNotFoundError(f"Node {node_id} does not exist.")
        
        outgoing_edges = [
            self.edges[edge_id] for edge_id in self.adjacency[node_id]
            if self.edges[edge_id].source == node_id
        ]
        logger.debug(f"Retrieved {len(outgoing_edges)} outgoing edges from node {node_id}.")
        return outgoing_edges

    def get_edges_to_node(self, node_id: str) -> List[Edge]:
        """
        Retrieves all incoming edges to a given node.

        Args:
            node_id (str): The unique identifier of the node.

        Returns:
            List[Edge]: A list of incoming Edge instances.

        Raises:
            NodeNotFoundError: If the node does not exist.
        """
        if node_id not in self.nodes:
            logger.error(f"Node {node_id} does not exist.")
            raise NodeNotFoundError(f"Node {node_id} does not exist.")
        
        incoming_edges = [
            self.edges[edge_id] for edge_id in self.adjacency[node_id]
            if self.edges[edge_id].target == node_id
        ]
        logger.debug(f"Retrieved {len(incoming_edges)} incoming edges to node {node_id}.")
        return incoming_edges

    def get_edges_of_node(self, node_id: str) -> List[Edge]:
        """
        Retrieves all edges connected to a given node.

        Args:
            node_id (str): The unique identifier of the node.

        Returns:
            List[Edge]: A list of connected Edge instances.

        Raises:
            NodeNotFoundError: If the node does not exist.
        """
        if node_id not in self.nodes:
            logger.error(f"Node {node_id} does not exist.")
            raise NodeNotFoundError(f"Node {node_id} does not exist.")
        
        connected_edges = [self.edges[edge_id] for edge_id in self.adjacency[node_id]]
        logger.debug(f"Retrieved {len(connected_edges)} edges connected to node {node_id}.")
        return connected_edges

    def remove_edge(self, source_id: str, target_id: str, rel_type: str) -> None:
        """
        Removes a specific edge based on source, target, and relationship type.

        Args:
            source_id (str): The unique identifier of the source node.
            target_id (str): The unique identifier of the target node.
            rel_type (str): The type of relationship the edge represents.

        Raises:
            EdgeNotFoundError: If no matching edge is found.
        """
        with self.lock:
            found_edges = [
                edge_id for edge_id, edge in self.edges.items()
                if edge.source == source_id and edge.target == target_id and edge.type == rel_type
            ]
            if not found_edges:
                logger.error(f"No edge found between {source_id} and {target_id} of type {rel_type}.")
                raise EdgeNotFoundError(f"No edge found between {source_id} and {target_id} of type {rel_type}.")
            
            for edge_id in found_edges:
                self.remove_edge_by_id(edge_id)
                logger.info(f"Removed edge {edge_id} between {source_id} and {target_id} of type {rel_type}.")

    def find_nodes_by_label(self, label: str) -> List[Node]:
        """
        Finds all nodes with a given label.

        Args:
            label (str): The label to search for.

        Returns:
            List[Node]: A list of Node instances matching the label.
        """
        node_ids = self.label_index.get(label, set())
        nodes = [self.nodes[node_id] for node_id in node_ids]
        logger.info(f"Found {len(nodes)} nodes with label '{label}'.")
        return nodes

    def find_nodes_by_property(self, property_name: str, property_value: Any) -> List[Node]:
        """
        Finds all nodes with a specific property name and value.

        Args:
            property_name (str): The name of the property to search for.
            property_value (Any): The value of the property to match.

        Returns:
            List[Node]: A list of Node instances matching the property criteria.
        """
        node_ids = self.property_index.get(property_name, {}).get(property_value, set())
        nodes = [self.nodes[node_id] for node_id in node_ids]
        logger.info(f"Found {len(nodes)} nodes with property '{property_name}: {property_value}'.")
        return nodes

    def find_nodes(self, labels: Optional[List[str]] = None, properties: Optional[Dict[str, Any]] = None) -> List[Node]:
        """
        Finds nodes based on a combination of labels and/or properties.

        Args:
            labels (Optional[List[str]]): A list of labels to filter nodes.
            properties (Optional[Dict[str, Any]]): A dictionary of properties to filter nodes.

        Returns:
            List[Node]: A list of Node instances matching the specified criteria.
        """
        if not labels and not properties:
            logger.warning("No search criteria specified. Returning all nodes.")
            return list(self.nodes.values())
        
        result_ids: Optional[Set[str]] = None

        if labels:
            for label in labels:
                label_ids = self.label_index.get(label, set())
                if result_ids is None:
                    result_ids = label_ids.copy()
                else:
                    result_ids &= label_ids
                logger.debug(f"Filtering by label '{label}', found {len(label_ids)} nodes.")

        if properties:
            for prop, value in properties.items():
                prop_ids = self.property_index.get(prop, {}).get(value, set())
                if result_ids is None:
                    result_ids = prop_ids.copy()
                else:
                    result_ids &= prop_ids
                logger.debug(f"Filtering by property '{prop}: {value}', found {len(prop_ids)} nodes.")

        if result_ids is None:
            result_ids = set()

        nodes = [self.nodes[node_id] for node_id in result_ids]
        logger.info(f"Found {len(nodes)} nodes matching the specified criteria.")
        return nodes

    def update_edge_properties(self, edge_id: str, new_properties: Dict[str, Any]) -> None:
        """
        Updates the properties of an existing edge.

        Args:
            edge_id (str): The unique identifier of the edge to update.
            new_properties (Dict[str, Any]): A dictionary of properties to update.

        Raises:
            EdgeNotFoundError: If the edge does not exist.
        """
        with self.lock:
            if edge_id not in self.edges:
                logger.error(f"Edge {edge_id} does not exist.")
                raise EdgeNotFoundError(f"Edge {edge_id} does not exist.")
            
            edge = self.edges[edge_id]
            edge.properties.update(new_properties)
            logger.info(f"Updated properties for edge {edge_id} with {new_properties}.")

    def save_to_json(self, file_path: str) -> None:
        """
        Saves the entire graph to a JSON file.

        Args:
            file_path (str): The path to the JSON file where the graph will be saved.

        Raises:
            KnowledgeGraphError: If an I/O error occurs during saving.
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
                logger.info(f"Graph saved to {file_path}.")
            except IOError as e:
                logger.error(f"Error saving graph to JSON: {e}")
                raise KnowledgeGraphError(f"Error saving graph to JSON: {e}")

    def load_from_json(self, file_path: str) -> None:
        """
        Loads a graph from a JSON file, resetting the current graph.

        Args:
            file_path (str): The path to the JSON file from which to load the graph.

        Raises:
            KnowledgeGraphError: If an I/O or JSON decoding error occurs during loading.
        """
        with self.lock:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except (IOError, json.JSONDecodeError) as e:
                logger.error(f"Error loading graph from JSON: {e}")
                raise KnowledgeGraphError(f"Error loading graph from JSON: {e}")
            
            # Reset existing structures
            self.clear_graph()
            logger.info(f"Loading graph from {file_path}.")

            # Load nodes
            for node_data in data.get("nodes", []):
                node = Node(
                    id=node_data["id"],
                    labels=set(node_data["labels"]),
                    properties=node_data["properties"]
                )
                self.nodes[node.id] = node
                logger.debug(f"Loaded node: {node.id}")

                # Update label index
                for label in node.labels:
                    self.label_index[label].add(node.id)
                    logger.debug(f"Indexed label '{label}' for node {node.id}.")

                # Update property index
                for prop, value in node.properties.items():
                    self.property_index[prop][value].add(node.id)
                    logger.debug(f"Indexed property '{prop}: {value}' for node {node.id}.")

            # Load edges
            for edge_data in data.get("edges", []):
                edge = Edge(
                    id=edge_data["id"],
                    source=edge_data["source"],
                    target=edge_data["target"],
                    type=edge_data["type"],
                    properties=edge_data.get("properties", {})
                )
                self.edges[edge.id] = edge
                logger.debug(f"Loaded edge: {edge.id}")

                # Update adjacency
                self.adjacency[edge.source].add(edge.id)
                self.adjacency[edge.target].add(edge.id)
                logger.debug(f"Updated adjacency for nodes {edge.source} and {edge.target} with edge {edge.id}.")

            logger.info("Graph loading completed.")

    def get_all_nodes(self) -> List[Node]:
        """
        Retrieves all nodes in the graph.

        Returns:
            List[Node]: A list of all Node instances in the graph.
        """
        logger.debug(f"Retrieving all nodes, total count: {len(self.nodes)}.")
        return list(self.nodes.values())

    def get_all_edges(self) -> List[Edge]:
        """
        Retrieves all edges in the graph.

        Returns:
            List[Edge]: A list of all Edge instances in the graph.
        """
        logger.debug(f"Retrieving all edges, total count: {len(self.edges)}.")
        return list(self.edges.values())

    def node_exists(self, node_id: str) -> bool:
        """
        Checks if a node exists in the graph.

        Args:
            node_id (str): The unique identifier of the node.

        Returns:
            bool: True if the node exists, False otherwise.
        """
        existence = node_id in self.nodes
        logger.debug(f"Node existence check for {node_id}: {existence}.")
        return existence

    def edge_exists(self, edge_id: str) -> bool:
        """
        Checks if a specific edge exists in the graph.

        Args:
            edge_id (str): The unique identifier of the edge.

        Returns:
            bool: True if the edge exists, False otherwise.
        """
        existence = edge_id in self.edges
        logger.debug(f"Edge existence check for {edge_id}: {existence}.")
        return existence

    def merge_properties(self, node_id: str, properties: Dict[str, Any]) -> None:
        """
        Merges the properties of a node with another dictionary.
        Existing keys are overwritten.

        Args:
            node_id (str): The unique identifier of the node.
            properties (Dict[str, Any]): A dictionary of properties to merge.

        Raises:
            NodeNotFoundError: If the node does not exist.
        """
        with self.lock:
            self.update_node_properties(node_id, properties)
            logger.info(f"Merged properties for node {node_id} with {properties}.")

    def clear_graph(self) -> None:
        """
        Completely clears the graph, removing all nodes and edges.
        """
        with self.lock:
            self.nodes.clear()
            self.edges.clear()
            self.label_index.clear()
            self.property_index.clear()
            self.adjacency.clear()
            logger.info("Cleared the entire graph.")

    def export_to_csv(self, nodes_file: str, edges_file: str) -> None:
        """
        Exports the nodes and edges of the graph to separate CSV files.

        Args:
            nodes_file (str): The path to the CSV file for nodes.
            edges_file (str): The path to the CSV file for edges.

        Raises:
            KnowledgeGraphError: If an I/O error occurs during export.
        """
        with self.lock:
            try:
                # Export nodes
                with open(nodes_file, 'w', encoding='utf-8', newline='') as f_nodes:
                    writer = csv.writer(f_nodes)
                    writer.writerow(['id', 'labels', 'properties'])
                    for node in self.nodes.values():
                        writer.writerow([node.id, ",".join(node.labels), json.dumps(node.properties, ensure_ascii=False)])
                logger.info(f"Exported nodes to {nodes_file}.")

                # Export edges
                with open(edges_file, 'w', encoding='utf-8', newline='') as f_edges:
                    writer = csv.writer(f_edges)
                    writer.writerow(['id', 'source', 'target', 'type', 'properties'])
                    for edge in self.edges.values():
                        writer.writerow([edge.id, edge.source, edge.target, edge.type, json.dumps(edge.properties, ensure_ascii=False)])
                logger.info(f"Exported edges to {edges_file}.")
            except IOError as e:
                logger.error(f"Error exporting to CSV: {e}")
                raise KnowledgeGraphError(f"Error exporting to CSV: {e}")

    def import_from_csv(self, nodes_file: str, edges_file: str) -> None:
        """
        Imports nodes and edges into the graph from separate CSV files.

        Args:
            nodes_file (str): The path to the CSV file containing nodes.
            edges_file (str): The path to the CSV file containing edges.

        Raises:
            KnowledgeGraphError: If an I/O, CSV parsing, or JSON decoding error occurs during import.
        """
        with self.lock:
            try:
                # Import nodes
                with open(nodes_file, 'r', encoding='utf-8') as f_nodes:
                    reader = csv.DictReader(f_nodes)
                    for row in reader:
                        node_id = row['id']
                        labels = row['labels'].split(",") if row['labels'] else []
                        properties = json.loads(row['properties']) if row['properties'] else {}
                        node = Node(id=node_id, labels=set(labels), properties=properties)
                        self.nodes[node.id] = node

                        # Update label index
                        for label in node.labels:
                            self.label_index[label].add(node.id)
                        
                        # Update property index
                        for prop, value in node.properties.items():
                            self.property_index[prop][value].add(node.id)
                logger.info(f"Imported nodes from {nodes_file}.")

                # Import edges
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

                        # Update adjacency
                        self.adjacency[source].add(edge.id)
                        self.adjacency[target].add(edge.id)
                logger.info(f"Imported edges from {edges_file}.")
            except (IOError, csv.Error, json.JSONDecodeError) as e:
                logger.error(f"Error importing from CSV: {e}")
                raise KnowledgeGraphError(f"Error importing from CSV: {e}")

    def find_shortest_path(self, start_node_id: str, end_node_id: str, weighted: bool = False) -> Optional[List[str]]:
        """
        Finds the shortest path between two nodes using BFS for unweighted graphs or Dijkstra for weighted graphs.

        Args:
            start_node_id (str): The unique identifier of the start node.
            end_node_id (str): The unique identifier of the end node.
            weighted (bool, optional): Whether to consider edge weights. Defaults to False.

        Returns:
            Optional[List[str]]: A list of node IDs representing the shortest path, or None if no path exists.

        Raises:
            NodeNotFoundError: If either the start or end node does not exist.
            KnowledgeGraphError: If nodes are not present in the NetworkX graph during weighted path finding.
        """
        if start_node_id not in self.nodes or end_node_id not in self.nodes:
            logger.error("One or both of the specified nodes do not exist.")
            raise NodeNotFoundError("One or both of the specified nodes do not exist.")
        
        if not weighted:
            # BFS algorithm for unweighted graphs
            visited = set()
            queue = deque([[start_node_id]])

            while queue:
                path = queue.popleft()
                node_id = path[-1]

                if node_id == end_node_id:
                    logger.info(f"Shortest path found between {start_node_id} and {end_node_id}: {path}")
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
            
            logger.info(f"No path found between {start_node_id} and {end_node_id}.")
            return None
        else:
            # Dijkstra's algorithm for weighted graphs
            G = self._to_networkx_graph(weighted=True)
            try:
                path = nx.dijkstra_path(G, start_node_id, end_node_id, weight='weight')
                logger.info(f"Weighted shortest path between {start_node_id} and {end_node_id}: {path}")
                return path
            except nx.NetworkXNoPath:
                logger.info(f"No weighted path found between {start_node_id} and {end_node_id}.")
                return None
            except nx.NodeNotFound:
                logger.error("One of the specified nodes does not exist in the NetworkX graph.")
                raise KnowledgeGraphError("One of the specified nodes does not exist in the NetworkX graph.")

    def has_cycle(self) -> bool:
        """
        Checks if the graph contains any cycles using NetworkX's cycle detection.

        Returns:
            bool: True if a cycle exists, False otherwise.
        """
        G = self._to_networkx_graph()
        try:
            cycles = list(nx.find_cycle(G, orientation='ignore'))
            if cycles:
                logger.info("Cycle detected in the graph.")
                return True
        except nx.NetworkXNoCycle:
            logger.info("No cycles detected in the graph.")
            return False
        return False

    def traverse_bfs(self, start_node_id: str, visit_func: Callable[[Node], None]) -> None:
        """
        Traverses the graph in breadth-first order starting from a specified node.
        Applies a user-defined function to each visited node.

        Args:
            start_node_id (str): The unique identifier of the starting node.
            visit_func (Callable[[Node], None]): A function to apply to each visited node.

        Raises:
            NodeNotFoundError: If the starting node does not exist.
        """
        if start_node_id not in self.nodes:
            logger.error(f"Starting node {start_node_id} does not exist.")
            raise NodeNotFoundError(f"Starting node {start_node_id} does not exist.")
        
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
        logger.info(f"BFS traversal completed starting from node {start_node_id}.")

    def traverse_dfs(self, start_node_id: str, visit_func: Callable[[Node], None]) -> None:
        """
        Traverses the graph in depth-first order starting from a specified node.
        Applies a user-defined function to each visited node.

        Args:
            start_node_id (str): The unique identifier of the starting node.
            visit_func (Callable[[Node], None]): A function to apply to each visited node.

        Raises:
            NodeNotFoundError: If the starting node does not exist.
        """
        if start_node_id not in self.nodes:
            logger.error(f"Starting node {start_node_id} does not exist.")
            raise NodeNotFoundError(f"Starting node {start_node_id} does not exist.")
        
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
        logger.info(f"DFS traversal completed starting from node {start_node_id}.")

    def add_label_to_node(self, node_id: str, label: str) -> None:
        """
        Adds a label to an existing node.

        Args:
            node_id (str): The unique identifier of the node.
            label (str): The label to add.

        Raises:
            NodeNotFoundError: If the node does not exist.
        """
        with self.lock:
            node = self.get_node(node_id)
            if node is None:
                logger.error(f"Node {node_id} does not exist.")
                raise NodeNotFoundError(f"Node {node_id} does not exist.")
            
            if label not in node.labels:
                node.labels.add(label)
                self.label_index[label].add(node_id)
                logger.info(f"Added label '{label}' to node {node_id}.")
            else:
                logger.warning(f"Node {node_id} already has label '{label}'.")

    def remove_label_from_node(self, node_id: str, label: str) -> None:
        """
        Removes a label from an existing node.

        Args:
            node_id (str): The unique identifier of the node.
            label (str): The label to remove.

        Raises:
            NodeNotFoundError: If the node does not exist.
        """
        with self.lock:
            node = self.get_node(node_id)
            if node is None:
                logger.error(f"Node {node_id} does not exist.")
                raise NodeNotFoundError(f"Node {node_id} does not exist.")
            
            if label in node.labels:
                node.labels.remove(label)
                self.label_index[label].discard(node_id)
                if not self.label_index[label]:
                    del self.label_index[label]
                logger.info(f"Removed label '{label}' from node {node_id}.")
            else:
                logger.warning(f"Node {node_id} does not have label '{label}'.")

    def get_degree_centrality(self) -> Dict[str, int]:
        """
        Calculates the degree centrality for each node in the graph.

        Returns:
            Dict[str, int]: A dictionary mapping node IDs to their degree centrality.
        """
        centrality = {node_id: len(edges) for node_id, edges in self.adjacency.items()}
        logger.info("Calculated degree centrality for all nodes.")
        return centrality

    def visualize_graph(self, with_labels: bool = True, node_size: int = 300, edge_color: str = 'gray') -> None:
        """
        Visualizes the graph using NetworkX and Matplotlib.

        Args:
            with_labels (bool, optional): Whether to display node labels. Defaults to True.
            node_size (int, optional): The size of the nodes in the visualization. Defaults to 300.
            edge_color (str, optional): The color of the edges in the visualization. Defaults to 'gray'.
        """
        G = self._to_networkx_graph()
        pos = nx.spring_layout(G)
        plt.figure(figsize=(12, 8))
        nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color='lightblue')
        nx.draw_networkx_edges(G, pos, edge_color=edge_color)
        if with_labels:
            labels = {node_id: self.nodes[node_id].properties.get('name', node_id) for node_id in G.nodes()}
            nx.draw_networkx_labels(G, pos, labels, font_size=10)
        plt.title("Knowledge Graph Visualization")
        plt.axis('off')
        plt.show()
        logger.info("Graph visualization completed.")

    def _to_networkx_graph(self, weighted: bool = False) -> nx.Graph:
        """
        Converts the KnowledgeGraph into a NetworkX Graph object.

        Args:
            weighted (bool, optional): Whether to include edge weights. Defaults to False.

        Returns:
            nx.Graph: The corresponding NetworkX Graph object.
        """
        G = nx.Graph()
        for node_id, node in self.nodes.items():
            G.add_node(node_id, **node.properties)
        for edge_id, edge in self.edges.items():
            if weighted and 'weight' in edge.properties:
                G.add_edge(edge.source, edge.target, key=edge_id, type=edge.type, weight=edge.properties['weight'], **edge.properties)
            else:
                G.add_edge(edge.source, edge.target, key=edge_id, type=edge.type, **edge.properties)
        logger.debug("Converted KnowledgeGraph to NetworkX Graph.")
        return G

    def execute_transaction(self, operations: List[Callable[[], None]]) -> None:
        """
        Executes a series of operations atomically. If any operation fails, 
        all previous operations in the transaction are rolled back.

        Args:
            operations (List[Callable[[], None]]): A list of functions representing operations to execute.

        Raises:
            TransactionError: If any operation within the transaction fails.
        """
        with self.lock:
            # Backup current state
            backup = {
                "nodes": {node_id: node.__dict__.copy() for node_id, node in self.nodes.items()},
                "edges": {edge_id: edge.__dict__.copy() for edge_id, edge in self.edges.items()},
                "label_index": {k: v.copy() for k, v in self.label_index.items()},
                "property_index": {k: {pk: pv.copy() for pk, pv in v.items()} for k, v in self.property_index.items()},
                "adjacency": {k: v.copy() for k, v in self.adjacency.items()}
            }
            logger.debug("Backup created before transaction.")

            try:
                for operation in operations:
                    operation()
                logger.info("Transaction executed successfully.")
            except Exception as e:
                # Restore from backup on failure
                self.nodes = {k: Node(**v) for k, v in backup["nodes"].items()}
                self.edges = {k: Edge(**v) for k, v in backup["edges"].items()}
                self.label_index = defaultdict(set, {k: set(v) for k, v in backup["label_index"].items()})
                self.property_index = defaultdict(lambda: defaultdict(set), {
                    k: defaultdict(set, {pk: set(pv) for pk, pv in v.items()}) 
                    for k, v in backup["property_index"].items()
                })
                self.adjacency = defaultdict(set, {k: set(v) for k, v in backup["adjacency"].items()})
                logger.error(f"Transaction failed: {e}. State restored from backup.")
                raise TransactionError(f"Transaction failed: {e}")

# Example usage of the KnowledgeGraph
if __name__ == "__main__":
    kg = KnowledgeGraph()
    
    # Adding nodes
    alice_id = kg.add_node(labels=["Person"], properties={"name": "Alice", "age": 30, "hobby": "chess"})
    bob_id = kg.add_node(labels=["Person"], properties={"name": "Bob", "age": 25, "hobby": "football"})
    charlie_id = kg.add_node(labels=["Person"], properties={"name": "Charlie", "age": 35, "hobby": "guitar"})
    diana_id = kg.add_node(labels=["Person"], properties={"name": "Diana", "age": 28, "hobby": "painting"})
    paris_id = kg.add_node(labels=["City"], properties={"name": "Paris", "country": "France", "population": 2148327})
    openai_id = kg.add_node(labels=["Company"], properties={"name": "OpenAI", "industry": "AI Research"})
    techcorp_id = kg.add_node(labels=["Company"], properties={"name": "TechCorp", "industry": "Software Development"})
    
    # Adding edges
    kg.add_edge(alice_id, bob_id, "KNOWS", properties={"since": 2015})
    kg.add_edge(alice_id, charlie_id, "KNOWS", properties={"since": 2018})
    kg.add_edge(bob_id, diana_id, "KNOWS", properties={"since": 2020})
    kg.add_edge(alice_id, paris_id, "LIVES_IN", properties={"since": 2020})
    kg.add_edge(bob_id, openai_id, "WORKS_AT", properties={"role": "Engineer"})
    kg.add_edge(charlie_id, techcorp_id, "WORKS_AT", properties={"role": "Developer"})
    kg.add_edge(openai_id, techcorp_id, "COLLABORATES_WITH", properties={"project": "AI Integration"})
    
    # Node queries
    persons = kg.find_nodes_by_label("Person")
    logger.info(f"List of persons in the graph: {[person.properties['name'] for person in persons]}")
    
    alice_nodes = kg.find_nodes_by_property("name", "Alice")
    logger.info(f"Search for Alice node: {[node.id for node in alice_nodes]}")
    
    # Combined search
    chess_players = kg.find_nodes(labels=["Person"], properties={"hobby": "chess"})
    logger.info(f"List of chess players: {[node.properties['name'] for node in chess_players]}")
    
    # Updating node properties
    kg.update_node_properties(alice_id, {"hobby": "chess", "profession": "Data Scientist"})
    updated_alice = kg.get_node(alice_id)
    logger.info(f"Updated properties for Alice: {updated_alice.properties}")
    
    # Managing labels
    kg.add_label_to_node(alice_id, "Expert")
    kg.remove_label_from_node(alice_id, "Person")
    updated_labels_alice = kg.get_node(alice_id).labels
    logger.info(f"Current labels for Alice: {updated_labels_alice}")
    
    # Retrieving outgoing edges
    alice_out_edges = kg.get_edges_from_node(alice_id)
    logger.info(f"Outgoing edges from Alice: {[edge.type for edge in alice_out_edges]}")
    
    # Updating edge properties
    lives_in_edge_id = next((edge.id for edge in kg.get_edges_from_node(alice_id) if edge.type == "LIVES_IN"), None)
    if lives_in_edge_id:
        kg.update_edge_properties(lives_in_edge_id, {"visited_landmarks": ["Eiffel Tower", "Louvre Museum"]})
        logger.info(f"Updated properties for LIVES_IN edge: {kg.edges[lives_in_edge_id].properties}")
    else:
        logger.warning("LIVES_IN edge not found for Alice.")
    
    # BFS Traversal
    def print_node(node: Node):
        print(f"Visited: {node.properties.get('name', node.id)}")
    
    logger.info("Starting BFS traversal from Alice.")
    kg.traverse_bfs(alice_id, print_node)
    
    # DFS Traversal
    logger.info("Starting DFS traversal from Alice.")
    kg.traverse_dfs(alice_id, print_node)
    
    # Finding the shortest path
    path = kg.find_shortest_path(alice_id, diana_id)
    if path:
        logger.info(f"Shortest path between Alice and Diana: {path}")
    else:
        logger.info("No path found between Alice and Diana.")
    
    # Finding the weighted shortest path (adding weights)
    # Adding weights to edges
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
        logger.info(f"Weighted shortest path between Alice and Diana: {weighted_path}")
    else:
        logger.info("No weighted path found between Alice and Diana.")
    
    # Calculating degree centrality
    centrality = kg.get_degree_centrality()
    logger.info(f"Degree centrality: {centrality}")
    
    # Checking for cycles
    has_cycle = kg.has_cycle()
    logger.info(f"Graph contains a cycle: {has_cycle}")
    
    # Visualizing the graph
    kg.visualize_graph()
    
    # Exporting the graph to CSV files
    kg.export_to_csv("nodes.csv", "edges.csv")
    
    # Saving the graph to a JSON file
    kg.save_to_json("graph.json")
    
    # Loading the graph from a JSON file
    kg2 = KnowledgeGraph()
    kg2.load_from_json("graph.json")
    logger.info(f"KG2 loaded nodes: {[node.properties['name'] for node in kg2.get_all_nodes()]}")
    logger.info(f"KG2 loaded edges: {[edge.type for edge in kg2.get_all_edges()]}")
    
    # Importing from CSV into a new graph
    kg3 = KnowledgeGraph()
    kg3.import_from_csv("nodes.csv", "edges.csv")
    logger.info(f"KG3 imported nodes: {[node.properties['name'] for node in kg3.get_all_nodes()]}")
    logger.info(f"KG3 imported edges: {[edge.type for edge in kg3.get_all_edges()]}")
    
    # Example of a transaction
    try:
        kg.execute_transaction([
            lambda: kg.add_node(labels=["Person"], properties={"name": "Eve", "age": 22, "hobby": "reading"}),
            lambda: kg.add_edge(alice_id, "non_existent_node", "KNOWS")
        ])
    except TransactionError as te:
        logger.error(f"Transaction failed: {te}")
    
    # Verifying that the transaction was rolled back
    eve_nodes = kg.find_nodes_by_property("name", "Eve")
    logger.info(f"Search for Eve node after transaction: {[node.id for node in eve_nodes]}")
