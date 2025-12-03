"""
Knowledge Graph Construction Module
Builds a directed graph from triples using NetworkX.
"""

import networkx as nx
from typing import List, Dict, Set, Tuple, Optional
import matplotlib.pyplot as plt


class KnowledgeGraph:
    """Constructs and manages a knowledge graph from triples."""
    
    def __init__(self):
        """Initialize an empty directed graph."""
        self.graph = nx.DiGraph()
        self.node_embeddings = {}  # Store embeddings for nodes
        
    def build_from_triples(self, triples: List[Dict]):
        """
        Build knowledge graph from normalized triples.
        
        Args:
            triples: List of triples with 'cause', 'relation', 'effect'
        """
        print(f"Building knowledge graph from {len(triples)} triples...")
        
        for triple in triples:
            cause = triple.get('cause')
            effect = triple.get('effect')
            relation = triple.get('relation', 'related_to')
            score = triple.get('score')

            if cause and effect:
                # Add nodes if they don't exist
                if not self.graph.has_node(cause):
                    self.graph.add_node(cause)
                if not self.graph.has_node(effect):
                    self.graph.add_node(effect)
                
                # Add edge with relation as attribute
                if score:
                    self.graph.add_edge(cause,effect, relation=relation, score=score)
                else:
                    self.graph.add_edge(cause, effect, relation=relation)
        
        print(f"Graph constructed: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")
    
    def get_nodes(self) -> List[str]:
        """Get all nodes in the graph."""
        return list(self.graph.nodes())
    
    def get_edges(self) -> List[Tuple[str, str]]:
        """Get all edges in the graph."""
        return list(self.graph.edges())
    
    def get_neighbors(self, node: str, direction: str = 'forward') -> List[str]:
        """
        Get neighbors of a node based on direction.
        
        Args:
            node: The node to get neighbors for
            direction: 'forward' (successors), 'backward' (predecessors), or 'bidirectional' (both)
            
        Returns:
            List of neighbor nodes
        """
        if not self.graph.has_node(node):
            return []
        
        if direction == 'forward':
            return list(self.graph.successors(node))
        elif direction == 'backward':
            return list(self.graph.predecessors(node))
        elif direction == 'bidirectional':
            return list(set(self.graph.successors(node)) | set(self.graph.predecessors(node)))
        else:
            raise ValueError(f"Invalid direction: {direction}")
    
    def get_edge_relation(self, source: str, target: str) -> Optional[str]:
        """Get the relation label of an edge."""
        if self.graph.has_edge(source, target):
            return self.graph[source][target].get('relation')
        return None
    
    def get_edge_relation_with_score(self, source: str, target:str) -> Optional[Tuple[str, float]]:
        if self.graph.has_edge(source,target):
            edge = self.graph[source][target]
            return (edge.get('relation'), edge.get('score'))
        return None
    
    def dfs_traversal(self, start_node: str, direction: str = 'forward', 
                     max_depth: int = 5, max_chains: int = 100) -> List[List[str]]:
        """
        Perform depth-first traversal to generate reasoning chains.
        
        Args:
            start_node: Starting node for traversal
            direction: Traversal direction ('forward', 'backward', 'bidirectional')
            max_depth: Maximum depth of traversal
            max_chains: Maximum number of chains to return
            
        Returns:
            List of reasoning chains (each chain is a list of nodes)
        """
        if not self.graph.has_node(start_node):
            return []
        
        chains = []
        visited_paths = set()
        
        def dfs(node: str, path: List[str], depth: int):
            if depth > max_depth or len(chains) >= max_chains:
                return
            
            # Add current path if it's longer than just the start node
            if len(path) > 1:
                path_tuple = tuple(path)
                if path_tuple not in visited_paths:
                    chains.append(path.copy())
                    visited_paths.add(path_tuple)
            
            # Get neighbors based on direction
            neighbors = self.get_neighbors(node, direction)
            
            for neighbor in neighbors:
                # Avoid cycles
                if neighbor not in path:
                    path.append(neighbor)
                    dfs(neighbor, path, depth + 1)
                    path.pop()
        
        dfs(start_node, [start_node], 0)
        return chains
    
    def get_subgraph(self, nodes: List[str]) -> nx.DiGraph:
        """Extract a subgraph containing only the specified nodes."""
        return self.graph.subgraph(nodes).copy()
    
    def visualize(self, output_file: Optional[str] = None, max_nodes: int = 50):
        """
        Visualize the knowledge graph (limited to max_nodes for readability).
        
        Args:
            output_file: If provided, save visualization to this file
            max_nodes: Maximum number of nodes to visualize
        """
        if self.graph.number_of_nodes() > max_nodes:
            print(f"Graph too large ({self.graph.number_of_nodes()} nodes). Visualizing first {max_nodes} nodes...")
            nodes = list(self.graph.nodes())[:max_nodes]
            subgraph = self.get_subgraph(nodes)
        else:
            subgraph = self.graph
        
        plt.figure(figsize=(15, 10))
        pos = nx.spring_layout(subgraph, k=0.5, iterations=50)
        
        nx.draw(subgraph, pos, with_labels=True, node_color='lightblue', 
                node_size=500, font_size=8, font_weight='bold',
                arrows=True, edge_color='gray', arrowsize=20)
        
        plt.title("Knowledge Graph Visualization")
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {output_file}")
        else:
            plt.show()
        plt.close()
    
    def get_graph_stats(self) -> Dict:
        """Get statistics about the knowledge graph."""
        return {
            'num_nodes': self.graph.number_of_nodes(),
            'num_edges': self.graph.number_of_edges(),
            'avg_degree': sum(dict(self.graph.degree()).values()) / max(self.graph.number_of_nodes(), 1),
            'density': nx.density(self.graph),
            'is_connected': nx.is_weakly_connected(self.graph) if self.graph.number_of_nodes() > 0 else False
        }

