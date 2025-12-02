"""
Query Processing Module
Finds starting nodes in the graph by computing cosine similarity
between query keywords and graph nodes.
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple, Dict
from knowledge_graph import KnowledgeGraph


class QueryProcessor:
    """Processes queries to find relevant starting nodes in the knowledge graph."""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', tau: float = 0.5):
        """
        Initialize the query processor.
        
        Args:
            model_name: Name of the SentenceBERT model to use
            tau: Similarity threshold for selecting starting nodes (default: 0.5)
        """
        self.model = SentenceTransformer(model_name)
        self.tau = tau
        self.node_embeddings = {}
        
    def compute_node_embeddings(self, nodes: List[str]):
        """
        Precompute embeddings for all graph nodes.
        
        Args:
            nodes: List of node names from the knowledge graph
        """
        print(f"Computing embeddings for {len(nodes)} graph nodes...")
        embeddings = self.model.encode(nodes, show_progress_bar=True)
        self.node_embeddings = {node: emb for node, emb in zip(nodes, embeddings)}
        
    def find_starting_nodes(self, query: str, kg: KnowledgeGraph) -> List[Tuple[str, float]]:
        """
        Find starting nodes for graph traversal based on query similarity.
        
        Args:
            query: User query string
            kg: Knowledge graph instance
            
        Returns:
            List of (node, similarity_score) tuples sorted by similarity (descending)
        """
        # Compute embeddings if not already done
        nodes = kg.get_nodes()
        if not self.node_embeddings:
            self.compute_node_embeddings(nodes)
        
        # Compute query embedding
        query_embedding = self.model.encode([query])[0]
        
        # Compute similarities with all nodes
        node_similarities = []
        for node in nodes:
            if node in self.node_embeddings:
                node_emb = self.node_embeddings[node]
                similarity = cosine_similarity([query_embedding], [node_emb])[0][0]
                
                if similarity >= self.tau:
                    node_similarities.append((node, float(similarity)))
        
        # Sort by similarity (descending)
        node_similarities.sort(key=lambda x: x[1], reverse=True)
        
        print(f"Found {len(node_similarities)} starting nodes above threshold {self.tau}")
        return node_similarities
    
    def extract_keywords(self, query: str) -> List[str]:
        """
        Extract keywords from query (simple implementation).
        
        Args:
            query: User query string
            
        Returns:
            List of keywords
        """
        # Simple keyword extraction - remove common words
        stop_words = {'what', 'how', 'why', 'when', 'where', 'is', 'are', 'the', 'a', 'an', 
                     'does', 'do', 'can', 'could', 'would', 'should', 'will', 'of', 'to', 'in',
                     'for', 'on', 'with', 'by', 'from', 'about', 'as', 'at', 'be', 'this', 'that'}
        
        words = query.lower().split()
        keywords = [w.strip('?.,!') for w in words if w.lower() not in stop_words]
        return keywords

