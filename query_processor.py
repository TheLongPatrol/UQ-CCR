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
        
        Following the paper's algorithm (equations 5-8):
        - Extract keywords q_1, ..., q_k from query
        - Encode each keyword separately: e_{q_i} = SentenceBERT(q_i)
        - For each node v, compute sim(q_i, v) = cosine similarity
        - S = union of {v ∈ V : sim(q_i, v) > τ} for all keywords
        
        Args:
            query: User query string
            kg: Knowledge graph instance
            
        Returns:
            List of (node, similarity_score) tuples sorted by similarity (descending)
        """
        # Compute node embeddings if not already done
        nodes = kg.get_nodes()
        if not self.node_embeddings:
            self.compute_node_embeddings(nodes)
        
        # Step 1: Extract keywords from query
        keywords = self.extract_keywords(query)
        
        # If no keywords extracted, fall back to using the full query
        if not keywords:
            keywords = [query]
        
        # Step 2: Encode each keyword separately (equation 5)
        keyword_embeddings = {kw: self.model.encode([kw])[0] for kw in keywords}
        
        # Step 3 & 4: For each node, compute similarity with all keywords
        # and take the union of nodes where any sim(q_i, v) > τ (equation 8)
        node_best_scores = {}  # Track best similarity for each node
        
        for node in nodes:
            if node not in self.node_embeddings:
                continue
                
            node_emb = self.node_embeddings[node]
            
            # Compute similarity with each keyword (equation 7)
            for kw, kw_emb in keyword_embeddings.items():
                similarity = cosine_similarity([kw_emb], [node_emb])[0][0]
                
                # Check if above threshold τ
                if similarity >= self.tau:
                    # Keep track of the best (max) similarity for this node
                    if node not in node_best_scores or similarity > node_best_scores[node]:
                        node_best_scores[node] = float(similarity)
        
        # Convert to list of tuples
        node_similarities = [(node, score) for node, score in node_best_scores.items()]
        
        # Sort by similarity (descending)
        node_similarities.sort(key=lambda x: x[1], reverse=True)
        
        print(f"Found {len(node_similarities)} starting nodes above threshold {self.tau}")
        if keywords:
            print(f"Keywords used: {keywords}")
        return node_similarities
    
    def extract_keywords(self, query: str) -> List[str]:
        """
        Extract key entities/keywords from query for semantic matching.
        
        Combines individual content words into meaningful phrases where possible.
        
        Args:
            query: User query string
            
        Returns:
            List of keywords/key phrases
        """
        # Extended stop words for query processing
        stop_words = {
            'what', 'how', 'why', 'when', 'where', 'who', 'which',
            'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did',
            'the', 'a', 'an', 'this', 'that', 'these', 'those',
            'can', 'could', 'would', 'should', 'will', 'may', 'might', 'must',
            'of', 'to', 'in', 'for', 'on', 'with', 'by', 'from', 'about',
            'as', 'at', 'into', 'through', 'during', 'before', 'after',
            'and', 'or', 'but', 'if', 'then', 'so', 'because',
            'it', 'its', 'their', 'there', 'here'
        }
        
        # Clean and tokenize
        cleaned = query.replace('?', ' ').replace('.', ' ').replace(',', ' ').replace('!', ' ')
        words = cleaned.split()
        
        # Extract content words (non-stopwords)
        content_words = []
        for word in words:
            word_clean = word.strip().lower()
            if word_clean and word_clean not in stop_words and len(word_clean) > 1:
                # Preserve original case for proper nouns
                content_words.append(word.strip())
        
        # Also try to extract multi-word phrases (bigrams of content words)
        keywords = []
        
        # Add individual content words
        keywords.extend(content_words)
        
        # Add bigrams of adjacent content words for better entity matching
        if len(content_words) >= 2:
            for i in range(len(content_words) - 1):
                bigram = f"{content_words[i]} {content_words[i+1]}"
                keywords.append(bigram)
        
        return keywords
