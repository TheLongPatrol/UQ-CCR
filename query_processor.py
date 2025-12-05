"""
Query Processing Module
Parses causal queries to extract cause and effect entities,
then finds starting nodes (cause) and target nodes (effect) in the graph.

For forward queries like "How did X cause/contribute to Y":
  - Starting nodes = nodes related to X (cause)
  - Target nodes = nodes related to Y (effect)
  
For backward queries like "What caused Y":
  - Starting nodes = nodes related to Y (effect)
  - Target nodes = nodes related to causes (discovered via traversal)
"""

import numpy as np
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple, Dict, Optional
from knowledge_graph import KnowledgeGraph


class QueryProcessor:
    """Processes queries to find relevant starting and target nodes in the knowledge graph."""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', tau: float = 0.5):
        """
        Initialize the query processor.
        
        Args:
            model_name: Name of the SentenceBERT model to use
            tau: Similarity threshold for selecting nodes (default: 0.5)
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
    
    def extract_cause_effect(self, query: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Extract cause and effect phrases from a causal query.
        
        For "How did X cause/contribute/affect Y":
          - cause = X
          - effect = Y
          
        For "What caused Y" / "Why did Y happen":
          - cause = None (to be discovered)
          - effect = Y
        
        Args:
            query: User query string
            
        Returns:
            Tuple of (cause_phrase, effect_phrase)
        """
        query_lower = query.lower()
        
        # Pattern 1: "How did X contribute/cause/affect/influence Y"
        # Captures: X = cause, Y = effect
        forward_patterns = [
            # "How did X cause/contribute to Y" - use greedy match for cause phrase
            (r"how did (.+) (cause|contribute to|lead to|affect|influence|impact|drive|shape) (.+?)[\?\.]?$", 0, 2),
            # "How does X affect Y"
            (r"how does (.+) (cause|contribute to|lead to|affect|influence|impact|drive|shape) (.+?)[\?\.]?$", 0, 2),
            # "What effect did X have on Y"
            (r"what (effect|impact|influence|role) did (.+?) have (?:on |in )?(.+?)[\?\.]?$", 1, 2),
            # "What were the effects of X on Y"
            (r"what were the (?:effects?|consequences?|impacts?|results?) of (.+?) on (.+?)[\?\.]?$", 0, 1),
        ]
        
        # Pattern 2: "What caused Y" / "Why did Y happen"
        # Captures: cause = None, effect = Y
        # Format: (pattern, effect_group_index)
        backward_patterns = [
            (r"what caused (.+?)[\?\.]?$", 0),
            (r"what causes (.+?)[\?\.]?$", 0),
            (r"why did (.+?) (?:happen|occur|rise|fall|increase|decrease|change)[\?\.]?$", 0),
            (r"why does (.+?) (?:happen|occur|rise|fall|increase|decrease|change)[\?\.]?$", 0),
            (r"what led to (.+?)[\?\.]?$", 0),
            (r"what (?:are|were) the (?:causes?|reasons?|factors?) (?:of|for|behind) (.+?)[\?\.]?$", 0),
        ]
        
        # Try forward patterns
        # Format: (pattern, cause_group_index, effect_group_index)
        for pattern, cause_idx, effect_idx in forward_patterns:
            match = re.search(pattern, query_lower)
            if match:
                groups = match.groups()
                cause = groups[cause_idx].strip()
                effect = groups[effect_idx].strip()
                
                # Clean up effect - remove leading prepositions
                effect = re.sub(r"^(to |on |in )", "", effect).strip()
                
                print(f"Extracted cause: '{cause}'")
                print(f"Extracted effect: '{effect}'")
                return (cause, effect)
        
        # Try backward patterns
        for pattern, effect_idx in backward_patterns:
            match = re.search(pattern, query_lower)
            if match:
                groups = match.groups()
                effect = groups[effect_idx].strip()
                
                print(f"Extracted cause: None (backward query)")
                print(f"Extracted effect: '{effect}'")
                return (None, effect)
        
        # Fallback: couldn't parse, return None
        print("Could not extract cause/effect from query")
        return (None, None)
    
    def find_matching_nodes(
        self, 
        phrase: str, 
        kg: KnowledgeGraph,
        top_k: int = 50
    ) -> List[Tuple[str, float]]:
        """
        Find nodes in the graph that semantically match a given phrase.
        
        Args:
            phrase: The phrase to match (e.g., cause or effect)
            kg: Knowledge graph instance
            top_k: Maximum number of nodes to return
            
        Returns:
            List of (node, similarity_score) tuples sorted by similarity (descending)
        """
        # Compute node embeddings if not already done
        nodes = kg.get_nodes()
        if not self.node_embeddings:
            self.compute_node_embeddings(nodes)
        
        # Extract keywords from the phrase
        keywords = self._extract_keywords_from_phrase(phrase)
        
        if not keywords:
            keywords = [phrase]
        
        # Encode each keyword
        keyword_embeddings = {kw: self.model.encode([kw])[0] for kw in keywords}
        
        # Find nodes matching any keyword above threshold
        node_best_scores = {}
        
        for node in nodes:
            if node not in self.node_embeddings:
                continue
                
            node_emb = self.node_embeddings[node]
            
            for kw, kw_emb in keyword_embeddings.items():
                similarity = cosine_similarity([kw_emb], [node_emb])[0][0]
                
                if similarity >= self.tau:
                    if node not in node_best_scores or similarity > node_best_scores[node]:
                        node_best_scores[node] = float(similarity)
        
        # Convert to sorted list
        node_similarities = [(node, score) for node, score in node_best_scores.items()]
        node_similarities.sort(key=lambda x: x[1], reverse=True)
        
        return node_similarities[:top_k]
    
    def find_starting_nodes(
        self, 
        query: str, 
        direction: str,
        kg: KnowledgeGraph
    ) -> List[Tuple[str, float]]:
        """
        Find starting nodes based on query and reasoning direction.
        
        For forward direction: starting nodes match the CAUSE
        For backward direction: starting nodes match the EFFECT
        
        Args:
            query: User query string
            direction: 'forward', 'backward', or 'bidirectional'
            kg: Knowledge graph instance
            
        Returns:
            List of (node, similarity_score) tuples
        """
        cause, effect = self.extract_cause_effect(query)
        
        if direction == 'forward':
            # Forward: start from cause nodes
            if cause:
                print(f"Finding starting nodes matching CAUSE: '{cause}'")
                nodes = self.find_matching_nodes(cause, kg)
                print(f"Found {len(nodes)} starting nodes (cause-related)")
                return nodes
            else:
                # Fallback to all keywords if can't extract cause
                print("Warning: Could not extract cause, using full query")
                return self._find_nodes_from_query(query, kg)
                
        elif direction == 'backward':
            # Backward: start from effect nodes, traverse backward to find causes
            if effect:
                print(f"Finding starting nodes matching EFFECT: '{effect}'")
                nodes = self.find_matching_nodes(effect, kg)
                print(f"Found {len(nodes)} starting nodes (effect-related)")
                return nodes
            else:
                print("Warning: Could not extract effect, using full query")
                return self._find_nodes_from_query(query, kg)
                
        else:  # bidirectional
            # Use all keywords for bidirectional
            return self._find_nodes_from_query(query, kg)
    
    def find_target_nodes(
        self, 
        query: str, 
        direction: str,
        kg: KnowledgeGraph
    ) -> List[Tuple[str, float]]:
        """
        Find target/ending nodes based on query and reasoning direction.
        
        For forward direction: target nodes match the EFFECT
        For backward direction: no specific target (exploring causes)
        
        Args:
            query: User query string
            direction: 'forward', 'backward', or 'bidirectional'
            kg: Knowledge graph instance
            
        Returns:
            List of (node, similarity_score) tuples, or empty list for backward
        """
        cause, effect = self.extract_cause_effect(query)
        
        if direction == 'forward':
            # Forward: target is the effect
            if effect:
                print(f"Finding target nodes matching EFFECT: '{effect}'")
                nodes = self.find_matching_nodes(effect, kg)
                print(f"Found {len(nodes)} target nodes (effect-related)")
                return nodes
            else:
                return []
                
        elif direction == 'backward':
            # Backward: target is the cause (but we're discovering it)
            if cause:
                print(f"Finding target nodes matching CAUSE: '{cause}'")
                nodes = self.find_matching_nodes(cause, kg)
                print(f"Found {len(nodes)} target nodes (cause-related)")
                return nodes
            else:
                return []
                
        else:  # bidirectional
            return []
    
    def _find_nodes_from_query(
        self, 
        query: str, 
        kg: KnowledgeGraph
    ) -> List[Tuple[str, float]]:
        """
        Fallback: Find nodes matching any keyword in the full query.
        
        Args:
            query: User query string
            kg: Knowledge graph instance
            
        Returns:
            List of (node, similarity_score) tuples
        """
        nodes = kg.get_nodes()
        if not self.node_embeddings:
            self.compute_node_embeddings(nodes)
        
        keywords = self._extract_keywords_from_phrase(query)
        if not keywords:
            keywords = [query]
        
        keyword_embeddings = {kw: self.model.encode([kw])[0] for kw in keywords}
        
        node_best_scores = {}
        for node in nodes:
            if node not in self.node_embeddings:
                continue
            node_emb = self.node_embeddings[node]
            
            for kw, kw_emb in keyword_embeddings.items():
                similarity = cosine_similarity([kw_emb], [node_emb])[0][0]
                if similarity >= self.tau:
                    if node not in node_best_scores or similarity > node_best_scores[node]:
                        node_best_scores[node] = float(similarity)
        
        node_similarities = [(node, score) for node, score in node_best_scores.items()]
        node_similarities.sort(key=lambda x: x[1], reverse=True)
        
        print(f"Found {len(node_similarities)} nodes above threshold {self.tau}")
        return node_similarities
    
    def _extract_keywords_from_phrase(self, phrase: str) -> List[str]:
        """
        Extract keywords from a phrase for semantic matching.
        
        Args:
            phrase: Input phrase
            
        Returns:
            List of keywords and bigrams
        """
        # Stop words to filter out
        stop_words = {
            'what', 'how', 'why', 'when', 'where', 'who', 'which',
            'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did',
            'the', 'a', 'an', 'this', 'that', 'these', 'those',
            'can', 'could', 'would', 'should', 'will', 'may', 'might', 'must',
            'of', 'to', 'in', 'for', 'on', 'with', 'by', 'from', 'about',
            'as', 'at', 'into', 'through', 'during', 'before', 'after',
            'and', 'or', 'but', 'if', 'then', 'so', 'because',
            'it', 'its', 'their', 'there', 'here', "'s"
        }
        
        # Clean and tokenize
        cleaned = phrase.replace('?', ' ').replace('.', ' ').replace(',', ' ').replace("'s", " 's ")
        words = cleaned.split()
        
        # Extract content words
        content_words = []
        for word in words:
            word_clean = word.strip().lower()
            if word_clean and word_clean not in stop_words and len(word_clean) > 1:
                content_words.append(word.strip())
        
        keywords = []
        
        # Add the full phrase first (most specific)
        keywords.append(phrase)
        
        # Add individual content words
        keywords.extend(content_words)
        
        # Add bigrams
        if len(content_words) >= 2:
            for i in range(len(content_words) - 1):
                bigram = f"{content_words[i]} {content_words[i+1]}"
                keywords.append(bigram)
        
        return keywords
    
    # Legacy method for backward compatibility
    def extract_keywords(self, query: str) -> List[str]:
        """Extract keywords from query (legacy method)."""
        return self._extract_keywords_from_phrase(query)
