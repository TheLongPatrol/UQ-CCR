"""
Chain Ranking Module
Ranks reasoning chains based on semantic similarity to the query.
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple, Dict, Optional
from knowledge_graph import KnowledgeGraph


class ChainRanker:
    """Ranks reasoning chains based on relevance to the query."""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', use_reliability_scores: bool = False):
        """
        Initialize the chain ranker.
        
        Args:
            model_name: Name of the SentenceBERT model to use
        """
        self.model = SentenceTransformer(model_name)
        self.use_reliab_score = use_reliability_scores
    def aggregate_chain_evidence(self, chain: List[str], kg: KnowledgeGraph) -> Tuple[str, Optional[float]]:
        """
        Aggregate evidence from a reasoning chain into a single text.
        
        Args:
            chain: List of nodes representing a reasoning chain
            kg: Knowledge graph instance
            
        Returns:
            Aggregated text representation of the chain
        """
        if not chain:
            return ""
        
        # Build chain description with relations
        chain_parts = []
        reliab_score = 1
        for i in range(len(chain)):
            chain_parts.append(chain[i])
            
            # Add relation if there's a next node
            if i < len(chain) - 1:
                if self.use_reliab_score:
                    result = kg.get_edge_relation_with_score(chain[i], chain[i+1])
                    if not result:
                        # Check reverse direction
                        result = kg.get_edge_relation_with_score(chain[i+1], chain[i])
                    relation, rel_score = result
                    if relation:
                        chain_parts.append(f"[{relation}]")
                    reliab_score*=rel_score
                else:
                    relation = kg.get_edge_relation(chain[i], chain[i+1])
                    if not relation:
                        # Check reverse direction
                        relation = kg.get_edge_relation(chain[i+1], chain[i])
                    if relation:
                        chain_parts.append(f"[{relation}]")
        
        # Join into coherent text
        aggregated = " → ".join(chain_parts)
        if self.use_reliab_score:
            return aggregated, reliab_score**(1/len(chain))
        else:
            return aggregated, None
    
    def score_chain(self, query: str, chain: List[str], kg: KnowledgeGraph) -> Tuple[float, Optional[float]]:
        """
        Score a single reasoning chain based on query relevance.
        
        Args:
            query: User query string
            chain: List of nodes representing a reasoning chain
            kg: Knowledge graph instance
            
        Returns:
            Similarity score between query and chain
        """
        # Aggregate chain evidence
        chain_text, reliab_score = self.aggregate_chain_evidence(chain, kg)
        
        if not chain_text:
            return 0.0
        
        # Compute embeddings
        query_embedding = self.model.encode([query])[0]
        chain_embedding = self.model.encode([chain_text])[0]
        
        # Compute cosine similarity
        similarity = cosine_similarity([query_embedding], [chain_embedding])[0][0]
        
        return float(similarity), reliab_score
    
    def score_chain_with_features(self, query: str, chain: List[str], 
                                  kg: KnowledgeGraph, weights: Dict[str, float] = None) -> Tuple[float, Optional[float]]:
        """
        Score a chain using multiple features.
        
        Args:
            query: User query string
            chain: List of nodes representing a reasoning chain
            kg: Knowledge graph instance
            weights: Dictionary of feature weights (default: equal weights)
            
        Returns:
            Weighted composite score
        """
        if weights is None:
            weights = {
                'semantic_similarity': 0.5,
                'chain_length': 0.2,
                'node_importance': 0.3
            }
        
        # Feature 1: Semantic similarity
        semantic_score, reliab_score = self.score_chain(query, chain, kg)
        
        # Feature 2: Chain length (normalized, prefer moderate lengths)
        optimal_length = 4
        length_score = 1.0 / (1.0 + abs(len(chain) - optimal_length) * 0.2)
        
        # Feature 3: Node importance (based on degree centrality)
        node_degrees = [kg.graph.degree(node) for node in chain if kg.graph.has_node(node)]
        avg_degree = np.mean(node_degrees) if node_degrees else 0
        max_possible_degree = max([d for n, d in kg.graph.degree()]) if kg.graph.number_of_nodes() > 0 else 1
        importance_score = avg_degree / max(max_possible_degree, 1)
        
        # Weighted combination
        total_score = (
            weights['semantic_similarity'] * semantic_score +
            weights['chain_length'] * length_score +
            weights['node_importance'] * importance_score
        )
        
        return float(total_score), reliab_score
    
    def rank_chains(self, query: str, chains: List[List[str]], 
                   kg: KnowledgeGraph, top_k: int = 10,
                   use_features: bool = True, thresh: float=0.85) -> List[Tuple[List[str], float]]:
        """
        Rank all reasoning chains and return top-k.
        
        Args:
            query: User query string
            chains: List of reasoning chains
            kg: Knowledge graph instance
            top_k: Number of top chains to return
            use_features: Whether to use multi-feature scoring
            
        Returns:
            List of (chain, score) tuples sorted by score (descending)
        """
        if not chains:
            return []
        
        print(f"Ranking {len(chains)} chains...")
        
        # Score all chains
        scored_chains = []
        for chain in chains:
            if use_features:
                score, reliab_score = self.score_chain_with_features(query, chain, kg)
            else:
                score, reliab_score = self.score_chain(query, chain, kg)
            if self.use_reliab_score and reliab_score > 0.85:
                scored_chains.append((chain, score, reliab_score))
            elif not self.use_reliab_score:
                scored_chains.append((chain, score, None))
        
        # Sort by score (descending)
        scored_chains.sort(key=lambda x: x[1], reverse=True)
        
        # Return top-k
        top_chains = scored_chains[:top_k]
        
        print(f"Top {len(top_chains)} chains selected")
        return top_chains
    
    def format_chain(self, chain: List[str], kg: KnowledgeGraph, score: float = None, reliab_score: float = None) -> str:
        """
        Format a reasoning chain for display.
        
        Args:
            chain: List of nodes representing a reasoning chain
            kg: Knowledge graph instance
            score: Optional score to include in formatting
            
        Returns:
            Formatted string representation of the chain
        """
        if not chain:
            return "Empty chain"
        
        formatted_parts = []
        raw_chain = []
        for i in range(len(chain)):
            formatted_parts.append(f"{chain[i]}")
            
            if i < len(chain) - 1:
                # Get relation
                relation = kg.get_edge_relation(chain[i], chain[i+1])
                forward_found = True
                if not relation:
                    forward_found = False
                    relation = kg.get_edge_relation(chain[i+1], chain[i])
                if forward_found:
                    raw_chain.append(f"<{chain[i]},{relation},{chain[i+1]}>")
                else:
                    raw_chain.append(f"<{chain[i+1]},{relation},{chain[i]}>")
                if relation and forward_found:
                    formatted_parts.append(f" --[{relation}]--> ")
                elif relation:
                    formatted_parts.append(f" --[{relation}]--> ")
                else:
                    formatted_parts.append(" --> ")
        
        chain_str = "".join(formatted_parts)
        
        if reliab_score is not None and score is not None:
            chain_str = f"[Reliability score: {reliab_score:.3f}] [Score: {score:.3f}] {chain_str}"
        elif reliab_score is not None:
            chain_str = f"[Reliability score: {reliab_score:.3f}] {chain_str}"
        elif score is not None:
            chain_str = f"[Score: {score:.3f}] {chain_str}"
        
        return chain_str, raw_chain

