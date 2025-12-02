"""
Entity Clustering Module
Clusters semantically similar entities using SentenceBERT embeddings.
Entities with cosine similarity > 0.85 are merged.
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Set, Tuple
from collections import defaultdict


class EntityClusterer:
    """Clusters entities based on semantic similarity using SentenceBERT."""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', similarity_threshold: float = 0.85):
        """
        Initialize the entity clusterer.
        
        Args:
            model_name: Name of the SentenceBERT model to use
            similarity_threshold: Cosine similarity threshold for merging entities (default: 0.85)
        """
        self.model = SentenceTransformer(model_name)
        self.similarity_threshold = similarity_threshold
        self.entity_to_cluster = {}  # Maps original entity to canonical entity
        self.cluster_to_entities = defaultdict(set)  # Maps canonical entity to all merged entities
        
    def extract_entities_from_triples(self, triples: List[Dict]) -> Set[str]:
        """
        Extract unique entities from triples.
        
        Args:
            triples: List of triples, each containing 'cause' and 'effect'
            
        Returns:
            Set of unique entity strings
        """
        entities = set()
        for triple in triples:
            if 'cause' in triple:
                entities.add(triple['cause'])
            if 'effect' in triple:
                entities.add(triple['effect'])
        return entities
    
    def cluster_entities(self, entities: List[str]) -> Dict[str, str]:
        """
        Cluster entities based on semantic similarity.
        
        Args:
            entities: List of entity strings
            
        Returns:
            Dictionary mapping each entity to its canonical (cluster representative) entity
        """
        if not entities:
            return {}
        
        # Compute embeddings for all entities
        print(f"Computing embeddings for {len(entities)} entities...")
        embeddings = self.model.encode(entities, show_progress_bar=True)
        
        # Compute pairwise cosine similarities
        print("Computing pairwise similarities...")
        similarities = cosine_similarity(embeddings)
        
        # Cluster entities using greedy approach
        entity_list = list(entities)
        visited = set()
        
        for i, entity in enumerate(entity_list):
            if entity in visited:
                continue
                
            # This entity becomes the canonical representative
            canonical = entity
            cluster_members = {entity}
            visited.add(entity)
            self.entity_to_cluster[entity] = canonical
            
            # Find all similar entities
            for j, other_entity in enumerate(entity_list):
                if j <= i or other_entity in visited:
                    continue
                    
                if similarities[i][j] >= self.similarity_threshold:
                    cluster_members.add(other_entity)
                    visited.add(other_entity)
                    self.entity_to_cluster[other_entity] = canonical
            
            self.cluster_to_entities[canonical] = cluster_members
        
        print(f"Clustered {len(entities)} entities into {len(self.cluster_to_entities)} clusters")
        return self.entity_to_cluster
    
    def get_canonical_entity(self, entity: str) -> str:
        """Get the canonical (cluster representative) entity for a given entity."""
        return self.entity_to_cluster.get(entity, entity)
    
    def get_cluster_members(self, canonical_entity: str) -> Set[str]:
        """Get all entities in a cluster given its canonical entity."""
        return self.cluster_to_entities.get(canonical_entity, {canonical_entity})
    
    def normalize_triples(self, triples: List[Dict]) -> List[Dict]:
        """
        Normalize triples by replacing entities with their canonical representatives.
        
        Args:
            triples: List of triples with 'cause', 'relation', 'effect'
            
        Returns:
            List of normalized triples
        """
        normalized = []
        for triple in triples:
            normalized_triple = triple.copy()
            if 'cause' in triple:
                normalized_triple['cause'] = self.get_canonical_entity(triple['cause'])
            if 'effect' in triple:
                normalized_triple['effect'] = self.get_canonical_entity(triple['effect'])
            normalized.append(normalized_triple)
        return normalized

