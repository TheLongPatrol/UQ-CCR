"""
Reasoning Direction Classifier
Classifies queries into forward, backward, or bidirectional reasoning directions.
"""

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import Literal, Dict, List
import numpy as np


class DirectionClassifier:
    """Classifies the required reasoning direction for a query."""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize the direction classifier.
        
        Args:
            model_name: Name of the SentenceBERT model to use
        """
        self.model = SentenceTransformer(model_name)
        
        # Define direction patterns and their embeddings
        self.direction_patterns = {
            'forward': [
                "What happens if",
                "What are the effects of",
                "What does this cause",
                "What are the consequences of",
                "What results from",
                "What leads to",
                "What is caused by this",
                "What follows from"
            ],
            'backward': [
                "What causes this",
                "Why does this happen",
                "What leads to this",
                "What are the reasons for",
                "What explains this",
                "What is the cause of",
                "What results in this",
                "What are the prerequisites for"
            ],
            'bidirectional': [
                "What is related to",
                "Tell me about",
                "Explain the relationship",
                "How is this connected",
                "What is associated with",
                "What are all factors involving",
                "Describe everything about",
                "What interactions involve"
            ]
        }
        
        # Precompute embeddings for patterns
        self.pattern_embeddings = {}
        for direction, patterns in self.direction_patterns.items():
            embeddings = self.model.encode(patterns)
            self.pattern_embeddings[direction] = embeddings
    
    def classify(self, query: str) -> Literal['forward', 'backward', 'bidirectional']:
        """
        Classify the reasoning direction for a query.
        
        Args:
            query: User query string
            
        Returns:
            Direction: 'forward', 'backward', or 'bidirectional'
        """
        # Encode query
        query_embedding = self.model.encode([query])[0]
        
        # Compute similarity with each direction's patterns
        direction_scores = {}
        for direction, pattern_embeddings in self.pattern_embeddings.items():
            similarities = cosine_similarity([query_embedding], pattern_embeddings)[0]
            direction_scores[direction] = float(np.max(similarities))
        
        # Select direction with highest score
        best_direction = max(direction_scores, key=direction_scores.get)
        
        print(f"Query direction classified as: {best_direction}")
        print(f"Scores: {direction_scores}")
        
        return best_direction
    
    def classify_with_rules(self, query: str) -> Literal['forward', 'backward', 'bidirectional']:
        """
        Classify using both semantic similarity and rule-based patterns.
        
        Args:
            query: User query string
            
        Returns:
            Direction: 'forward', 'backward', or 'bidirectional'
        """
        query_lower = query.lower()
        
        # Rule-based patterns
        forward_keywords = ['effect', 'consequence', 'result', 'leads to', 'causes what', 
                          'happens if', 'outcome', 'impact']
        backward_keywords = ['cause', 'reason', 'why', 'leads to this', 'results in this',
                           'explanation', 'source', 'origin', 'prerequisite']
        bidirectional_keywords = ['relationship', 'connection', 'associated', 'related',
                                 'interaction', 'everything about', 'tell me about']
        
        # Check for rule-based matches
        forward_count = sum(1 for kw in forward_keywords if kw in query_lower)
        backward_count = sum(1 for kw in backward_keywords if kw in query_lower)
        bidirectional_count = sum(1 for kw in bidirectional_keywords if kw in query_lower)
        
        # If clear rule-based match, use it
        rule_scores = {
            'forward': forward_count,
            'backward': backward_count,
            'bidirectional': bidirectional_count
        }
        
        max_rule_score = max(rule_scores.values())
        if max_rule_score > 0:
            best_rule_direction = max(rule_scores, key=rule_scores.get)
            print(f"Rule-based classification: {best_rule_direction}")
            return best_rule_direction
        
        # Otherwise, use semantic similarity
        return self.classify(query)

