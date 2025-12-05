"""
Reasoning Direction Classifier
Classifies queries into forward, backward, or bidirectional reasoning directions.

Direction Classification Logic:
- Forward: Query asks about effects/consequences of X on Y (X → Y)
  "How did X contribute to Y?" "What effect did X have on Y?"
  
- Backward: Query asks about causes/reasons for Y (? → Y)
  "What caused Y?" "Why did Y happen?"
  
- Bidirectional: Query asks about general relationships
  "What is the relationship between X and Y?"
"""

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import Literal, Dict, List, Tuple
import numpy as np
import re


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
        # Forward: X causes/affects/contributes to Y (traverse from X to find effects)
        # These patterns describe queries where we START from a cause and find effects
        self.direction_patterns = {
            'forward': [
                # "How did X contribute to Y" patterns - X is cause, Y is effect
                "How did something contribute to the outcome",
                "How did this affect that",
                "How did X influence Y",
                "What effect did X have on Y",
                "How did X lead to Y",
                "What impact did X have",
                "How did X cause Y to happen",
                "What were the effects of X on Y",
                "How did X shape Y",
                "What role did X play in Y",
                "How did X drive Y",
                "What consequences did X have for Y",
            ],
            'backward': [
                # "What caused Y" patterns - Y is the effect, looking for causes
                "What caused this to happen",
                "Why did this occur",
                "What were the reasons for this",
                "What led to this outcome",
                "What explains this result",
                "What factors caused this",
                "What were the origins of this",
                "What triggered this event",
                "What are the causes behind this",
                "What made this happen",
            ],
            'bidirectional': [
                "What is the relationship between X and Y",
                "Tell me about X and Y",
                "Explain the connection between these",
                "How are X and Y connected",
                "What is associated with this",
                "Describe the interactions",
                "What are all the relationships",
                "How do these relate to each other",
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
        Classify using both regex patterns and semantic similarity.
        
        Key insight:
        - "How did X contribute/affect/influence Y" → FORWARD (X causes Y, traverse from X)
        - "What caused Y" / "Why did Y happen" → BACKWARD (find causes of Y, traverse to Y)
        - "Relationship between X and Y" → BIDIRECTIONAL
        
        Args:
            query: User query string
            
        Returns:
            Direction: 'forward', 'backward', or 'bidirectional'
        """
        query_lower = query.lower()
        
        # FORWARD patterns: "How did X contribute/affect/influence/lead to/cause Y"
        # These ask about the EFFECT of X on Y, so we traverse FROM X
        forward_patterns = [
            r'how did .+ (contribute|affect|influence|impact|lead|drive|shape|cause|play.+role)',
            r'how does .+ (contribute|affect|influence|impact|lead|drive|shape|cause)',
            r'how .+ (contribute|affect|influence|impact|cause) .+',
            r'what (effect|impact|influence|role) did .+ have',
            r'what were the (effects|consequences|impacts|results) of .+ on',
            r'how .+ (contributed|affected|influenced|impacted|led|shaped|caused)',
            r'what happens (if|when)',
            r'what (would|could|might) .+ cause',
        ]
        
        # BACKWARD patterns: "What caused Y" / "Why did Y happen"
        # These ask about the CAUSE of Y, so we traverse TO Y
        backward_patterns = [
            r'^what (caused|causes|led to|triggered|explains?|made)',
            r'^why (did|does|is|was|were|has|have)',
            r'what (are|were) the (causes?|reasons?|factors?|origins?) (of|for|behind)',
            r'what (led|leads) to (this|the|that)',
            r'what (is|was) (the|a) (cause|reason|source|origin) of',
            r'what (drove|drives|prompted|triggered)',
        ]
        
        # BIDIRECTIONAL patterns: general relationship queries
        bidirectional_patterns = [
            r'(what is|explain|describe) the (relationship|connection|link) between',
            r'how (are|is) .+ (related|connected|associated)',
            r'tell me about .+ and',
            r'what (are|is) .+ associated with',
        ]
        
        # Check patterns with priority: forward > backward > bidirectional
        for pattern in forward_patterns:
            if re.search(pattern, query_lower):
                print(f"Rule-based classification: forward (matched: {pattern})")
                return 'forward'
        
        for pattern in backward_patterns:
            if re.search(pattern, query_lower):
                print(f"Rule-based classification: backward (matched: {pattern})")
                return 'backward'
        
        for pattern in bidirectional_patterns:
            if re.search(pattern, query_lower):
                print(f"Rule-based classification: bidirectional (matched: {pattern})")
                return 'bidirectional'
        
        # Fall back to semantic similarity
        return self.classify(query)

