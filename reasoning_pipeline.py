"""
Main Reasoning Pipeline
Integrates all components to perform knowledge graph reasoning.
"""

from typing import List, Dict, Tuple, Optional
from entity_clustering import EntityClusterer
from knowledge_graph import KnowledgeGraph
from query_processor import QueryProcessor
from direction_classifier import DirectionClassifier
from chain_ranker import ChainRanker
from frequency_reliability import website_reliability_map
import json
from context_scorer import get_context_probs
import pickle
import os

class ReasoningPipeline:
    """Main pipeline for knowledge graph-based reasoning."""
    
    def __init__(self, 
                 similarity_threshold: float = 0.85,
                 tau: float = 0.5,
                 model_name: str = 'all-MiniLM-L6-v2',
                 use_scores: bool = False):
        """
        Initialize the reasoning pipeline.
        
        Args:
            similarity_threshold: Threshold for entity clustering (default: 0.85)
            tau: Threshold for selecting starting nodes (default: 0.5)
            model_name: SentenceBERT model name
        """
        self.entity_clusterer = EntityClusterer(model_name, similarity_threshold)
        self.knowledge_graph = KnowledgeGraph()
        self.query_processor = QueryProcessor(model_name, tau)
        self.direction_classifier = DirectionClassifier(model_name)
        self.chain_ranker = ChainRanker(model_name, use_reliability_scores=use_scores)
        
        self.triples = []
        self.is_built = False
        self.use_scores = use_scores

    def load_triples_from_json(self, filepath: str):
        """
        Load triples from a JSON file.
        
        Args:
            filepath: Path to JSON file containing triples
        """
        print(f"Loading triples from {filepath}...")
        with open(filepath, 'r') as f:
            data = json.load(f)
            
        # Handle different JSON formats
        if isinstance(data, dict) and 'triples' in data:
            self.triples = data['triples']
        elif isinstance(data, list):
            self.triples = data
        else:
            raise ValueError("JSON must contain 'triples' key or be a list of triples")
        
        print(f"Loaded {len(self.triples)} triples")
        
    def load_triples_from_list(self, triples: List[Dict]):
        """
        Load triples from a Python list.
        
        Args:
            triples: List of triple dictionaries
        """
        self.triples = triples
        print(f"Loaded {len(self.triples)} triples")
        
    def load_triples_by_articles(self, article_dir, relation_dir):
        article_names = sorted(os.listdir(article_dir))
        relation_files = sorted(os.listdir(relation_dir))
        article_relations = []
        for i in range(len(article_names)):
            article_name = article_names[i]
            relation_fname = relation_files[i]
            
            with open(relation_dir+relation_fname) as f:
                data = json.load(f)
                if isinstance(data, dict) and 'triples' in data:
                    relations = data['triples']
                elif isinstance(data, list):
                    relations = data
                else:
                    raise ValueError("JSON must contain 'triples' key or be a list of triples")
            article_relations.append((article_name, relations))
        return article_relations

    def build_knowledge_graph(self, compute_scores_online = False, use_website_rel_scores = False):
        """Build the knowledge graph from loaded triples."""
        if not self.triples:
            raise ValueError("No triples loaded. Use load_triples_from_json() or load_triples_from_list() first.")
        
        print("\n" + "="*60)
        print("STEP 1: Entity Clustering")
        print("="*60)
        
        # Extract entities
        entities = self.entity_clusterer.extract_entities_from_triples(self.triples)
        print(f"Extracted {len(entities)} unique entities")
        
        # Cluster entities
        self.entity_clusterer.cluster_entities(list(entities))
        
        if self.use_scores:
            all_triples = []
            articles_triples = self.load_triples_by_articles("bitcoin_docs/", "new_relations_json/")
            if compute_scores_online:
                context_scores = get_context_probs("bitcoin_docs/", "new_relations/", "out/")
            else:
                with open('context_scores.pkl', "rb") as f:
                    context_scores = pickle.load(f)
            for article_triple in articles_triples:
                article_name = article_triple[0]
                triples = article_triple[1]
                article_scores = context_scores[article_name]
                for triple in triples:
                    triple_as_tuple = (triple['cause'], triple['relation'], triple['effect'])
                    if use_website_rel_scores:
                        act_article_name = article_name.split("_")[0]
                        if act_article_name in website_reliability_map:
                            web_rel_score, bias_score = website_reliability_map[act_article_name]
                            if web_rel_score < 30:
                                web_rel_score = 0.25
                            elif web_rel_score < 32:
                                web_rel_score = 0.3
                            elif web_rel_score < 36:
                                web_rel_score = 0.6
                            elif web_rel_score < 40:
                                web_rel_score = 0.85
                            else:
                                web_rel_score = 0.95
                        else:
                            web_rel_score = 0.5
                        triple['score'] = 0.7*article_scores[triple_as_tuple]+0.2*web_rel_score+0.1*(1-abs(bias_score/21))
                    else:
                        triple['score'] = article_scores[triple_as_tuple]
                    all_triples.append(triple)
            self.triples = all_triples

        # Normalize triples
        if self.use_scores:
            normalized_triples = self.entity_clusterer.normalize_triples_with_context_scores(self.triples)
        else:
            normalized_triples = self.entity_clusterer.normalize_triples(self.triples)
        print(f"Normalized triples to use canonical entities")

        
        print("\n" + "="*60)
        print("STEP 2: Knowledge Graph Construction")
        print("="*60)
        
        # Build graph
        self.knowledge_graph.build_from_triples(normalized_triples)
        
        # Compute node embeddings for query processing
        nodes = self.knowledge_graph.get_nodes()
        self.query_processor.compute_node_embeddings(nodes)
        
        # Print statistics
        stats = self.knowledge_graph.get_graph_stats()
        print(f"\nGraph Statistics:")
        print(f"  - Nodes: {stats['num_nodes']}")
        print(f"  - Edges: {stats['num_edges']}")
        print(f"  - Avg Degree: {stats['avg_degree']:.2f}")
        print(f"  - Density: {stats['density']:.4f}")
        print(f"  - Weakly Connected: {stats['is_connected']}")
        
        self.is_built = True
        
    def query(self, 
             query_text: str,
             max_depth: int = 5,
             max_chains_per_node: int = 50,
             top_k: int = 10,
             use_rule_based_direction: bool = True) -> Dict:
        """
        Process a query and return reasoning chains.
        
        Args:
            query_text: User query string
            max_depth: Maximum depth for graph traversal
            max_chains_per_node: Maximum chains to generate per starting node
            top_k: Number of top reasoning chains to return
            use_rule_based_direction: Whether to use rule-based direction classification
            
        Returns:
            Dictionary containing query results
        """
        if not self.is_built:
            raise ValueError("Knowledge graph not built. Call build_knowledge_graph() first.")
        
        print("\n" + "="*60)
        print(f"QUERY: {query_text}")
        print("="*60)
        
        # Step 1: Find starting nodes
        print("\n[1] Finding starting nodes...")
        starting_nodes = self.query_processor.find_starting_nodes(query_text, self.knowledge_graph)
        
        if not starting_nodes:
            print("No starting nodes found!")
            return {
                'query': query_text,
                'starting_nodes': [],
                'direction': None,
                'chains': [],
                'top_chains': []
            }
        
        print(f"Top starting nodes:")
        for node, score in starting_nodes[:5]:
            print(f"  - {node} (similarity: {score:.3f})")
        
        # Step 2: Classify reasoning direction
        print("\n[2] Classifying reasoning direction...")
        if use_rule_based_direction:
            direction = self.direction_classifier.classify_with_rules(query_text)
        else:
            direction = self.direction_classifier.classify(query_text)
        
        # Step 3: Perform graph traversal from each starting node
        print(f"\n[3] Performing {direction} traversal...")
        all_chains = []
        for node, score in starting_nodes[:10]:  # Limit to top 10 starting nodes
            chains = self.knowledge_graph.dfs_traversal(
                node, 
                direction=direction,
                max_depth=max_depth,
                max_chains=max_chains_per_node
            )
            all_chains.extend(chains)
            print(f"  - From '{node}': found {len(chains)} chains")
        
        print(f"Total chains generated: {len(all_chains)}")
        
        # Step 4: Rank chains
        print(f"\n[4] Ranking chains and selecting top-{top_k}...")
        top_chains = self.chain_ranker.rank_chains(
            query_text,
            all_chains,
            self.knowledge_graph,
            top_k=top_k,
            use_features=True
        )
        
        # Format results
        results = {
            'query': query_text,
            'starting_nodes': starting_nodes[:10],
            'direction': direction,
            'total_chains': len(all_chains),
            'top_chains': top_chains
        }
        
        return results
    
    def display_results(self, results: Dict, show_all_chains: bool = False):
        """
        Display query results in a readable format.
        
        Args:
            results: Results dictionary from query()
            show_all_chains: Whether to show all top chains (default: False, shows top 5)
        """
        print("\n" + "="*60)
        print("RESULTS")
        print("="*60)
        
        print(f"\nQuery: {results['query']}")
        print(f"Direction: {results['direction']}")
        print(f"Total chains found: {results['total_chains']}")
        
        print(f"\nTop Starting Nodes:")
        for node, score in results['starting_nodes'][:5]:
            print(f"  - {node} (score: {score:.3f})")
        
        print(f"\nTop Reasoning Chains:")
        num_to_show = len(results['top_chains']) if show_all_chains else min(5, len(results['top_chains']))
        
        for i, (chain, score, reliab_score) in enumerate(results['top_chains'][:num_to_show], 1):
            formatted = self.chain_ranker.format_chain(chain, self.knowledge_graph, score, reliab_score)
            print(f"\n{i}. {formatted}")
        
        if len(results['top_chains']) > num_to_show:
            print(f"\n... and {len(results['top_chains']) - num_to_show} more chains")
    
    def save_results_to_json(self, results: Dict, filepath: str):
        """
        Save query results to a JSON file.
        
        Args:
            results: Results dictionary from query()
            filepath: Output file path
        """
        # Convert chains to serializable format
        serializable_results = {
            'query': results['query'],
            'direction': results['direction'],
            'total_chains': results['total_chains'],
            'starting_nodes': [{'node': node, 'score': score} for node, score in results['starting_nodes']],
            'top_chains': [
                {
                    'chain': chain,
                    'score': score,
                    'formatted': self.chain_ranker.format_chain(chain, self.knowledge_graph)
                }
                for chain, score in results['top_chains']
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"\nResults saved to {filepath}")

