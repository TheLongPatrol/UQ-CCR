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
import itertools
from nltk.tokenize import sent_tokenize
import torch
from sentence_transformers import SentenceTransformer, util
import os

class ReasoningPipeline:
    """Main pipeline for knowledge graph-based reasoning."""
    
    def __init__(self, 
                 similarity_threshold: float = 0.85,
                 tau: float = 0.5,
                 model_name: str = 'all-MiniLM-L6-v2',
                 use_scores: bool = False,
                 articles_dir: str = "bitcoin_docs/",
                 relations_dir: str = "relations/",
                 relations_json_dir: str = "relations_json/"):
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
        self.triples_to_articles = {}
        self.article_dir = articles_dir
        self.relation_dir = relations_dir
        self.relations_json_dir = relations_json_dir
        self.triples_to_sent_per_art = None

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
        
    def load_triples_by_articles(self):
        article_names = sorted(os.listdir(self.article_dir))
        relation_files = sorted(os.listdir(self.relations_json_dir))
        article_relations = []
        for i in range(len(article_names)):
            article_name = article_names[i]
            relation_fname = relation_files[i]
            
            with open(self.relations_json_dir+relation_fname) as f:
                data = json.load(f)
                if isinstance(data, dict) and 'triples' in data:
                    relations = data['triples']
                elif isinstance(data, list):
                    relations = data
                else:
                    raise ValueError("JSON must contain 'triples' key or be a list of triples")
            article_relations.append((article_name, relations))
        return article_relations

    def map_triples_to_article(self, articles_triples: List):
        if self.triples_to_sent_per_art is not None:
            return 
        article_to_triples = {}
        article_to_relations = {}
        for article, triples in articles_triples:
            triples_as_sents = []
            triples_as_tuple = []
            for triple in triples:
                sent = (" ".join([triple['cause'], triple['relation'], triple['effect']])).strip()
                triples_as_sents.append(sent)
                triples_as_tuple.append((triple['cause'], triple['relation'], triple['effect']))
            article_to_triples[article] = triples_as_sents
            article_to_relations[article] = triples_as_tuple
        article_names = sorted(os.listdir(self.article_dir))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        sentence_model = SentenceTransformer('msmarco-MiniLM-L6-cos-v5').to(device)
        sentence_model.eval()
        self.triples_to_sent_per_art = {}
        for i in range(len(article_names)):
            article_name = article_names[i]
            with open(self.article_dir+article_name, "r") as f:
                passage = f.read()
                sentences = sent_tokenize(passage)
            relations_as_str = article_to_triples[article_name]
            relations = article_to_relations[article_name]
            article_mat = sentence_model.encode_document(sentences, convert_to_tensor=True, device=device, normalize_embeddings=True)
            relations_as_str_mat = sentence_model.encode_query(relations_as_str, convert_to_tensor=True, device=device, normalize_embeddings=True)
            best_fit_sents = util.semantic_search(relations_as_str_mat, article_mat, top_k=5)
            triples_to_sentences = {}
            for j in range(len(best_fit_sents)):
                cause,relation,effect = relations[j]
                ind_found = 0
                for k in range(len(best_fit_sents[j])):
                    match = best_fit_sents[j][k]
                    matched_sent = sentences[int(match['corpus_id'])]
                    if relation in matched_sent:
                        ind_found = int(match['corpus_id'])
                        break
                matched_sent = sentences[ind_found]
                triples_to_sentences[(cause,relation,effect)] = matched_sent
            self.triples_to_sent_per_art[article_name] = triples_to_sentences

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
        
        articles_triples = self.load_triples_by_articles()
        self.map_triples_to_article(articles_triples)
        if self.use_scores:
            all_triples = []
            if compute_scores_online:
                context_scores = get_context_probs(self.article_dir, self.relation_dir, "out/")
            else:
                scores_filename = 'context_scores.pkl'
                if "misinfo" in self.article_dir:
                    scores_filename = 'context_scores_misinfo.pkl'
                with open(scores_filename, "rb") as f:
                    context_scores = pickle.load(f)
            for article_triple in articles_triples:
                article_name = article_triple[0]
                triples = article_triple[1]
                article_scores = context_scores[article_name]
                for triple in triples:
                    triple_as_tuple = (triple['cause'], triple['relation'], triple['effect'])
                    if triple_as_tuple not in self.triples_to_articles:
                        self.triples_to_articles[triple_as_tuple] = [article_name]
                    else:
                        self.triples_to_articles[triple_as_tuple].append(article_name)
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
        else:
            for article_triple in articles_triples:
                article_name = article_triple[0]
                triples = article_triple[1]
                for triple in triples:
                    triple_as_tuple = (triple['cause'], triple['relation'], triple['effect'])
                    if triple_as_tuple not in self.triples_to_articles:
                        self.triples_to_articles[triple_as_tuple] = [article_name]
                    else:
                        self.triples_to_articles[triple_as_tuple].append(article_name)
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
    
    def process_results(self, results:Dict):
        top_chains = results['top_chains']
        final_chains = []
        ind = 0
        while len(final_chains) < 50 and ind < len(top_chains):
            chain, score, reliab_score = top_chains[ind]
            _, chain_as_triple = self.chain_ranker.format_chain(chain, self.knowledge_graph, score, reliab_score)
            final_chains.extend(chain_as_triple)
            ind+=1
        return final_chains
    
    def find_original_triple(self, chains: List):
        original_triples = []
        triples_and_arts = []
        for triple in chains:
            cause,relation,effect = triple
            all_causes = self.entity_clusterer.get_cluster_members(cause)
            all_effects = self.entity_clusterer.get_cluster_members(effect)
            for pot_cause, pot_effect in itertools.product(all_causes, all_effects):
                if (pot_cause, relation, pot_effect) in self.triples_to_articles:
                    trip = (pot_cause, relation, pot_effect)
                    original_triples.append(trip)
                    triples_and_arts.append((trip, self.triples_to_articles[trip]))
        return original_triples, triples_and_arts
    
    def get_orig_sentence(self, triples_and_articles):
        sentences = []
        for triple_as_tuple, article_names in triples_and_articles:
            for article_name in article_names:
                sentences.append(self.triples_to_sent_per_art[article_name][triple_as_tuple])
        return sentences
    
    def query_for_context(self, 
             query_text: str,
             max_depth: int = 5,
             max_chains_per_node: int = 50,
             top_k: int = 10,
             use_rule_based_direction: bool = True):
        results = self.query(query_text=query_text, max_depth=max_depth, max_chains_per_node=max_chains_per_node, top_k=top_k, use_rule_based_direction=use_rule_based_direction)
        chains = self.process_results(results)
        original_triples, triples_and_articles = self.find_original_triple(chains)
        sentences = self.get_orig_sentence(triples_and_articles)
        triples_as_str = []
        for triple in original_triples:
            cause, relation, effect = triple
            triples_as_str.append(f"<{cause}, {relation}, {effect}>")
        context = ["Relation chains:", " ".join(triples_as_str), "\nSentences", " ".join(sentences)]
        return "Context\n" +"\n".join(context), results
    
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
            formatted,_ = self.chain_ranker.format_chain(chain, self.knowledge_graph, score, reliab_score)
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

