"""
Example Usage of the Knowledge Graph Reasoning Pipeline

This script demonstrates how to use the reasoning pipeline to:
1. Load triples from JSON
2. Build a knowledge graph with entity clustering
3. Query the graph with different types of questions
4. Visualize and save results
"""

from reasoning_pipeline import ReasoningPipeline
import json


def main():
    # Initialize the pipeline
    print("Initializing Knowledge Graph Reasoning Pipeline...")
    pipeline = ReasoningPipeline(
        similarity_threshold=0.85,  # Entity clustering threshold
        tau=0.5,                    # Starting node similarity threshold
        model_name='all-MiniLM-L6-v2',
        use_scores=True,
        articles_dir='bitcoin_docs/',
        relations_dir='relations/',
        relations_json_dir='relations_json/'
    )
    # pipeline to use with misinfo docs
    # pipeline = ReasoningPipeline(
    #     similarity_threshold=0.85,  # Entity clustering threshold
    #     tau=0.5,                    # Starting node similarity threshold
    #     model_name='all-MiniLM-L6-v2',
    #     use_scores=True,
    #     articles_dir='bitcoin_misinfo_docs/',
    #     relations_dir='relations_misinfo/',
    #     relations_json_dir='relations_misinfo_json/'
    # )
    
    # Load triples from JSON file
    pipeline.load_triples_from_json('all_relations.json')
    
    # Build the knowledge graph
    pipeline.build_knowledge_graph()
    
    # Example queries demonstrating different reasoning directions
    
    # Query 1: Forward reasoning (cause to effect)
    print("\n\n" + "="*80)
    print("EXAMPLE 1")
    print("="*80)
    
    query1 = "How did Andrew Yang’s advocacy for blockchain technology influence Bitcoin’s adoption?"
    context1, results1 = pipeline.query_for_context(
        query1,
        max_depth=5,
        max_chains_per_node=50,
        top_k=10
    )
    print(context1)
    pipeline.display_results(results1)
    
    # Query 2: Backward reasoning (effect to cause)
    print("\n\n" + "="*80)
    print("EXAMPLE 2")
    print("="*80)
    
    query2 = "How did global economic uncertainty contribute to Bitcoin’s price rise?"
    context2, results2 = pipeline.query_for_context(
        query2,
        max_depth=5,
        max_chains_per_node=50,
        top_k=10
    )
    print(context2)
    pipeline.display_results(results2)
    
    # # Query 3: Bidirectional reasoning (exploring relationships)
    # print("\n\n" + "="*80)
    # print("EXAMPLE 3: Bidirectional Reasoning (Relationships)")
    # print("="*80)
    
    # query3 = "What is the relationship between alpha-synuclein and GCase?"
    # results3 = pipeline.query(
    #     query3,
    #     max_depth=4,
    #     max_chains_per_node=50,
    #     top_k=10
    # )
    # pipeline.display_results(results3)
    
    # # Query 4: Complex multi-hop reasoning
    # print("\n\n" + "="*80)
    # print("EXAMPLE 4: Complex Multi-hop Reasoning")
    # print("="*80)
    
    # query4 = "How does low GCase activity lead to Parkinson's disease?"
    # results4 = pipeline.query(
    #     query4,
    #     max_depth=6,
    #     max_chains_per_node=100,
    #     top_k=15
    # )
    # pipeline.display_results(results4, show_all_chains=False)
    
    # # Save results to JSON
    # pipeline.save_results_to_json(results4, 'results_example4.json')
    
    # Visualize the knowledge graph (limited to 50 nodes for readability)
    print("\n\nGenerating knowledge graph visualization...")
    pipeline.knowledge_graph.visualize('knowledge_graph.png', max_nodes=30)
    
    # Print graph statistics
    print("\n" + "="*80)
    print("KNOWLEDGE GRAPH STATISTICS")
    print("="*80)
    stats = pipeline.knowledge_graph.get_graph_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")


def custom_triples_example():
    """Example using custom triples defined in code."""
    
    print("\n\n" + "="*80)
    print("CUSTOM TRIPLES EXAMPLE")
    print("="*80)
    
    # Define custom triples
    custom_triples = [
        {
            "cause": "climate change",
            "relation": "causes",
            "effect": "rising temperatures"
        },
        {
            "cause": "rising temperatures",
            "relation": "leads to",
            "effect": "melting ice caps"
        },
        {
            "cause": "melting ice caps",
            "relation": "results in",
            "effect": "sea level rise"
        },
        {
            "cause": "sea level rise",
            "relation": "threatens",
            "effect": "coastal cities"
        },
        {
            "cause": "rising temperatures",
            "relation": "increases",
            "effect": "extreme weather events"
        },
        {
            "cause": "extreme weather events",
            "relation": "damages",
            "effect": "infrastructure"
        },
        {
            "cause": "deforestation",
            "relation": "contributes to",
            "effect": "climate change"
        },
        {
            "cause": "fossil fuel burning",
            "relation": "accelerates",
            "effect": "global warming"
        },
        {
            "cause": "global warming",
            "relation": "is equivalent to",
            "effect": "climate change"
        }
    ]
    
    # Initialize pipeline
    pipeline = ReasoningPipeline(similarity_threshold=0.85, tau=0.4)
    
    # Load custom triples
    pipeline.load_triples_from_list(custom_triples)
    
    # Build graph
    pipeline.build_knowledge_graph()
    
    # Query the custom graph
    query = "What are the consequences of climate change?"
    results = pipeline.query(query, max_depth=5, top_k=10)
    pipeline.display_results(results)


if __name__ == "__main__":
    # Run main examples with the provided medical knowledge graph
    main()
    
    # Optionally, uncomment to run the custom triples example
    # custom_triples_example()

