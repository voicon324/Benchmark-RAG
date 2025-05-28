#!/usr/bin/env python3
"""
Demo script for BM25Model integration in NewAIBench.

This script demonstrates how to configure, initialize, index corpus,
and perform retrieval using the BM25Model implementation.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

from newaibench.models.sparse import BM25Model
from newaibench.evaluation.metrics import IRMetrics
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_sample_corpus():
    """Create a sample corpus for demonstration."""
    return {
        "doc_001": {
            "text": "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed.",
            "title": "Introduction to Machine Learning",
            "metadata": {"domain": "AI", "year": 2023}
        },
        "doc_002": {
            "text": "Deep learning neural networks with multiple layers can automatically learn complex patterns in data, making them particularly effective for image recognition and natural language processing.",
            "title": "Deep Learning Fundamentals",
            "metadata": {"domain": "AI", "year": 2023}
        },
        "doc_003": {
            "text": "Natural language processing techniques allow computers to understand, interpret, and generate human language in a valuable way.",
            "title": "NLP Overview",
            "metadata": {"domain": "NLP", "year": 2023}
        },
        "doc_004": {
            "text": "Information retrieval systems help users find relevant information from large document collections using various ranking algorithms.",
            "title": "Information Retrieval Systems",
            "metadata": {"domain": "IR", "year": 2023}
        },
        "doc_005": {
            "text": "Computer vision algorithms enable machines to interpret and understand visual information from the world, including image and video analysis.",
            "title": "Computer Vision Basics",
            "metadata": {"domain": "CV", "year": 2023}
        },
        "doc_006": {
            "text": "Data mining and knowledge discovery involve extracting useful patterns and insights from large datasets using statistical and machine learning methods.",
            "title": "Data Mining Techniques",
            "metadata": {"domain": "Data Science", "year": 2023}
        },
        "doc_007": {
            "text": "Supervised learning algorithms learn from labeled training data to make predictions on new, unseen data examples.",
            "title": "Supervised Learning",
            "metadata": {"domain": "ML", "year": 2023}
        },
        "doc_008": {
            "text": "Unsupervised learning discovers hidden patterns in data without using labeled examples, including clustering and dimensionality reduction.",
            "title": "Unsupervised Learning",
            "metadata": {"domain": "ML", "year": 2023}
        }
    }


def create_sample_queries():
    """Create sample queries for demonstration."""
    return [
        {
            "query_id": "query_001",
            "text": "machine learning algorithms",
            "intent": "Find information about ML algorithms"
        },
        {
            "query_id": "query_002", 
            "text": "deep neural networks",
            "intent": "Find information about deep learning"
        },
        {
            "query_id": "query_003",
            "text": "natural language processing",
            "intent": "Find information about NLP"
        },
        {
            "query_id": "query_004",
            "text": "computer vision image recognition",
            "intent": "Find information about computer vision"
        },
        {
            "query_id": "query_005",
            "text": "unsupervised learning clustering",
            "intent": "Find information about unsupervised learning"
        }
    ]


def demo_basic_usage():
    """Demonstrate basic BM25Model usage."""
    print("=" * 60)
    print("BM25Model Basic Usage Demo")
    print("=" * 60)
    
    # 1. Configuration
    config = {
        "name": "bm25_demo",
        "type": "sparse",
        "parameters": {
            "k1": 1.6,
            "b": 0.75,
            "tokenizer": "simple",
            "lowercase": True,
            "remove_stopwords": False,
            "min_token_length": 2
        }
    }
    
    print(f"📋 Configuration: {config}")
    
    # 2. Initialize model
    print("\n🚀 Initializing BM25Model...")
    model = BM25Model(config)
    
    # 3. Load model
    print("📥 Loading model...")
    model.load_model()
    
    # 4. Prepare data
    corpus = create_sample_corpus()
    queries = create_sample_queries()
    
    print(f"\n📚 Corpus: {len(corpus)} documents")
    print(f"❓ Queries: {len(queries)} queries")
    
    # 5. Index corpus
    print("\n🔍 Indexing corpus...")
    model.index_corpus(corpus, show_progress=True)
    
    # 6. Display model info
    info = model.get_model_info()
    print(f"\n📊 Model Info:")
    print(f"  - Model Name: {info['model_name']}")
    print(f"  - Model Type: {info['model_type']}")
    print(f"  - Documents Indexed: {info['num_documents']}")
    print(f"  - Vocabulary Size: {info['vocabulary_size']}")
    print(f"  - Parameters: k1={info['parameters']['k1']}, b={info['parameters']['b']}")
    
    return model, corpus, queries


def demo_retrieval(model, corpus, queries):
    """Demonstrate retrieval functionality."""
    print("\n" + "=" * 60)
    print("BM25 Retrieval Demo")
    print("=" * 60)
    
    # Perform retrieval
    print("🔎 Performing retrieval...")
    results = model.predict(queries, corpus, top_k=3)
    
    # Display results
    print("\n📋 Retrieval Results:")
    for query_data in queries:
        query_id = query_data["query_id"]
        query_text = query_data["text"]
        
        print(f"\n🔹 Query {query_id}: '{query_text}'")
        
        if query_id in results and results[query_id]:
            for rank, (doc_id, score) in enumerate(results[query_id].items(), 1):
                doc_title = corpus[doc_id].get("title", "No title")
                print(f"   {rank}. {doc_id}: {score:.4f} - {doc_title}")
        else:
            print("   No results found")
    
    return results


def demo_advanced_features(model, corpus, queries):
    """Demonstrate advanced BM25 features."""
    print("\n" + "=" * 60)
    print("BM25 Advanced Features Demo")
    print("=" * 60)
    
    # 1. Score normalization
    print("\n🔧 Testing score normalization...")
    normalized_results = model.predict(
        queries[:2], corpus, 
        top_k=3, 
        normalize_scores=True
    )
    
    print("📊 Normalized scores (max=1.0):")
    for query_id, doc_scores in normalized_results.items():
        max_score = max(doc_scores.values()) if doc_scores else 0
        print(f"  {query_id}: max_score = {max_score:.4f}")
    
    # 2. Minimum score threshold
    print("\n🎯 Testing minimum score threshold...")
    threshold = 2.0
    filtered_results = model.predict(
        queries[:2], corpus,
        top_k=5,
        min_score=threshold
    )
    
    print(f"📊 Results with min_score >= {threshold}:")
    for query_id, doc_scores in filtered_results.items():
        print(f"  {query_id}: {len(doc_scores)} documents")
        for doc_id, score in doc_scores.items():
            print(f"    {doc_id}: {score:.4f}")
    
    # 3. Test with different corpus (should trigger re-indexing)
    print("\n🔄 Testing corpus consistency check...")
    modified_corpus = corpus.copy()
    modified_corpus["doc_new"] = {
        "text": "This is a new document about artificial intelligence and robotics.",
        "title": "New AI Document"
    }
    
    consistency_results = model.predict(queries[:1], modified_corpus, top_k=2)
    print(f"✅ Retrieval with modified corpus completed")


def demo_different_configurations():
    """Demonstrate different BM25 configurations."""
    print("\n" + "=" * 60)
    print("BM25 Configuration Comparison")
    print("=" * 60)
    
    corpus = create_sample_corpus()
    test_query = [{"query_id": "test", "text": "machine learning algorithms"}]
    
    configs = [
        {
            "name": "bm25_standard",
            "type": "sparse",
            "parameters": {"k1": 1.2, "b": 0.75}
        },
        {
            "name": "bm25_high_k1",
            "type": "sparse", 
            "parameters": {"k1": 2.0, "b": 0.75}
        },
        {
            "name": "bm25_low_b",
            "type": "sparse",
            "parameters": {"k1": 1.2, "b": 0.25}
        },
        {
            "name": "bm25_with_stopwords",
            "type": "sparse",
            "parameters": {
                "k1": 1.2, "b": 0.75,
                "remove_stopwords": True
            }
        }
    ]
    
    results_comparison = {}
    
    for config in configs:
        print(f"\n🔧 Testing configuration: {config['name']}")
        model = BM25Model(config)
        model.load_model()
        model.index_corpus(corpus)
        
        results = model.predict(test_query, corpus, top_k=3)
        results_comparison[config['name']] = results['test']
        
        # Show top result
        if results['test']:
            top_doc_id = list(results['test'].keys())[0]
            top_score = results['test'][top_doc_id]
            print(f"   Top result: {top_doc_id} (score: {top_score:.4f})")
    
    # Compare results
    print("\n📊 Configuration Comparison Summary:")
    for config_name, doc_scores in results_comparison.items():
        top_score = max(doc_scores.values()) if doc_scores else 0
        print(f"  {config_name}: max_score = {top_score:.4f}")


def demo_evaluation_integration():
    """Demonstrate integration with evaluation metrics."""
    print("\n" + "=" * 60)
    print("BM25 Evaluation Integration Demo")
    print("=" * 60)
    
    # Setup
    corpus = create_sample_corpus()
    queries = create_sample_queries()
    
    config = {
        "name": "bm25_eval",
        "type": "sparse",
        "parameters": {"k1": 1.6, "b": 0.75}
    }
    
    model = BM25Model(config)
    model.load_model()
    model.index_corpus(corpus)
    
    # Get results
    results = model.predict(queries, corpus, top_k=5)
    
    # Create mock qrels (ground truth)
    qrels = {
        "query_001": {"doc_001": 3, "doc_007": 2, "doc_008": 1},
        "query_002": {"doc_002": 3, "doc_001": 1},
        "query_003": {"doc_003": 3, "doc_002": 1},
        "query_004": {"doc_005": 3, "doc_002": 1},
        "query_005": {"doc_008": 3, "doc_007": 2}
    }
    
    print("📊 Evaluating BM25 performance...")
    
    # Convert results to evaluation format
    eval_results = {}
    for query_id, doc_scores in results.items():
        eval_results[query_id] = [
            (doc_id, score) for doc_id, score in doc_scores.items()
        ]
    
    # Calculate metrics
    try:
        ndcg_3 = IRMetrics.ndcg_at_k(qrels, eval_results, k=3)
        map_3 = IRMetrics.map_at_k(qrels, eval_results, k=3)
        recall_3 = IRMetrics.recall_at_k(qrels, eval_results, k=3)
        precision_3 = IRMetrics.precision_at_k(qrels, eval_results, k=3)
        
        print(f"📈 Evaluation Results:")
        print(f"  - nDCG@3: {ndcg_3:.4f}")
        print(f"  - MAP@3: {map_3:.4f}")
        print(f"  - Recall@3: {recall_3:.4f}")
        print(f"  - Precision@3: {precision_3:.4f}")
        
    except Exception as e:
        print(f"⚠️ Evaluation failed: {e}")
        print("Note: This might be due to evaluation module dependencies")


def main():
    """Main demo function."""
    print("🎯 NewAIBench BM25Model Integration Demo")
    print("=" * 60)
    
    try:
        # Basic usage demo
        model, corpus, queries = demo_basic_usage()
        
        # Retrieval demo
        results = demo_retrieval(model, corpus, queries)
        
        # Advanced features demo
        demo_advanced_features(model, corpus, queries)
        
        # Different configurations demo
        demo_different_configurations()
        
        # Evaluation integration demo
        demo_evaluation_integration()
        
        print("\n" + "=" * 60)
        print("✅ Demo completed successfully!")
        print("📝 Key takeaways:")
        print("  - BM25Model integrates seamlessly with NewAIBench")
        print("  - Supports flexible configuration and advanced features")
        print("  - Compatible with evaluation framework")
        print("  - Efficient corpus indexing and retrieval")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
