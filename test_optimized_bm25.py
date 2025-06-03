#!/usr/bin/env python3
"""
Quick validation test for OptimizedBM25Model
"""

import time
import logging
from typing import Dict, List, Any

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import models
try:
    from src.newaibench.models import BM25Model, OptimizedBM25Model
    from src.newaibench.models.optimized_sparse import BM25OptimizationConfig
    print("✓ Successfully imported models")
except ImportError as e:
    print(f"✗ Import error: {e}")
    exit(1)


def create_test_data():
    """Create small test corpus and queries."""
    
    # Test corpus
    corpus = {}
    for i in range(100):
        corpus[f"doc_{i}"] = {
            'text': f"This is document {i} about information retrieval and search algorithms. "
                   f"It contains relevant information for testing BM25 scoring functions. "
                   f"Document {i} focuses on {'machine learning' if i % 2 == 0 else 'natural language processing'}."
        }
    
    # Test queries
    queries = [
        {'query_id': 'q1', 'text': 'information retrieval'},
        {'query_id': 'q2', 'text': 'machine learning algorithms'},
        {'query_id': 'q3', 'text': 'search functions'},
        {'query_id': 'q4', 'text': 'document scoring'},
        {'query_id': 'q5', 'text': 'natural language processing'}
    ]
    
    return corpus, queries


def test_original_bm25():
    """Test original BM25Model."""
    print("\n=== Testing Original BM25Model ===")
    
    config = {
        'name': 'test_bm25',
        'type': 'sparse',
        'parameters': {
            'k1': 1.6,
            'b': 0.75,
            'tokenizer': 'simple'
        }
    }
    
    corpus, queries = create_test_data()
    
    try:
        # Initialize and load model
        model = BM25Model(config)
        model.load_model()
        print("✓ Model loaded successfully")
        
        # Index corpus
        start_time = time.time()
        model.index_corpus(corpus)
        index_time = time.time() - start_time
        print(f"✓ Corpus indexed in {index_time:.3f}s")
        
        # Run queries
        start_time = time.time()
        results = model.predict(queries, corpus, top_k=10)
        query_time = time.time() - start_time
        print(f"✓ Queries processed in {query_time:.3f}s")
        
        # Show sample results
        print(f"✓ Retrieved {len(results)} query results")
        sample_query = list(results.keys())[0]
        sample_results = results[sample_query]
        print(f"  Sample query '{sample_query}' returned {len(sample_results)} documents")
        
        # Get model info
        info = model.get_model_info()
        print(f"✓ Model info: {info['num_documents']} documents, {info['vocabulary_size']} vocabulary")
        
        return index_time, query_time, results
        
    except Exception as e:
        print(f"✗ Error testing original BM25: {e}")
        return None, None, None


def test_optimized_bm25():
    """Test OptimizedBM25Model."""
    print("\n=== Testing OptimizedBM25Model ===")
    
    config = {
        'name': 'test_optimized_bm25',
        'type': 'sparse',
        'parameters': {
            'k1': 1.6,
            'b': 0.75,
            'tokenizer': 'simple'
        }
    }
    
    opt_config = BM25OptimizationConfig(
        use_parallel_indexing=True,
        num_workers=2,
        use_caching=True,
        cache_size=1000,
        batch_size=50
    )
    
    corpus, queries = create_test_data()
    
    try:
        # Initialize and load model
        model = OptimizedBM25Model(config, opt_config)
        model.load_model()
        print("✓ Optimized model loaded successfully")
        
        # Index corpus
        start_time = time.time()
        model.index_corpus(corpus)
        index_time = time.time() - start_time
        print(f"✓ Corpus indexed in {index_time:.3f}s")
        
        # Run queries
        start_time = time.time()
        results = model.predict(queries, corpus, top_k=10)
        query_time = time.time() - start_time
        print(f"✓ Queries processed in {query_time:.3f}s")
        
        # Show sample results
        print(f"✓ Retrieved {len(results)} query results")
        sample_query = list(results.keys())[0]
        sample_results = results[sample_query]
        print(f"  Sample query '{sample_query}' returned {len(sample_results)} documents")
        
        # Get model info
        info = model.get_model_info()
        print(f"✓ Model info: {info['num_documents']} documents, {info['vocabulary_size']} vocabulary")
        
        # Show optimization stats
        if 'optimization_stats' in info:
            stats = info['optimization_stats']
            print(f"✓ Optimization stats:")
            print(f"  - Indexing time: {stats['indexing_time']:.3f}s")
            print(f"  - Cache hits: {stats['cache_hits']}")
            print(f"  - Cache misses: {stats['cache_misses']}")
        
        return index_time, query_time, results
        
    except Exception as e:
        print(f"✗ Error testing optimized BM25: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


def compare_results(original_results, optimized_results):
    """Compare results between models."""
    print("\n=== Comparing Results ===")
    
    if original_results is None or optimized_results is None:
        print("✗ Cannot compare - one of the models failed")
        return
    
    # Check if same queries processed
    orig_queries = set(original_results.keys())
    opt_queries = set(optimized_results.keys())
    
    if orig_queries != opt_queries:
        print(f"✗ Different queries processed: {len(orig_queries)} vs {len(opt_queries)}")
        return
    
    print(f"✓ Both models processed {len(orig_queries)} queries")
    
    # Compare document overlap for each query
    total_overlap = 0
    for query_id in orig_queries:
        orig_docs = set(original_results[query_id].keys())
        opt_docs = set(optimized_results[query_id].keys())
        
        if len(orig_docs) > 0:
            overlap = len(orig_docs & opt_docs) / len(orig_docs | opt_docs)
            total_overlap += overlap
    
    avg_overlap = total_overlap / len(orig_queries)
    print(f"✓ Average document overlap: {avg_overlap:.2%}")
    
    # Show sample score comparison
    sample_query = list(orig_queries)[0]
    orig_scores = original_results[sample_query]
    opt_scores = optimized_results[sample_query]
    
    common_docs = set(orig_scores.keys()) & set(opt_scores.keys())
    if common_docs:
        sample_doc = list(common_docs)[0]
        orig_score = orig_scores[sample_doc]
        opt_score = opt_scores[sample_doc]
        print(f"✓ Sample score comparison for '{sample_doc}': {orig_score:.4f} vs {opt_score:.4f}")


def main():
    """Run the validation test."""
    print("BM25 Optimization Validation Test")
    print("="*40)
    
    # Test original model
    orig_index_time, orig_query_time, orig_results = test_original_bm25()
    
    # Test optimized model
    opt_index_time, opt_query_time, opt_results = test_optimized_bm25()
    
    # Compare results
    compare_results(orig_results, opt_results)
    
    # Show performance comparison
    if orig_index_time and opt_index_time:
        print(f"\n=== Performance Summary ===")
        print(f"Indexing time:")
        print(f"  Original: {orig_index_time:.3f}s")
        print(f"  Optimized: {opt_index_time:.3f}s")
        print(f"  Speedup: {orig_index_time / opt_index_time:.2f}x")
        
    if orig_query_time and opt_query_time:
        print(f"Query time:")
        print(f"  Original: {orig_query_time:.3f}s")
        print(f"  Optimized: {opt_query_time:.3f}s")
        print(f"  Speedup: {orig_query_time / opt_query_time:.2f}x")
    
    print("\n✓ Validation test completed successfully!")


if __name__ == "__main__":
    main()
