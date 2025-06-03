#!/usr/bin/env python3
"""
Performance test for OptimizedBM25Model query processing improvements
"""

import time
import logging
from typing import Dict, List, Any
import numpy as np

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


def create_large_test_data():
    """Create larger test corpus and queries for performance testing."""
    # Create larger corpus
    corpus = {}
    for i in range(5000):  # 5K documents
        corpus[f"doc_{i}"] = {
            "text": f"This is document {i} about topic {i % 10}. " * 20,  # Longer documents
            "title": f"Document {i} Title"
        }
    
    # Create more queries
    queries = []
    for i in range(500):  # 500 queries
        queries.append({
            "query_id": f"query_{i}",
            "text": f"topic {i % 10} document search information"
        })
    
    return corpus, queries


def test_performance_improvements():
    """Test performance improvements with larger dataset."""
    print("\n=== Performance Test with Large Dataset ===")
    
    # Configuration for both models
    config = {
        'name': 'performance_test_bm25',
        'type': 'sparse',
        'parameters': {
            'k1': 1.6,
            'b': 0.75,
            'tokenizer': 'simple'
        }
    }
    
    # Create test data
    corpus, queries = create_large_test_data()
    print(f"Created test data: {len(corpus)} documents, {len(queries)} queries")
    
    # Test original BM25Model
    print("\n--- Testing Original BM25Model ---")
    original_model = BM25Model(config)
    original_model.load_model()
    
    # Index corpus
    start_time = time.time()
    original_model.index_corpus(corpus, show_progress=False)
    original_index_time = time.time() - start_time
    print(f"Original indexing time: {original_index_time:.3f}s")
    
    # Run queries
    start_time = time.time()
    original_results = original_model.predict(queries, corpus, top_k=20, show_progress=False)
    original_query_time = time.time() - start_time
    original_avg_time = original_query_time / len(queries)
    print(f"Original query time: {original_query_time:.3f}s ({original_avg_time:.6f}s per query)")
    
    # Test optimized BM25Model with different configurations
    configs_to_test = [
        {
            'name': 'Basic Optimized',
            'config': BM25OptimizationConfig(
                use_parallel_indexing=True,
                use_caching=True,
                use_vectorized_scoring=False,
                reduce_progress_overhead=False
            )
        },
        {
            'name': 'Vectorized Optimized',
            'config': BM25OptimizationConfig(
                use_parallel_indexing=True,
                use_caching=True,
                use_vectorized_scoring=True,
                reduce_progress_overhead=True,
                aggressive_pruning=True
            )
        },
        {
            'name': 'Full Optimized',
            'config': BM25OptimizationConfig(
                use_parallel_indexing=True,
                use_caching=True,
                use_vectorized_scoring=True,
                reduce_progress_overhead=True,
                aggressive_pruning=True,
                enable_pruning=True,
                early_termination_k=5000
            )
        }
    ]
    
    for test_config in configs_to_test:
        print(f"\n--- Testing {test_config['name']} ---")
        opt_config = test_config['config']
        
        optimized_model = OptimizedBM25Model(config, opt_config)
        optimized_model.load_model()
        
        # Index corpus
        start_time = time.time()
        optimized_model.index_corpus(corpus, show_progress=False)
        opt_index_time = time.time() - start_time
        print(f"Optimized indexing time: {opt_index_time:.3f}s")
        
        # Run queries
        start_time = time.time()
        optimized_results = optimized_model.predict(queries, corpus, top_k=20, show_progress=False)
        opt_query_time = time.time() - start_time
        opt_avg_time = opt_query_time / len(queries)
        print(f"Optimized query time: {opt_query_time:.3f}s ({opt_avg_time:.6f}s per query)")
        
        # Calculate speedup
        index_speedup = original_index_time / opt_index_time if opt_index_time > 0 else 0
        query_speedup = original_query_time / opt_query_time if opt_query_time > 0 else 0
        
        print(f"Indexing speedup: {index_speedup:.2f}x")
        print(f"Query speedup: {query_speedup:.2f}x")
        
        # Get optimization stats
        model_info = optimized_model.get_model_info()
        if 'optimization_stats' in model_info:
            stats = model_info['optimization_stats']
            cache_hit_rate = stats['cache_hits'] / (stats['cache_hits'] + stats['cache_misses']) if (stats['cache_hits'] + stats['cache_misses']) > 0 else 0
            print(f"Cache hit rate: {cache_hit_rate:.2%}")
        
        # Verify results are similar
        if len(original_results) > 0 and len(optimized_results) > 0:
            sample_query = list(original_results.keys())[0]
            orig_docs = set(original_results[sample_query].keys())
            opt_docs = set(optimized_results[sample_query].keys())
            if orig_docs and opt_docs:
                overlap = len(orig_docs & opt_docs) / len(orig_docs | opt_docs)
                print(f"Result similarity: {overlap:.2%}")
        
        print("-" * 50)


def test_query_scalability():
    """Test how performance scales with number of queries."""
    print("\n=== Query Scalability Test ===")
    
    config = {
        'name': 'scalability_test_bm25',
        'type': 'sparse',
        'parameters': {
            'k1': 1.6,
            'b': 0.75,
            'tokenizer': 'simple'
        }
    }
    
    opt_config = BM25OptimizationConfig(
        use_parallel_indexing=True,
        use_caching=True,
        use_vectorized_scoring=True,
        reduce_progress_overhead=True,
        aggressive_pruning=True
    )
    
    # Create corpus
    corpus = {}
    for i in range(1000):
        corpus[f"doc_{i}"] = {
            "text": f"This is document {i} about topic {i % 20}. " * 10,
            "title": f"Document {i} Title"
        }
    
    # Test with different query counts
    query_counts = [10, 50, 100, 250, 500]
    
    for query_count in query_counts:
        print(f"\n--- Testing with {query_count} queries ---")
        
        # Create queries
        queries = []
        for i in range(query_count):
            queries.append({
                "query_id": f"query_{i}",
                "text": f"topic {i % 20} document search"
            })
        
        # Test optimized model
        optimized_model = OptimizedBM25Model(config, opt_config)
        optimized_model.load_model()
        optimized_model.index_corpus(corpus, show_progress=False)
        
        start_time = time.time()
        results = optimized_model.predict(queries, corpus, top_k=10, show_progress=False)
        query_time = time.time() - start_time
        avg_time = query_time / query_count
        
        print(f"Total time: {query_time:.3f}s")
        print(f"Average per query: {avg_time:.6f}s")
        print(f"Queries per second: {query_count / query_time:.1f}")


def main():
    """Run performance tests."""
    print("OptimizedBM25Model Query Performance Test")
    print("=" * 50)
    
    test_performance_improvements()
    test_query_scalability()
    
    print("\n✓ Performance testing completed!")


if __name__ == "__main__":
    main()
