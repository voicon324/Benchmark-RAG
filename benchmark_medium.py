#!/usr/bin/env python3
"""
Medium-scale BM25 performance benchmark
"""

import time
import logging
import numpy as np
from typing import Dict, List, Any

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import models
from src.newaibench.models import BM25Model, OptimizedBM25Model
from src.newaibench.models.optimized_sparse import BM25OptimizationConfig


def generate_test_corpus(num_docs: int) -> Dict[str, Dict[str, str]]:
    """Generate synthetic test corpus."""
    print(f"Generating {num_docs} test documents...")
    
    # Realistic words for document generation
    words = [
        'algorithm', 'analysis', 'application', 'approach', 'architecture', 'artificial',
        'classification', 'computer', 'data', 'database', 'deep', 'development',
        'evaluation', 'feature', 'framework', 'function', 'implementation', 'information',
        'intelligence', 'language', 'learning', 'machine', 'method', 'model', 'natural',
        'network', 'neural', 'optimization', 'performance', 'processing', 'recognition',
        'research', 'retrieval', 'search', 'system', 'technique', 'technology', 'training'
    ]
    
    corpus = {}
    for i in range(num_docs):
        # Generate document with varying length (50-200 words)
        doc_length = np.random.randint(50, 201)
        doc_words = np.random.choice(words, size=doc_length, replace=True)
        doc_text = ' '.join(doc_words)
        
        corpus[f"doc_{i:06d}"] = {
            'text': doc_text,
            'title': f"Document {i}",
            'metadata': {'length': doc_length}
        }
    
    return corpus


def generate_test_queries(num_queries: int) -> List[Dict[str, str]]:
    """Generate test queries."""
    query_terms = [
        'machine learning algorithm',
        'neural network training',
        'information retrieval system',
        'data processing method',
        'artificial intelligence application',
        'deep learning model',
        'computer vision technique',
        'natural language processing',
        'performance optimization',
        'database search algorithm'
    ]
    
    queries = []
    for i in range(num_queries):
        query_text = np.random.choice(query_terms)
        queries.append({
            'query_id': f"query_{i:04d}",
            'text': query_text
        })
    
    return queries


def benchmark_models(corpus_size: int, num_queries: int):
    """Benchmark both models with given parameters."""
    print(f"\n{'='*60}")
    print(f"BENCHMARK: {corpus_size} documents, {num_queries} queries")
    print(f"{'='*60}")
    
    # Generate test data
    corpus = generate_test_corpus(corpus_size)
    queries = generate_test_queries(num_queries)
    
    # Model configurations
    base_config = {
        'name': 'benchmark_bm25',
        'type': 'sparse',
        'parameters': {
            'k1': 1.6,
            'b': 0.75,
            'tokenizer': 'simple'
        }
    }
    
    opt_config = BM25OptimizationConfig(
        use_parallel_indexing=True,
        num_workers=4,
        use_sparse_matrix=True,
        use_caching=True,
        cache_size=5000,
        batch_size=500,
        enable_pruning=True,
        early_termination_k=5000
    )
    
    results = {}
    
    # Test Original BM25Model
    print(f"\n--- Original BM25Model ---")
    try:
        model = BM25Model(base_config)
        model.load_model()
        
        # Indexing
        start_time = time.time()
        model.index_corpus(corpus)
        orig_index_time = time.time() - start_time
        print(f"Indexing time: {orig_index_time:.3f}s")
        
        # Querying  
        start_time = time.time()
        orig_results = model.predict(queries, corpus, top_k=100)
        orig_query_time = time.time() - start_time
        print(f"Query time: {orig_query_time:.3f}s ({orig_query_time/num_queries:.4f}s per query)")
        
        results['original'] = {
            'index_time': orig_index_time,
            'query_time': orig_query_time,
            'results': orig_results
        }
        
    except Exception as e:
        print(f"Error with original model: {e}")
        results['original'] = None
    
    # Test Optimized BM25Model
    print(f"\n--- OptimizedBM25Model ---")
    try:
        model = OptimizedBM25Model(base_config, opt_config)
        model.load_model()
        
        # Indexing
        start_time = time.time()
        model.index_corpus(corpus)
        opt_index_time = time.time() - start_time
        print(f"Indexing time: {opt_index_time:.3f}s")
        
        # Querying
        start_time = time.time()
        opt_results = model.predict(queries, corpus, top_k=100)
        opt_query_time = time.time() - start_time
        print(f"Query time: {opt_query_time:.3f}s ({opt_query_time/num_queries:.4f}s per query)")
        
        # Show optimization stats
        info = model.get_model_info()
        if 'optimization_stats' in info:
            stats = info['optimization_stats']
            cache_hit_rate = stats['cache_hits'] / (stats['cache_hits'] + stats['cache_misses']) if (stats['cache_hits'] + stats['cache_misses']) > 0 else 0
            print(f"Cache hit rate: {cache_hit_rate:.1%}")
        
        results['optimized'] = {
            'index_time': opt_index_time,
            'query_time': opt_query_time,
            'results': opt_results
        }
        
    except Exception as e:
        print(f"Error with optimized model: {e}")
        import traceback
        traceback.print_exc()
        results['optimized'] = None
    
    # Compare results
    print(f"\n--- Performance Comparison ---")
    if results['original'] and results['optimized']:
        index_speedup = orig_index_time / opt_index_time
        query_speedup = orig_query_time / opt_query_time
        
        print(f"Indexing speedup: {index_speedup:.2f}x")
        print(f"Query speedup: {query_speedup:.2f}x")
        
        # Check result similarity
        orig_res = results['original']['results']
        opt_res = results['optimized']['results']
        
        # Calculate overlap for first query
        if len(orig_res) > 0 and len(opt_res) > 0:
            sample_query = list(orig_res.keys())[0]
            orig_docs = set(orig_res[sample_query].keys())
            opt_docs = set(opt_res[sample_query].keys())
            overlap = len(orig_docs & opt_docs) / len(orig_docs | opt_docs) if (orig_docs | opt_docs) else 0
            print(f"Result overlap: {overlap:.1%}")
    
    return results


def main():
    """Run progressive benchmarks."""
    print("BM25 Optimization Performance Benchmark")
    print("="*50)
    
    # Progressive benchmarks
    test_cases = [
        (1000, 50),    # Small
        (5000, 100),   # Medium 
        (10000, 200),  # Large
        (25000, 500),  # Very Large
    ]
    
    all_results = []
    
    for corpus_size, num_queries in test_cases:
        result = benchmark_models(corpus_size, num_queries)
        all_results.append({
            'corpus_size': corpus_size,
            'num_queries': num_queries,
            'results': result
        })
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    print(f"{'Corpus Size':<12} {'Queries':<8} {'Index Speedup':<14} {'Query Speedup':<14}")
    print(f"{'-'*60}")
    
    for benchmark in all_results:
        corpus_size = benchmark['corpus_size']
        num_queries = benchmark['num_queries']
        results = benchmark['results']
        
        if results['original'] and results['optimized']:
            index_speedup = results['original']['index_time'] / results['optimized']['index_time']
            query_speedup = results['original']['query_time'] / results['optimized']['query_time']
            
            print(f"{corpus_size:<12} {num_queries:<8} {index_speedup:<14.2f} {query_speedup:<14.2f}")
        else:
            print(f"{corpus_size:<12} {num_queries:<8} {'ERROR':<14} {'ERROR':<14}")
    
    # Overall averages
    valid_benchmarks = [b for b in all_results if b['results']['original'] and b['results']['optimized']]
    if valid_benchmarks:
        avg_index_speedup = np.mean([
            b['results']['original']['index_time'] / b['results']['optimized']['index_time']
            for b in valid_benchmarks
        ])
        avg_query_speedup = np.mean([
            b['results']['original']['query_time'] / b['results']['optimized']['query_time']
            for b in valid_benchmarks
        ])
        
        print(f"\nAverage speedups:")
        print(f"  Indexing: {avg_index_speedup:.2f}x")
        print(f"  Querying: {avg_query_speedup:.2f}x")


if __name__ == "__main__":
    main()
