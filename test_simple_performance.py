#!/usr/bin/env python3
"""
Simple test script to validate OptimizedBM25Model performance improvements
using the YAML test configuration.
"""

import sys
import os
import time
import yaml
import traceback
from pathlib import Path

# Add the src directory to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

try:
    from newaibench.models.optimized_sparse import OptimizedBM25Model, BM25OptimizationConfig
    from newaibench.models.sparse import BM25Model
    from newaibench.tokenizers.simple import SimpleTokenizer
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running from the correct directory")
    sys.exit(1)

def load_yaml_test(yaml_path):
    """Load test configuration from YAML file."""
    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f)

def prepare_test_data(test_config):
    """Extract queries and corpus from test configuration."""
    dataset = test_config['datasets'][0]  # Use first dataset
    
    queries = []
    for query in dataset['queries']:
        queries.append({
            'query_id': query['query_id'],
            'text': query['text']
        })
    
    corpus = {}
    for doc in dataset['corpus']:
        corpus[doc['doc_id']] = {
            'title': doc['title'],
            'text': doc['text']
        }
    
    return queries, corpus

def create_model(model_config):
    """Create a model instance from configuration."""
    model_type = model_config['type']
    params = model_config['parameters']
    
    if model_type == "BM25Model":
        return BM25Model(
            k1=params['k1'],
            b=params['b'],
            tokenizer=SimpleTokenizer()
        )
    elif model_type == "OptimizedBM25Model":
        opt_config = BM25OptimizationConfig(**params.get('optimization_config', {}))
        return OptimizedBM25Model(
            k1=params['k1'],
            b=params['b'],
            tokenizer=SimpleTokenizer(),
            optimization_config=opt_config
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def benchmark_model(model, queries, corpus, top_k=10):
    """Benchmark a model and return timing results."""
    print(f"\nTesting {model.name}...")
    
    # Index corpus
    index_start = time.time()
    model.index_corpus(corpus, show_progress=False)
    index_time = time.time() - index_start
    
    # Process queries
    query_start = time.time()
    results = model.predict(queries, corpus, top_k=top_k, show_progress=False)
    query_time = time.time() - query_start
    
    # Calculate average query time
    avg_query_time = query_time / len(queries) if queries else 0
    
    return {
        'model_name': model.name,
        'index_time': index_time,
        'total_query_time': query_time,
        'avg_query_time': avg_query_time,
        'num_queries': len(queries),
        'results': results
    }

def compare_results(baseline_results, optimized_results, tolerance=0.1):
    """Compare results between baseline and optimized models."""
    print("\nComparing results between models...")
    
    baseline_data = baseline_results['results']
    optimized_data = optimized_results['results']
    
    all_match = True
    for query_id in baseline_data:
        if query_id not in optimized_data:
            print(f"❌ Query {query_id} missing in optimized results")
            all_match = False
            continue
        
        baseline_docs = baseline_data[query_id]
        optimized_docs = optimized_data[query_id]
        
        # Check if top documents are similar (order might differ slightly due to optimizations)
        baseline_top = sorted(baseline_docs.items(), key=lambda x: x[1], reverse=True)[:5]
        optimized_top = sorted(optimized_docs.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # Check if top docs are in similar ranges
        for i, ((b_doc, b_score), (o_doc, o_score)) in enumerate(zip(baseline_top, optimized_top)):
            score_diff = abs(b_score - o_score)
            if score_diff > tolerance:
                print(f"⚠️  Query {query_id}: Score difference {score_diff:.3f} for top-{i+1} docs")
                all_match = False
    
    if all_match:
        print("✅ Results are consistent between models")
    
    return all_match

def print_performance_summary(results_list):
    """Print a performance comparison summary."""
    print("\n" + "="*60)
    print("PERFORMANCE SUMMARY")
    print("="*60)
    
    print(f"{'Model':<25} {'Index (s)':<10} {'Query (s)':<10} {'Speedup':<10}")
    print("-" * 55)
    
    baseline_time = None
    for result in results_list:
        model_name = result['model_name']
        index_time = result['index_time']
        avg_query_time = result['avg_query_time']
        
        if baseline_time is None:
            baseline_time = avg_query_time
            speedup = "1.0x"
        else:
            speedup = f"{baseline_time / avg_query_time:.2f}x" if avg_query_time > 0 else "∞"
        
        print(f"{model_name:<25} {index_time:<10.4f} {avg_query_time:<10.6f} {speedup:<10}")

def main():
    """Main test function."""
    yaml_path = "/home/hkduy/NewAI/new_bench/test_optimized_bm25.yaml"
    
    if not os.path.exists(yaml_path):
        print(f"Test file not found: {yaml_path}")
        return
    
    try:
        # Load test configuration
        print("Loading test configuration...")
        test_config = load_yaml_test(yaml_path)
        queries, corpus = prepare_test_data(test_config)
        
        print(f"Test setup:")
        print(f"  - Queries: {len(queries)}")
        print(f"  - Documents: {len(corpus)}")
        
        # Run benchmarks for each model
        results = []
        for model_config in test_config['models']:
            try:
                model = create_model(model_config)
                result = benchmark_model(model, queries, corpus, top_k=10)
                results.append(result)
            except Exception as e:
                print(f"Error testing {model_config['name']}: {e}")
                traceback.print_exc()
        
        # Compare results between baseline and optimized models
        if len(results) >= 2:
            compare_results(results[0], results[1])
        
        # Print performance summary
        print_performance_summary(results)
        
        # Additional model info for optimized models
        for i, model_config in enumerate(test_config['models']):
            if 'OptimizedBM25' in model_config['type'] and i < len(results):
                print(f"\n{model_config['name']} optimization stats:")
                try:
                    model = create_model(model_config)
                    model.index_corpus(corpus, show_progress=False)
                    info = model.get_model_info()
                    opt_stats = info.get('optimization_stats', {})
                    for key, value in opt_stats.items():
                        print(f"  - {key}: {value}")
                except Exception as e:
                    print(f"  Error getting model info: {e}")
        
        print("\n✅ Test completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
