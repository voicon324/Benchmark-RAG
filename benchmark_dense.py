#!/usr/bin/env python3
"""
Performance benchmarking script for Dense Retrieval models.

This script provides comprehensive performance analysis including:
- Encoding speed benchmarks
- Search latency measurements  
- Memory usage profiling
- Scalability analysis
- Model comparison studies
"""

import sys
import os
sys.path.append('src')

import time
import psutil
import numpy as np
from typing import List, Dict, Any, Tuple
import logging
import json
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PerformanceBenchmark:
    """Comprehensive performance benchmarking suite."""
    
    def __init__(self):
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "system_info": self._get_system_info(),
            "benchmarks": {}
        }
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Collect system information."""
        return {
            "cpu_count": psutil.cpu_count(),
            "memory_total": psutil.virtual_memory().total,
            "python_version": sys.version,
            "platform": os.name
        }
    
    def _measure_memory(self, func, *args, **kwargs):
        """Measure memory usage of a function."""
        process = psutil.Process(os.getpid())
        
        # Baseline memory
        baseline = process.memory_info().rss
        
        # Execute function
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        # Peak memory
        peak = process.memory_info().rss
        
        return result, {
            "execution_time": end_time - start_time,
            "memory_baseline": baseline,
            "memory_peak": peak,
            "memory_delta": peak - baseline
        }
    
    def benchmark_model_loading(self, model_configs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Benchmark model loading times."""
        print("\n=== Benchmarking Model Loading ===")
        
        results = {}
        
        for config in model_configs:
            model_name = config["model_name_or_path"]
            print(f"Testing model: {model_name}")
            
            try:
                from newaibench.models.dense import DenseTextRetriever
                
                def load_model():
                    model = DenseTextRetriever(config)
                    model.load_model()
                    return model
                
                model, metrics = self._measure_memory(load_model)
                
                results[model_name] = {
                    "load_time": metrics["execution_time"],
                    "memory_usage": metrics["memory_delta"],
                    "success": True
                }
                
                print(f"  ✓ Load time: {metrics['execution_time']:.2f}s")
                print(f"  ✓ Memory: {metrics['memory_delta'] / 1024**2:.1f}MB")
                
            except Exception as e:
                results[model_name] = {
                    "error": str(e),
                    "success": False
                }
                print(f"  ✗ Failed: {e}")
        
        return results
    
    def benchmark_encoding_speed(self, model_configs: List[Dict[str, Any]], 
                                test_sizes: List[int] = [10, 50, 100, 500]) -> Dict[str, Any]:
        """Benchmark text encoding speeds."""
        print("\n=== Benchmarking Encoding Speed ===")
        
        results = {}
        
        # Generate test texts of varying lengths
        test_texts = {
            "short": ["Short text"] * max(test_sizes),
            "medium": ["This is a medium length text with several words and concepts"] * max(test_sizes),
            "long": ["This is a much longer text that contains many more words and should test the model's ability to handle variable length inputs effectively while maintaining good performance characteristics and demonstrating scalability"] * max(test_sizes)
        }
        
        for config in model_configs:
            model_name = config["model_name_or_path"]
            print(f"Testing model: {model_name}")
            
            try:
                from newaibench.models.dense import DenseTextRetriever
                
                model = DenseTextRetriever(config)
                model.load_model()
                
                model_results = {}
                
                for text_type, texts in test_texts.items():
                    type_results = {}
                    
                    for size in test_sizes:
                        if size > len(texts):
                            continue
                            
                        test_batch = texts[:size]
                        
                        # Warmup
                        model.encode_texts(test_batch[:min(5, size)])
                        
                        # Actual benchmark
                        start_time = time.time()
                        embeddings = model.encode_texts(test_batch)
                        end_time = time.time()
                        
                        encoding_time = end_time - start_time
                        texts_per_sec = size / encoding_time
                        
                        type_results[size] = {
                            "time": encoding_time,
                            "texts_per_sec": texts_per_sec,
                            "embedding_shape": embeddings.shape
                        }
                        
                        print(f"  {text_type} x{size}: {texts_per_sec:.0f} texts/sec")
                    
                    model_results[text_type] = type_results
                
                results[model_name] = model_results
                
            except Exception as e:
                results[model_name] = {"error": str(e)}
                print(f"  ✗ Failed: {e}")
        
        return results
    
    def benchmark_search_performance(self, index_sizes: List[int] = [100, 1000, 5000],
                                   query_counts: List[int] = [1, 10, 50]) -> Dict[str, Any]:
        """Benchmark search performance across different corpus sizes."""
        print("\n=== Benchmarking Search Performance ===")
        
        results = {}
        
        # Test configurations
        search_configs = [
            {
                "name": "brute_force",
                "config": {
                    "name": "brute_test",
                    "type": "dense",
                    "model_name_or_path": "all-MiniLM-L6-v2",
                    "parameters": {"use_ann_index": False}
                }
            },
            {
                "name": "hnswlib",
                "config": {
                    "name": "hnsw_test", 
                    "type": "dense",
                    "model_name_or_path": "all-MiniLM-L6-v2",
                    "parameters": {
                        "use_ann_index": True,
                        "ann_backend": "hnswlib",
                        "m_parameter_hnsw": 16,
                        "ef_search_hnsw": 50
                    }
                }
            }
        ]
        
        for search_config in search_configs:
            search_name = search_config["name"]
            print(f"Testing search method: {search_name}")
            
            try:
                from newaibench.models.dense import DenseTextRetriever
                
                model = DenseTextRetriever(search_config["config"])
                model.load_model()
                
                method_results = {}
                
                for index_size in index_sizes:
                    print(f"  Index size: {index_size}")
                    
                    # Generate corpus
                    corpus = [
                        {
                            "id": f"doc_{i}",
                            "text": f"Document {i} about topic {i % 10} with content related to subject {i % 20}"
                        }
                        for i in range(index_size)
                    ]
                    
                    # Index corpus
                    index_start = time.time()
                    model.index_corpus(corpus)
                    index_time = time.time() - index_start
                    
                    size_results = {
                        "index_time": index_time,
                        "queries": {}
                    }
                    
                    for query_count in query_counts:
                        queries = [f"query about topic {i}" for i in range(query_count)]
                        
                        # Warmup
                        model.predict(queries[:min(2, query_count)], corpus, top_k=10)
                        
                        # Benchmark search
                        search_start = time.time()
                        results_list = model.predict(queries, corpus, top_k=10)
                        search_time = time.time() - search_start
                        
                        queries_per_sec = query_count / search_time
                        
                        size_results["queries"][query_count] = {
                            "search_time": search_time,
                            "queries_per_sec": queries_per_sec,
                            "results_returned": len(results_list[0]) if results_list else 0
                        }
                        
                        print(f"    {query_count} queries: {queries_per_sec:.1f} queries/sec")
                    
                    method_results[index_size] = size_results
                
                results[search_name] = method_results
                
            except Exception as e:
                results[search_name] = {"error": str(e)}
                print(f"  ✗ Failed: {e}")
        
        return results
    
    def benchmark_batch_sizes(self, batch_sizes: List[int] = [1, 4, 8, 16, 32, 64]) -> Dict[str, Any]:
        """Benchmark optimal batch sizes for encoding."""
        print("\n=== Benchmarking Batch Sizes ===")
        
        try:
            from newaibench.models.dense import DenseTextRetriever
            
            config = {
                "name": "batch_test",
                "type": "dense", 
                "model_name_or_path": "all-MiniLM-L6-v2",
                "parameters": {}
            }
            
            model = DenseTextRetriever(config)
            model.load_model()
            
            # Generate test texts
            test_texts = [
                f"This is test document {i} with some content to encode"
                for i in range(128)
            ]
            
            results = {}
            
            for batch_size in batch_sizes:
                print(f"Testing batch size: {batch_size}")
                
                # Warmup
                model.encode_texts(test_texts[:batch_size], batch_size=batch_size)
                
                # Benchmark
                start_time = time.time()
                embeddings = model.encode_texts(test_texts, batch_size=batch_size)
                end_time = time.time()
                
                encoding_time = end_time - start_time
                texts_per_sec = len(test_texts) / encoding_time
                
                results[batch_size] = {
                    "time": encoding_time,
                    "texts_per_sec": texts_per_sec,
                    "embedding_shape": embeddings.shape
                }
                
                print(f"  {texts_per_sec:.0f} texts/sec")
            
            return results
            
        except Exception as e:
            return {"error": str(e)}
    
    def benchmark_memory_scaling(self, corpus_sizes: List[int] = [100, 500, 1000, 2000]) -> Dict[str, Any]:
        """Benchmark memory usage scaling with corpus size."""
        print("\n=== Benchmarking Memory Scaling ===")
        
        try:
            from newaibench.models.dense import DenseTextRetriever
            
            config = {
                "name": "memory_test",
                "type": "dense",
                "model_name_or_path": "all-MiniLM-L6-v2", 
                "parameters": {"use_ann_index": False}
            }
            
            model = DenseTextRetriever(config)
            model.load_model()
            
            results = {}
            process = psutil.Process(os.getpid())
            
            for corpus_size in corpus_sizes:
                print(f"Testing corpus size: {corpus_size}")
                
                # Generate corpus
                corpus = [
                    {
                        "id": f"doc_{i}",
                        "text": f"Document {i} content " * 10  # Longer docs
                    }
                    for i in range(corpus_size)
                ]
                
                # Measure indexing
                memory_before = process.memory_info().rss
                
                start_time = time.time()
                model.index_corpus(corpus)
                index_time = time.time() - start_time
                
                memory_after = process.memory_info().rss
                memory_used = memory_after - memory_before
                
                results[corpus_size] = {
                    "index_time": index_time,
                    "memory_used": memory_used,
                    "memory_per_doc": memory_used / corpus_size,
                    "docs_per_sec": corpus_size / index_time
                }
                
                print(f"  Memory: {memory_used / 1024**2:.1f}MB ({memory_used / corpus_size / 1024:.1f}KB/doc)")
                print(f"  Speed: {corpus_size / index_time:.0f} docs/sec")
                
                # Clear for next iteration
                model.doc_embeddings.clear()
                model.doc_ids_list.clear()
            
            return results
            
        except Exception as e:
            return {"error": str(e)}
    
    def run_full_benchmark(self) -> Dict[str, Any]:
        """Run complete benchmark suite."""
        print("Dense Retrieval Performance Benchmark Suite")
        print("=" * 60)
        
        # Model configurations to test
        model_configs = [
            {
                "name": "lightweight",
                "type": "dense",
                "model_name_or_path": "all-MiniLM-L6-v2",
                "parameters": {}
            },
            {
                "name": "standard", 
                "type": "dense",
                "model_name_or_path": "all-mpnet-base-v2",
                "parameters": {}
            }
        ]
        
        # Run benchmarks
        self.results["benchmarks"]["model_loading"] = self.benchmark_model_loading(model_configs)
        self.results["benchmarks"]["encoding_speed"] = self.benchmark_encoding_speed(model_configs)
        self.results["benchmarks"]["search_performance"] = self.benchmark_search_performance()
        self.results["benchmarks"]["batch_sizes"] = self.benchmark_batch_sizes()
        self.results["benchmarks"]["memory_scaling"] = self.benchmark_memory_scaling()
        
        return self.results
    
    def save_results(self, filepath: str):
        """Save benchmark results to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"Results saved to: {filepath}")
    
    def print_summary(self):
        """Print benchmark summary."""
        print(f"\n{'='*60}")
        print("BENCHMARK SUMMARY")
        print(f"{'='*60}")
        
        # Model loading summary
        if "model_loading" in self.results["benchmarks"]:
            print("\nModel Loading Performance:")
            for model, data in self.results["benchmarks"]["model_loading"].items():
                if data.get("success"):
                    print(f"  {model}: {data['load_time']:.2f}s, {data['memory_usage']/1024**2:.1f}MB")
        
        # Encoding speed summary
        if "encoding_speed" in self.results["benchmarks"]:
            print("\nEncoding Speed (texts/sec):")
            for model, data in self.results["benchmarks"]["encoding_speed"].items():
                if "medium" in data and 100 in data["medium"]:
                    speed = data["medium"][100]["texts_per_sec"]
                    print(f"  {model}: {speed:.0f} texts/sec")
        
        # Search performance summary
        if "search_performance" in self.results["benchmarks"]:
            print("\nSearch Performance (queries/sec @ 1000 docs):")
            for method, data in self.results["benchmarks"]["search_performance"].items():
                if 1000 in data and 10 in data[1000]["queries"]:
                    speed = data[1000]["queries"][10]["queries_per_sec"]
                    print(f"  {method}: {speed:.1f} queries/sec")
        
        # Memory scaling summary
        if "memory_scaling" in self.results["benchmarks"]:
            print("\nMemory Usage (KB per document):")
            data = self.results["benchmarks"]["memory_scaling"]
            if 1000 in data:
                mem_per_doc = data[1000]["memory_per_doc"] / 1024
                print(f"  Average: {mem_per_doc:.1f} KB/doc")

def main():
    """Run benchmarks."""
    benchmark = PerformanceBenchmark()
    
    try:
        results = benchmark.run_full_benchmark()
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"benchmark_results_{timestamp}.json"
        benchmark.save_results(results_file)
        
        # Print summary
        benchmark.print_summary()
        
        print(f"\n✓ Benchmark completed successfully!")
        print(f"  Full results saved to: {results_file}")
        
    except Exception as e:
        print(f"✗ Benchmark failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
