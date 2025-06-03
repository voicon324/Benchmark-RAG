#!/usr/bin/env python3
"""
Comprehensive BM25 Optimization Benchmark Script

This script compares the performance between the original BM25Model 
and the new OptimizedBM25Model to validate speed improvements.
"""

import time
import logging
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Tuple
from collections import defaultdict
import memory_profiler
import psutil
import multiprocessing as mp

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import models
from src.newaibench.models import BM25Model, OptimizedBM25Model
from src.newaibench.models.optimized_sparse import BM25OptimizationConfig


class BM25PerformanceBenchmark:
    """Comprehensive performance benchmark for BM25 implementations."""
    
    def __init__(self):
        self.results = defaultdict(list)
        self.memory_usage = defaultdict(list)
        
    def generate_test_corpus(self, num_docs: int, 
                           avg_doc_length: int = 100) -> Dict[str, Dict[str, str]]:
        """Generate synthetic test corpus for benchmarking."""
        logger.info(f"Generating test corpus with {num_docs} documents")
        
        # Common words for generating realistic text
        words = [
            'document', 'retrieval', 'information', 'search', 'text', 'query',
            'algorithm', 'ranking', 'scoring', 'relevance', 'model', 'system',
            'analysis', 'processing', 'classification', 'machine', 'learning',
            'data', 'mining', 'extraction', 'indexing', 'corpus', 'collection',
            'evaluation', 'performance', 'benchmark', 'testing', 'optimization',
            'computer', 'science', 'technology', 'research', 'method', 'approach',
            'technique', 'implementation', 'framework', 'library', 'tool',
            'application', 'software', 'database', 'storage', 'memory', 'cache'
        ]
        
        corpus = {}
        
        for i in range(num_docs):
            # Generate document with varying length
            doc_length = np.random.poisson(avg_doc_length)
            doc_length = max(10, doc_length)  # Minimum 10 words
            
            # Generate text by randomly sampling words
            doc_text = ' '.join(np.random.choice(words, size=doc_length))
            
            corpus[f"doc_{i:06d}"] = {
                'text': doc_text,
                'title': f"Document {i}",
                'metadata': {'length': doc_length}
            }
            
            if (i + 1) % 10000 == 0:
                logger.info(f"Generated {i + 1} documents")
        
        return corpus
    
    def generate_test_queries(self, num_queries: int, 
                            corpus: Dict[str, Dict[str, str]]) -> List[Dict[str, str]]:
        """Generate test queries based on corpus content."""
        logger.info(f"Generating {num_queries} test queries")
        
        # Extract words from corpus for realistic queries
        all_words = []
        doc_samples = list(corpus.values())[:min(1000, len(corpus))]  # Sample for efficiency
        
        for doc in doc_samples:
            words = doc['text'].split()
            all_words.extend(words)
        
        # Get most common words for realistic queries
        from collections import Counter
        word_counts = Counter(all_words)
        common_words = [word for word, count in word_counts.most_common(200)]
        
        queries = []
        for i in range(num_queries):
            # Generate queries of varying length (1-5 words)
            query_length = np.random.randint(1, 6)
            query_words = np.random.choice(common_words, size=query_length, replace=False)
            query_text = ' '.join(query_words)
            
            queries.append({
                'query_id': f"query_{i:04d}",
                'text': query_text
            })
        
        return queries
    
    def create_model_configs(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Create configurations for both models."""
        
        base_config = {
            'name': 'bm25_test',
            'type': 'sparse',
            'parameters': {
                'k1': 1.6,
                'b': 0.75,
                'tokenizer': 'simple',
                'lowercase': True,
                'remove_stopwords': False
            }
        }
        
        # Optimization configuration for optimized model
        opt_config = BM25OptimizationConfig(
            use_parallel_indexing=True,
            num_workers=min(mp.cpu_count(), 8),
            use_sparse_matrix=True,
            use_caching=True,
            cache_size=10000,
            batch_size=1000,
            enable_pruning=True,
            use_fast_tokenizer=True,
            early_termination_k=10000
        )
        
        return base_config, opt_config
    
    def measure_memory_usage(self, func, *args, **kwargs):
        """Measure memory usage of a function."""
        process = psutil.Process()
        mem_before = process.memory_info().rss / 1024 / 1024  # MB
        
        result = func(*args, **kwargs)
        
        mem_after = process.memory_info().rss / 1024 / 1024  # MB
        memory_usage = mem_after - mem_before
        
        return result, memory_usage
    
    def benchmark_indexing(self, corpus_sizes: List[int]) -> Dict[str, Any]:
        """Benchmark corpus indexing performance."""
        logger.info("Benchmarking indexing performance")
        
        indexing_results = {
            'corpus_sizes': corpus_sizes,
            'original_times': [],
            'optimized_times': [],
            'original_memory': [],
            'optimized_memory': [],
            'speedup_factors': []
        }
        
        base_config, opt_config = self.create_model_configs()
        
        for corpus_size in corpus_sizes:
            logger.info(f"Testing with corpus size: {corpus_size}")
            
            # Generate test corpus
            corpus = self.generate_test_corpus(corpus_size)
            
            # Test original BM25Model
            logger.info("Testing original BM25Model indexing")
            original_model = BM25Model(base_config)
            original_model.load_model()
            
            start_time = time.time()
            _, original_memory = self.measure_memory_usage(
                original_model.index_corpus, corpus
            )
            original_time = time.time() - start_time
            
            logger.info(f"Original indexing: {original_time:.2f}s, {original_memory:.1f}MB")
            
            # Test optimized BM25Model
            logger.info("Testing OptimizedBM25Model indexing")
            optimized_model = OptimizedBM25Model(base_config, opt_config)
            optimized_model.load_model()
            
            start_time = time.time()
            _, optimized_memory = self.measure_memory_usage(
                optimized_model.index_corpus, corpus
            )
            optimized_time = time.time() - start_time
            
            logger.info(f"Optimized indexing: {optimized_time:.2f}s, {optimized_memory:.1f}MB")
            
            # Calculate speedup
            speedup = original_time / optimized_time if optimized_time > 0 else 0
            
            # Store results
            indexing_results['original_times'].append(original_time)
            indexing_results['optimized_times'].append(optimized_time)
            indexing_results['original_memory'].append(original_memory)
            indexing_results['optimized_memory'].append(optimized_memory)
            indexing_results['speedup_factors'].append(speedup)
            
            logger.info(f"Speedup factor: {speedup:.2f}x")
            
            # Clean up
            del original_model
            del optimized_model
            del corpus
        
        return indexing_results
    
    def benchmark_querying(self, corpus_size: int, query_counts: List[int]) -> Dict[str, Any]:
        """Benchmark query processing performance."""
        logger.info("Benchmarking query performance")
        
        query_results = {
            'query_counts': query_counts,
            'original_times': [],
            'optimized_times': [],
            'original_avg_times': [],
            'optimized_avg_times': [],
            'speedup_factors': []
        }
        
        # Generate test corpus and index both models
        corpus = self.generate_test_corpus(corpus_size)
        base_config, opt_config = self.create_model_configs()
        
        # Prepare models
        original_model = BM25Model(base_config)
        original_model.load_model()
        original_model.index_corpus(corpus)
        
        optimized_model = OptimizedBM25Model(base_config, opt_config)
        optimized_model.load_model()
        optimized_model.index_corpus(corpus)
        
        for query_count in query_counts:
            logger.info(f"Testing with {query_count} queries")
            
            # Generate test queries
            queries = self.generate_test_queries(query_count, corpus)
            
            # Test original BM25Model
            logger.info("Testing original BM25Model querying")
            start_time = time.time()
            original_results = original_model.predict(queries, corpus, top_k=100)
            original_time = time.time() - start_time
            original_avg_time = original_time / query_count
            
            # Test optimized BM25Model
            logger.info("Testing OptimizedBM25Model querying")
            start_time = time.time()
            optimized_results = optimized_model.predict(queries, corpus, top_k=100)
            optimized_time = time.time() - start_time
            optimized_avg_time = optimized_time / query_count
            
            # Calculate speedup
            speedup = original_time / optimized_time if optimized_time > 0 else 0
            
            # Store results
            query_results['original_times'].append(original_time)
            query_results['optimized_times'].append(optimized_time)
            query_results['original_avg_times'].append(original_avg_time)
            query_results['optimized_avg_times'].append(optimized_avg_time)
            query_results['speedup_factors'].append(speedup)
            
            logger.info(f"Original: {original_time:.2f}s ({original_avg_time:.4f}s/query)")
            logger.info(f"Optimized: {optimized_time:.2f}s ({optimized_avg_time:.4f}s/query)")
            logger.info(f"Query speedup: {speedup:.2f}x")
            
            # Verify results are similar (basic sanity check)
            if len(original_results) > 0 and len(optimized_results) > 0:
                sample_query = list(original_results.keys())[0]
                orig_docs = set(original_results[sample_query].keys())
                opt_docs = set(optimized_results[sample_query].keys())
                overlap = len(orig_docs & opt_docs) / len(orig_docs | opt_docs)
                logger.info(f"Result overlap: {overlap:.2%}")
        
        return query_results
    
    def benchmark_memory_usage(self, corpus_sizes: List[int]) -> Dict[str, Any]:
        """Benchmark memory usage patterns."""
        logger.info("Benchmarking memory usage")
        
        memory_results = {
            'corpus_sizes': corpus_sizes,
            'original_peak_memory': [],
            'optimized_peak_memory': [],
            'memory_savings': []
        }
        
        base_config, opt_config = self.create_model_configs()
        
        for corpus_size in corpus_sizes:
            logger.info(f"Memory test with corpus size: {corpus_size}")
            
            corpus = self.generate_test_corpus(corpus_size)
            queries = self.generate_test_queries(50, corpus)
            
            # Test original model memory usage
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024
            
            original_model = BM25Model(base_config)
            original_model.load_model()
            original_model.index_corpus(corpus)
            _ = original_model.predict(queries, corpus, top_k=100)
            
            original_peak = process.memory_info().rss / 1024 / 1024 - initial_memory
            del original_model
            
            # Test optimized model memory usage
            initial_memory = process.memory_info().rss / 1024 / 1024
            
            optimized_model = OptimizedBM25Model(base_config, opt_config)
            optimized_model.load_model()
            optimized_model.index_corpus(corpus)
            _ = optimized_model.predict(queries, corpus, top_k=100)
            
            optimized_peak = process.memory_info().rss / 1024 / 1024 - initial_memory
            del optimized_model
            
            memory_saving = (original_peak - optimized_peak) / original_peak if original_peak > 0 else 0
            
            memory_results['original_peak_memory'].append(original_peak)
            memory_results['optimized_peak_memory'].append(optimized_peak)
            memory_results['memory_savings'].append(memory_saving)
            
            logger.info(f"Original peak: {original_peak:.1f}MB")
            logger.info(f"Optimized peak: {optimized_peak:.1f}MB") 
            logger.info(f"Memory saving: {memory_saving:.1%}")
        
        return memory_results
    
    def create_visualizations(self, results: Dict[str, Any], output_dir: Path):
        """Create performance visualization charts."""
        output_dir.mkdir(exist_ok=True)
        
        # Indexing performance chart
        if 'indexing' in results:
            plt.figure(figsize=(12, 8))
            
            plt.subplot(2, 2, 1)
            data = results['indexing']
            plt.plot(data['corpus_sizes'], data['original_times'], 'o-', label='Original BM25', linewidth=2)
            plt.plot(data['corpus_sizes'], data['optimized_times'], 's-', label='Optimized BM25', linewidth=2)
            plt.xlabel('Corpus Size (documents)')
            plt.ylabel('Indexing Time (seconds)')
            plt.title('Indexing Performance Comparison')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.subplot(2, 2, 2)
            plt.plot(data['corpus_sizes'], data['speedup_factors'], 'g^-', linewidth=2)
            plt.xlabel('Corpus Size (documents)')
            plt.ylabel('Speedup Factor')
            plt.title('Indexing Speedup Factor')
            plt.grid(True, alpha=0.3)
            plt.axhline(y=1, color='r', linestyle='--', alpha=0.5)
            
            plt.subplot(2, 2, 3)
            plt.plot(data['corpus_sizes'], data['original_memory'], 'o-', label='Original BM25', linewidth=2)
            plt.plot(data['corpus_sizes'], data['optimized_memory'], 's-', label='Optimized BM25', linewidth=2)
            plt.xlabel('Corpus Size (documents)')
            plt.ylabel('Memory Usage (MB)')
            plt.title('Memory Usage During Indexing')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(output_dir / 'indexing_performance.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # Querying performance chart
        if 'querying' in results:
            plt.figure(figsize=(12, 6))
            
            plt.subplot(1, 2, 1)
            data = results['querying']
            plt.plot(data['query_counts'], data['original_avg_times'], 'o-', label='Original BM25', linewidth=2)
            plt.plot(data['query_counts'], data['optimized_avg_times'], 's-', label='Optimized BM25', linewidth=2)
            plt.xlabel('Number of Queries')
            plt.ylabel('Average Query Time (seconds)')
            plt.title('Query Performance Comparison')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.yscale('log')
            
            plt.subplot(1, 2, 2)
            plt.plot(data['query_counts'], data['speedup_factors'], 'g^-', linewidth=2)
            plt.xlabel('Number of Queries')
            plt.ylabel('Speedup Factor')
            plt.title('Query Speedup Factor')
            plt.grid(True, alpha=0.3)
            plt.axhline(y=1, color='r', linestyle='--', alpha=0.5)
            
            plt.tight_layout()
            plt.savefig(output_dir / 'querying_performance.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # Memory usage chart
        if 'memory' in results:
            plt.figure(figsize=(10, 6))
            
            data = results['memory']
            x = np.arange(len(data['corpus_sizes']))
            width = 0.35
            
            plt.bar(x - width/2, data['original_peak_memory'], width, label='Original BM25', alpha=0.7)
            plt.bar(x + width/2, data['optimized_peak_memory'], width, label='Optimized BM25', alpha=0.7)
            
            plt.xlabel('Corpus Size')
            plt.ylabel('Peak Memory Usage (MB)')
            plt.title('Memory Usage Comparison')
            plt.xticks(x, [f"{size:,}" for size in data['corpus_sizes']])
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(output_dir / 'memory_usage.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def run_comprehensive_benchmark(self):
        """Run the complete benchmark suite."""
        logger.info("Starting comprehensive BM25 optimization benchmark")
        
        results = {}
        
        # Benchmark 1: Indexing performance with varying corpus sizes
        logger.info("=== Indexing Performance Benchmark ===")
        corpus_sizes = [1000, 5000, 10000, 25000, 50000]
        results['indexing'] = self.benchmark_indexing(corpus_sizes)
        
        # Benchmark 2: Query performance with fixed corpus, varying query count
        logger.info("=== Query Performance Benchmark ===")
        query_counts = [10, 50, 100, 500, 1000]
        results['querying'] = self.benchmark_querying(10000, query_counts)
        
        # Benchmark 3: Memory usage analysis
        logger.info("=== Memory Usage Benchmark ===")
        memory_corpus_sizes = [5000, 10000, 25000, 50000]
        results['memory'] = self.benchmark_memory_usage(memory_corpus_sizes)
        
        # Save results
        output_dir = Path('benchmark_results')
        output_dir.mkdir(exist_ok=True)
        
        with open(output_dir / 'benchmark_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        # Create visualizations
        self.create_visualizations(results, output_dir)
        
        # Generate summary report
        self.generate_summary_report(results, output_dir)
        
        logger.info(f"Benchmark completed! Results saved to {output_dir}")
        
        return results
    
    def generate_summary_report(self, results: Dict[str, Any], output_dir: Path):
        """Generate a comprehensive summary report."""
        
        report = []
        report.append("# BM25 Optimization Benchmark Report")
        report.append("")
        report.append("## Executive Summary")
        report.append("")
        
        # Calculate average improvements
        if 'indexing' in results:
            avg_indexing_speedup = np.mean(results['indexing']['speedup_factors'])
            report.append(f"- **Average Indexing Speedup**: {avg_indexing_speedup:.2f}x")
        
        if 'querying' in results:
            avg_query_speedup = np.mean(results['querying']['speedup_factors'])
            report.append(f"- **Average Query Speedup**: {avg_query_speedup:.2f}x")
        
        if 'memory' in results:
            avg_memory_saving = np.mean(results['memory']['memory_savings'])
            report.append(f"- **Average Memory Savings**: {avg_memory_saving:.1%}")
        
        report.append("")
        report.append("## Detailed Results")
        report.append("")
        
        # Indexing results
        if 'indexing' in results:
            report.append("### Indexing Performance")
            report.append("")
            data = results['indexing']
            
            df = pd.DataFrame({
                'Corpus Size': data['corpus_sizes'],
                'Original Time (s)': data['original_times'],
                'Optimized Time (s)': data['optimized_times'],
                'Speedup Factor': data['speedup_factors'],
                'Original Memory (MB)': data['original_memory'],
                'Optimized Memory (MB)': data['optimized_memory']
            })
            
            report.append(df.to_string(index=False))
            report.append("")
        
        # Query results  
        if 'querying' in results:
            report.append("### Query Performance")
            report.append("")
            data = results['querying']
            
            df = pd.DataFrame({
                'Query Count': data['query_counts'],
                'Original Time (s)': data['original_times'],
                'Optimized Time (s)': data['optimized_times'],
                'Speedup Factor': data['speedup_factors'],
                'Original Avg (s/query)': data['original_avg_times'],
                'Optimized Avg (s/query)': data['optimized_avg_times']
            })
            
            report.append(df.to_string(index=False))
            report.append("")
        
        # Memory results
        if 'memory' in results:
            report.append("### Memory Usage")
            report.append("")
            data = results['memory']
            
            df = pd.DataFrame({
                'Corpus Size': data['corpus_sizes'],
                'Original Peak (MB)': data['original_peak_memory'],
                'Optimized Peak (MB)': data['optimized_peak_memory'],
                'Memory Savings': [f"{saving:.1%}" for saving in data['memory_savings']]
            })
            
            report.append(df.to_string(index=False))
            report.append("")
        
        report.append("## Optimization Features")
        report.append("")
        report.append("The OptimizedBM25Model includes the following enhancements:")
        report.append("- Parallel corpus indexing using multiprocessing")
        report.append("- Vectorized operations with NumPy/SciPy")
        report.append("- LRU caching for query tokenization")
        report.append("- Memory-efficient sparse matrix storage")
        report.append("- Batch query processing")
        report.append("- Early termination and pruning strategies")
        report.append("- Optional GPU acceleration (CuPy)")
        report.append("- Performance statistics tracking")
        report.append("")
        
        # Save report
        with open(output_dir / 'benchmark_report.md', 'w') as f:
            f.write('\n'.join(report))


def main():
    """Run the benchmark."""
    benchmark = BM25PerformanceBenchmark()
    results = benchmark.run_comprehensive_benchmark()
    
    print("\n" + "="*50)
    print("BENCHMARK SUMMARY")
    print("="*50)
    
    if 'indexing' in results:
        avg_speedup = np.mean(results['indexing']['speedup_factors'])
        print(f"Average Indexing Speedup: {avg_speedup:.2f}x")
    
    if 'querying' in results:
        avg_speedup = np.mean(results['querying']['speedup_factors'])
        print(f"Average Query Speedup: {avg_speedup:.2f}x")
        
    if 'memory' in results:
        avg_saving = np.mean(results['memory']['memory_savings'])
        print(f"Average Memory Savings: {avg_saving:.1%}")
    
    print("\nDetailed results saved to benchmark_results/")


if __name__ == "__main__":
    main()
