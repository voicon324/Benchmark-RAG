"""
Optimization utilities for large-scale Information Retrieval evaluation.

This module provides memory-efficient and performance-optimized implementations
for evaluating very large datasets that don't fit in memory.
"""

import numpy as np
import json
from typing import Dict, List, Any, Iterator, Optional, Tuple
from pathlib import Path
import logging
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from functools import partial
import gc

from .metrics import IRMetrics, EvaluationConfig, MetricResult

logger = logging.getLogger(__name__)


@dataclass
class OptimizationConfig:
    """Configuration for large-scale evaluation optimization."""
    
    # Memory management
    max_memory_mb: int = 4096  # Maximum memory usage in MB
    batch_size: int = 1000     # Number of queries per batch
    chunk_size: int = 100      # Chunk size for streaming
    
    # Parallel processing
    num_workers: int = mp.cpu_count()
    use_multiprocessing: bool = True
    
    # Storage optimization
    use_memory_mapping: bool = True
    cache_intermediate: bool = False
    compress_results: bool = True
    
    # Performance tuning
    early_termination_k: int = 1000  # Stop processing after k docs per query
    approximate_computation: bool = False
    precision_threshold: float = 1e-6


class StreamingEvaluator:
    """Memory-efficient evaluator for large-scale datasets."""
    
    def __init__(self, config: EvaluationConfig, opt_config: OptimizationConfig = None):
        self.config = config
        self.opt_config = opt_config or OptimizationConfig()
        self.logger = logging.getLogger(__name__)
        
    def evaluate_streaming(self, 
                         qrels_file: Path, 
                         results_file: Path, 
                         output_file: Optional[Path] = None) -> Dict[str, float]:
        """
        Evaluate large datasets using streaming to minimize memory usage.
        
        Args:
            qrels_file: Path to qrels file (TREC format)
            results_file: Path to results file (TREC format)
            output_file: Optional path to save results
            
        Returns:
            Dictionary with aggregated metrics
        """
        self.logger.info(f"Starting streaming evaluation")
        self.logger.info(f"Memory limit: {self.opt_config.max_memory_mb}MB")
        self.logger.info(f"Batch size: {self.opt_config.batch_size}")
        
        # Load qrels in chunks
        qrels_stream = self._stream_qrels(qrels_file)
        results_stream = self._stream_results(results_file)
        
        # Process in batches
        all_metrics = []
        batch_count = 0
        
        for qrels_batch, results_batch in self._batch_iterator(qrels_stream, results_stream):
            batch_count += 1
            self.logger.info(f"Processing batch {batch_count} with {len(qrels_batch)} queries")
            
            # Evaluate batch
            batch_metrics = self._evaluate_batch(qrels_batch, results_batch)
            all_metrics.append(batch_metrics)
            
            # Memory cleanup
            gc.collect()
            
            # Check memory usage
            if self._get_memory_usage() > self.opt_config.max_memory_mb:
                self.logger.warning("Memory usage exceeded limit, forcing garbage collection")
                gc.collect()
        
        # Aggregate results across batches
        final_metrics = self._aggregate_batch_metrics(all_metrics)
        
        # Save results if requested
        if output_file:
            self._save_streaming_results(final_metrics, output_file)
        
        self.logger.info("Streaming evaluation completed")
        return final_metrics
    
    def _stream_qrels(self, qrels_file: Path) -> Iterator[Dict[str, Dict[str, int]]]:
        """Stream qrels file in TREC format."""
        current_batch = {}
        batch_size = 0
        
        with open(qrels_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 4:
                    qid, _, docid, rel = parts[0], parts[1], parts[2], int(parts[3])
                    
                    if qid not in current_batch:
                        current_batch[qid] = {}
                        batch_size += 1
                    
                    current_batch[qid][docid] = rel
                    
                    # Yield batch when size limit reached
                    if batch_size >= self.opt_config.batch_size:
                        yield current_batch
                        current_batch = {}
                        batch_size = 0
        
        # Yield remaining batch
        if current_batch:
            yield current_batch
    
    def _stream_results(self, results_file: Path) -> Iterator[Dict[str, List[Tuple[str, float]]]]:
        """Stream results file in TREC format."""
        current_batch = {}
        current_qid = None
        
        with open(results_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 6:
                    qid, _, docid, rank, score, _ = parts
                    score = float(score)
                    
                    if qid != current_qid:
                        # New query - yield previous batch if exists
                        if current_batch and len(current_batch) >= self.opt_config.batch_size:
                            yield current_batch
                            current_batch = {}
                        
                        current_qid = qid
                    
                    if qid not in current_batch:
                        current_batch[qid] = []
                    
                    current_batch[qid].append((docid, score))
                    
                    # Apply early termination
                    if len(current_batch[qid]) >= self.opt_config.early_termination_k:
                        continue  # Skip remaining docs for this query
        
        # Yield remaining batch
        if current_batch:
            yield current_batch
    
    def _batch_iterator(self, qrels_stream, results_stream):
        """Coordinate streaming of qrels and results."""
        # This is a simplified implementation
        # In practice, you'd need more sophisticated coordination
        qrels_batches = list(qrels_stream)
        results_batches = list(results_stream)
        
        # Align batches by query IDs
        for qrels_batch in qrels_batches:
            # Find matching results batch
            matching_results = {}
            for results_batch in results_batches:
                for qid in qrels_batch:
                    if qid in results_batch:
                        matching_results[qid] = results_batch[qid]
            
            yield qrels_batch, matching_results
    
    def _evaluate_batch(self, qrels_batch: Dict, results_batch: Dict) -> Dict[str, List[float]]:
        """Evaluate a single batch and return per-query metrics."""
        batch_metrics = {
            f'ndcg@{k}': [] for k in self.config.k_values
        }
        batch_metrics.update({
            f'map@{k}': [] for k in self.config.k_values
        })
        batch_metrics.update({
            f'recall@{k}': [] for k in self.config.k_values
        })
        batch_metrics.update({
            f'precision@{k}': [] for k in self.config.k_values
        })
        batch_metrics.update({
            f'mrr@{k}': [] for k in self.config.k_values
        })
        batch_metrics.update({
            f'success@{k}': [] for k in self.config.k_values
        })
        
        for qid in qrels_batch:
            if qid in results_batch:
                qrels_single = {qid: qrels_batch[qid]}
                results_single = {qid: results_batch[qid]}
                
                # Compute metrics for this query
                for k in self.config.k_values:
                    batch_metrics[f'ndcg@{k}'].append(
                        IRMetrics.ndcg_at_k(qrels_single, results_single, k)
                    )
                    batch_metrics[f'map@{k}'].append(
                        IRMetrics.map_at_k(qrels_single, results_single, k)
                    )
                    batch_metrics[f'recall@{k}'].append(
                        IRMetrics.recall_at_k(qrels_single, results_single, k)
                    )
                    batch_metrics[f'precision@{k}'].append(
                        IRMetrics.precision_at_k(qrels_single, results_single, k)
                    )
                    batch_metrics[f'mrr@{k}'].append(
                        IRMetrics.mrr_at_k(qrels_single, results_single, k)
                    )
                    batch_metrics[f'success@{k}'].append(
                        IRMetrics.success_at_k(qrels_single, results_single, k)
                    )
        
        return batch_metrics
    
    def _aggregate_batch_metrics(self, all_metrics: List[Dict]) -> Dict[str, float]:
        """Aggregate metrics across all batches."""
        final_metrics = {}
        
        # Get all metric names from first batch
        if all_metrics:
            for metric_name in all_metrics[0]:
                # Collect all values for this metric across batches
                all_values = []
                for batch_metrics in all_metrics:
                    all_values.extend(batch_metrics[metric_name])
                
                # Compute mean
                if all_values:
                    final_metrics[metric_name] = np.mean(all_values)
                else:
                    final_metrics[metric_name] = 0.0
        
        return final_metrics
    
    def _get_memory_usage(self) -> int:
        """Get current memory usage in MB."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss // (1024 * 1024)
        except ImportError:
            return 0  # Can't measure without psutil
    
    def _save_streaming_results(self, metrics: Dict, output_file: Path):
        """Save streaming evaluation results."""
        result_data = {
            'metrics': metrics,
            'config': {
                'evaluation': self.config.__dict__,
                'optimization': self.opt_config.__dict__
            },
            'timestamp': str(pd.Timestamp.now()) if 'pd' in globals() else 'unknown'
        }
        
        if self.opt_config.compress_results:
            import gzip
            with gzip.open(f"{output_file}.gz", 'wt') as f:
                json.dump(result_data, f, indent=2)
        else:
            with open(output_file, 'w') as f:
                json.dump(result_data, f, indent=2)


class ParallelEvaluator:
    """Multi-process evaluator for CPU-intensive evaluation tasks."""
    
    def __init__(self, config: EvaluationConfig, opt_config: OptimizationConfig = None):
        self.config = config
        self.opt_config = opt_config or OptimizationConfig()
        
    def evaluate_parallel(self, qrels: Dict, results: Dict) -> Dict[str, float]:
        """
        Evaluate using multiple processes for improved performance.
        
        Args:
            qrels: Query relevance judgments
            results: Model retrieval results
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Split queries into chunks for parallel processing
        query_ids = list(qrels.keys())
        chunk_size = max(1, len(query_ids) // self.opt_config.num_workers)
        query_chunks = [
            query_ids[i:i + chunk_size] 
            for i in range(0, len(query_ids), chunk_size)
        ]
        
        # Create partial function for worker processes
        worker_func = partial(
            self._evaluate_chunk,
            qrels=qrels,
            results=results,
            config=self.config
        )
        
        # Use multiprocessing for CPU-bound tasks
        if self.opt_config.use_multiprocessing:
            with ProcessPoolExecutor(max_workers=self.opt_config.num_workers) as executor:
                chunk_results = list(executor.map(worker_func, query_chunks))
        else:
            # Use threading for I/O-bound tasks
            with ThreadPoolExecutor(max_workers=self.opt_config.num_workers) as executor:
                chunk_results = list(executor.map(worker_func, query_chunks))
        
        # Aggregate results from all chunks
        return self._aggregate_parallel_results(chunk_results)
    
    @staticmethod
    def _evaluate_chunk(query_chunk: List[str], 
                       qrels: Dict, 
                       results: Dict, 
                       config: EvaluationConfig) -> Dict[str, List[float]]:
        """Evaluate a chunk of queries in a worker process."""
        chunk_metrics = {}
        
        for qid in query_chunk:
            if qid in results:
                qrels_single = {qid: qrels[qid]}
                results_single = {qid: results[qid]}
                
                # Compute metrics for this query
                for k in config.k_values:
                    metric_name = f'ndcg@{k}'
                    if metric_name not in chunk_metrics:
                        chunk_metrics[metric_name] = []
                    chunk_metrics[metric_name].append(
                        IRMetrics.ndcg_at_k(qrels_single, results_single, k)
                    )
                    
                    # Add other metrics as needed
                    for metric_func, metric_prefix in [
                        (IRMetrics.map_at_k, 'map'),
                        (IRMetrics.recall_at_k, 'recall'),
                        (IRMetrics.precision_at_k, 'precision'),
                        (IRMetrics.mrr_at_k, 'mrr'),
                        (IRMetrics.success_at_k, 'success')
                    ]:
                        metric_name = f'{metric_prefix}@{k}'
                        if metric_name not in chunk_metrics:
                            chunk_metrics[metric_name] = []
                        chunk_metrics[metric_name].append(
                            metric_func(qrels_single, results_single, k)
                        )
        
        return chunk_metrics
    
    def _aggregate_parallel_results(self, chunk_results: List[Dict]) -> Dict[str, float]:
        """Aggregate results from parallel chunks."""
        final_metrics = {}
        
        # Collect all metric names
        all_metric_names = set()
        for chunk_result in chunk_results:
            all_metric_names.update(chunk_result.keys())
        
        # Aggregate each metric
        for metric_name in all_metric_names:
            all_values = []
            for chunk_result in chunk_results:
                if metric_name in chunk_result:
                    all_values.extend(chunk_result[metric_name])
            
            if all_values:
                final_metrics[metric_name] = np.mean(all_values)
            else:
                final_metrics[metric_name] = 0.0
        
        return final_metrics


def optimize_for_large_scale():
    """
    Demonstration of optimization techniques for large-scale evaluation.
    """
    print("Large-scale Optimization Demo")
    print("=" * 40)
    
    # Create optimization config
    opt_config = OptimizationConfig(
        max_memory_mb=2048,
        batch_size=500,
        num_workers=mp.cpu_count(),
        use_multiprocessing=True,
        early_termination_k=100
    )
    
    eval_config = EvaluationConfig(k_values=[10, 100])
    
    # Demo parallel evaluation
    print("Testing parallel evaluation...")
    
    # Create synthetic large dataset
    np.random.seed(42)
    num_queries = 10000
    docs_per_query = 50
    
    qrels = {}
    results = {}
    
    for i in range(num_queries):
        qid = f"q{i}"
        qrels[qid] = {f"doc{j}": np.random.randint(0, 4) for j in range(docs_per_query)}
        
        scores = np.random.random(docs_per_query)
        results[qid] = [(f"doc{j}", scores[j]) for j in np.argsort(scores)[::-1]]
    
    # Time parallel vs sequential evaluation
    import time
    
    # Sequential evaluation
    start_time = time.time()
    sequential_metrics = {}
    for k in eval_config.k_values:
        sequential_metrics[f'ndcg@{k}'] = IRMetrics.ndcg_at_k(qrels, results, k)
    sequential_time = time.time() - start_time
    
    # Parallel evaluation  
    parallel_evaluator = ParallelEvaluator(eval_config, opt_config)
    start_time = time.time()
    parallel_metrics = parallel_evaluator.evaluate_parallel(qrels, results)
    parallel_time = time.time() - start_time
    
    print(f"Sequential time: {sequential_time:.2f}s")
    print(f"Parallel time: {parallel_time:.2f}s")
    print(f"Speedup: {sequential_time/parallel_time:.2f}x")
    
    # Verify results are similar
    for metric in sequential_metrics:
        if metric in parallel_metrics:
            diff = abs(sequential_metrics[metric] - parallel_metrics[metric])
            print(f"{metric}: diff = {diff:.6f}")


if __name__ == "__main__":
    optimize_for_large_scale()
