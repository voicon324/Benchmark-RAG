"""
Performance and stress tests for evaluation module.

Tests the evaluation module under high load and validates performance characteristics.
"""

import pytest
import time
import psutil
import os
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import random

from newaibench.evaluation import Evaluator, EvaluationConfig, IRMetrics


class TestEvaluationPerformance:
    """Performance tests for evaluation metrics."""
    
    def setup_method(self):
        """Set up performance test data."""
        self.process = psutil.Process(os.getpid())
        
    def measure_memory_usage(self):
        """Get current memory usage in MB."""
        return self.process.memory_info().rss / 1024 / 1024
    
    def create_large_dataset(self, num_queries, num_docs_per_query, num_relevant_per_query=5):
        """Create large synthetic dataset for performance testing."""
        qrels = {}
        results = {}
        
        random.seed(42)  # Reproducible
        
        for q in range(num_queries):
            query_id = f'q{q}'
            
            # Create relevance judgments
            qrels[query_id] = {}
            for d in range(num_docs_per_query):
                doc_id = f'd{d}'
                # First num_relevant_per_query docs are relevant
                qrels[query_id][doc_id] = 1 if d < num_relevant_per_query else 0
            
            # Create results with some noise
            results[query_id] = {}
            for d in range(min(num_docs_per_query, 1000)):  # Limit results to 1000
                doc_id = f'd{d}'
                # Relevant docs get higher scores on average
                base_score = 0.8 if d < num_relevant_per_query else 0.3
                noise = random.uniform(-0.1, 0.1)
                score = max(0.01, base_score + noise - d * 0.001)
                results[query_id][doc_id] = score
        
        return qrels, results
    
    @pytest.mark.performance
    def test_large_scale_evaluation_speed(self):
        """Test evaluation speed on large datasets."""
        # Create large dataset
        num_queries = 1000
        num_docs = 5000
        
        print(f"\nCreating dataset: {num_queries} queries, {num_docs} docs each")
        start_time = time.time()
        qrels, results = self.create_large_dataset(num_queries, num_docs)
        dataset_time = time.time() - start_time
        print(f"Dataset creation: {dataset_time:.2f}s")
        
        # Configure for speed
        config = EvaluationConfig(
            k_values=[10, 100],
            include_per_query=False,  # Skip per-query for speed
            relevance_threshold=1
        )
        
        evaluator = Evaluator(config)
        
        # Measure evaluation time
        start_time = time.time()
        initial_memory = self.measure_memory_usage()
        
        evaluation_results = evaluator.evaluate(qrels, results)
        
        end_time = time.time()
        final_memory = self.measure_memory_usage()
        
        evaluation_time = end_time - start_time
        memory_increase = final_memory - initial_memory
        
        print(f"Evaluation time: {evaluation_time:.2f}s")
        print(f"Memory increase: {memory_increase:.1f} MB")
        print(f"Queries per second: {num_queries / evaluation_time:.1f}")
        
        # Performance thresholds (adjust based on hardware)
        assert evaluation_time < 60.0, f"Evaluation too slow: {evaluation_time:.2f}s"
        assert memory_increase < 1000, f"Excessive memory: {memory_increase:.1f} MB"
        
        # Validate results make sense
        assert 0.0 <= evaluation_results['metrics']['ndcg@10'] <= 1.0
        assert evaluation_results['evaluation_time'] > 0
    
    @pytest.mark.performance  
    def test_individual_metric_performance(self):
        """Test performance of individual metrics."""
        # Medium-sized dataset for detailed timing
        qrels, results = self.create_large_dataset(500, 1000)
        
        metrics_timing = {}
        
        # Test each metric individually
        for metric_name in ['ndcg', 'map', 'recall', 'precision', 'mrr', 'success']:
            print(f"\nTesting {metric_name} performance...")
            
            start_time = time.time()
            
            if metric_name == 'ndcg':
                result = IRMetrics.ndcg_at_k(qrels, results, 100)
            else:
                metric_func = getattr(IRMetrics, f'{metric_name}_at_k')
                result = metric_func(qrels, results, 100, relevance_threshold=1)
            
            end_time = time.time()
            timing = end_time - start_time
            metrics_timing[metric_name] = timing
            
            print(f"{metric_name}: {timing:.3f}s, result: {result:.4f}")
            
            # Individual metric should be reasonably fast
            assert timing < 30.0, f"{metric_name} too slow: {timing:.3f}s"
            assert 0.0 <= result <= 1.0, f"Invalid {metric_name} result: {result}"
        
        # nDCG and MAP are typically slower due to more complex computation
        print("\nMetric performance summary:")
        for metric, timing in sorted(metrics_timing.items(), key=lambda x: x[1]):
            print(f"{metric}: {timing:.3f}s")
    
    @pytest.mark.performance
    def test_memory_scalability(self):
        """Test memory usage scaling with dataset size."""
        dataset_sizes = [100, 500, 1000]
        memory_usage = []
        
        for size in dataset_sizes:
            print(f"\nTesting memory usage with {size} queries...")
            
            # Clear any existing data
            import gc
            gc.collect()
            
            initial_memory = self.measure_memory_usage()
            
            # Create dataset
            qrels, results = self.create_large_dataset(size, 1000)
            
            # Evaluate
            config = EvaluationConfig(k_values=[10], include_per_query=False)
            evaluator = Evaluator(config)
            evaluator.evaluate(qrels, results)
            
            final_memory = self.measure_memory_usage()
            memory_increase = final_memory - initial_memory
            memory_usage.append(memory_increase)
            
            print(f"Memory increase for {size} queries: {memory_increase:.1f} MB")
        
        # Memory should scale roughly linearly (with some overhead)
        # Check that memory doesn't explode quadratically
        ratio_1_to_2 = memory_usage[1] / memory_usage[0] if memory_usage[0] > 0 else float('inf')
        ratio_2_to_3 = memory_usage[2] / memory_usage[1] if memory_usage[1] > 0 else float('inf')
        
        print(f"Memory scaling ratios: {ratio_1_to_2:.1f}x, {ratio_2_to_3:.1f}x")
        
        # Should be roughly linear scaling (factor of ~5x for 5x data)
        assert ratio_1_to_2 < 10, f"Memory scaling too steep: {ratio_1_to_2:.1f}x"
        assert ratio_2_to_3 < 5, f"Memory scaling too steep: {ratio_2_to_3:.1f}x"
    
    @pytest.mark.performance
    def test_k_value_scaling(self):
        """Test performance scaling with different k values."""
        qrels, results = self.create_large_dataset(200, 2000)
        
        k_values_list = [
            [10],
            [10, 100], 
            [1, 5, 10, 20, 50, 100],
            [1, 3, 5, 10, 20, 50, 100, 200, 500, 1000]
        ]
        
        for k_values in k_values_list:
            print(f"\nTesting with k_values: {k_values}")
            
            config = EvaluationConfig(k_values=k_values, include_per_query=False)
            evaluator = Evaluator(config)
            
            start_time = time.time()
            results_eval = evaluator.evaluate(qrels, results)
            end_time = time.time()
            
            timing = end_time - start_time
            print(f"Time for {len(k_values)} k-values: {timing:.3f}s")
            
            # Should scale reasonably with number of k values
            assert timing < 60.0, f"Too slow for {len(k_values)} k-values: {timing:.3f}s"
            
            # All metrics should be present
            expected_metrics = len(k_values) * 6  # 6 metrics per k value
            actual_metrics = len(results_eval['metrics'])
            assert actual_metrics == expected_metrics
    
    @pytest.mark.performance
    def test_concurrent_evaluation(self):
        """Test thread safety and concurrent evaluation performance."""
        # Create shared dataset
        qrels, results = self.create_large_dataset(100, 500)
        
        def evaluate_model(model_id):
            """Evaluate model in separate thread."""
            # Add some noise to results to simulate different models
            noisy_results = {}
            for query_id, query_results in results.items():
                noisy_results[query_id] = {}
                for doc_id, score in query_results.items():
                    noise = random.uniform(-0.1, 0.1) * model_id * 0.01
                    noisy_results[query_id][doc_id] = max(0.01, score + noise)
            
            config = EvaluationConfig(k_values=[10, 100], include_per_query=False)
            evaluator = Evaluator(config)
            return evaluator.evaluate(qrels, noisy_results)
        
        # Test concurrent evaluation
        num_threads = 4
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(evaluate_model, i) for i in range(num_threads)]
            results_list = [future.result() for future in futures]
        
        end_time = time.time()
        concurrent_time = end_time - start_time
        
        print(f"Concurrent evaluation ({num_threads} threads): {concurrent_time:.3f}s")
        
        # Should complete in reasonable time
        assert concurrent_time < 120.0, f"Concurrent evaluation too slow: {concurrent_time:.3f}s"
        
        # All evaluations should succeed
        assert len(results_list) == num_threads
        for result in results_list:
            assert 'metrics' in result
            assert 'ndcg@10' in result['metrics']
    
    @pytest.mark.performance
    def test_edge_case_performance(self):
        """Test performance with edge cases that might be slow."""
        
        # Test 1: Many queries with few results each
        print("\nTest 1: Many queries, few results")
        qrels_sparse = {f'q{i}': {f'd{i}': 1} for i in range(1000)}
        results_sparse = {f'q{i}': {f'd{i}': 0.9} for i in range(1000)}
        
        start_time = time.time()
        evaluator = Evaluator(EvaluationConfig(k_values=[1, 10], include_per_query=False))
        eval_result = evaluator.evaluate(qrels_sparse, results_sparse)
        sparse_time = time.time() - start_time
        print(f"Sparse dataset time: {sparse_time:.3f}s")
        assert sparse_time < 30.0
        
        # Test 2: Few queries with many results each  
        print("\nTest 2: Few queries, many results")
        qrels_dense = {f'q{i}': {f'd{j}': 1 if j < 10 else 0 for j in range(5000)} for i in range(10)}
        results_dense = {f'q{i}': {f'd{j}': 1.0 - j/5000 for j in range(1000)} for i in range(10)}
        
        start_time = time.time() 
        eval_result = evaluator.evaluate(qrels_dense, results_dense)
        dense_time = time.time() - start_time
        print(f"Dense dataset time: {dense_time:.3f}s")
        assert dense_time < 30.0
        
        # Test 3: Very high k values
        print("\nTest 3: Very high k values")
        start_time = time.time()
        config_high_k = EvaluationConfig(k_values=[100, 500, 1000], include_per_query=False)
        evaluator_high_k = Evaluator(config_high_k)
        eval_result = evaluator_high_k.evaluate(qrels_dense, results_dense)
        high_k_time = time.time() - start_time
        print(f"High k values time: {high_k_time:.3f}s")
        assert high_k_time < 45.0
    
    @pytest.mark.performance 
    def test_numerical_stability_performance(self):
        """Test performance with numerically challenging cases."""
        
        # Create dataset with very small differences in scores
        num_queries = 100
        qrels = {}
        results = {}
        
        for q in range(num_queries):
            query_id = f'q{q}'
            qrels[query_id] = {f'd{i}': 1 if i < 5 else 0 for i in range(1000)}
            
            # Results with very small differences (numerical precision challenge)
            results[query_id] = {}
            base_score = 0.5
            for i in range(100):
                doc_id = f'd{i}'
                # Very small decrements that might cause precision issues
                score = base_score - i * 1e-10
                results[query_id][doc_id] = score
        
        start_time = time.time()
        config = EvaluationConfig(k_values=[10, 100], include_per_query=False)
        evaluator = Evaluator(config)
        eval_result = evaluator.evaluate(qrels, results)
        end_time = time.time()
        
        numerical_time = end_time - start_time
        print(f"Numerical precision test time: {numerical_time:.3f}s")
        
        # Should handle numerical precision without significant slowdown
        assert numerical_time < 30.0
        
        # Results should be valid (not NaN or infinite)
        for metric_value in eval_result['metrics'].values():
            assert not np.isnan(metric_value), "NaN metric value detected"
            assert not np.isinf(metric_value), "Infinite metric value detected"
            assert 0.0 <= metric_value <= 1.0, f"Invalid metric value: {metric_value}"


class TestStressTesting:
    """Stress tests for extreme conditions."""
    
    @pytest.mark.stress
    def test_extreme_dataset_sizes(self):
        """Test with extremely large datasets."""
        # This test is expensive and might be skipped in regular CI
        
        # Very large dataset
        num_queries = 5000
        num_docs = 10000
        
        print(f"\nStress test: {num_queries} queries, {num_docs} docs")
        
        # Generate in chunks to avoid memory issues during creation
        qrels = {}
        results = {}
        
        random.seed(42)
        chunk_size = 500
        
        for chunk_start in range(0, num_queries, chunk_size):
            chunk_end = min(chunk_start + chunk_size, num_queries)
            print(f"Generating queries {chunk_start} to {chunk_end}...")
            
            for q in range(chunk_start, chunk_end):
                query_id = f'q{q}'
                
                # Only store a subset of documents to manage memory
                qrels[query_id] = {f'd{i}': 1 if i < 10 else 0 for i in range(min(1000, num_docs))}
                results[query_id] = {f'd{i}': 1.0 - i/500 for i in range(min(200, num_docs))}
        
        # Evaluate with minimal configuration
        config = EvaluationConfig(k_values=[10], include_per_query=False)
        evaluator = Evaluator(config)
        
        start_time = time.time()
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        eval_result = evaluator.evaluate(qrels, results)
        
        end_time = time.time()
        final_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        stress_time = end_time - start_time
        memory_increase = final_memory - initial_memory
        
        print(f"Stress test time: {stress_time:.2f}s")
        print(f"Memory increase: {memory_increase:.1f} MB")
        print(f"Queries per second: {num_queries / stress_time:.1f}")
        
        # Should complete even under stress (generous thresholds)
        assert stress_time < 300.0, f"Stress test too slow: {stress_time:.2f}s"
        assert memory_increase < 2000, f"Excessive memory in stress test: {memory_increase:.1f} MB"
    
    @pytest.mark.stress
    def test_memory_pressure_handling(self):
        """Test behavior under memory pressure."""
        # Gradually increase dataset size and monitor memory
        max_safe_memory = 1000  # MB threshold
        
        query_counts = [100, 500, 1000, 2000, 5000]
        
        for num_queries in query_counts:
            print(f"\nTesting {num_queries} queries...")
            
            # Check current memory
            current_memory = psutil.Process().memory_info().rss / 1024 / 1024
            if current_memory > max_safe_memory:
                print(f"Stopping test due to memory pressure: {current_memory:.1f} MB")
                break
            
            # Create dataset
            qrels = {f'q{i}': {f'd{j}': 1 if j < 5 else 0 for j in range(100)} 
                    for i in range(num_queries)}
            results = {f'q{i}': {f'd{j}': 1.0 - j/100 for j in range(50)} 
                      for i in range(num_queries)}
            
            # Evaluate
            try:
                config = EvaluationConfig(k_values=[10], include_per_query=False)
                evaluator = Evaluator(config)
                eval_result = evaluator.evaluate(qrels, results)
                
                print(f"Success with {num_queries} queries")
                
            except MemoryError:
                print(f"Memory limit reached at {num_queries} queries")
                break
            except Exception as e:
                print(f"Other error at {num_queries} queries: {str(e)}")
                break


if __name__ == "__main__":
    # Run performance tests
    pytest.main([__file__, "-v", "-m", "performance"])
