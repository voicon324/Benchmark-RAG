"""
Integration tests for evaluation module.

Tests the evaluation module in realistic scenarios and validates against
reference implementations when possible.
"""

import pytest
import tempfile
import json
from pathlib import Path
import time

from newaibench.evaluation import Evaluator, BatchEvaluator, EvaluationConfig, quick_evaluate
from newaibench.evaluation.metrics import IRMetrics


class TestEvaluationIntegration:
    """Integration tests for the full evaluation pipeline."""
    
    def setup_method(self):
        """Set up realistic test data."""
        # Simulate MS MARCO-like data
        self.msmarco_qrels = {
            f'q{i}': {
                f'd{j}': 1 if j <= 3 else 0  # First 3 docs relevant
                for j in range(1, 21)  # 20 docs per query
            }
            for i in range(1, 51)  # 50 queries
        }
        
        # Simulate BM25-like results (decent but not perfect)
        self.bm25_results = {
            f'q{i}': {
                f'd{j}': max(0.1, 1.0 - (j-1)*0.05 + (0.1 if j <= 5 else 0))  # Boost top-5
                for j in range(1, 101)  # 100 results per query
            }
            for i in range(1, 51)
        }
        
        # Simulate DPR-like results (better at top ranks)
        self.dpr_results = {
            f'q{i}': {
                f'd{j}': max(0.1, 1.0 - (j-1)*0.03 + (0.2 if j <= 3 else 0))  # Strong boost top-3
                for j in range(1, 101)
            }
            for i in range(1, 51)
        }
        
        # BEIR-like configuration
        self.beir_config = EvaluationConfig(
            k_values=[1, 3, 5, 10, 100],
            relevance_threshold=1,
            include_per_query=True
        )
    
    def test_single_model_evaluation(self):
        """Test complete evaluation of a single model."""
        evaluator = Evaluator(self.beir_config)
        results = evaluator.evaluate(self.msmarco_qrels, self.bm25_results)
        
        # Validate structure
        assert 'metrics' in results
        assert 'per_query_metrics' in results
        assert 'dataset_stats' in results
        assert 'timestamp' in results
        assert 'evaluation_time' in results
        
        # Check all expected metrics are present
        expected_metrics = []
        for k in self.beir_config.k_values:
            for metric in ['ndcg', 'map', 'recall', 'precision', 'mrr', 'success']:
                expected_metrics.append(f'{metric}@{k}')
        
        for metric in expected_metrics:
            assert metric in results['metrics'], f"Missing metric: {metric}"
            assert 0.0 <= results['metrics'][metric] <= 1.0, f"Invalid value for {metric}"
        
        # Validate per-query metrics
        for metric in expected_metrics:
            assert metric in results['per_query_metrics']
            assert len(results['per_query_metrics'][metric]) <= len(self.msmarco_qrels)
    
    def test_batch_evaluation(self):
        """Test batch evaluation comparing multiple models."""
        batch_evaluator = BatchEvaluator(self.beir_config)
        
        model_results = {
            'BM25': self.bm25_results,
            'DPR': self.dpr_results
        }
        
        comparison = batch_evaluator.evaluate_multiple_models(
            self.msmarco_qrels, model_results
        )
        
        # Validate structure
        assert 'individual_results' in comparison
        assert 'comparative_analysis' in comparison
        
        # Check individual results
        individual = comparison['individual_results']
        assert 'BM25' in individual
        assert 'DPR' in individual
        
        for model_name, result in individual.items():
            assert 'model_name' in result
            assert result['model_name'] == model_name
            assert 'metrics' in result
        
        # Check comparative analysis
        analysis = comparison['comparative_analysis']
        assert 'num_models' in analysis
        assert analysis['num_models'] == 2
        assert 'best_model_by_metric' in analysis
        assert 'metric_rankings' in analysis
        
        # DPR should generally outperform BM25 on nDCG@10
        if 'ndcg@10' in analysis['best_model_by_metric']:
            # This is expected but not guaranteed due to synthetic data
            best_ndcg_model = analysis['best_model_by_metric']['ndcg@10']['model']
            print(f"Best nDCG@10 model: {best_ndcg_model}")
    
    def test_quick_evaluate_function(self):
        """Test the quick_evaluate convenience function."""
        metrics = quick_evaluate(
            self.msmarco_qrels, 
            self.bm25_results, 
            k_values=[1, 10, 100]
        )
        
        expected_metrics = []
        for k in [1, 10, 100]:
            for metric in ['ndcg', 'map', 'recall', 'precision', 'mrr', 'success']:
                expected_metrics.append(f'{metric}@{k}')
        
        for metric in expected_metrics:
            assert metric in metrics
            assert 0.0 <= metrics[metric] <= 1.0
    
    def test_result_saving(self):
        """Test saving evaluation results to file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "test_results.json"
            
            evaluator = Evaluator(self.beir_config)
            results = evaluator.evaluate(
                self.msmarco_qrels, 
                self.bm25_results, 
                save_path=save_path
            )
            
            # Check file was created
            assert save_path.exists()
            
            # Validate saved content
            with open(save_path, 'r') as f:
                saved_results = json.load(f)
            
            assert saved_results['metrics'] == results['metrics']
            assert 'timestamp' in saved_results
    
    def test_batch_saving(self):
        """Test saving batch evaluation results."""
        with tempfile.TemporaryDirectory() as temp_dir:
            batch_evaluator = BatchEvaluator(self.beir_config)
            
            model_results = {
                'BM25': self.bm25_results,
                'DPR': self.dpr_results
            }
            
            comparison = batch_evaluator.evaluate_multiple_models(
                self.msmarco_qrels, 
                model_results,
                save_dir=temp_dir
            )
            
            # Check individual model files
            bm25_file = Path(temp_dir) / "BM25_evaluation.json"
            dpr_file = Path(temp_dir) / "DPR_evaluation.json"
            comp_file = Path(temp_dir) / "comparative_analysis.json"
            
            assert bm25_file.exists()
            assert dpr_file.exists()
            assert comp_file.exists()
            
            # Validate comparative analysis file
            with open(comp_file, 'r') as f:
                comp_data = json.load(f)
            
            assert 'num_models' in comp_data
            assert comp_data['num_models'] == 2
    
    def test_performance_benchmarking(self):
        """Test evaluation performance on larger datasets."""
        # Create larger dataset
        large_qrels = {
            f'q{i}': {
                f'd{j}': 1 if j <= 5 else 0
                for j in range(1, 1001)  # 1000 docs per query
            }
            for i in range(1, 501)  # 500 queries
        }
        
        large_results = {
            f'q{i}': {
                f'd{j}': max(0.01, 1.0 - j*0.001)
                for j in range(1, 101)  # 100 results per query
            }
            for i in range(1, 501)
        }
        
        config = EvaluationConfig(
            k_values=[10, 100],
            include_per_query=False  # Faster without per-query
        )
        
        evaluator = Evaluator(config)
        
        start_time = time.time()
        results = evaluator.evaluate(large_qrels, large_results)
        end_time = time.time()
        
        execution_time = end_time - start_time
        print(f"Large dataset evaluation time: {execution_time:.2f}s")
        
        # Should complete in reasonable time (adjust threshold as needed)
        assert execution_time < 30.0, f"Evaluation too slow: {execution_time:.2f}s"
        
        # Validate results
        assert 'metrics' in results
        for metric in ['ndcg@10', 'map@10', 'recall@100']:
            assert metric in results['metrics']
    
    def test_realistic_ranking_scenario(self):
        """Test with realistic ranking scenario matching real IR systems."""
        # Create more realistic qrels and results
        realistic_qrels = {}
        realistic_results = {}
        
        for q in range(1, 21):  # 20 queries
            query_id = f'query_{q}'
            
            # Realistic relevance distribution
            relevant_docs = [f'doc_{q}_{i}' for i in range(1, 6)]  # 5 relevant
            irrelevant_docs = [f'doc_{q}_{i}' for i in range(6, 51)]  # 45 irrelevant
            
            realistic_qrels[query_id] = {}
            for doc in relevant_docs:
                realistic_qrels[query_id][doc] = 2 if doc.endswith('1') else 1  # Graded relevance
            for doc in irrelevant_docs:
                realistic_qrels[query_id][doc] = 0
            
            # Realistic system results (good system finds ~60% of relevant docs)
            all_docs = relevant_docs + irrelevant_docs
            import random
            random.seed(42)  # Reproducible
            random.shuffle(all_docs)
            
            realistic_results[query_id] = {}
            for i, doc in enumerate(all_docs[:20]):  # Return top 20
                # Relevant docs get higher scores on average but with noise
                base_score = 0.8 if doc in relevant_docs else 0.3
                noise = random.uniform(-0.2, 0.2)
                score = max(0.1, min(1.0, base_score + noise - i*0.02))
                realistic_results[query_id][doc] = score
        
        # Evaluate with BEIR-like configuration
        config = EvaluationConfig(
            k_values=[1, 3, 5, 10, 20],
            relevance_threshold=1
        )
        
        evaluator = Evaluator(config)
        results = evaluator.evaluate(realistic_qrels, realistic_results)
        
        # Sanity checks for realistic values
        assert 0.1 <= results['metrics']['ndcg@10'] <= 0.9
        assert 0.1 <= results['metrics']['map@10'] <= 0.8
        assert 0.1 <= results['metrics']['recall@20'] <= 1.0
        
        print(f"Realistic scenario results:")
        print(f"nDCG@10: {results['metrics']['ndcg@10']:.4f}")
        print(f"MAP@10: {results['metrics']['map@10']:.4f}")
        print(f"Recall@20: {results['metrics']['recall@20']:.4f}")
    
    def test_cross_validation_with_reference(self):
        """Test against manually calculated reference values."""
        # Simple case where we can manually verify results
        simple_qrels = {
            'q1': {'d1': 2, 'd2': 1, 'd3': 0}
        }
        
        perfect_results = {
            'q1': {'d1': 1.0, 'd2': 0.8, 'd3': 0.6}  # Perfect ranking
        }
        
        imperfect_results = {
            'q1': {'d2': 1.0, 'd1': 0.8, 'd3': 0.6}  # Swapped top 2
        }
        
        evaluator = Evaluator(EvaluationConfig(k_values=[2, 3]))
        
        # Perfect ranking should give nDCG = 1.0
        perfect_eval = evaluator.evaluate(simple_qrels, perfect_results)
        assert abs(perfect_eval['metrics']['ndcg@2'] - 1.0) < 1e-10
        assert abs(perfect_eval['metrics']['ndcg@3'] - 1.0) < 1e-10
        
        # Imperfect ranking should give lower nDCG
        imperfect_eval = evaluator.evaluate(simple_qrels, imperfect_results)
        assert imperfect_eval['metrics']['ndcg@2'] < perfect_eval['metrics']['ndcg@2']
        assert imperfect_eval['metrics']['ndcg@3'] < perfect_eval['metrics']['ndcg@3']
        
        # Manual calculation for imperfect ranking:
        # DCG@2 = 1 + 2/log2(3) = 1 + 1.262 = 2.262
        # IDCG@2 = 2 + 1/log2(3) = 2 + 1.262 = 3.262
        # nDCG@2 = 2.262 / 3.262 = 0.693
        import math
        expected_ndcg2 = (1 + 2/math.log2(3)) / (2 + 1/math.log2(3))
        assert abs(imperfect_eval['metrics']['ndcg@2'] - expected_ndcg2) < 1e-10
    
    def test_error_handling_and_recovery(self):
        """Test error handling in various scenarios."""
        evaluator = Evaluator(self.beir_config)
        
        # Empty results for some queries
        partial_results = {
            'q1': {'d1': 0.9, 'd2': 0.8},
            'q2': {},  # Empty results
            # q3 missing entirely
        }
        
        partial_qrels = {
            'q1': {'d1': 1, 'd2': 0},
            'q2': {'d1': 1, 'd2': 1},
            'q3': {'d1': 1}
        }
        
        # Should handle gracefully without crashing
        results = evaluator.evaluate(partial_qrels, partial_results)
        
        assert 'metrics' in results
        # Metrics should be computed for valid overlapping queries
        assert results['dataset_stats']['num_overlapping_queries'] >= 1
    
    def test_memory_efficiency(self):
        """Test memory usage doesn't grow excessively."""
        # This test is more observational - in production you'd use memory profiling
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create moderately large dataset
        large_qrels = {
            f'q{i}': {f'd{j}': 1 if j <= 10 else 0 for j in range(1, 1001)}
            for i in range(1, 101)
        }
        
        large_results = {
            f'q{i}': {f'd{j}': 1.0 - j/1000 for j in range(1, 101)}
            for i in range(1, 101)
        }
        
        evaluator = Evaluator(EvaluationConfig(
            k_values=[10, 100],
            include_per_query=False
        ))
        
        results = evaluator.evaluate(large_qrels, large_results)
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        print(f"Memory increase: {memory_increase:.1f} MB")
        
        # Should not use excessive memory (threshold may need adjustment)
        assert memory_increase < 500, f"Excessive memory usage: {memory_increase:.1f} MB"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
