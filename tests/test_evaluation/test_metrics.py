"""
Comprehensive unit tests for NewAIBench IR evaluation metrics.

This test suite validates the mathematical correctness of all implemented
IR metrics against hand-calculated examples and edge cases.
"""

import pytest
import math
import numpy as np
from typing import Dict, List

from newaibench.evaluation.metrics import IRMetrics, EvaluationConfig
from newaibench.evaluation.evaluator import Evaluator


class TestIRMetrics:
    """Test suite for IRMetrics class."""
    
    def setup_method(self):
        """Set up test data for each test method."""
        # Simple test case with known results
        self.simple_qrels = {
            'q1': {'d1': 3, 'd2': 2, 'd3': 1, 'd4': 0},
            'q2': {'d1': 1, 'd2': 0, 'd3': 2, 'd4': 3}
        }
        
        self.simple_results = {
            'q1': {'d1': 0.9, 'd2': 0.8, 'd3': 0.7, 'd4': 0.6},  # Perfect ranking
            'q2': {'d4': 0.9, 'd3': 0.8, 'd1': 0.7, 'd2': 0.6}   # Perfect ranking
        }
        
        # Binary relevance test case
        self.binary_qrels = {
            'q1': {'d1': 1, 'd2': 1, 'd3': 0, 'd4': 0},
            'q2': {'d1': 0, 'd2': 1, 'd3': 1, 'd4': 0}
        }
        
        self.binary_results = {
            'q1': {'d1': 0.9, 'd3': 0.8, 'd2': 0.7, 'd4': 0.6},  # Imperfect ranking
            'q2': {'d2': 0.9, 'd3': 0.8, 'd1': 0.7, 'd4': 0.6}   # Good ranking
        }
        
        # Edge case: no relevant documents
        self.no_relevant_qrels = {
            'q1': {'d1': 0, 'd2': 0, 'd3': 0}
        }
        
        self.no_relevant_results = {
            'q1': {'d1': 0.9, 'd2': 0.8, 'd3': 0.7}
        }
        
        # Edge case: empty results
        self.empty_results = {
            'q1': {}
        }
    
    def test_dcg_computation(self):
        """Test DCG computation with hand-calculated example."""
        # Test case: relevance scores [3, 2, 1] for positions 1, 2, 3
        relevance_scores = [3, 2, 1]
        
        # Hand-calculated DCG@3: 3 + 2/log2(3) + 1/log2(4) = 3 + 1.262 + 0.5 = 4.762
        expected_dcg = 3 + 2/math.log2(3) + 1/math.log2(4)
        actual_dcg = IRMetrics._dcg_at_k(relevance_scores, 3)
        
        assert abs(actual_dcg - expected_dcg) < 1e-10, f"Expected {expected_dcg}, got {actual_dcg}"
    
    def test_dcg_edge_cases(self):
        """Test DCG computation edge cases."""
        # Empty list
        assert IRMetrics._dcg_at_k([], 5) == 0.0
        
        # k = 0
        assert IRMetrics._dcg_at_k([3, 2, 1], 0) == 0.0
        
        # k > length
        assert IRMetrics._dcg_at_k([3, 2], 5) == 3 + 2/math.log2(3)
        
        # All zeros
        assert IRMetrics._dcg_at_k([0, 0, 0], 3) == 0.0
    
    def test_ideal_dcg_computation(self):
        """Test IDCG computation."""
        relevance_scores = [1, 3, 2, 0]  # Unsorted
        
        # IDCG should sort to [3, 2, 1, 0] and compute DCG
        expected_idcg = 3 + 2/math.log2(3) + 1/math.log2(4)
        actual_idcg = IRMetrics._ideal_dcg_at_k(relevance_scores, 3)
        
        assert abs(actual_idcg - expected_idcg) < 1e-10
    
    def test_ndcg_perfect_ranking(self):
        """Test nDCG with perfect ranking (should be 1.0)."""
        qrels = {'q1': {'d1': 3, 'd2': 2, 'd3': 1}}
        results = {'q1': {'d1': 0.9, 'd2': 0.8, 'd3': 0.7}}  # Perfect order
        
        ndcg = IRMetrics.ndcg_at_k(qrels, results, 3)
        assert abs(ndcg - 1.0) < 1e-10, f"Perfect ranking should give nDCG=1.0, got {ndcg}"
    
    def test_ndcg_worst_ranking(self):
        """Test nDCG with worst possible ranking."""
        qrels = {'q1': {'d1': 3, 'd2': 2, 'd3': 1}}
        results = {'q1': {'d3': 0.9, 'd2': 0.8, 'd1': 0.7}}  # Reverse order
        
        ndcg = IRMetrics.ndcg_at_k(qrels, results, 3)
        
        # Hand calculate: DCG = 1 + 2/log2(3) + 3/log2(4), IDCG = 3 + 2/log2(3) + 1/log2(4)
        dcg = 1 + 2/math.log2(3) + 3/math.log2(4)
        idcg = 3 + 2/math.log2(3) + 1/math.log2(4)
        expected_ndcg = dcg / idcg
        
        assert abs(ndcg - expected_ndcg) < 1e-10
    
    def test_ndcg_no_relevant_docs(self):
        """Test nDCG when query has no relevant documents."""
        ndcg = IRMetrics.ndcg_at_k(self.no_relevant_qrels, self.no_relevant_results, 3)
        assert ndcg == 0.0
    
    def test_ndcg_per_query(self):
        """Test nDCG with per-query breakdown."""
        ndcg, per_query = IRMetrics.ndcg_at_k(self.simple_qrels, self.simple_results, 3, per_query=True)
        
        assert isinstance(per_query, dict)
        assert 'q1' in per_query
        assert 'q2' in per_query
        assert abs(per_query['q1'] - 1.0) < 1e-10  # Perfect ranking
        assert abs(per_query['q2'] - 1.0) < 1e-10  # Perfect ranking
    
    def test_precision_at_k_perfect(self):
        """Test Precision@k with perfect results."""
        # All top-k documents are relevant
        qrels = {'q1': {'d1': 1, 'd2': 1, 'd3': 0, 'd4': 0}}
        results = {'q1': {'d1': 0.9, 'd2': 0.8, 'd3': 0.7, 'd4': 0.6}}
        
        precision = IRMetrics.precision_at_k(qrels, results, 2)
        assert precision == 1.0  # 2 relevant out of top 2
    
    def test_precision_at_k_mixed(self):
        """Test Precision@k with mixed results."""
        precision = IRMetrics.precision_at_k(self.binary_qrels, self.binary_results, 2)
        
        # q1: top-2 are [d1, d3], relevance [1, 0] -> precision = 1/2 = 0.5
        # q2: top-2 are [d2, d3], relevance [1, 1] -> precision = 2/2 = 1.0
        # Average: (0.5 + 1.0) / 2 = 0.75
        expected_precision = 0.75
        assert abs(precision - expected_precision) < 1e-10
    
    def test_precision_at_k_zero_results(self):
        """Test Precision@k when no results returned."""
        precision = IRMetrics.precision_at_k(self.binary_qrels, self.empty_results, 2)
        # q1 has empty results -> precision = 0
        # q2 not in results -> precision = 0  
        # Average across all queries: 0
        assert precision == 0.0
    
    def test_recall_at_k_perfect(self):
        """Test Recall@k with perfect recall."""
        qrels = {'q1': {'d1': 1, 'd2': 1, 'd3': 0}}
        results = {'q1': {'d1': 0.9, 'd2': 0.8, 'd3': 0.7}}
        
        recall = IRMetrics.recall_at_k(qrels, results, 2)
        assert recall == 1.0  # Found 2 out of 2 relevant documents
    
    def test_recall_at_k_partial(self):
        """Test Recall@k with partial recall."""
        recall = IRMetrics.recall_at_k(self.binary_qrels, self.binary_results, 2)
        
        # q1: 2 relevant docs total, found 1 in top-2 -> recall = 1/2 = 0.5
        # q2: 2 relevant docs total, found 2 in top-2 -> recall = 2/2 = 1.0
        # Average: (0.5 + 1.0) / 2 = 0.75
        expected_recall = 0.75
        assert abs(recall - expected_recall) < 1e-10
    
    def test_recall_at_k_no_relevant(self):
        """Test Recall@k when query has no relevant documents."""
        recall = IRMetrics.recall_at_k(self.no_relevant_qrels, self.no_relevant_results, 2)
        assert recall == 0.0
    
    def test_average_precision_single_query(self):
        """Test Average Precision computation for single query."""
        qrels = {'d1': 1, 'd2': 0, 'd3': 1, 'd4': 0}
        results = {'d1': 0.9, 'd3': 0.8, 'd2': 0.7, 'd4': 0.6}
        
        # Ranking: [d1(1), d3(1), d2(0), d4(0)]
        # Precision at positions where relevant doc found:
        # P@1 = 1/1 = 1.0 (d1 is relevant)
        # P@2 = 2/2 = 1.0 (d3 is relevant)
        # AP = (1.0 + 1.0) / min(4, 2) = 2.0 / 2 = 1.0
        
        ap = IRMetrics.average_precision_at_k(qrels, results, 4)
        assert abs(ap - 1.0) < 1e-10
    
    def test_average_precision_imperfect(self):
        """Test Average Precision with imperfect ranking."""
        qrels = {'d1': 1, 'd2': 1, 'd3': 0, 'd4': 0}
        results = {'d1': 0.9, 'd3': 0.8, 'd2': 0.7, 'd4': 0.6}
        
        # Ranking: [d1(1), d3(0), d2(1), d4(0)]
        # P@1 = 1/1 = 1.0, P@3 = 2/3 = 0.667
        # AP = (1.0 + 0.667) / min(4, 2) = 1.667 / 2 = 0.833
        
        ap = IRMetrics.average_precision_at_k(qrels, results, 4)
        expected_ap = (1.0 + 2.0/3.0) / 2.0
        assert abs(ap - expected_ap) < 1e-10
    
    def test_map_at_k(self):
        """Test Mean Average Precision."""
        map_score = IRMetrics.map_at_k(self.binary_qrels, self.binary_results, 4)
        
        # Calculate AP for each query manually
        # q1: ranking [d1(1), d3(0), d2(1), d4(0)] -> AP = (1.0 + 2/3) / 2 = 0.833
        # q2: ranking [d2(1), d3(1), d1(0), d4(0)] -> AP = (1.0 + 1.0) / 2 = 1.0
        # MAP = (0.833 + 1.0) / 2 = 0.917
        
        expected_map = ((1.0 + 2.0/3.0) / 2.0 + 1.0) / 2.0
        assert abs(map_score - expected_map) < 1e-10
    
    def test_mrr_at_k_perfect(self):
        """Test MRR with first relevant at rank 1."""
        qrels = {'q1': {'d1': 1, 'd2': 0}}
        results = {'q1': {'d1': 0.9, 'd2': 0.8}}
        
        mrr = IRMetrics.mrr_at_k(qrels, results, 2)
        assert mrr == 1.0  # First relevant at rank 1 -> RR = 1/1 = 1.0
    
    def test_mrr_at_k_mixed(self):
        """Test MRR with mixed first relevant positions."""
        mrr = IRMetrics.mrr_at_k(self.binary_qrels, self.binary_results, 4)
        
        # q1: first relevant (d1) at rank 1 -> RR = 1/1 = 1.0
        # q2: first relevant (d2) at rank 1 -> RR = 1/1 = 1.0
        # MRR = (1.0 + 1.0) / 2 = 1.0
        
        assert abs(mrr - 1.0) < 1e-10
    
    def test_mrr_at_k_no_relevant_found(self):
        """Test MRR when no relevant documents found in top-k."""
        qrels = {'q1': {'d1': 1, 'd2': 1}}
        results = {'q1': {'d3': 0.9, 'd4': 0.8}}  # No relevant docs returned
        
        mrr = IRMetrics.mrr_at_k(qrels, results, 2)
        assert mrr == 0.0
    
    def test_success_at_k_all_success(self):
        """Test Success@k when all queries have relevant documents."""
        success = IRMetrics.success_at_k(self.binary_qrels, self.binary_results, 2)
        
        # Both q1 and q2 have at least one relevant doc in top-2
        assert success == 1.0
    
    def test_success_at_k_partial(self):
        """Test Success@k with partial success."""
        qrels = {'q1': {'d1': 1}, 'q2': {'d1': 1}}
        results = {'q1': {'d1': 0.9}, 'q2': {'d2': 0.9}}  # q2 has no relevant docs
        
        success = IRMetrics.success_at_k(qrels, results, 1)
        assert success == 0.5  # 1 out of 2 queries successful
    
    def test_relevance_threshold(self):
        """Test metrics with different relevance thresholds."""
        qrels = {'q1': {'d1': 3, 'd2': 2, 'd3': 1, 'd4': 0}}
        results = {'q1': {'d1': 0.9, 'd2': 0.8, 'd3': 0.7, 'd4': 0.6}}
        
        # With threshold=1, all docs with rel >= 1 are relevant (d1, d2, d3)
        recall_thresh1 = IRMetrics.recall_at_k(qrels, results, 3, relevance_threshold=1)
        assert recall_thresh1 == 1.0
        
        # With threshold=2, only docs with rel >= 2 are relevant (d1, d2)
        recall_thresh2 = IRMetrics.recall_at_k(qrels, results, 3, relevance_threshold=2)
        assert recall_thresh2 == 1.0  # Found 2 out of 2 relevant
        
        # With threshold=3, only docs with rel >= 3 are relevant (d1)
        recall_thresh3 = IRMetrics.recall_at_k(qrels, results, 3, relevance_threshold=3)
        assert recall_thresh3 == 1.0  # Found 1 out of 1 relevant
    
    def test_input_validation(self):
        """Test input validation."""
        with pytest.raises(TypeError):
            IRMetrics.ndcg_at_k("invalid", {}, 10)
        
        with pytest.raises(TypeError):
            IRMetrics.ndcg_at_k({}, "invalid", 10)
        
        with pytest.raises(ValueError):
            IRMetrics.ndcg_at_k({}, {}, -1)  # Negative k
        
        with pytest.raises(ValueError):
            IRMetrics.ndcg_at_k({}, {}, 0)   # Zero k
    
    def test_missing_queries(self):
        """Test handling of missing queries in results."""
        qrels = {'q1': {'d1': 1}, 'q2': {'d1': 1}}
        results = {'q1': {'d1': 0.9}}  # Missing q2
        
        # Should handle missing q2 gracefully
        ndcg = IRMetrics.ndcg_at_k(qrels, results, 1)
        assert 0.0 <= ndcg <= 1.0
        
        # With per_query, missing queries should have score 0 or be excluded
        ndcg, per_query = IRMetrics.ndcg_at_k(qrels, results, 1, per_query=True)
        assert 'q1' in per_query
        # q2 may or may not be in per_query depending on implementation
    
    def test_empty_qrels_values(self):
        """Test queries with empty relevance judgments."""
        qrels = {'q1': {}}  # Empty judgments
        results = {'q1': {'d1': 0.9}}
        
        ndcg = IRMetrics.ndcg_at_k(qrels, results, 1)
        assert ndcg == 0.0
    
    def test_unsorted_results(self):
        """Test that metrics handle unsorted results correctly."""
        qrels = {'q1': {'d1': 1, 'd2': 0, 'd3': 1}}
        results = {'q1': {'d2': 0.9, 'd1': 0.8, 'd3': 0.7}}  # d1 should rank higher than d3
        
        # Results should be sorted by score internally
        precision = IRMetrics.precision_at_k(qrels, results, 2)
        # Top-2 after sorting: [d2(0.9, rel=0), d1(0.8, rel=1)]
        # Precision@2 = 1/2 = 0.5
        assert abs(precision - 0.5) < 1e-10


class TestEvaluator:
    """Test suite for Evaluator class."""
    
    def setup_method(self):
        """Set up test data."""
        self.qrels = {
            'q1': {'d1': 3, 'd2': 2, 'd3': 1, 'd4': 0},
            'q2': {'d1': 1, 'd2': 0, 'd3': 2, 'd4': 3}
        }
        
        self.results = {
            'q1': {'d1': 0.9, 'd2': 0.8, 'd3': 0.7, 'd4': 0.6},
            'q2': {'d4': 0.9, 'd3': 0.8, 'd1': 0.7, 'd2': 0.6}
        }
        
        self.config = EvaluationConfig(
            k_values=[1, 5, 10],
            relevance_threshold=1,
            include_per_query=True
        )
    
    def test_evaluator_basic(self):
        """Test basic evaluator functionality."""
        evaluator = Evaluator(self.config)
        results = evaluator.evaluate(self.qrels, self.results)
        
        assert 'metrics' in results
        assert 'ndcg@1' in results['metrics']
        assert 'ndcg@5' in results['metrics'] 
        assert 'ndcg@10' in results['metrics']
        assert 'map@1' in results['metrics']
        assert 'recall@1' in results['metrics']
        assert 'precision@1' in results['metrics']
        assert 'mrr@1' in results['metrics']
        assert 'success@1' in results['metrics']
        
        assert 'per_query_metrics' in results
        assert results['per_query_metrics'] is not None
        
        assert 'timestamp' in results
        assert 'evaluation_time' in results
        assert 'dataset_stats' in results
    
    def test_evaluator_no_per_query(self):
        """Test evaluator without per-query metrics."""
        config = EvaluationConfig(
            k_values=[5, 10],
            include_per_query=False
        )
        evaluator = Evaluator(config)
        results = evaluator.evaluate(self.qrels, self.results)
        
        assert results['per_query_metrics'] is None
    
    def test_dataset_stats(self):
        """Test dataset statistics computation."""
        evaluator = Evaluator(self.config)
        results = evaluator.evaluate(self.qrels, self.results)
        
        stats = results['dataset_stats']
        assert stats['num_queries_qrels'] == 2
        assert stats['num_queries_results'] == 2
        assert stats['num_overlapping_queries'] == 2
        assert stats['total_qrels'] == 8  # 4 + 4 judgments
        assert stats['queries_with_relevant_docs'] == 2  # Both queries have relevant docs
    
    def test_evaluator_input_validation(self):
        """Test input validation in evaluator."""
        evaluator = Evaluator(self.config)
        
        with pytest.raises(ValueError):
            evaluator.evaluate({}, self.results)  # Empty qrels
        
        with pytest.raises(ValueError):
            evaluator.evaluate(self.qrels, {})  # Empty results
        
        with pytest.raises(ValueError):
            # No overlapping queries
            evaluator.evaluate({'q999': {'d1': 1}}, {'q888': {'d1': 0.5}})
    
    def test_single_metric_evaluation(self):
        """Test single metric evaluation."""
        evaluator = Evaluator(self.config)
        
        ndcg_results = evaluator.evaluate_single_metric('ndcg', self.qrels, self.results)
        assert 'ndcg@1' in ndcg_results
        assert 'ndcg@5' in ndcg_results
        assert 'ndcg@10' in ndcg_results
        
        map_results = evaluator.evaluate_single_metric('map', self.qrels, self.results, k_values=[10])
        assert 'map@10' in map_results
        assert len(map_results) == 1


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_large_k_values(self):
        """Test with k values larger than available documents."""
        qrels = {'q1': {'d1': 1}}
        results = {'q1': {'d1': 0.9}}
        
        # k=100 but only 1 document
        ndcg = IRMetrics.ndcg_at_k(qrels, results, 100)
        assert ndcg == 1.0  # Should still work correctly
    
    def test_all_zero_relevance(self):
        """Test queries where all documents have zero relevance."""
        qrels = {'q1': {'d1': 0, 'd2': 0, 'd3': 0}}
        results = {'q1': {'d1': 0.9, 'd2': 0.8, 'd3': 0.7}}
        
        ndcg = IRMetrics.ndcg_at_k(qrels, results, 3)
        assert ndcg == 0.0
        
        map_score = IRMetrics.map_at_k(qrels, results, 3)
        assert map_score == 0.0
    
    def test_single_document(self):
        """Test with single document per query."""
        qrels = {'q1': {'d1': 2}}
        results = {'q1': {'d1': 0.9}}
        
        ndcg = IRMetrics.ndcg_at_k(qrels, results, 1)
        assert ndcg == 1.0
        
        recall = IRMetrics.recall_at_k(qrels, results, 1)
        assert recall == 1.0
        
        precision = IRMetrics.precision_at_k(qrels, results, 1)
        assert precision == 1.0
    
    def test_duplicate_scores(self):
        """Test handling of duplicate scores in results."""
        qrels = {'q1': {'d1': 1, 'd2': 1, 'd3': 0}}
        results = {'q1': {'d1': 0.8, 'd2': 0.8, 'd3': 0.8}}  # All same score
        
        # Should handle ties consistently (behavior may vary but should not crash)
        precision = IRMetrics.precision_at_k(qrels, results, 2)
        assert 0.0 <= precision <= 1.0
    
    def test_very_large_dataset(self):
        """Test performance with larger dataset."""
        # Create larger test dataset
        num_queries = 100
        num_docs_per_query = 50
        
        qrels = {}
        results = {}
        
        for q in range(num_queries):
            query_id = f'q{q}'
            qrels[query_id] = {}
            results[query_id] = {}
            
            for d in range(num_docs_per_query):
                doc_id = f'd{d}'
                qrels[query_id][doc_id] = 1 if d < 5 else 0  # First 5 docs relevant
                results[query_id][doc_id] = 1.0 - (d / num_docs_per_query)  # Decreasing scores
        
        # Should complete in reasonable time
        import time
        start_time = time.time()
        
        ndcg = IRMetrics.ndcg_at_k(qrels, results, 10)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        assert execution_time < 5.0  # Should complete within 5 seconds
        assert 0.0 <= ndcg <= 1.0
    
    def test_graded_relevance(self):
        """Test with graded relevance (0-3 scale)."""
        qrels = {'q1': {'d1': 3, 'd2': 2, 'd3': 1, 'd4': 0}}
        results = {'q1': {'d1': 0.9, 'd2': 0.8, 'd3': 0.7, 'd4': 0.6}}
        
        # Test with different relevance thresholds
        recall_thresh1 = IRMetrics.recall_at_k(qrels, results, 4, relevance_threshold=1)
        recall_thresh2 = IRMetrics.recall_at_k(qrels, results, 4, relevance_threshold=2) 
        recall_thresh3 = IRMetrics.recall_at_k(qrels, results, 4, relevance_threshold=3)
        
        # Higher thresholds should give same or lower recall
        assert recall_thresh1 >= recall_thresh2 >= recall_thresh3
    
    def test_float_precision(self):
        """Test numerical precision and stability."""
        # Create case that might cause floating point issues
        qrels = {'q1': {f'd{i}': 1 for i in range(1000)}}
        results = {'q1': {f'd{i}': 1.0/i for i in range(1, 1001)}}
        
        ndcg = IRMetrics.ndcg_at_k(qrels, results, 100)
        assert 0.0 <= ndcg <= 1.0
        assert not math.isnan(ndcg)
        assert not math.isinf(ndcg)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
