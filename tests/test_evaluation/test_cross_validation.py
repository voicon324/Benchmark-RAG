"""
Cross-validation tests for IR metrics against reference libraries.

This module validates our custom IR metrics implementation against established
libraries like ranx and ir_measures to ensure mathematical correctness.
"""

import pytest
import numpy as np
from typing import Dict, List, Any
import warnings

# Suppress warnings from reference libraries
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

try:
    import ranx
    RANX_AVAILABLE = True
except ImportError:
    RANX_AVAILABLE = False

try:
    import ir_measures
    from ir_measures import evaluator
    IR_MEASURES_AVAILABLE = True
except ImportError:
    IR_MEASURES_AVAILABLE = False

from newaibench.evaluation import IRMetrics, EvaluationConfig


class CrossValidationTestSuite:
    """Test suite for cross-validation with reference libraries."""
    
    def __init__(self):
        self.tolerance = 1e-6  # Numerical tolerance for comparisons
        
    def create_test_data(self) -> tuple:
        """Create standardized test data for cross-validation."""
        # Query relevance judgments (qrels)
        qrels = {
            'q1': {'doc1': 3, 'doc2': 2, 'doc3': 1, 'doc4': 0, 'doc5': 1},
            'q2': {'doc1': 1, 'doc2': 0, 'doc3': 3, 'doc4': 2, 'doc5': 0},
            'q3': {'doc1': 0, 'doc2': 1, 'doc3': 0, 'doc4': 1, 'doc5': 2}
        }
        
        # Model results (rankings with scores)
        results = {
            'q1': [('doc1', 0.9), ('doc3', 0.7), ('doc2', 0.6), ('doc5', 0.4), ('doc4', 0.2)],
            'q2': [('doc3', 0.95), ('doc4', 0.8), ('doc1', 0.5), ('doc2', 0.3), ('doc5', 0.1)],
            'q3': [('doc5', 0.85), ('doc2', 0.7), ('doc4', 0.6), ('doc1', 0.4), ('doc3', 0.2)]
        }
        
        return qrels, results
    
    @pytest.mark.skipif(not RANX_AVAILABLE, reason="ranx library not available")
    def test_ndcg_vs_ranx(self):
        """Cross-validate nDCG computation with ranx library."""
        qrels, results = self.create_test_data()
        
        # Convert to ranx format
        ranx_qrels = ranx.Qrels(qrels)
        ranx_run = ranx.Run(results)
        
        # Test multiple k values
        for k in [1, 3, 5, 10]:
            # Our implementation
            our_ndcg = IRMetrics.ndcg_at_k(qrels, results, k)
            
            # ranx implementation
            ranx_ndcg = ranx.evaluate(ranx_qrels, ranx_run, f"ndcg@{k}")
            
            # Compare
            assert abs(our_ndcg - ranx_ndcg) < self.tolerance, \
                f"nDCG@{k} mismatch: ours={our_ndcg:.6f}, ranx={ranx_ndcg:.6f}"
    
    @pytest.mark.skipif(not RANX_AVAILABLE, reason="ranx library not available")
    def test_map_vs_ranx(self):
        """Cross-validate MAP computation with ranx library."""
        qrels, results = self.create_test_data()
        
        # Convert to ranx format
        ranx_qrels = ranx.Qrels(qrels)
        ranx_run = ranx.Run(results)
        
        # Test multiple k values
        for k in [5, 10, 100]:
            # Our implementation
            our_map = IRMetrics.map_at_k(qrels, results, k)
            
            # ranx implementation
            ranx_map = ranx.evaluate(ranx_qrels, ranx_run, f"map@{k}")
            
            # Compare
            assert abs(our_map - ranx_map) < self.tolerance, \
                f"MAP@{k} mismatch: ours={our_map:.6f}, ranx={ranx_map:.6f}"
    
    @pytest.mark.skipif(not RANX_AVAILABLE, reason="ranx library not available")
    def test_mrr_vs_ranx(self):
        """Cross-validate MRR computation with ranx library."""
        qrels, results = self.create_test_data()
        
        # Convert to ranx format
        ranx_qrels = ranx.Qrels(qrels)
        ranx_run = ranx.Run(results)
        
        # Test multiple k values
        for k in [5, 10, 100]:
            # Our implementation
            our_mrr = IRMetrics.mrr_at_k(qrels, results, k)
            
            # ranx implementation
            ranx_mrr = ranx.evaluate(ranx_qrels, ranx_run, f"mrr@{k}")
            
            # Compare
            assert abs(our_mrr - ranx_mrr) < self.tolerance, \
                f"MRR@{k} mismatch: ours={our_mrr:.6f}, ranx={ranx_mrr:.6f}"
    
    @pytest.mark.skipif(not IR_MEASURES_AVAILABLE, reason="ir_measures library not available")
    def test_ndcg_vs_ir_measures(self):
        """Cross-validate nDCG computation with ir_measures library."""
        qrels, results = self.create_test_data()
        
        # Convert to ir_measures format
        ir_qrels = []
        for qid, docs in qrels.items():
            for docid, rel in docs.items():
                ir_qrels.append((qid, docid, rel))
        
        ir_run = []
        for qid, docs in results.items():
            for rank, (docid, score) in enumerate(docs, 1):
                ir_run.append((qid, docid, score, rank))
        
        # Test multiple k values
        for k in [1, 3, 5, 10]:
            # Our implementation
            our_ndcg = IRMetrics.ndcg_at_k(qrels, results, k)
            
            # ir_measures implementation
            ev = evaluator.Evaluator([ir_measures.nDCG @ k])
            ir_results = ev.evaluate(ir_run, ir_qrels)
            ir_ndcg = ir_results[ir_measures.nDCG @ k]
            
            # Compare
            assert abs(our_ndcg - ir_ndcg) < self.tolerance, \
                f"nDCG@{k} mismatch: ours={our_ndcg:.6f}, ir_measures={ir_ndcg:.6f}"
    
    def test_edge_cases_consistency(self):
        """Test edge cases for consistency across implementations."""
        # Empty results
        qrels = {'q1': {'doc1': 1}}
        results = {'q1': []}
        
        our_ndcg = IRMetrics.ndcg_at_k(qrels, results, 10)
        assert our_ndcg == 0.0, "Empty results should give nDCG=0"
        
        # No relevant documents
        qrels = {'q1': {'doc1': 0, 'doc2': 0}}
        results = {'q1': [('doc1', 0.9), ('doc2', 0.8)]}
        
        our_ndcg = IRMetrics.ndcg_at_k(qrels, results, 10)
        assert our_ndcg == 0.0, "No relevant docs should give nDCG=0"
        
        # Perfect ranking
        qrels = {'q1': {'doc1': 3, 'doc2': 2, 'doc3': 1}}
        results = {'q1': [('doc1', 0.9), ('doc2', 0.8), ('doc3', 0.7)]}
        
        our_ndcg = IRMetrics.ndcg_at_k(qrels, results, 10)
        assert abs(our_ndcg - 1.0) < self.tolerance, "Perfect ranking should give nDCG≈1"
    
    def run_comprehensive_validation(self):
        """Run comprehensive cross-validation tests."""
        print("Running comprehensive cross-validation...")
        
        if RANX_AVAILABLE:
            print("✓ Testing against ranx library")
            self.test_ndcg_vs_ranx()
            self.test_map_vs_ranx()
            self.test_mrr_vs_ranx()
        else:
            print("⚠ ranx library not available")
        
        if IR_MEASURES_AVAILABLE:
            print("✓ Testing against ir_measures library")
            self.test_ndcg_vs_ir_measures()
        else:
            print("⚠ ir_measures library not available")
        
        print("✓ Testing edge cases")
        self.test_edge_cases_consistency()
        
        print("✅ All cross-validation tests passed!")


def benchmark_performance():
    """Benchmark performance against reference libraries."""
    import time
    
    # Create larger test dataset
    np.random.seed(42)
    num_queries = 1000
    docs_per_query = 100
    
    qrels = {}
    results = {}
    
    for i in range(num_queries):
        qid = f"q{i}"
        # Generate random relevance judgments
        qrels[qid] = {f"doc{j}": np.random.randint(0, 4) for j in range(docs_per_query)}
        # Generate random rankings
        scores = np.random.random(docs_per_query)
        results[qid] = [(f"doc{j}", scores[j]) for j in np.argsort(scores)[::-1]]
    
    print(f"Benchmarking with {num_queries} queries, {docs_per_query} docs each...")
    
    # Benchmark our implementation
    start_time = time.time()
    our_ndcg = IRMetrics.ndcg_at_k(qrels, results, 10)
    our_time = time.time() - start_time
    
    print(f"Our implementation: {our_time:.4f}s, nDCG@10={our_ndcg:.4f}")
    
    # Benchmark ranx if available
    if RANX_AVAILABLE:
        ranx_qrels = ranx.Qrels(qrels)
        ranx_run = ranx.Run(results)
        
        start_time = time.time()
        ranx_ndcg = ranx.evaluate(ranx_qrels, ranx_run, "ndcg@10")
        ranx_time = time.time() - start_time
        
        print(f"ranx implementation: {ranx_time:.4f}s, nDCG@10={ranx_ndcg:.4f}")
        print(f"Speed ratio: {ranx_time/our_time:.2f}x")


if __name__ == "__main__":
    # Run cross-validation tests
    validator = CrossValidationTestSuite()
    validator.run_comprehensive_validation()
    
    # Run performance benchmark
    print("\n" + "="*50)
    benchmark_performance()
