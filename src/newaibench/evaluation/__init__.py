"""
NewAIBench - Evaluation Module

This module provides comprehensive Information Retrieval evaluation capabilities
including standard IR metrics computation and batch evaluation utilities.
"""

from .metrics import IRMetrics, EvaluationConfig, MetricResult
from .evaluator import Evaluator, BatchEvaluator
from .optimization import StreamingEvaluator, ParallelEvaluator, OptimizationConfig

__all__ = [
    'IRMetrics',
    'EvaluationConfig', 
    'MetricResult',
    'Evaluator',
    'BatchEvaluator',
    'StreamingEvaluator',
    'ParallelEvaluator',
    'OptimizationConfig'
]

# Version information
__version__ = '1.0.0'

# Default evaluation configuration
DEFAULT_CONFIG = EvaluationConfig(
    k_values=[1, 3, 5, 10, 20, 100],
    relevance_threshold=1,
    include_per_query=True,
    statistical_tests=False
)

def quick_evaluate(qrels, results, k_values=None, relevance_threshold=1):
    """
    Quick evaluation function for common use cases.
    
    Args:
        qrels: Query relevance judgments
        results: Model results  
        k_values: List of k values (default: [1, 5, 10, 100])
        relevance_threshold: Minimum relevance score (default: 1)
        
    Returns:
        Dictionary with evaluation metrics
        
    Example:
        >>> metrics = quick_evaluate(qrels, results, k_values=[10, 100])
        >>> print(f"nDCG@10: {metrics['ndcg@10']:.4f}")
    """
    config = EvaluationConfig(
        k_values=k_values or [1, 5, 10, 100],
        relevance_threshold=relevance_threshold,
        include_per_query=False
    )
    evaluator = Evaluator(config)
    results = evaluator.evaluate(qrels, results)
    return results['metrics']
