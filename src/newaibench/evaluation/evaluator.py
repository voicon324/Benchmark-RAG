"""
NewAIBench - Evaluation Engine

Main evaluation engine that orchestrates metric computation and provides
a high-level interface for batch evaluation of IR models.
"""

import time
import logging
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import json
from datetime import datetime

from .metrics import IRMetrics, EvaluationConfig, MetricResult

logger = logging.getLogger(__name__)


class Evaluator:
    """
    High-level evaluation engine for Information Retrieval models.
    
    This class provides a convenient interface for computing multiple IR metrics
    across different k values with comprehensive error handling and reporting.
    
    Example:
        >>> config = EvaluationConfig(k_values=[1, 5, 10, 100])
        >>> evaluator = Evaluator(config)
        >>> results = evaluator.evaluate(qrels, model_results)
        >>> print(f"nDCG@10: {results['ndcg@10']:.4f}")
    """
    
    def __init__(self, config: EvaluationConfig = None):
        """
        Initialize evaluator with configuration.
        
        Args:
            config: Evaluation configuration. If None, uses default config.
        """
        self.config = config or EvaluationConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def evaluate(self, 
                 qrels: Dict[str, Dict[str, int]], 
                 results: Dict[str, Dict[str, float]],
                 save_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
        """
        Evaluate model results against relevance judgments.
        
        Args:
            qrels: Query relevance judgments {query_id: {doc_id: relevance_score}}
            results: Model results {query_id: {doc_id: score}}
            save_path: Optional path to save detailed results
            
        Returns:
            Dictionary containing all computed metrics and metadata
            
        Example:
            >>> qrels = {
            ...     'q1': {'d1': 2, 'd2': 1, 'd3': 0},
            ...     'q2': {'d1': 1, 'd4': 3}
            ... }
            >>> results = {
            ...     'q1': {'d1': 0.9, 'd2': 0.7, 'd3': 0.5},
            ...     'q2': {'d4': 0.8, 'd1': 0.6}
            ... }
            >>> evaluator = Evaluator()
            >>> metrics = evaluator.evaluate(qrels, results)
        """
        start_time = time.time()
        
        # Validate inputs
        self._validate_inputs(qrels, results)
        
        # Initialize results dictionary
        evaluation_results = {
            'timestamp': datetime.now().isoformat(),
            'config': self.config.__dict__.copy(),
            'dataset_stats': self._compute_dataset_stats(qrels, results),
            'metrics': {},
            'per_query_metrics': {} if self.config.include_per_query else None,
            'evaluation_time': 0.0
        }
        
        self.logger.info(f"Starting evaluation with {len(qrels)} queries, "
                        f"{len(results)} result sets, k_values={self.config.k_values}")
        
        # Compute metrics for each k value
        for k in self.config.k_values:
            self.logger.debug(f"Computing metrics for k={k}")
            
            try:
                # nDCG@k
                if self.config.include_per_query:
                    ndcg, ndcg_per_query = IRMetrics.ndcg_at_k(qrels, results, k, per_query=True)
                    evaluation_results['per_query_metrics'][f'ndcg@{k}'] = ndcg_per_query
                else:
                    ndcg = IRMetrics.ndcg_at_k(qrels, results, k, per_query=False)
                evaluation_results['metrics'][f'ndcg@{k}'] = ndcg
                
                # MAP@k
                if self.config.include_per_query:
                    map_score, map_per_query = IRMetrics.map_at_k(
                        qrels, results, k, self.config.relevance_threshold, per_query=True
                    )
                    evaluation_results['per_query_metrics'][f'map@{k}'] = map_per_query
                else:
                    map_score = IRMetrics.map_at_k(
                        qrels, results, k, self.config.relevance_threshold, per_query=False
                    )
                evaluation_results['metrics'][f'map@{k}'] = map_score
                
                # Recall@k
                if self.config.include_per_query:
                    recall, recall_per_query = IRMetrics.recall_at_k(
                        qrels, results, k, self.config.relevance_threshold, per_query=True
                    )
                    evaluation_results['per_query_metrics'][f'recall@{k}'] = recall_per_query
                else:
                    recall = IRMetrics.recall_at_k(
                        qrels, results, k, self.config.relevance_threshold, per_query=False
                    )
                evaluation_results['metrics'][f'recall@{k}'] = recall
                
                # Precision@k
                if self.config.include_per_query:
                    precision, precision_per_query = IRMetrics.precision_at_k(
                        qrels, results, k, self.config.relevance_threshold, per_query=True
                    )
                    evaluation_results['per_query_metrics'][f'precision@{k}'] = precision_per_query
                else:
                    precision = IRMetrics.precision_at_k(
                        qrels, results, k, self.config.relevance_threshold, per_query=False
                    )
                evaluation_results['metrics'][f'precision@{k}'] = precision
                
                # MRR@k
                if self.config.include_per_query:
                    mrr, mrr_per_query = IRMetrics.mrr_at_k(
                        qrels, results, k, self.config.relevance_threshold, per_query=True
                    )
                    evaluation_results['per_query_metrics'][f'mrr@{k}'] = mrr_per_query
                else:
                    mrr = IRMetrics.mrr_at_k(
                        qrels, results, k, self.config.relevance_threshold, per_query=False
                    )
                evaluation_results['metrics'][f'mrr@{k}'] = mrr
                
                # Success@k
                if self.config.include_per_query:
                    success, success_per_query = IRMetrics.success_at_k(
                        qrels, results, k, self.config.relevance_threshold, per_query=True
                    )
                    evaluation_results['per_query_metrics'][f'success@{k}'] = success_per_query
                else:
                    success = IRMetrics.success_at_k(
                        qrels, results, k, self.config.relevance_threshold, per_query=False
                    )
                evaluation_results['metrics'][f'success@{k}'] = success
                
            except Exception as e:
                self.logger.error(f"Error computing metrics for k={k}: {str(e)}")
                # Set error values
                for metric in ['ndcg', 'map', 'recall', 'precision', 'mrr', 'success']:
                    evaluation_results['metrics'][f'{metric}@{k}'] = 0.0
        
        evaluation_results['evaluation_time'] = time.time() - start_time
        
        self.logger.info(f"Evaluation completed in {evaluation_results['evaluation_time']:.2f}s")
        self._log_metric_summary(evaluation_results['metrics'])
        
        # Save results if path provided
        if save_path:
            self._save_results(evaluation_results, save_path)
        
        return evaluation_results
    
    def evaluate_single_metric(self, 
                              metric_name: str,
                              qrels: Dict[str, Dict[str, int]], 
                              results: Dict[str, Dict[str, float]],
                              k_values: Optional[List[int]] = None) -> Dict[str, float]:
        """
        Evaluate a single metric across multiple k values.
        
        Args:
            metric_name: Name of metric ('ndcg', 'map', 'recall', 'precision', 'mrr', 'success')
            qrels: Query relevance judgments
            results: Model results
            k_values: List of k values. If None, uses config k_values
            
        Returns:
            Dictionary with metric values for each k
        """
        k_values = k_values or self.config.k_values
        metric_results = {}
        
        metric_func = getattr(IRMetrics, f'{metric_name}_at_k')
        
        for k in k_values:
            if metric_name == 'ndcg':
                value = metric_func(qrels, results, k, per_query=False)
            else:
                value = metric_func(qrels, results, k, self.config.relevance_threshold, per_query=False)
            metric_results[f'{metric_name}@{k}'] = value
            
        return metric_results
    
    def _validate_inputs(self, qrels: Dict, results: Dict) -> None:
        """Validate input formats."""
        if not qrels:
            raise ValueError("qrels cannot be empty")
        if not results:
            raise ValueError("results cannot be empty")
            
        # Check for overlapping queries
        qrels_queries = set(qrels.keys())
        results_queries = set(results.keys())
        overlap = qrels_queries.intersection(results_queries)
        
        if not overlap:
            raise ValueError("No overlapping queries between qrels and results")
            
        self.logger.info(f"Found {len(overlap)} overlapping queries out of "
                        f"{len(qrels_queries)} qrels queries and {len(results_queries)} result queries")
    
    def _compute_dataset_stats(self, qrels: Dict, results: Dict) -> Dict[str, Any]:
        """Compute dataset statistics."""
        stats = {
            'num_queries_qrels': len(qrels),
            'num_queries_results': len(results),
            'num_overlapping_queries': len(set(qrels.keys()) & set(results.keys())),
            'total_qrels': sum(len(judgments) for judgments in qrels.values()),
            'total_results': sum(len(res) for res in results.values()),
            'avg_qrels_per_query': 0.0,
            'avg_results_per_query': 0.0,
            'queries_with_relevant_docs': 0,
            'queries_with_no_results': 0
        }
        
        if qrels:
            stats['avg_qrels_per_query'] = stats['total_qrels'] / len(qrels)
            
            # Count queries with relevant documents
            stats['queries_with_relevant_docs'] = sum(
                1 for judgments in qrels.values() 
                if any(rel >= self.config.relevance_threshold for rel in judgments.values())
            )
        
        if results:
            stats['avg_results_per_query'] = stats['total_results'] / len(results)
            
            # Count queries with no results
            stats['queries_with_no_results'] = sum(
                1 for res in results.values() if not res
            )
        
        return stats
    
    def _log_metric_summary(self, metrics: Dict[str, float]) -> None:
        """Log summary of key metrics."""
        key_metrics = []
        
        # Find common k values to report
        for k in [1, 5, 10, 100]:
            if f'ndcg@{k}' in metrics:
                key_metrics.append(f"nDCG@{k}: {metrics[f'ndcg@{k}']:.4f}")
            if f'map@{k}' in metrics and k in [10, 100]:
                key_metrics.append(f"MAP@{k}: {metrics[f'map@{k}']:.4f}")
            if f'recall@{k}' in metrics and k in [100]:
                key_metrics.append(f"Recall@{k}: {metrics[f'recall@{k}']:.4f}")
        
        if key_metrics:
            self.logger.info("Key metrics: " + ", ".join(key_metrics))
    
    def _save_results(self, results: Dict[str, Any], save_path: Union[str, Path]) -> None:
        """Save evaluation results to file."""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            self.logger.info(f"Results saved to {save_path}")
        except Exception as e:
            self.logger.error(f"Failed to save results to {save_path}: {str(e)}")


class BatchEvaluator:
    """
    Evaluator for comparing multiple models or running multiple experiments.
    
    Useful for benchmark comparisons and systematic evaluation across multiple
    models, datasets, or parameter configurations.
    """
    
    def __init__(self, config: EvaluationConfig = None):
        """Initialize batch evaluator."""
        self.config = config or EvaluationConfig()
        self.evaluator = Evaluator(self.config)
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def evaluate_multiple_models(self, 
                                 qrels: Dict[str, Dict[str, int]],
                                 model_results: Dict[str, Dict[str, Dict[str, float]]],
                                 save_dir: Optional[Union[str, Path]] = None) -> Dict[str, Dict[str, Any]]:
        """
        Evaluate multiple models against the same qrels.
        
        Args:
            qrels: Query relevance judgments  
            model_results: {model_name: {query_id: {doc_id: score}}}
            save_dir: Optional directory to save individual model results
            
        Returns:
            Dictionary with evaluation results for each model
        """
        all_results = {}
        
        for model_name, results in model_results.items():
            self.logger.info(f"Evaluating model: {model_name}")
            
            save_path = None
            if save_dir:
                save_path = Path(save_dir) / f"{model_name}_evaluation.json"
            
            model_eval = self.evaluator.evaluate(qrels, results, save_path)
            model_eval['model_name'] = model_name
            all_results[model_name] = model_eval
        
        # Compute comparative analysis
        comparative_results = self._compute_comparative_analysis(all_results)
        
        # Save comparative results
        if save_dir:
            comp_save_path = Path(save_dir) / "comparative_analysis.json"
            try:
                with open(comp_save_path, 'w', encoding='utf-8') as f:
                    json.dump(comparative_results, f, indent=2, ensure_ascii=False)
                self.logger.info(f"Comparative analysis saved to {comp_save_path}")
            except Exception as e:
                self.logger.error(f"Failed to save comparative analysis: {str(e)}")
        
        return {
            'individual_results': all_results,
            'comparative_analysis': comparative_results
        }
    
    def _compute_comparative_analysis(self, 
                                    all_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Compute comparative analysis across models."""
        if len(all_results) < 2:
            return {'note': 'Need at least 2 models for comparative analysis'}
        
        analysis = {
            'num_models': len(all_results),
            'models': list(all_results.keys()),
            'best_model_by_metric': {},
            'metric_rankings': {},
            'performance_summary': {}
        }
        
        # Find best model for each metric
        for metric_name in ['ndcg@10', 'map@10', 'recall@100', 'mrr@10']:
            if metric_name in list(all_results.values())[0]['metrics']:
                best_model = max(all_results.keys(), 
                               key=lambda m: all_results[m]['metrics'][metric_name])
                best_score = all_results[best_model]['metrics'][metric_name]
                analysis['best_model_by_metric'][metric_name] = {
                    'model': best_model,
                    'score': best_score
                }
                
                # Create ranking for this metric
                ranking = sorted(all_results.items(), 
                               key=lambda x: x[1]['metrics'][metric_name], 
                               reverse=True)
                analysis['metric_rankings'][metric_name] = [
                    {'model': model, 'score': results['metrics'][metric_name]} 
                    for model, results in ranking
                ]
        
        # Performance summary table
        for model_name, results in all_results.items():
            analysis['performance_summary'][model_name] = {
                metric: results['metrics'].get(metric, 0.0)
                for metric in ['ndcg@1', 'ndcg@5', 'ndcg@10', 'map@10', 'recall@100']
                if metric in results['metrics']
            }
        
        return analysis
