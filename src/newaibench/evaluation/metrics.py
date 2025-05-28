"""
NewAIBench - Information Retrieval Evaluation Metrics Module

This module implements standard IR evaluation metrics with focus on:
- Mathematical accuracy following standard IR formulas
- High performance for large-scale evaluation
- Comprehensive edge case handling
- Clear documentation and examples

Author: NewAIBench Team
"""

import math
import numpy as np
from typing import Dict, List, Union, Optional, Tuple, Any
from collections import defaultdict
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class EvaluationConfig:
    """Configuration for evaluation metrics."""
    k_values: List[int] = None
    relevance_threshold: int = 1  # Minimum relevance score considered as relevant
    include_per_query: bool = True
    statistical_tests: bool = False
    
    def __post_init__(self):
        if self.k_values is None:
            self.k_values = [1, 3, 5, 10, 20, 100]


@dataclass 
class MetricResult:
    """Result of metric computation."""
    value: float
    per_query_values: Optional[Dict[str, float]] = None
    num_queries: int = 0
    num_valid_queries: int = 0  # Queries that have relevant documents


class IRMetrics:
    """
    Core Information Retrieval evaluation metrics implementation.
    
    This class implements standard IR metrics following established formulas:
    - nDCG@k: Normalized Discounted Cumulative Gain
    - MAP@k: Mean Average Precision  
    - Recall@k: Proportion of relevant documents retrieved
    - Precision@k: Proportion of retrieved documents that are relevant
    - MRR@k: Mean Reciprocal Rank
    - Success@k: Proportion of queries with at least one relevant document
    
    All methods handle edge cases gracefully and follow standard IR conventions.
    """
    
    @staticmethod
    def _validate_inputs(qrels: Dict[str, Dict[str, int]], 
                        results: Dict[str, Dict[str, float]], 
                        k: int) -> None:
        """Validate input formats and parameters."""
        if not isinstance(qrels, dict):
            raise TypeError("qrels must be a dictionary")
        if not isinstance(results, dict):
            raise TypeError("results must be a dictionary") 
        if k <= 0:
            raise ValueError("k must be positive")
            
        # Check if qrels and results have overlapping queries
        qrels_queries = set(qrels.keys())
        results_queries = set(results.keys())
        
        if not qrels_queries.intersection(results_queries):
            logger.warning("No overlapping queries between qrels and results")
    
    @staticmethod
    def _get_relevance_scores(qrels: Dict[str, Dict[str, int]], 
                             query_id: str, 
                             doc_ids: List[str]) -> List[int]:
        """Get relevance scores for documents in order."""
        query_qrels = qrels.get(query_id, {})
        return [query_qrels.get(doc_id, 0) for doc_id in doc_ids]
    
    @staticmethod
    def _dcg_at_k(relevance_scores: List[int], k: int) -> float:
        """
        Compute Discounted Cumulative Gain at k.
        
        DCG@k = rel_1 + sum_{i=2}^k (rel_i / log2(i+1))
        
        Args:
            relevance_scores: List of relevance scores in ranked order
            k: Cutoff rank
            
        Returns:
            DCG@k value
        """
        if not relevance_scores or k <= 0:
            return 0.0
            
        dcg = 0.0
        for i, rel in enumerate(relevance_scores[:k]):
            if i == 0:
                dcg += rel
            else:
                dcg += rel / math.log2(i + 2)  # i+2 because i is 0-based
                
        return dcg
    
    @staticmethod
    def _ideal_dcg_at_k(relevance_scores: List[int], k: int) -> float:
        """
        Compute Ideal DCG at k (DCG with perfect ranking).
        
        Args:
            relevance_scores: List of all relevance scores for the query
            k: Cutoff rank
            
        Returns:
            IDCG@k value
        """
        # Sort relevance scores in descending order for ideal ranking
        ideal_scores = sorted(relevance_scores, reverse=True)
        return IRMetrics._dcg_at_k(ideal_scores, k)
    
    @staticmethod
    def ndcg_at_k(qrels: Dict[str, Dict[str, int]], 
                  results: Dict[str, Dict[str, float]], 
                  k: int,
                  per_query: bool = False) -> Union[float, Tuple[float, Dict[str, float]]]:
        """
        Compute Normalized Discounted Cumulative Gain at k.
        
        nDCG@k = DCG@k / IDCG@k
        
        Args:
            qrels: Query relevance judgments {query_id: {doc_id: relevance_score}}
            results: Model results {query_id: {doc_id: score}} (assumed sorted by score desc)  
            k: Cutoff rank
            per_query: Whether to return per-query scores
            
        Returns:
            nDCG@k score (and per-query scores if requested)
            
        Example:
            >>> qrels = {'q1': {'d1': 3, 'd2': 2, 'd3': 0}}
            >>> results = {'q1': {'d1': 0.9, 'd3': 0.8, 'd2': 0.7}}
            >>> ndcg = IRMetrics.ndcg_at_k(qrels, results, k=3)
            >>> print(f"nDCG@3: {ndcg:.4f}")
        """
        IRMetrics._validate_inputs(qrels, results, k)
        
        ndcg_scores = {}
        valid_queries = []
        
        for query_id in qrels.keys():
            if query_id not in results:
                continue
                
            # Get all relevance scores for this query to compute IDCG
            all_relevance_scores = list(qrels[query_id].values())
            
            # Skip queries with no relevant documents
            if not any(score > 0 for score in all_relevance_scores):
                if per_query:
                    ndcg_scores[query_id] = 0.0
                continue
                
            # Get ranked document IDs from results
            result_items = list(results[query_id].items())
            # Ensure results are sorted by score descending
            result_items.sort(key=lambda x: x[1], reverse=True)
            doc_ids = [doc_id for doc_id, _ in result_items]
            
            # Get relevance scores in ranked order
            relevance_scores = IRMetrics._get_relevance_scores(qrels, query_id, doc_ids)
            
            # Compute DCG@k
            dcg = IRMetrics._dcg_at_k(relevance_scores, k)
            
            # Compute IDCG@k  
            idcg = IRMetrics._ideal_dcg_at_k(all_relevance_scores, k)
            
            # Compute nDCG@k
            if idcg > 0:
                ndcg = dcg / idcg
                valid_queries.append(query_id)
            else:
                ndcg = 0.0
                
            if per_query:
                ndcg_scores[query_id] = ndcg
        
        # Compute mean nDCG@k across valid queries
        if valid_queries:
            if per_query:
                mean_ndcg = sum(ndcg_scores[qid] for qid in valid_queries) / len(valid_queries)
                return mean_ndcg, ndcg_scores
            else:
                mean_ndcg = sum(ndcg_scores.get(qid, 0.0) for qid in valid_queries) / len(valid_queries)
                return mean_ndcg
        else:
            if per_query:
                return 0.0, ndcg_scores
            else:
                return 0.0
    
    @staticmethod
    def precision_at_k(qrels: Dict[str, Dict[str, int]], 
                       results: Dict[str, Dict[str, float]], 
                       k: int,
                       relevance_threshold: int = 1,
                       per_query: bool = False) -> Union[float, Tuple[float, Dict[str, float]]]:
        """
        Compute Precision at k.
        
        P@k = (Number of relevant documents in top-k) / k
        
        Args:
            qrels: Query relevance judgments
            results: Model results  
            k: Cutoff rank
            relevance_threshold: Minimum relevance score considered relevant
            per_query: Whether to return per-query scores
            
        Returns:
            Precision@k score (and per-query scores if requested)
        """
        IRMetrics._validate_inputs(qrels, results, k)
        
        precision_scores = {}
        
        for query_id in qrels.keys():
            if query_id not in results:
                if per_query:
                    precision_scores[query_id] = 0.0
                continue
                
            # Get top-k documents
            result_items = list(results[query_id].items())
            result_items.sort(key=lambda x: x[1], reverse=True)
            top_k_docs = result_items[:k]
            
            # Count relevant documents in top-k
            relevant_count = 0
            for doc_id, _ in top_k_docs:
                relevance = qrels[query_id].get(doc_id, 0)
                if relevance >= relevance_threshold:
                    relevant_count += 1
            
            # Compute precision@k
            precision = relevant_count / min(k, len(top_k_docs)) if top_k_docs else 0.0
            
            if per_query:
                precision_scores[query_id] = precision
        
        # Compute mean precision@k
        if qrels:
            if per_query:
                mean_precision = sum(precision_scores.values()) / len(qrels)
                return mean_precision, precision_scores
            else:
                mean_precision = sum(precision_scores.values()) / len(qrels)
                return mean_precision
        else:
            if per_query:
                return 0.0, precision_scores
            else:
                return 0.0
    
    @staticmethod
    def recall_at_k(qrels: Dict[str, Dict[str, int]], 
                    results: Dict[str, Dict[str, float]], 
                    k: int,
                    relevance_threshold: int = 1,
                    per_query: bool = False) -> Union[float, Tuple[float, Dict[str, float]]]:
        """
        Compute Recall at k.
        
        Recall@k = (Number of relevant documents in top-k) / (Total relevant documents)
        
        Args:
            qrels: Query relevance judgments
            results: Model results
            k: Cutoff rank  
            relevance_threshold: Minimum relevance score considered relevant
            per_query: Whether to return per-query scores
            
        Returns:
            Recall@k score (and per-query scores if requested)
        """
        IRMetrics._validate_inputs(qrels, results, k)
        
        recall_scores = {}
        valid_queries = []
        
        for query_id in qrels.keys():
            # Count total relevant documents for this query
            total_relevant = sum(1 for rel in qrels[query_id].values() 
                               if rel >= relevance_threshold)
            
            if total_relevant == 0:
                if per_query:
                    recall_scores[query_id] = 0.0
                continue
                
            valid_queries.append(query_id)
            
            if query_id not in results:
                if per_query:
                    recall_scores[query_id] = 0.0
                continue
                
            # Get top-k documents
            result_items = list(results[query_id].items())
            result_items.sort(key=lambda x: x[1], reverse=True)
            top_k_docs = result_items[:k]
            
            # Count relevant documents in top-k
            relevant_found = 0
            for doc_id, _ in top_k_docs:
                relevance = qrels[query_id].get(doc_id, 0)
                if relevance >= relevance_threshold:
                    relevant_found += 1
            
            # Compute recall@k
            recall = relevant_found / total_relevant
            
            if per_query:
                recall_scores[query_id] = recall
        
        # Compute mean recall@k across queries with relevant documents
        if valid_queries:
            if per_query:
                mean_recall = sum(recall_scores[qid] for qid in valid_queries) / len(valid_queries)
                return mean_recall, recall_scores
            else:
                mean_recall = sum(recall_scores.get(qid, 0.0) for qid in valid_queries) / len(valid_queries)
                return mean_recall
        else:
            if per_query:
                return 0.0, recall_scores
            else:
                return 0.0
    
    @staticmethod
    def average_precision_at_k(qrels: Dict[str, int], 
                               results: Dict[str, float], 
                               k: int,
                               relevance_threshold: int = 1) -> float:
        """
        Compute Average Precision at k for a single query.
        
        AP@k = (sum_{i=1}^k (P@i * rel_i)) / min(k, |relevant_docs|)
        
        Args:
            qrels: Relevance judgments for single query {doc_id: relevance}
            results: Results for single query {doc_id: score}
            k: Cutoff rank
            relevance_threshold: Minimum relevance score considered relevant
            
        Returns:
            Average Precision@k for the query
        """
        # Count total relevant documents
        total_relevant = sum(1 for rel in qrels.values() if rel >= relevance_threshold)
        
        if total_relevant == 0:
            return 0.0
            
        # Get ranked documents
        result_items = list(results.items())
        result_items.sort(key=lambda x: x[1], reverse=True)
        
        ap_sum = 0.0
        relevant_found = 0
        
        for i, (doc_id, _) in enumerate(result_items[:k]):
            relevance = qrels.get(doc_id, 0)
            if relevance >= relevance_threshold:
                relevant_found += 1
                precision_at_i = relevant_found / (i + 1)
                ap_sum += precision_at_i
        
        # Normalize by min(k, total_relevant)
        ap = ap_sum / min(k, total_relevant)
        return ap
    
    @staticmethod  
    def map_at_k(qrels: Dict[str, Dict[str, int]], 
                 results: Dict[str, Dict[str, float]], 
                 k: int,
                 relevance_threshold: int = 1,
                 per_query: bool = False) -> Union[float, Tuple[float, Dict[str, float]]]:
        """
        Compute Mean Average Precision at k.
        
        MAP@k = mean(AP@k) across all queries
        
        Args:
            qrels: Query relevance judgments
            results: Model results
            k: Cutoff rank
            relevance_threshold: Minimum relevance score considered relevant  
            per_query: Whether to return per-query scores
            
        Returns:
            MAP@k score (and per-query scores if requested)
        """
        IRMetrics._validate_inputs(qrels, results, k)
        
        ap_scores = {}
        valid_queries = []
        
        for query_id in qrels.keys():
            # Skip queries with no relevant documents
            total_relevant = sum(1 for rel in qrels[query_id].values() 
                               if rel >= relevance_threshold)
            if total_relevant == 0:
                if per_query:
                    ap_scores[query_id] = 0.0
                continue
                
            valid_queries.append(query_id)
            
            if query_id not in results:
                if per_query:
                    ap_scores[query_id] = 0.0
                continue
            
            # Compute AP@k for this query
            ap = IRMetrics.average_precision_at_k(
                qrels[query_id], results[query_id], k, relevance_threshold
            )
            
            if per_query:
                ap_scores[query_id] = ap
        
        # Compute MAP@k across queries with relevant documents
        if valid_queries:
            if per_query:
                mean_ap = sum(ap_scores[qid] for qid in valid_queries) / len(valid_queries)
                return mean_ap, ap_scores
            else:
                mean_ap = sum(ap_scores.get(qid, 0.0) for qid in valid_queries) / len(valid_queries)
                return mean_ap
        else:
            if per_query:
                return 0.0, ap_scores
            else:
                return 0.0
    
    @staticmethod
    def mrr_at_k(qrels: Dict[str, Dict[str, int]], 
                 results: Dict[str, Dict[str, float]], 
                 k: int,
                 relevance_threshold: int = 1,
                 per_query: bool = False) -> Union[float, Tuple[float, Dict[str, float]]]:
        """
        Compute Mean Reciprocal Rank at k.
        
        MRR@k = mean(1/rank_first_relevant) across all queries
        
        Args:
            qrels: Query relevance judgments
            results: Model results
            k: Cutoff rank
            relevance_threshold: Minimum relevance score considered relevant
            per_query: Whether to return per-query scores
            
        Returns:
            MRR@k score (and per-query scores if requested)
        """
        IRMetrics._validate_inputs(qrels, results, k)
        
        rr_scores = {}
        valid_queries = []
        
        for query_id in qrels.keys():
            # Skip queries with no relevant documents  
            has_relevant = any(rel >= relevance_threshold for rel in qrels[query_id].values())
            if not has_relevant:
                if per_query:
                    rr_scores[query_id] = 0.0
                continue
                
            valid_queries.append(query_id)
            
            if query_id not in results:
                if per_query:
                    rr_scores[query_id] = 0.0
                continue
            
            # Get ranked documents
            result_items = list(results[query_id].items())
            result_items.sort(key=lambda x: x[1], reverse=True)
            
            # Find rank of first relevant document
            rr = 0.0
            for i, (doc_id, _) in enumerate(result_items[:k]):
                relevance = qrels[query_id].get(doc_id, 0)
                if relevance >= relevance_threshold:
                    rr = 1.0 / (i + 1)  # i+1 because rank is 1-based
                    break
            
            if per_query:
                rr_scores[query_id] = rr
        
        # Compute MRR@k across queries with relevant documents
        if valid_queries:
            if per_query:
                mean_rr = sum(rr_scores[qid] for qid in valid_queries) / len(valid_queries)
                return mean_rr, rr_scores
            else:
                mean_rr = sum(rr_scores.get(qid, 0.0) for qid in valid_queries) / len(valid_queries)
                return mean_rr
        else:
            if per_query:
                return 0.0, rr_scores
            else:
                return 0.0
    
    @staticmethod
    def success_at_k(qrels: Dict[str, Dict[str, int]], 
                     results: Dict[str, Dict[str, float]], 
                     k: int,
                     relevance_threshold: int = 1,
                     per_query: bool = False) -> Union[float, Tuple[float, Dict[str, float]]]:
        """
        Compute Success at k (Hit Rate).
        
        Success@k = Proportion of queries with at least one relevant document in top-k
        
        Args:
            qrels: Query relevance judgments
            results: Model results
            k: Cutoff rank
            relevance_threshold: Minimum relevance score considered relevant
            per_query: Whether to return per-query scores
            
        Returns:
            Success@k score (and per-query scores if requested)
        """
        IRMetrics._validate_inputs(qrels, results, k)
        
        success_scores = {}
        valid_queries = []
        
        for query_id in qrels.keys():
            # Skip queries with no relevant documents
            has_relevant = any(rel >= relevance_threshold for rel in qrels[query_id].values())
            if not has_relevant:
                if per_query:
                    success_scores[query_id] = 0.0
                continue
                
            valid_queries.append(query_id)
            
            if query_id not in results:
                if per_query:
                    success_scores[query_id] = 0.0
                continue
            
            # Check if any document in top-k is relevant
            result_items = list(results[query_id].items())
            result_items.sort(key=lambda x: x[1], reverse=True)
            
            success = 0.0
            for doc_id, _ in result_items[:k]:
                relevance = qrels[query_id].get(doc_id, 0)
                if relevance >= relevance_threshold:
                    success = 1.0
                    break
            
            if per_query:
                success_scores[query_id] = success
        
        # Compute Success@k across queries with relevant documents
        if valid_queries:
            if per_query:
                mean_success = sum(success_scores[qid] for qid in valid_queries) / len(valid_queries)
                return mean_success, success_scores
            else:
                mean_success = sum(success_scores.get(qid, 0.0) for qid in valid_queries) / len(valid_queries)
                return mean_success
        else:
            if per_query:
                return 0.0, success_scores
            else:
                return 0.0
