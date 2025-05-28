"""
Results aggregation and analysis tool for NewAIBench.

This module provides functionality to:
- Aggregate results from multiple experiments
- Calculate descriptive statistics and comparisons
- Create summary tables by different criteria
- Analyze correlations and trends
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from collections import defaultdict
from dataclasses import dataclass
import statistics

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    pd = None

from .storage import ResultsStorage, ExperimentMetadata, RunResult


@dataclass
class AggregationConfig:
    """Configuration for result aggregation."""
    experiments: List[str] = None  # Filter by experiment names
    models: List[str] = None       # Filter by model names  
    datasets: List[str] = None     # Filter by dataset names
    metrics: List[str] = None      # Metrics to aggregate
    include_failed: bool = False   # Include failed runs
    
    def __post_init__(self):
        if self.metrics is None:
            self.metrics = ['ndcg@10', 'map@10', 'recall@10']


class ResultsAggregator:
    """Main class for aggregating results."""
    
    def __init__(self, storage: ResultsStorage):
        """
        Initialize results aggregator.
        
        Args:
            storage: Storage backend for accessing results
        """
        self.storage = storage
        self.logger = logging.getLogger(__name__)
    
    def get_comparison_table(self, config: AggregationConfig) -> Any:
        """
        Get comparison table of results.
        
        Args:
            config: Aggregation configuration
            
        Returns:
            DataFrame with comparison results or None if pandas not available
        """
        if not HAS_PANDAS:
            raise ImportError("pandas is required for comparison tables")
        
        # Get filtered results
        results_df = self.storage.query_results(
            experiments=config.experiments,
            models=config.models,
            datasets=config.datasets,
            metrics=config.metrics
        )
        
        return results_df
    
    def get_summary_statistics(self, config: AggregationConfig) -> Any:
        """
        Get summary statistics for metrics.
        
        Args:
            config: Aggregation configuration
            
        Returns:
            DataFrame with summary statistics or None if pandas not available
        """
        if not HAS_PANDAS:
            raise ImportError("pandas is required for summary statistics")
        
        results_df = self.get_comparison_table(config)
        
        if results_df.empty:
            return pd.DataFrame()
        
        # Select numeric columns (metrics)
        numeric_cols = results_df.select_dtypes(include=['float64', 'int64']).columns
        metrics_cols = [col for col in numeric_cols if col in (config.metrics or [])]
        
        if not metrics_cols:
            return pd.DataFrame()
        
        # Calculate statistics grouped by model and dataset
        summary = results_df.groupby(['model', 'dataset'])[metrics_cols].agg([
            'mean', 'std', 'count', 'min', 'max'
        ]).round(4)
        
        return summary
    
    def get_best_performers(
        self, 
        config: AggregationConfig,
        metric: str = 'ndcg@10',
        top_k: int = 5
    ) -> Any:
        """
        Get top performing models for a specific metric.
        
        Args:
            config: Aggregation configuration
            metric: Metric to rank by
            top_k: Number of top performers to return
            
        Returns:
            DataFrame with top performers
        """
        if not HAS_PANDAS:
            raise ImportError("pandas is required for best performers analysis")
        
        results_df = self.get_comparison_table(config)
        
        if results_df.empty or metric not in results_df.columns:
            return pd.DataFrame()
        
        # Calculate mean performance per model-dataset combination
        mean_performance = results_df.groupby(['model', 'dataset'])[metric].mean().reset_index()
        
        # Sort by metric descending and take top_k
        best_performers = mean_performance.sort_values(
            metric, ascending=False
        ).head(top_k)
        
        return best_performers
    
    def get_model_comparison(
        self,
        config: AggregationConfig,
        models: List[str] = None
    ) -> Any:
        """
        Compare specific models across datasets and metrics.
        
        Args:
            config: Aggregation configuration
            models: Specific models to compare (if None, use all)
            
        Returns:
            Dictionary with model comparison for each metric
        """
        if not HAS_PANDAS:
            raise ImportError("pandas is required for model comparison")
        
        # Update config to filter by specific models if provided
        if models:
            config.models = models
        
        results_df = self.get_comparison_table(config)
        
        if results_df.empty:
            return {}
        
        # Pivot table with models as index, datasets as columns, metrics as values
        metrics_cols = [col for col in results_df.columns 
                       if col in (config.metrics or [])]
        
        comparison_tables = {}
        for metric in metrics_cols:
            pivot = results_df.pivot_table(
                values=metric,
                index='model',
                columns='dataset',
                aggfunc='mean'
            ).round(4)
            comparison_tables[metric] = pivot
        
        return comparison_tables
    
    def get_dataset_analysis(
        self,
        config: AggregationConfig,
        datasets: List[str] = None
    ) -> Any:
        """
        Analyze performance across different datasets.
        
        Args:
            config: Aggregation configuration
            datasets: Specific datasets to analyze (if None, use all)
            
        Returns:
            DataFrame with dataset analysis
        """
        if not HAS_PANDAS:
            raise ImportError("pandas is required for dataset analysis")
        
        if datasets:
            config.datasets = datasets
        
        results_df = self.get_comparison_table(config)
        
        if results_df.empty:
            return pd.DataFrame()
        
        # Calculate statistics per dataset
        metrics_cols = [col for col in results_df.columns 
                       if col in (config.metrics or [])]
        
        dataset_stats = results_df.groupby('dataset')[metrics_cols].agg([
            'mean', 'std', 'min', 'max', 'count'
        ]).round(4)
        
        return dataset_stats
    
    def get_time_series_analysis(
        self,
        config: AggregationConfig,
        time_window: Optional[timedelta] = None
    ) -> Dict[str, Any]:
        """
        Analyze performance trends over time.
        
        Args:
            config: Aggregation configuration
            time_window: Time window to analyze (if None, use all time)
            
        Returns:
            Dictionary with time series analysis
        """
        results = []
        
        # Get all experiments matching filters
        experiments = self.storage.list_experiments()
        if config.experiments:
            experiments = [exp for exp in experiments if exp in config.experiments]
        
        for exp_name in experiments:
            exp_results = self.storage.get_experiment_results(exp_name)
            for run in exp_results:
                # Apply filters
                if config.models and run.model_name not in config.models:
                    continue
                if config.datasets and run.dataset_name not in config.datasets:
                    continue
                if not config.include_failed and not run.success:
                    continue
                
                # Apply time window filter
                if time_window:
                    cutoff_time = datetime.now() - time_window
                    if run.created_at < cutoff_time:
                        continue
                
                results.append({
                    'timestamp': run.created_at,
                    'experiment': exp_name,
                    'model': run.model_name,
                    'dataset': run.dataset_name,
                    'metrics': run.metrics
                })
        
        # Basic analysis without pandas
        if not results:
            return {}
        
        # Group by time periods (daily)
        daily_stats = defaultdict(list)
        for result in results:
            day = result['timestamp'].date()
            daily_stats[day].append(result)
        
        # Calculate daily averages for each metric
        time_series = {}
        for metric in config.metrics or ['ndcg@10']:
            daily_averages = {}
            for day, day_results in daily_stats.items():
                metric_values = []
                for result in day_results:
                    if result['metrics'] and metric in result['metrics']:
                        metric_values.append(result['metrics'][metric])
                
                if metric_values:
                    daily_averages[day] = statistics.mean(metric_values)
            
            time_series[metric] = daily_averages
        
        return {
            'time_series': time_series,
            'total_runs': len(results),
            'date_range': {
                'start': min(r['timestamp'] for r in results).date(),
                'end': max(r['timestamp'] for r in results).date()
            }
        }
    
    def get_correlation_analysis(
        self,
        config: AggregationConfig
    ) -> Any:
        """
        Analyze correlations between different metrics.
        
        Args:
            config: Aggregation configuration
            
        Returns:
            DataFrame with correlation matrix
        """
        if not HAS_PANDAS:
            self.logger.warning("pandas not available, skipping correlation analysis")
            return None
        
        results_df = self.get_comparison_table(config)
        
        if results_df.empty:
            return pd.DataFrame()
        
        # Select only numeric metric columns
        metrics_cols = [col for col in results_df.columns 
                       if col in (config.metrics or [])]
        
        if len(metrics_cols) < 2:
            return pd.DataFrame()
        
        correlation_matrix = results_df[metrics_cols].corr().round(3)
        return correlation_matrix
    
    def export_aggregated_results(
        self,
        config: AggregationConfig,
        output_path: Path,
        format: str = 'json'
    ) -> None:
        """
        Export aggregated results to file.
        
        Args:
            config: Aggregation configuration
            output_path: Path to save results
            format: Export format ('json', 'csv')
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == 'json':
            # Export as JSON (works without pandas)
            results = []
            experiments = self.storage.list_experiments()
            
            if config.experiments:
                experiments = [exp for exp in experiments if exp in config.experiments]
            
            for exp_name in experiments:
                exp_results = self.storage.get_experiment_results(exp_name)
                for run in exp_results:
                    # Apply filters
                    if config.models and run.model_name not in config.models:
                        continue
                    if config.datasets and run.dataset_name not in config.datasets:
                        continue
                    if not config.include_failed and not run.success:
                        continue
                    
                    run_data = {
                        'experiment': exp_name,
                        'model': run.model_name,
                        'dataset': run.dataset_name,
                        'timestamp': run.created_at.isoformat(),
                        'status': 'completed' if run.success else 'failed',
                        'metrics': run.metrics
                    }
                    
                    # Filter metrics if specified
                    if config.metrics:
                        filtered_metrics = {
                            k: v for k, v in run_data['metrics'].items()
                            if k in config.metrics
                        }
                        run_data['metrics'] = filtered_metrics
                    
                    results.append(run_data)
            
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
                
        elif format == 'csv' and HAS_PANDAS:
            # Export as CSV (requires pandas)
            results_df = self.get_comparison_table(config)
            if not results_df.empty:
                results_df.to_csv(output_path, index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        self.logger.info(f"Exported aggregated results to {output_path}")


def create_aggregation_config(
    experiments: List[str] = None,
    models: List[str] = None,
    datasets: List[str] = None,
    metrics: List[str] = None,
    include_failed: bool = False
) -> AggregationConfig:
    """
    Convenience function to create aggregation configuration.
    
    Args:
        experiments: Filter by experiment names
        models: Filter by model names
        datasets: Filter by dataset names
        metrics: Metrics to aggregate
        include_failed: Include failed runs
        
    Returns:
        AggregationConfig instance
    """
    return AggregationConfig(
        experiments=experiments,
        models=models,
        datasets=datasets,
        metrics=metrics,
        include_failed=include_failed
    )


def create_aggregator(storage_path: Union[str, Path]) -> ResultsAggregator:
    """
    Create ResultsAggregator with default storage.
    
    Args:
        storage_path: Path to storage directory
        
    Returns:
        ResultsAggregator instance
    """
    storage = ResultsStorage(storage_path)
    return ResultsAggregator(storage)
