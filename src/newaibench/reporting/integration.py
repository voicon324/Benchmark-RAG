
"""
Integration module for connecting the reporting system with NewAIBench ExperimentRunner.

This module provides utilities to automatically capture results from experiment runs
and store them in the reporting system.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List

from .storage import ResultsStorage
from .aggregator import ResultsAggregator, AggregationConfig, create_aggregation_config
from .reporter import ReportGenerator, ReportConfig


class ReportingIntegration:
    """Integration layer between ExperimentRunner and reporting system."""
    
    def __init__(self, storage: ResultsStorage):
        """
        Initialize reporting integration.
        
        Args:
            storage: Storage backend for results
        """
        self.storage = storage
        self.aggregator = ResultsAggregator(storage)
        self.report_generator = ReportGenerator(storage, self.aggregator)
    
    def capture_experiment_result(
        self,
        experiment_name: str,
        model: str,
        dataset: str,
        metrics: Dict[str, float],
        config: Optional[Dict[str, Any]] = None,
        output_dir: Optional[str] = None
    ) -> None:
        """
        Capture and store experiment result.
        
        Args:
            experiment_name: Name of the experiment
            model: Model identifier
            dataset: Dataset identifier  
            metrics: Dictionary of evaluation metrics
            config: Optional experiment configuration
            output_dir: Optional output directory path
        """
        metadata = {
            'timestamp': datetime.now().isoformat(),
            'config': config or {},
        }
        
        if output_dir:
            metadata['output_dir'] = str(output_dir)
        
        self.storage.store_result(
            experiment_name=experiment_name,
            model=model,
            dataset=dataset,
            metrics=metrics,
            metadata=metadata
        )
    
    def capture_from_output_dir(
        self,
        output_dir: str,
        experiment_name: Optional[str] = None
    ) -> int:
        """
        Capture results from existing NewAIBench output directory.
        
        Args:
            output_dir: Path to output directory containing results
            experiment_name: Optional experiment name (inferred from path if not provided)
            
        Returns:
            Number of results captured
        """
        output_path = Path(output_dir)
        if not output_path.exists():
            raise ValueError(f"Output directory does not exist: {output_dir}")
        
        captured_count = 0
        
        # If experiment_name not provided, use directory name
        if experiment_name is None:
            experiment_name = output_path.name
        
        # Look for evaluation.json files in subdirectories
        for item in output_path.iterdir():
            if item.is_dir():
                eval_file = item / 'evaluation.json'
                if eval_file.exists():
                    try:
                        with open(eval_file, 'r') as f:
                            eval_data = json.load(f)
                        
                        # Extract model and dataset from directory name
                        dir_parts = item.name.split('_')
                        if len(dir_parts) >= 2:
                            model = dir_parts[0]
                            dataset = dir_parts[1]
                        else:
                            model = item.name
                            dataset = "unknown"
                        
                        # Store result
                        self.capture_experiment_result(
                            experiment_name=experiment_name,
                            model=model,
                            dataset=dataset,
                            metrics=eval_data.get('metrics', {}),
                            config=eval_data.get('config', {}),
                            output_dir=str(item)
                        )
                        captured_count += 1
                        
                    except Exception as e:
                        print(f"Warning: Failed to capture result from {eval_file}: {e}")
        
        return captured_count
    
    def generate_experiment_report(
        self,
        experiment_name: str,
        output_dir: str,
        formats: Optional[List[str]] = None
    ) -> List[str]:
        """
        Generate comprehensive report for a specific experiment.
        
        Args:
            experiment_name: Name of the experiment to report on
            output_dir: Directory to save report files
            formats: List of formats to generate ('csv', 'markdown', 'latex')
            
        Returns:
            List of generated file paths
        """
        if formats is None:
            formats = ['markdown', 'csv']
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create aggregation config for this experiment
        config = create_aggregation_config(
            experiments=[experiment_name],
            metrics=["ndcg@10", "map@10", "recall@10"],
            include_failed=False
        )
        
        # Create report config
        report_config = ReportConfig(
            title=f"Report for {experiment_name}",
            include_metadata=True,
            include_statistics=True,
            include_comparison=True
        )
        
        generated_files = []
        
        # Generate CSV report
        if 'csv' in formats:
            try:
                csv_path = output_path / f"{experiment_name}_report.csv"
                self.report_generator.generate_csv_report(
                    output_path=str(csv_path),
                    config=report_config,
                    aggregation_config=config
                )
                generated_files.append(str(csv_path))
            except Exception as e:
                print(f"Warning: Failed to generate CSV report: {e}")
        
        # Generate Markdown report
        if 'markdown' in formats:
            try:
                md_path = output_path / f"{experiment_name}_report.md"
                self.report_generator.generate_markdown_report(
                    output_path=str(md_path),
                    config=report_config,
                    aggregation_config=config
                )
                generated_files.append(str(md_path))
            except Exception as e:
                print(f"Warning: Failed to generate Markdown report: {e}")
        
        # Generate LaTeX report
        if 'latex' in formats:
            try:
                tex_path = output_path / f"{experiment_name}_report.tex"
                self.report_generator.generate_latex_report(
                    output_path=str(tex_path),
                    config=report_config,
                    aggregation_config=config
                )
                generated_files.append(str(tex_path))
            except Exception as e:
                print(f"Warning: Failed to generate LaTeX report: {e}")
        
        return generated_files
    
    def setup_auto_capture(
        self,
        experiment_runner,
        experiment_name: str
    ) -> None:
        """
        Setup automatic result capture for ExperimentRunner.
        
        Args:
            experiment_runner: Instance of ExperimentRunner
            experiment_name: Name for the experiment
        """
        # Store original evaluate method
        original_evaluate = experiment_runner.evaluate
        integration = self
        
        def wrapped_evaluate(model, dataset, output_dir=None):
            """Wrapped evaluate method that captures results."""
            # Call original evaluate method
            result = original_evaluate(model, dataset, output_dir)
            
            # Capture result if evaluation was successful
            if hasattr(result, 'metrics') and result.metrics:
                try:
                    integration.capture_experiment_result(
                        experiment_name=experiment_name,
                        model=model.name if hasattr(model, 'name') else str(model),
                        dataset=dataset.name if hasattr(dataset, 'name') else str(dataset),
                        metrics=result.metrics,
                        config=getattr(result, 'config', {}),
                        output_dir=output_dir
                    )
                except Exception as e:
                    print(f"Warning: Failed to capture result automatically: {e}")
            
            return result
        
        # Replace evaluate method
        experiment_runner.evaluate = wrapped_evaluate


def setup_default_reporting(results_dir: str = "./results") -> ReportingIntegration:
    """
    Setup default reporting integration with filesystem storage backend.
    
    Args:
        results_dir: Directory for storing results
        
    Returns:
        Configured ReportingIntegration instance
    """
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)
    
    # Setup filesystem storage
    storage = ResultsStorage(str(results_path))
    
    return ReportingIntegration(storage)


def migrate_existing_results(
    old_results_dir: str,
    new_results_dir: str = "./results"
) -> int:
    """
    Migrate existing NewAIBench results to new reporting system.
    
    Args:
        old_results_dir: Directory containing existing results
        new_results_dir: Directory for new reporting system storage
        
    Returns:
        Number of results migrated
    """
    integration = setup_default_reporting(new_results_dir)
    
    old_path = Path(old_results_dir)
    migrated_count = 0
    
    # Look for experiment directories
    for exp_dir in old_path.iterdir():
        if exp_dir.is_dir():
            try:
                count = integration.capture_from_output_dir(
                    str(exp_dir),
                    experiment_name=exp_dir.name
                )
                migrated_count += count
                print(f"Migrated {count} results from experiment: {exp_dir.name}")
            except Exception as e:
                print(f"Warning: Failed to migrate {exp_dir}: {e}")
    
    print(f"Total migrated: {migrated_count} results")
    return migrated_count
