"""
Storage system for NewAIBench benchmark results.

This module provides classes and functions to:
- Store experiment results in an organized manner
- Manage metadata and configuration information
- Provide interface for accessing and querying results

Storage structure:
results/
└── experiments/
    └── {experiment_name}/
        ├── metadata.json
        └── runs/
            └── {model}_{dataset}_{timestamp}/
                ├── evaluation.json
                └── run_file.trec (optional)
"""

import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict

import pandas as pd


@dataclass
class ExperimentMetadata:
    """Metadata for an experiment."""
    experiment_id: str
    experiment_name: str
    description: str
    created_at: datetime
    author: Optional[str] = None
    tags: List[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []


@dataclass
class EvaluationResults:
    """Results from an evaluation run."""
    metrics: Dict[str, float]
    execution_time: float
    additional_info: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.additional_info is None:
            self.additional_info = {}


@dataclass
class RunResult:
    """Result from a single experiment run (model + dataset)."""
    run_id: str
    experiment_id: str
    model_name: str
    dataset_name: str
    metrics: Dict[str, float]
    execution_time: float
    created_at: datetime
    config: Dict[str, Any]
    metadata: Dict[str, Any]
    run_file_path: Optional[str] = None
    error_message: Optional[str] = None
    success: bool = True


class ResultsStorage:
    """Main class for managing benchmark results storage."""
    
    def __init__(self, base_path: Union[str, Path]):
        """
        Initialize results storage.
        
        Args:
            base_path: Base directory for storing results
        """
        self.base_path = Path(base_path)
        self.experiments_path = self.base_path / "experiments"
        
        # Create directories if they don't exist
        self.experiments_path.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
    
    def store_result(
        self,
        experiment_name: str,
        model: str,
        dataset: str,
        metrics: Dict[str, float],
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Store a single benchmark result.
        
        Args:
            experiment_name: Name of the experiment
            model: Model identifier
            dataset: Dataset identifier
            metrics: Dictionary of evaluation metrics
            metadata: Optional metadata dictionary
            
        Returns:
            Path to stored result
        """
        timestamp = datetime.now()
        run_id = f"{model}_{dataset}_{timestamp.strftime('%Y%m%d_%H%M%S')}"
        
        # Create experiment directory
        exp_dir = self.experiments_path / experiment_name
        exp_dir.mkdir(exist_ok=True)
        
        # Create runs directory
        runs_dir = exp_dir / "runs"
        runs_dir.mkdir(exist_ok=True)
        
        # Create run directory
        run_dir = runs_dir / run_id
        run_dir.mkdir(exist_ok=True)
        
        # Create run result
        result = RunResult(
            run_id=run_id,
            experiment_id=experiment_name,
            model_name=model,
            dataset_name=dataset,
            metrics=metrics,
            execution_time=metadata.get('execution_time', 0.0) if metadata else 0.0,
            created_at=timestamp,
            config=metadata.get('config', {}) if metadata else {},
            metadata=metadata or {},
            success=True
        )
        
        # Save result
        eval_path = run_dir / "evaluation.json"
        with open(eval_path, 'w', encoding='utf-8') as f:
            result_dict = asdict(result)
            result_dict['created_at'] = result.created_at.isoformat()
            json.dump(result_dict, f, indent=2, ensure_ascii=False)
        
        # Save experiment metadata if it doesn't exist
        metadata_path = exp_dir / "metadata.json"
        if not metadata_path.exists():
            exp_metadata = ExperimentMetadata(
                experiment_id=experiment_name,
                experiment_name=experiment_name,
                description=metadata.get('description', '') if metadata else '',
                created_at=timestamp,
                author=metadata.get('author') if metadata else None
            )
            
            with open(metadata_path, 'w', encoding='utf-8') as f:
                metadata_dict = asdict(exp_metadata)
                metadata_dict['created_at'] = exp_metadata.created_at.isoformat()
                json.dump(metadata_dict, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Stored result: {eval_path}")
        return str(eval_path)
    
    def store_experiment_metadata(self, metadata: ExperimentMetadata) -> str:
        """
        Store experiment metadata.
        
        Args:
            metadata: ExperimentMetadata object
            
        Returns:
            Path to stored metadata file
        """
        exp_dir = self.experiments_path / metadata.experiment_id
        exp_dir.mkdir(exist_ok=True)
        
        metadata_path = exp_dir / "metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            metadata_dict = asdict(metadata)
            metadata_dict['created_at'] = metadata.created_at.isoformat()
            json.dump(metadata_dict, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Stored experiment metadata: {metadata_path}")
        return str(metadata_path)
    
    def get_experiment_metadata(self, experiment_id: str) -> Optional[ExperimentMetadata]:
        """
        Get experiment metadata.
        
        Args:
            experiment_id: ID of the experiment
            
        Returns:
            ExperimentMetadata object if found, None otherwise
        """
        metadata_path = self.experiments_path / experiment_id / "metadata.json"
        if not metadata_path.exists():
            return None
        
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            data['created_at'] = datetime.fromisoformat(data['created_at'])
            return ExperimentMetadata(**data)
        except Exception as e:
            self.logger.warning(f"Failed to load metadata from {metadata_path}: {e}")
            return None
    
    def store_run_result(self, run_result: RunResult) -> str:
        """
        Store a run result.
        
        Args:
            run_result: RunResult object
            
        Returns:
            Path to stored result file
        """
        # Create experiment directory
        exp_dir = self.experiments_path / run_result.experiment_id
        exp_dir.mkdir(exist_ok=True)
        
        # Create runs directory
        runs_dir = exp_dir / "runs"
        runs_dir.mkdir(exist_ok=True)
        
        # Create run directory
        run_dir = runs_dir / run_result.run_id
        run_dir.mkdir(exist_ok=True)
        
        # Save result
        eval_path = run_dir / "evaluation.json"
        with open(eval_path, 'w', encoding='utf-8') as f:
            result_dict = asdict(run_result)
            result_dict['created_at'] = run_result.created_at.isoformat()
            json.dump(result_dict, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Stored run result: {eval_path}")
        return str(eval_path)
    
    def get_run_results(self, experiment_id: str) -> List[RunResult]:
        """
        Get all run results for an experiment.
        
        Args:
            experiment_id: ID of the experiment
            
        Returns:
            List of RunResult objects
        """
        return self.get_experiment_results(experiment_id)
    
    def get_experiment_results(self, experiment_name: str) -> List[RunResult]:
        """
        Get all results for an experiment.
        
        Args:
            experiment_name: Name of the experiment
            
        Returns:
            List of RunResult objects
        """
        runs_dir = self.experiments_path / experiment_name / "runs"
        if not runs_dir.exists():
            return []
        
        results = []
        for run_dir in runs_dir.iterdir():
            if run_dir.is_dir():
                eval_path = run_dir / "evaluation.json"
                if eval_path.exists():
                    try:
                        with open(eval_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        
                        data['created_at'] = datetime.fromisoformat(data['created_at'])
                        results.append(RunResult(**data))
                    except Exception as e:
                        self.logger.warning(f"Failed to load result from {eval_path}: {e}")
        
        return sorted(results, key=lambda x: x.created_at, reverse=True)
    
    def list_experiments(self) -> List[str]:
        """
        List all experiment names.
        
        Returns:
            List of experiment names
        """
        experiments = []
        for exp_dir in self.experiments_path.iterdir():
            if exp_dir.is_dir():
                experiments.append(exp_dir.name)
        
        return sorted(experiments)
    
    def get_all_results(self) -> List[RunResult]:
        """
        Get all results from all experiments.
        
        Returns:
            List of all RunResult objects
        """
        all_results = []
        for experiment_name in self.list_experiments():
            results = self.get_experiment_results(experiment_name)
            all_results.extend(results)
        
        return sorted(all_results, key=lambda x: x.created_at, reverse=True)
    
    def query_results(
        self,
        experiments: Optional[List[str]] = None,
        models: Optional[List[str]] = None,
        datasets: Optional[List[str]] = None,
        metrics: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Query results with filtering.
        
        Args:
            experiments: Filter by experiment names
            models: Filter by model names
            datasets: Filter by dataset names
            metrics: Filter by specific metrics
            
        Returns:
            DataFrame containing filtered results
        """
        all_results = self.get_all_results()
        
        # Convert to list of dictionaries
        data = []
        for result in all_results:
            if not result.success:
                continue
                
            # Apply filters
            if experiments and result.experiment_id not in experiments:
                continue
            if models and result.model_name not in models:
                continue
            if datasets and result.dataset_name not in datasets:
                continue
            
            # Create result dictionary
            result_dict = {
                'experiment': result.experiment_id,
                'model': result.model_name,
                'dataset': result.dataset_name,
                'created_at': result.created_at,
                'execution_time': result.execution_time,
            }
            
            # Add metrics
            for metric_name, metric_value in result.metrics.items():
                if metrics is None or metric_name in metrics:
                    result_dict[metric_name] = metric_value
            
            data.append(result_dict)
        
        return pd.DataFrame(data)
    
    def delete_experiment(self, experiment_name: str) -> bool:
        """
        Delete an experiment and all its results.
        
        Args:
            experiment_name: Name of experiment to delete
            
        Returns:
            True if successful, False otherwise
        """
        exp_dir = self.experiments_path / experiment_name
        if exp_dir.exists():
            try:
                shutil.rmtree(exp_dir)
                self.logger.info(f"Deleted experiment: {experiment_name}")
                return True
            except Exception as e:
                self.logger.error(f"Failed to delete experiment {experiment_name}: {e}")
                return False
        return False