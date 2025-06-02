"""
Main experiment runner implementation for NewAIBench.

This module provides the ExperimentRunner class that orchestrates the execution
of benchmark experiments across different models and datasets.
"""

import json
import logging
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict

# NewAIBench imports
from ..datasets import create_dataset_loader, DatasetConfig, DocumentImageDatasetConfig
from ..models import BaseRetrievalModel, BM25Model, DenseTextRetriever
from ..models.image_retrieval import OCRBasedDocumentRetriever, ImageEmbeddingDocumentRetriever
from ..models.colvintern_retrieval import ColVinternDocumentRetriever
from ..evaluation import Evaluator, EvaluationConfig

from .config import ExperimentConfig, ModelConfiguration, DatasetConfiguration


class ExperimentError(Exception):
    """Exception raised during experiment execution."""
    pass


@dataclass
class ExperimentResult:
    """Result of a single experiment run (model + dataset combination).
    
    Attributes:
        model_name: Name of the model used
        dataset_name: Name of the dataset used
        metrics: Computed evaluation metrics
        run_file_path: Path to saved run file
        execution_time: Time taken for the experiment
        index_time: Time taken to index the corpus
        retrieval_time: Time taken for actual retrieval (excluding indexing)
        metadata: Additional metadata about the experiment
        error: Error message if experiment failed
        success: Whether the experiment completed successfully
    """
    model_name: str
    dataset_name: str
    metrics: Dict[str, float]
    run_file_path: Optional[str] = None
    execution_time: float = 0.0
    index_time: float = 0.0
    retrieval_time: float = 0.0
    metadata: Dict[str, Any] = None
    error: Optional[str] = None
    success: bool = True
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class ExperimentRunner:
    """Main experiment runner class.
    
    This class orchestrates the execution of benchmark experiments by:
    1. Loading and configuring datasets
    2. Initializing and configuring models  
    3. Running retrieval experiments
    4. Evaluating results
    5. Saving outputs and logs
    """
    
    def __init__(self, config: ExperimentConfig):
        """Initialize the experiment runner.
        
        Args:
            config: Experiment configuration
        """
        self.config = config
        self.logger = self._setup_logging()
        self.results: List[ExperimentResult] = []
        
        # Validate configuration
        self.config.validate_cross_compatibility()
        
        # Create experiment directory
        self.experiment_dir = Path(config.output.output_dir) / config.output.experiment_name
        self.experiment_dir.mkdir(parents=True, exist_ok=config.output.overwrite)
        
        # Save experiment configuration
        self._save_experiment_config()
        
        self.logger.info(f"Initialized experiment: {config.output.experiment_name}")
        self.logger.info(f"Output directory: {self.experiment_dir}")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the experiment."""
        logger = logging.getLogger(f"experiment.{self.config.output.experiment_name}")
        logger.setLevel(getattr(logging, self.config.output.log_level))
        
        # Remove existing handlers
        logger.handlers.clear()
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # File handler (will be created when experiment starts)
        return logger
    
    def _save_experiment_config(self):
        """Save experiment configuration to file."""
        config_path = self.experiment_dir / "experiment_config.json"
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(self.config.to_dict(), f, indent=2, ensure_ascii=False)
        self.logger.info(f"Saved experiment configuration to {config_path}")
    
    def _create_model(self, model_config: ModelConfiguration) -> BaseRetrievalModel:
        """Create and configure a model instance.
        
        Args:
            model_config: Model configuration
            
        Returns:
            Configured model instance
            
        Raises:
            ExperimentError: If model creation fails
        """
        try:
            if model_config.type == 'sparse':
                # Create BM25 model
                config = {
                    'name': model_config.name,
                    'k1': model_config.parameters.get('k1', 1.2),
                    'b': model_config.parameters.get('b', 0.75),
                    'device': model_config.device,
                    **model_config.parameters
                }
                model = BM25Model(config)
                
            elif model_config.type == 'dense':
                # Create dense text retriever
                config = {
                    'name': model_config.name,
                    'model_name_or_path': model_config.model_name_or_path,
                    'device': model_config.device,
                    'batch_size': model_config.batch_size,
                    'max_seq_length': model_config.max_seq_length,
                    'parameters': model_config.parameters
                }
                model = DenseTextRetriever(config)
                
            elif model_config.type == 'image_retrieval':
                # Determine which image retrieval model to use
                retrieval_method = model_config.parameters.get('retrieval_method', 'ocr')
                
                config = {
                    'name': model_config.name,
                    'device': model_config.device,
                    'batch_size': model_config.batch_size,
                    **model_config.parameters
                }
                
                if retrieval_method == 'ocr':
                    model = OCRBasedDocumentRetriever(config)
                elif retrieval_method == 'embedding':
                    config['model_name_or_path'] = model_config.model_name_or_path
                    model = ImageEmbeddingDocumentRetriever(config)
                else:
                    raise ValueError(f"Unknown image retrieval method: {retrieval_method}")
                    
            elif model_config.type == 'multimodal':
                # Create multimodal retriever (e.g., ColVintern)
                config = {
                    'name': model_config.name,
                    'model_name_or_path': model_config.model_name_or_path,
                    'device': model_config.device,
                    'batch_size': model_config.batch_size,
                    'max_seq_length': model_config.max_seq_length,
                    'parameters': model_config.parameters
                }
                model = ColVinternDocumentRetriever(config)
                
            else:
                raise ValueError(f"Unknown model type: {model_config.type}")
            
            self.logger.info(f"Created model: {model_config.name} ({model_config.type})")
            return model
            
        except Exception as e:
            raise ExperimentError(f"Failed to create model '{model_config.name}': {str(e)}")
    
    def _create_dataset(self, dataset_config: DatasetConfiguration):
        """Create and configure a dataset loader.
        
        Args:
            dataset_config: Dataset configuration
            
        Returns:
            Tuple of (loader, corpus, queries, qrels)
            
        Raises:
            ExperimentError: If dataset loading fails
        """
        try:
            # Create base dataset config
            if dataset_config.type == 'text':
                config = DatasetConfig(
                    dataset_path=dataset_config.data_dir,
                    max_samples=dataset_config.max_samples,
                    **dataset_config.config_overrides
                )
                loader = create_dataset_loader('text', config)
                
            elif dataset_config.type == 'image':
                config = DocumentImageDatasetConfig(
                    dataset_path=dataset_config.data_dir,
                    max_samples=dataset_config.max_samples,
                    **dataset_config.config_overrides
                )
                loader = create_dataset_loader('image', config)
                
            elif dataset_config.type in ['huggingface', 'hf']:
                from newaibench.datasets import HuggingFaceDatasetConfig
                config = HuggingFaceDatasetConfig(
                    dataset_path=dataset_config.data_dir,
                    max_samples=dataset_config.max_samples,
                    **dataset_config.config_overrides
                )
                loader = create_dataset_loader('huggingface', config)
                
            else:
                raise ValueError(f"Unknown dataset type: {dataset_config.type}")
            
            # Load dataset components
            self.logger.info(f"Loading dataset: {dataset_config.name}")
            start_time = time.time()
            
            corpus = loader.load_corpus()
            queries = loader.load_queries()
            qrels = loader.load_qrels()
            
            load_time = time.time() - start_time
            
            # Validate data
            loader.validate_data(corpus, queries, qrels)
            
            # Log statistics
            stats = loader.get_statistics()
            self.logger.info(f"Loaded dataset '{dataset_config.name}' in {load_time:.2f}s")
            self.logger.info(f"Dataset stats: {stats['total_documents']} docs, "
                           f"{stats['total_queries']} queries, {stats['total_qrels']} qrels")
            
            return loader, corpus, queries, qrels
            
        except Exception as e:
            raise ExperimentError(f"Failed to load dataset '{dataset_config.name}': {str(e)}")
    
    def _run_retrieval(self, model: BaseRetrievalModel, corpus, queries, top_k: int) -> Tuple[Dict[str, Dict[str, float]], float, float]:
        """Run retrieval with the given model.
        
        Args:
            model: Configured model instance
            corpus: Document corpus
            queries: Query set
            top_k: Number of top documents to retrieve
            
        Returns:
            Tuple of (retrieval_results, index_time, retrieval_time)
            
        Raises:
            ExperimentError: If retrieval fails
        """
        try:
            self.logger.info(f"Running retrieval with model: {model.config.name}")
            start_time = time.time()
            
            # Load model if needed
            if hasattr(model, 'load_model') and callable(model.load_model):
                model.load_model()

            # Index corpus if needed
            index_time = 0.0
            if hasattr(model, 'index_corpus') and callable(model.index_corpus):
                self.logger.info("Indexing corpus...")
                index_start = time.time()
                model.index_corpus(corpus)
                index_time = time.time() - index_start
                self.logger.info(f"Corpus indexed in {index_time:.2f}s")
            
            # Convert queries format for model prediction
            # Dataset loader returns Dict[str, str] (query_id -> query_text)
            # Models expect List[Dict[str, str]] with 'query_id' and 'text' keys
            if isinstance(queries, dict):
                queries_list = [
                    {"query_id": query_id, "text": query_text}
                    for query_id, query_text in queries.items()
                ]
            else:
                queries_list = queries  # Already in correct format
            
            # Run predictions (time actual retrieval separately)
            retrieval_start = time.time()
            results = model.predict(queries_list, corpus, top_k=top_k)
            retrieval_time = time.time() - retrieval_start
            
            total_time = time.time() - start_time
            self.logger.info(f"Retrieval completed in {total_time:.2f}s (index: {index_time:.2f}s, retrieval: {retrieval_time:.2f}s)")
            
            return results, index_time, retrieval_time
            
        except Exception as e:
            raise ExperimentError(f"Retrieval failed: {str(e)}")
    
    def _save_run_file(self, results: Dict[str, Dict[str, float]], model_name: str, 
                      dataset_name: str) -> str:
        """Save retrieval results to run file.
        
        Args:
            results: Retrieval results
            model_name: Name of the model
            dataset_name: Name of the dataset
            
        Returns:
            Path to saved run file
        """
        # Create run file name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_filename = f"run_{model_name}_{dataset_name}_{timestamp}.{self.config.evaluation.run_file_format}"
        run_path = self.experiment_dir / "runs" / run_filename
        run_path.parent.mkdir(parents=True, exist_ok=True)
        
        if self.config.evaluation.run_file_format == 'trec':
            # Save in TREC format
            with open(run_path, 'w') as f:
                for query_id, doc_scores in results.items():
                    # Sort by score descending
                    sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
                    for rank, (doc_id, score) in enumerate(sorted_docs, 1):
                        f.write(f"{query_id} Q0 {doc_id} {rank} {score:.6f} {model_name}\n")
        
        elif self.config.evaluation.run_file_format == 'json':
            # Save in JSON format
            with open(run_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Saved run file: {run_path}")
        return str(run_path)
    
    def _evaluate_results(self, results: Dict[str, Dict[str, float]], qrels: Dict[str, Dict[str, int]]) -> Dict[str, float]:
        """Evaluate retrieval results.
        
        Args:
            results: Retrieval results
            qrels: Query relevance judgments
            
        Returns:
            Evaluation metrics
            
        Raises:
            ExperimentError: If evaluation fails
        """
        try:
            # Create evaluation configuration
            eval_config = EvaluationConfig(
                k_values=self.config.evaluation.k_values,
                relevance_threshold=self.config.evaluation.relevance_threshold,
                include_per_query=self.config.evaluation.include_per_query
            )
            
            # Create evaluator
            evaluator = Evaluator(eval_config)
            
            # Run evaluation
            self.logger.info("Evaluating results...")
            eval_results = evaluator.evaluate(qrels, results)
            
            # Extract main metrics
            metrics = eval_results['metrics']
            
            # Log key metrics
            for metric_name, value in metrics.items():
                if '@' in metric_name:  # Show @k metrics
                    self.logger.info(f"{metric_name}: {value:.4f}")
            
            return metrics
            
        except Exception as e:
            raise ExperimentError(f"Evaluation failed: {str(e)}")
    
    def _run_single_experiment(self, model_config: ModelConfiguration, 
                              dataset_config: DatasetConfiguration) -> ExperimentResult:
        """Run a single experiment with one model and one dataset.
        
        Args:
            model_config: Model configuration
            dataset_config: Dataset configuration
            
        Returns:
            Experiment result
        """
        experiment_start = time.time()
        
        try:
            self.logger.info(f"Starting experiment: {model_config.name} on {dataset_config.name}")
            
            # Create dataset
            loader, corpus, queries, qrels = self._create_dataset(dataset_config)
            
            # Create model
            model = self._create_model(model_config)
            
            # Run retrieval
            results, index_time, retrieval_time = self._run_retrieval(model, corpus, queries, self.config.evaluation.top_k)
            
            # Save run file if requested
            run_file_path = None
            if self.config.evaluation.save_run_file:
                run_file_path = self._save_run_file(results, model_config.name, dataset_config.name)
            
            # Evaluate results
            metrics = self._evaluate_results(results, qrels)
            
            execution_time = time.time() - experiment_start
            
            # Get model information including parameter count
            model_info = model.get_model_info() if hasattr(model, 'get_model_info') else {}
            
            # Create result
            result = ExperimentResult(
                model_name=model_config.name,
                dataset_name=dataset_config.name,
                metrics=metrics,
                run_file_path=run_file_path,
                execution_time=execution_time,
                index_time=index_time,
                retrieval_time=retrieval_time,
                metadata={
                    'model_type': model_config.type,
                    'dataset_type': dataset_config.type,
                    'num_documents': len(corpus),
                    'num_queries': len(queries),
                    'num_qrels': len(qrels),
                    'model_info': model_info,  # Include complete model information
                    'timing_breakdown': {
                        'total_execution_time': execution_time,
                        'index_time': index_time,
                        'retrieval_time': retrieval_time,
                        'evaluation_time': execution_time - index_time - retrieval_time
                    }
                },
                success=True
            )
            
            self.logger.info(f"Experiment completed successfully in {execution_time:.2f}s")
            return result
            
        except Exception as e:
            execution_time = time.time() - experiment_start
            error_msg = f"Experiment failed: {str(e)}"
            self.logger.error(error_msg)
            self.logger.debug(traceback.format_exc())
            
            return ExperimentResult(
                model_name=model_config.name,
                dataset_name=dataset_config.name,
                metrics={},
                execution_time=execution_time,
                index_time=0.0,  # Failed before indexing
                retrieval_time=0.0,  # Failed before retrieval
                error=error_msg,
                success=False
            )
    
    def run(self) -> List[ExperimentResult]:
        """Run all experiments defined in the configuration.
        
        Returns:
            List of experiment results
        """
        self.logger.info(f"Starting experiment suite: {self.config.output.experiment_name}")
        
        # Setup file logging now that experiment dir exists
        log_file = self.experiment_dir / "experiment.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        self.logger.addHandler(file_handler)
        
        total_experiments = len(self.config.models) * len(self.config.datasets)
        self.logger.info(f"Total experiments to run: {total_experiments}")
        
        # Run all combinations of models and datasets
        for i, model_config in enumerate(self.config.models):
            for j, dataset_config in enumerate(self.config.datasets):
                experiment_num = i * len(self.config.datasets) + j + 1
                self.logger.info(f"Running experiment {experiment_num}/{total_experiments}")
                
                result = self._run_single_experiment(model_config, dataset_config)
                self.results.append(result)
                
                # Save intermediate results if requested
                if self.config.output.save_intermediate:
                    self._save_results()
        
        # Save final results
        self._save_results()
        
        # Log summary
        self._log_summary()
        
        return self.results
    
    def _save_results(self):
        """Save experiment results to file."""
        results_path = self.experiment_dir / "results.json"
        
        # Convert results to serializable format
        results_data = {
            'experiment_config': self.config.to_dict(),
            'results': [asdict(result) for result in self.results],
            'summary': self._generate_summary()
        }
        
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Saved results to {results_path}")
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate summary statistics for the experiment."""
        if not self.results:
            return {}
        
        successful_results = [r for r in self.results if r.success]
        failed_results = [r for r in self.results if not r.success]
        
        summary = {
            'total_experiments': len(self.results),
            'successful_experiments': len(successful_results),
            'failed_experiments': len(failed_results),
            'total_execution_time': sum(r.execution_time for r in self.results),
            'average_execution_time': sum(r.execution_time for r in self.results) / len(self.results),
        }
        
        if successful_results:
            # Add timing statistics
            summary['timing_statistics'] = {
                'total_index_time': sum(r.index_time for r in successful_results),
                'total_retrieval_time': sum(r.retrieval_time for r in successful_results),
                'average_index_time': sum(r.index_time for r in successful_results) / len(successful_results),
                'average_retrieval_time': sum(r.retrieval_time for r in successful_results) / len(successful_results),
                'index_time_per_experiment': [r.index_time for r in successful_results],
                'retrieval_time_per_experiment': [r.retrieval_time for r in successful_results],
            }
            
            # Aggregate metrics across successful experiments
            all_metrics = {}
            for result in successful_results:
                for metric_name, value in result.metrics.items():
                    if metric_name not in all_metrics:
                        all_metrics[metric_name] = []
                    all_metrics[metric_name].append(value)
            
            # Compute averages
            summary['average_metrics'] = {
                metric_name: sum(values) / len(values)
                for metric_name, values in all_metrics.items()
            }
        
        if failed_results:
            summary['failed_experiments_details'] = [
                {
                    'model': r.model_name,
                    'dataset': r.dataset_name,
                    'error': r.error
                } for r in failed_results
            ]
        
        return summary
    
    def _log_summary(self):
        """Log experiment summary."""
        summary = self._generate_summary()
        
        self.logger.info("=" * 60)
        self.logger.info("EXPERIMENT SUMMARY")
        self.logger.info("=" * 60)
        self.logger.info(f"Total experiments: {summary['total_experiments']}")
        self.logger.info(f"Successful: {summary['successful_experiments']}")
        self.logger.info(f"Failed: {summary['failed_experiments']}")
        self.logger.info(f"Total execution time: {summary['total_execution_time']:.2f}s")
        self.logger.info(f"Average execution time: {summary['average_execution_time']:.2f}s")
        
        # Log timing statistics
        if 'timing_statistics' in summary:
            timing = summary['timing_statistics']
            self.logger.info("\nTiming breakdown:")
            self.logger.info(f"  Total index time: {timing['total_index_time']:.2f}s")
            self.logger.info(f"  Total retrieval time: {timing['total_retrieval_time']:.2f}s")
            self.logger.info(f"  Average index time: {timing['average_index_time']:.2f}s")
            self.logger.info(f"  Average retrieval time: {timing['average_retrieval_time']:.2f}s")
        
        if 'average_metrics' in summary:
            self.logger.info("\nAverage metrics across all successful experiments:")
            for metric_name, value in summary['average_metrics'].items():
                self.logger.info(f"  {metric_name}: {value:.4f}")
        
        if summary['failed_experiments'] > 0:
            self.logger.warning(f"\n{summary['failed_experiments']} experiments failed. Check logs for details.")
        
        self.logger.info("=" * 60)
