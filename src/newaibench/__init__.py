"""
NewAIBench - A Comprehensive AI Benchmarking Framework

This framework provides unified interfaces for dataset loading, model evaluation,
and Information Retrieval metrics computation.
"""

# Version information
__version__ = "1.0.0"

# Import core modules
from . import datasets, models, evaluation
from . import experiment  # Uncommented for running experiments

# Import key classes for convenient access
from .datasets import (
    BaseDatasetLoader,
    DatasetConfig,
    TextDatasetLoader,
    DocumentImageDatasetLoader
)

from .models import (
    BaseRetrievalModel,
    BM25Model,
    DenseTextRetriever
)

from .evaluation import (
    IRMetrics,
    EvaluationConfig,
    Evaluator,
    BatchEvaluator,
    quick_evaluate
)

from .experiment import (
    ExperimentRunner,
    ExperimentConfig,
    ModelConfiguration,
    DatasetConfiguration,
    EvaluationConfiguration,
    OutputConfiguration
)

# Public API
__all__ = [
    # Modules
    "datasets",
    "models", 
    "evaluation",
    "experiment"
    
    # Dataset classes
    "BaseDatasetLoader",
    "DatasetConfig",
    "TextDatasetLoader",
    "DocumentImageDatasetLoader",
    
    # Model classes
    "BaseRetrievalModel",
    "BM25Model", 
    "DenseTextRetriever",
    
    # Evaluation classes
    "IRMetrics",
    "EvaluationConfig",
    "Evaluator",
    "BatchEvaluator",
    "quick_evaluate",
    
    # Experiment classes
    "ExperimentRunner",
    "ExperimentConfig",
    "ModelConfiguration",
    "DatasetConfiguration", 
    "EvaluationConfiguration",
    "OutputConfiguration"
]

# Framework-level configuration
DEFAULT_EVALUATION_CONFIG = evaluation.DEFAULT_CONFIG
