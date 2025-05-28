"""
NewAIBench Experiment Module

This module provides experiment running capabilities for the NewAIBench framework,
allowing users to define and execute benchmark experiments with different datasets,
models, and evaluation configurations.
"""

from .config import (
    ExperimentConfig,
    ModelConfiguration,
    DatasetConfiguration,
    EvaluationConfiguration,
    OutputConfiguration
)

from .runner import (
    ExperimentRunner,
    ExperimentResult,
    ExperimentError
)

from .utils import (
    load_experiment_config,
    save_experiment_config,
    save_experiment_results,
    merge_configs,
    create_experiment_template,
    validate_device_availability,
    setup_experiment_logging
)

__all__ = [
    # Configuration classes
    "ExperimentConfig",
    "ModelConfiguration", 
    "DatasetConfiguration",
    "EvaluationConfiguration",
    "OutputConfiguration",
    
    # Runner classes
    "ExperimentRunner",
    "ExperimentResult",
    "ExperimentError",
    
    # Utility functions
    "load_experiment_config",
    "save_experiment_config",
    "save_experiment_results",
    "merge_configs",
    "create_experiment_template",
    "validate_device_availability",
    "setup_experiment_logging"

]

__version__ = "1.0.0"
