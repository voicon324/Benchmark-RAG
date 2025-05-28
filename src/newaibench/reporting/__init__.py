"""
NewAIBench Reporting Module

This module provides comprehensive reporting, aggregation, and visualization
capabilities for NewAIBench benchmark results.

Key Components:
- Storage: Flexible storage backends (filesystem, SQLite, dual)
- Aggregation: Result filtering, grouping, and statistical analysis  
- Reporting: Multi-format report generation (CSV, Markdown, LaTeX)
- Visualization: Chart generation with matplotlib/seaborn
- CLI: Command-line interface for all functionality
"""

from .storage import (
    ExperimentMetadata,
    EvaluationResults,
    RunResult,
    ResultsStorage
)

from .aggregator import (
    AggregationConfig,
    ResultsAggregator,
    create_aggregation_config
)

from .reporter import (
    ReportConfig,
    ReportGenerator
)

from .integration import (
    ReportingIntegration
)

__all__ = [
    # Storage
    'ExperimentMetadata',
    'RunResult',
    'ResultsStorage',
    
    # Aggregation
    'AggregationConfig',
    'ResultsAggregator',
    'create_aggregation_config',
    
    # Reporting
    'ReportConfig',
    'ReportGenerator',
    
    # Integration
    'ReportingIntegration',
]

__version__ = "1.0.0"