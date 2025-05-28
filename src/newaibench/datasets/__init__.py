"""
NewAIBench Dataset Loaders

This package provides dataset loading capabilities for the NewAIBench framework,
supporting various information retrieval dataset formats including text and
document image datasets.
"""

# Version information
__version__ = "0.1.0"

# Import base classes first
from .base import (
    BaseDatasetLoader,
    DatasetConfig,
    DatasetLoader,
    DatasetLoadingError,
    DataValidationError
)

# Import concrete implementations
from .text import TextDatasetLoader
from .image import (
    DocumentImageDatasetLoader,
    DocumentImageDatasetConfig
)

# Public API
__all__ = [
    # Base classes and interfaces
    "BaseDatasetLoader",
    "DatasetConfig", 
    "DatasetLoader",
    "DatasetLoadingError",
    "DataValidationError",
    
    # Concrete implementations
    "TextDatasetLoader",
    "DocumentImageDatasetLoader", 
    "DocumentImageDatasetConfig",
    
    # Version
    "__version__"
]

# Convenience factory function
def create_dataset_loader(dataset_type: str, config: DatasetConfig) -> BaseDatasetLoader:
    """Create a dataset loader based on dataset type.
    
    Args:
        dataset_type: Type of dataset ('text' or 'image')
        config: Dataset configuration object
        
    Returns:
        Appropriate dataset loader instance
        
    Raises:
        ValueError: If dataset_type is not supported
    """
    if dataset_type.lower() == "text":
        return TextDatasetLoader(config)
    elif dataset_type.lower() in ["image", "document_image"]:
        if not isinstance(config, DocumentImageDatasetConfig):
            # Convert base config to image config
            config = DocumentImageDatasetConfig(
                dataset_path=config.dataset_path,
                corpus_file=config.corpus_file,
                queries_file=config.queries_file,
                qrels_file=config.qrels_file,
                format_type=config.format_type,
                encoding=config.encoding,
                preprocessing_options=config.preprocessing_options,
                validation_enabled=config.validation_enabled,
                cache_enabled=config.cache_enabled,
                max_samples=config.max_samples,
                metadata=config.metadata
            )
        return DocumentImageDatasetLoader(config)
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}. "
                        f"Supported types: 'text', 'image', 'document_image'")

# Add factory function to __all__
__all__.append("create_dataset_loader")