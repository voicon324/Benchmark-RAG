"""
NewAIBench Models

This package provides base classes and implementations for various retrieval models
in the NewAIBench framework.
"""

# Version information
__version__ = "1.0.0"

# Import base classes
from .base import (
    BaseRetrievalModel,
    ModelType,
    ModelConfig
)

# Import concrete implementations
from .sparse import BM25Model
from .optimized_sparse import OptimizedBM25Model
from .dense import (
    DenseTextRetriever,
    SentenceBERTModel,
    DPRModel,
    TransformersModel
)

# Import image retrieval models
from .image_retrieval import (
    OCRBasedDocumentRetriever,
    ImageEmbeddingDocumentRetriever,
    MultimodalDocumentRetriever
)

# Import ColVintern retrieval model
from .colvintern_retrieval import ColVinternDocumentRetriever

# Import ColPali retrieval model
from .colpali_retrieval import ColPaliDocumentRetriever

# Public API
__all__ = [
    # Base classes
    "BaseRetrievalModel",
    "ModelType",
    "ModelConfig",
    
    # Sparse models
    "BM25Model",
    "OptimizedBM25Model",
    
    # Dense models
    "DenseTextRetriever",
    "SentenceBERTModel", 
    "DPRModel",
    "TransformersModel",
    
    # Image retrieval models
    "OCRBasedDocumentRetriever",
    "ImageEmbeddingDocumentRetriever", 
    "MultimodalDocumentRetriever",
    
    # ColVintern retrieval model
    "ColVinternDocumentRetriever",
    
    # ColPali retrieval model  
    "ColPaliDocumentRetriever"
]