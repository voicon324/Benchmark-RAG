"""
Base dataset loader for NewAIBench framework.

This module provides the foundational classes and interfaces for loading
various types of datasets in the NewAIBench information retrieval framework.
"""

import os
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple, Protocol
from collections import defaultdict

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class DatasetConfig:
    """Configuration class for dataset loading parameters.
    
    This class encapsulates all configuration parameters needed for dataset loading,
    providing type safety and validation for dataset operations.
    
    Attributes:
        dataset_path: Path to the dataset directory or file
        corpus_file: Name or path of the corpus file
        queries_file: Name or path of the queries file
        qrels_file: Name or path of the relevance judgments file
        format_type: Format of the dataset ('jsonl', 'tsv', 'json', 'csv')
        encoding: Text encoding for file reading (default: 'utf-8')
        preprocessing_options: Dictionary of preprocessing options
        validation_enabled: Whether to enable data validation
        cache_enabled: Whether to enable caching of loaded data
        max_samples: Maximum number of samples to load (None for all) - legacy parameter
        max_corpus_samples: Maximum number of corpus documents to load (None for all)
        max_query_samples: Maximum number of queries to load (None for all)
        metadata: Additional metadata for the dataset
    """
    dataset_path: Union[str, Path]
    corpus_file: str = "corpus.jsonl"
    queries_file: str = "queries.jsonl"
    qrels_file: str = "qrels.txt"
    format_type: str = "jsonl"
    encoding: str = "utf-8"
    preprocessing_options: Dict[str, Any] = field(default_factory=dict)
    validation_enabled: bool = True
    cache_enabled: bool = True
    max_samples: Optional[int] = None  # Legacy support - applies to both corpus and queries
    max_corpus_samples: Optional[int] = None  # Specific limit for corpus
    max_query_samples: Optional[int] = None   # Specific limit for queries
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self.dataset_path = Path(self.dataset_path)
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset path does not exist: {self.dataset_path}")
        
        if self.format_type not in ["jsonl", "tsv", "json", "csv"]:
            raise ValueError(f"Unsupported format type: {self.format_type}")


class DataValidationError(Exception):
    """Custom exception for data validation errors."""
    pass


class DatasetLoadingError(Exception):
    """Custom exception for dataset loading errors."""
    pass


class DatasetLoader(Protocol):
    """Protocol defining the interface for dataset loaders."""
    
    def load_corpus(self) -> Dict[str, Dict[str, Any]]:
        """Load the document corpus."""
        ...
    
    def load_queries(self) -> Dict[str, str]:
        """Load the query set."""
        ...
    
    def load_qrels(self) -> Dict[str, Dict[str, int]]:
        """Load the relevance judgments."""
        ...


class BaseDatasetLoader(ABC):
    """Abstract base class for all dataset loaders in NewAIBench.
    
    This class provides the fundamental interface and common functionality
    for loading different types of information retrieval datasets. All specific
    dataset loaders should inherit from this class.
    
    The class implements the Template Method pattern, providing a consistent
    interface while allowing subclasses to customize specific loading behaviors.
    
    Attributes:
        config: Configuration object containing dataset parameters
        _cache: Internal cache for loaded data
        _corpus: Cached corpus data
        _queries: Cached queries data
        _qrels: Cached relevance judgments data
    """
    
    def __init__(self, config: DatasetConfig) -> None:
        """Initialize the dataset loader with configuration.
        
        Args:
            config: DatasetConfig object containing loading parameters
            
        Raises:
            ValueError: If config is invalid
            FileNotFoundError: If dataset path doesn't exist
        """
        if not isinstance(config, DatasetConfig):
            raise ValueError("config must be an instance of DatasetConfig")
        
        self.config = config
        self._cache: Dict[str, Any] = {}
        self._corpus: Optional[Dict[str, Dict[str, Any]]] = None
        self._queries: Optional[Dict[str, str]] = None
        self._qrels: Optional[Dict[str, Dict[str, int]]] = None
        
        logger.info(f"Initialized {self.__class__.__name__} with dataset: {config.dataset_path}")
    
    @abstractmethod
    def load_corpus(self) -> Dict[str, Dict[str, Any]]:
        """Load the document corpus from the dataset.
        
        This method should be implemented by subclasses to handle
        specific corpus formats and structures.
        
        Returns:
            Dictionary mapping document IDs to document dictionaries.
            Each document dict should contain at least 'text' field.
            
        Raises:
            DatasetLoadingError: If corpus cannot be loaded
            DataValidationError: If loaded data fails validation
        """
        pass
    
    @abstractmethod
    def load_queries(self) -> Dict[str, str]:
        """Load the query set from the dataset.
        
        This method should be implemented by subclasses to handle
        specific query formats.
        
        Returns:
            Dictionary mapping query IDs to query text strings.
            
        Raises:
            DatasetLoadingError: If queries cannot be loaded
            DataValidationError: If loaded data fails validation
        """
        pass
    
    @abstractmethod
    def load_qrels(self) -> Dict[str, Dict[str, int]]:
        """Load the relevance judgments from the dataset.
        
        This method should be implemented by subclasses to handle
        specific qrels formats.
        
        Returns:
            Nested dictionary: {query_id: {doc_id: relevance_score}}
            
        Raises:
            DatasetLoadingError: If qrels cannot be loaded
            DataValidationError: If loaded data fails validation
        """
        pass
    
    def load_all(self) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, str], Dict[str, Dict[str, int]]]:
        """Load all dataset components (corpus, queries, qrels).
        
        This is a convenience method that loads all three main components
        of an IR dataset in the correct order with proper caching.
        
        Returns:
            Tuple of (corpus, queries, qrels)
            
        Raises:
            DatasetLoadingError: If any component cannot be loaded
        """
        logger.info("Loading all dataset components...")
        
        try:
            corpus = self.load_corpus()
            queries = self.load_queries()
            qrels = self.load_qrels()
            
            logger.info(f"Successfully loaded dataset: {len(corpus)} docs, {len(queries)} queries, {len(qrels)} qrels")
            return corpus, queries, qrels
            
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise DatasetLoadingError(f"Failed to load complete dataset: {e}") from e
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the loaded dataset.
        
        Returns:
            Dictionary containing dataset statistics including counts,
            average lengths, and other relevant metrics.
        """
        stats = {
            "dataset_path": str(self.config.dataset_path),
            "format_type": self.config.format_type,
            "encoding": self.config.encoding,
        }
        
        if self._corpus is not None:
            stats["corpus_size"] = len(self._corpus)
            stats["total_documents"] = len(self._corpus)
            if self._corpus:
                # Calculate average document length
                doc_lengths = [len(doc.get("text", "").split()) for doc in self._corpus.values()]
                stats["avg_doc_length"] = sum(doc_lengths) / len(doc_lengths) if doc_lengths else 0
                stats["max_doc_length"] = max(doc_lengths) if doc_lengths else 0
                stats["min_doc_length"] = min(doc_lengths) if doc_lengths else 0
        
        if self._queries is not None:
            stats["queries_count"] = len(self._queries)
            stats["total_queries"] = len(self._queries)
            if self._queries:
                # Calculate average query length
                query_lengths = [len(query.split()) for query in self._queries.values()]
                stats["avg_query_length"] = sum(query_lengths) / len(query_lengths) if query_lengths else 0
        
        if self._qrels is not None:
            stats["qrels_count"] = len(self._qrels)
            stats["total_qrels"] = len(self._qrels)
            if self._qrels:
                # Calculate relevance statistics
                total_judgments = sum(len(judgments) for judgments in self._qrels.values())
                stats["total_judgments"] = total_judgments
                
                # Count by relevance level
                relevance_counts = defaultdict(int)
                for judgments in self._qrels.values():
                    for score in judgments.values():
                        relevance_counts[score] += 1
                stats["relevance_distribution"] = dict(relevance_counts)
        
        return stats
    
    def validate_data(self, corpus: Dict[str, Dict[str, Any]], 
                     queries: Dict[str, str], 
                     qrels: Dict[str, Dict[str, int]]) -> bool:
        """Validate the consistency and integrity of loaded data.
        
        Args:
            corpus: Loaded corpus data
            queries: Loaded queries data
            qrels: Loaded relevance judgments
            
        Returns:
            True if validation passes
            
        Raises:
            DataValidationError: If validation fails
        """
        if not self.config.validation_enabled:
            logger.debug("Data validation is disabled")
            return True
        
        logger.info("Validating dataset consistency...")
        
        # Validate corpus
        if not corpus:
            raise DataValidationError("Corpus is empty")
        
        for doc_id, doc in corpus.items():
            if not isinstance(doc_id, str):
                raise DataValidationError(f"Document ID must be string: {doc_id}")
            if not isinstance(doc, dict):
                raise DataValidationError(f"Document must be dict: {doc_id}")
            if "text" not in doc:
                raise DataValidationError(f"Document missing 'text' field: {doc_id}")
        
        # Validate queries
        if not queries:
            raise DataValidationError("Queries are empty")
        
        for query_id, query_text in queries.items():
            if not isinstance(query_id, str):
                raise DataValidationError(f"Query ID must be string: {query_id}")
            if not isinstance(query_text, str):
                raise DataValidationError(f"Query text must be string: {query_id}")
        
        # Validate qrels
        if not qrels:
            raise DataValidationError("Qrels are empty")
        
        corpus_ids = set(corpus.keys())
        query_ids = set(queries.keys())
        
        for query_id, judgments in qrels.items():
            if str(query_id) not in query_ids:
                continue
                logger.warning(f"Qrel query ID not found in queries: {query_id}")
            
            for doc_id, score in judgments.items():
                if str(doc_id) not in corpus_ids:
                    continue
                    logger.warning(f"Qrel document ID not found in corpus: {doc_id}")
                    pass
                if not isinstance(score, int):
                    raise DataValidationError(f"Relevance score must be integer: {query_id}-{doc_id}")
        
        logger.info("Data validation completed successfully")
        return True
    
    def _read_file_safely(self, file_path: Path, max_retries: int = 3) -> str:
        """Safely read a file with error handling and retries.
        
        Args:
            file_path: Path to the file to read
            max_retries: Maximum number of retry attempts
            
        Returns:
            File content as string
            
        Raises:
            DatasetLoadingError: If file cannot be read after retries
        """
        for attempt in range(max_retries):
            try:
                with open(file_path, 'r', encoding=self.config.encoding) as f:
                    return f.read()
            except UnicodeDecodeError as e:
                if attempt == max_retries - 1:
                    raise DatasetLoadingError(f"Cannot decode file {file_path} with encoding {self.config.encoding}: {e}")
                logger.warning(f"Encoding error on attempt {attempt + 1}, retrying...")
            except Exception as e:
                if attempt == max_retries - 1:
                    raise DatasetLoadingError(f"Cannot read file {file_path}: {e}")
                logger.warning(f"Read error on attempt {attempt + 1}, retrying...")
        
        raise DatasetLoadingError(f"Failed to read file after {max_retries} attempts: {file_path}")
    
    def _apply_preprocessing(self, text: str) -> str:
        """Apply preprocessing options to text.
        
        Args:
            text: Input text to preprocess
            
        Returns:
            Preprocessed text
        """
        if not self.config.preprocessing_options:
            return text
        
        processed_text = text
        
        # Apply lowercase if enabled
        if self.config.preprocessing_options.get("lowercase", False):
            processed_text = processed_text.lower()
        
        # Remove special characters if enabled
        if self.config.preprocessing_options.get("remove_special_chars", False):
            import re
            processed_text = re.sub(r'[^\w\s]', ' ', processed_text)
        
        # Normalize whitespace if enabled
        if self.config.preprocessing_options.get("normalize_whitespace", True):
            import re
            processed_text = re.sub(r'\s+', ' ', processed_text).strip()
        
        return processed_text
    
    def clear_cache(self) -> None:
        """Clear all cached data to free memory."""
        self._cache.clear()
        self._corpus = None
        self._queries = None
        self._qrels = None
        logger.info("Dataset cache cleared")
    
    def __repr__(self) -> str:
        """String representation of the dataset loader."""
        return f"{self.__class__.__name__}(dataset_path='{self.config.dataset_path}')"