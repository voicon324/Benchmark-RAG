"""
Hugging Face Dataset Loader for NewAIBench framework.

This module provides a dataset loader that can load and convert datasets directly
from Hugging Face Hub to NewAIBench's standardized format (corpus.jsonl, queries.jsonl, qrels.txt).
"""

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
from collections import defaultdict

try:
    from datasets import load_dataset, Dataset, DatasetDict
    HF_DATASETS_AVAILABLE = True
except ImportError:
    HF_DATASETS_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

from .base import BaseDatasetLoader, DatasetConfig, DatasetLoadingError, DataValidationError

logger = logging.getLogger(__name__)


@dataclass
class HuggingFaceDatasetConfig(DatasetConfig):
    """Configuration class for Hugging Face dataset loading.
    
    Extends DatasetConfig with Hugging Face-specific parameters for loading
    datasets directly from the Hugging Face Hub.
    
    Attributes:
        hf_path: Path to dataset on Hugging Face Hub (e.g., "username/dataset_name")
        hf_config_name: Name of configuration/subset (optional)
        hf_split: Name of split to load (default: "train")
        corpus_feature_mapping: Dictionary to map HF columns to NewAIBench corpus fields
        queries_feature_mapping: Dictionary to map HF columns to NewAIBench query fields
        qrels_feature_mapping: Dictionary to map HF columns to NewAIBench qrels fields
        image_field_hf: Name of image field in HF dataset (for image retrieval)
        images_output_subdir: Subdirectory to save images (default: "hf_images")
        hf_use_streaming: Whether to use streaming for large datasets
        hf_trust_remote_code: Whether to trust remote code execution
        hf_auto_detect_columns: Whether to auto-detect column names
        hf_default_relevance_score: Default relevance score when not provided
        hf_separate_splits: Whether corpus, queries, qrels are in separate splits
        hf_corpus_split: Split name for corpus data
        hf_queries_split: Split name for queries data
        hf_qrels_split: Split name for qrels data
        nested_field_separator: Separator for accessing nested fields (e.g., ".")
    """
    
    # Core Hugging Face dataset parameters - Updated naming per specification
    hf_path: str = ""  # Required: e.g., "squad", "ms_marco", "username/my_dataset"
    hf_config_name: Optional[str] = None  # Optional config name for datasets with multiple configs
    hf_split: str = "train"  # Split to load: "train", "validation", "test"
    
    # Feature mapping for corpus (documents) - Updated naming per specification
    corpus_feature_mapping: Dict[str, str] = field(default_factory=lambda: {
        "doc_id": "id",               # Column containing document IDs
        "text": "text",               # Column containing document text content
        "title": "title",             # Column containing document titles (optional)
        "image": "image",             # Column containing PIL images (for image datasets)
    })
    
    # Feature mapping for queries - Updated naming per specification
    queries_feature_mapping: Dict[str, str] = field(default_factory=lambda: {
        "query_id": "id",             # Column containing query IDs
        "text": "question",           # Column containing query text
    })
    
    # Feature mapping for qrels (relevance judgments) - Updated naming per specification
    qrels_feature_mapping: Dict[str, str] = field(default_factory=lambda: {
        "query_id": "query_id",       # Column containing query IDs
        "doc_id": "doc_id",           # Column containing document IDs
        "relevance": "relevance",     # Column containing relevance scores
        "positive_doc_col": "positive_ctxs",  # Column containing list of positive document IDs
        "negative_doc_col": "negative_ctxs",  # Column containing list of negative document IDs
    })
    
    # Image handling options - Updated naming per specification
    image_field_hf: Optional[str] = None      # Name of image field in HF dataset
    images_output_subdir: str = "hf_images"   # Subdirectory for saved images
    
    # Processing options
    hf_use_streaming: bool = False    # Use streaming for large datasets
    hf_trust_remote_code: bool = False  # Trust remote code execution for some datasets
    hf_auto_detect_columns: bool = True  # Automatically detect column names if mappings fail
    hf_default_relevance_score: int = 1   # Default relevance score when not explicitly provided
    
    # Support for separate splits per component
    hf_separate_splits: bool = False          # Whether corpus, queries, qrels are in separate splits
    hf_corpus_split: Optional[str] = None     # Split name for corpus data
    hf_queries_split: Optional[str] = None    # Split name for queries data
    hf_qrels_split: Optional[str] = None      # Split name for qrels data
    
    # Support for nested field access
    nested_field_separator: str = "."         # Separator for accessing nested fields
    
    def __post_init__(self):
        """Validate Hugging Face-specific configuration."""
        super().__post_init__()
        
        if not self.hf_path:
            raise ValueError("hf_path must be provided")
        
        # Parse dataset identifier for config name if provided in format "dataset,config"
        if "," in self.hf_path and self.hf_config_name is None:
            parts = self.hf_path.split(",", 1)
            self.hf_path = parts[0]
            self.hf_config_name = parts[1].strip()
        
        # Set up separate splits if specified
        if self.hf_separate_splits:
            if not self.hf_corpus_split:
                self.hf_corpus_split = self.hf_split
            if not self.hf_queries_split:
                self.hf_queries_split = self.hf_split
            if not self.hf_qrels_split:
                self.hf_qrels_split = self.hf_split
        
        # Validate image configuration
        if self.image_field_hf and not PIL_AVAILABLE:
            logger.warning("PIL not available for image processing, image features will be disabled")
        
        logger.info(f"HuggingFace dataset config: {self.hf_path}, "
                   f"config: {self.hf_config_name}, split: {self.hf_split}")
        if self.hf_separate_splits:
            logger.info(f"Using separate splits - corpus: {self.hf_corpus_split}, "
                       f"queries: {self.hf_queries_split}, qrels: {self.hf_qrels_split}")


class HuggingFaceDatasetLoader(BaseDatasetLoader):
    """Dataset loader for datasets from Hugging Face Hub.
    
    This loader can load any dataset from Hugging Face Hub and convert it to
    NewAIBench's standardized format. It supports both text and image datasets,
    with flexible column mapping and automatic format detection.
    
    Features:
    - Load any public dataset from Hugging Face Hub
    - Flexible column mapping for different dataset structures
    - Support for text and image datasets
    - Automatic PIL image saving for image datasets
    - Streaming support for large datasets
    - Intelligent qrels generation from various relevance annotation formats
    - Auto-detection of column names when mappings fail
    
    Example usage:
        >>> config = HuggingFaceDatasetConfig(
        ...     dataset_path="/path/to/output",
        ...     hf_dataset_identifier="squad",
        ...     hf_dataset_split="validation",
        ...     hf_query_column_mapping={"query_id_col": "id", "text_col": "question"},
        ...     hf_corpus_column_mapping={"doc_id_col": "context_id", "text_col": "context"}
        ... )
        >>> loader = HuggingFaceDatasetLoader(config)
        >>> corpus, queries, qrels = loader.load_all()
    """
    
    def __init__(self, config: HuggingFaceDatasetConfig) -> None:
        """Initialize HuggingFace dataset loader.
        
        Args:
            config: HuggingFaceDatasetConfig with HF-specific settings
            
        Raises:
            ImportError: If datasets library is not available
            ValueError: If config is not HuggingFaceDatasetConfig
        """
        if not HF_DATASETS_AVAILABLE:
            raise ImportError(
                "Hugging Face datasets library not available. "
                "Please install with: pip install datasets"
            )
        
        if not isinstance(config, HuggingFaceDatasetConfig):
            raise ValueError("config must be an instance of HuggingFaceDatasetConfig")
        
        super().__init__(config)
        self.config: HuggingFaceDatasetConfig = config
        
        # Initialize dataset variables
        self._hf_dataset = None
        self._hf_corpus_dataset = None
        self._hf_queries_dataset = None
        self._hf_qrels_dataset = None
        self._detected_columns = {}
        
        logger.info(f"Initialized HuggingFaceDatasetLoader for dataset: {config.hf_path}")
    
    def _load_hf_dataset(self, component: str = None) -> Union[Dataset, DatasetDict]:
        """Load the dataset from Hugging Face Hub.
        
        Args:
            component: Specific component to load ('corpus', 'queries', 'qrels'), 
                      None for default split
        
        Returns:
            Loaded Hugging Face dataset
            
        Raises:
            DatasetLoadingError: If dataset cannot be loaded
        """
        # Handle caching for different components
        if component == 'corpus' and self._hf_corpus_dataset is not None:
            return self._hf_corpus_dataset
        elif component == 'queries' and self._hf_queries_dataset is not None:
            return self._hf_queries_dataset
        elif component == 'qrels' and self._hf_qrels_dataset is not None:
            return self._hf_qrels_dataset
        elif component is None and self._hf_dataset is not None:
            return self._hf_dataset
        
        try:
            # Determine which split to load
            if component and self.config.hf_separate_splits:
                if component == 'corpus':
                    split = self.config.hf_corpus_split
                elif component == 'queries':
                    split = self.config.hf_queries_split
                elif component == 'qrels':
                    split = self.config.hf_qrels_split
                else:
                    split = self.config.hf_split
            else:
                split = self.config.hf_split
            
            logger.info(f"Loading HF dataset: {self.config.hf_path}, split: {split}")
            
            kwargs = {
                "streaming": self.config.hf_use_streaming,
                "trust_remote_code": self.config.hf_trust_remote_code
            }
            
            if self.config.hf_config_name:
                kwargs["name"] = self.config.hf_config_name
            
            dataset = load_dataset(
                self.config.hf_path,
                split=split,
                **kwargs
            )
            
            # Cache the loaded dataset
            if component == 'corpus':
                self._hf_corpus_dataset = dataset
            elif component == 'queries':
                self._hf_queries_dataset = dataset
            elif component == 'qrels':
                self._hf_qrels_dataset = dataset
            else:
                self._hf_dataset = dataset
            
            logger.info(f"Successfully loaded HF dataset with {len(dataset) if hasattr(dataset, '__len__') else 'streaming'} samples")
            return dataset
            
        except Exception as e:
            raise DatasetLoadingError(f"Failed to load HF dataset {self.config.hf_path}: {e}")
    
    def _detect_columns(self, dataset: Dataset, target_mapping: Dict[str, str]) -> Dict[str, str]:
        """Auto-detect column names when mapping fails.
        
        Args:
            dataset: Hugging Face dataset
            target_mapping: Target column mapping
            
        Returns:
            Updated column mapping with detected columns
        """
        if not self.config.hf_auto_detect_columns:
            return target_mapping
        
        detected_mapping = target_mapping.copy()
        column_names = dataset.column_names
        
        # Common patterns for auto-detection
        detection_patterns = {
            "doc_id_col": ["id", "doc_id", "document_id", "passage_id", "_id"],
            "text_col": ["text", "content", "passage", "document", "context", "body"],
            "title_col": ["title", "heading", "name", "subject"],
            "query_id_col": ["id", "query_id", "question_id", "qid", "_id"],
            "text_col": ["question", "query", "text", "content"],
            "relevance_col": ["relevance", "label", "score", "rating"],
            "positive_doc_col": ["positive_ctxs", "positive_passages", "positive_docs"],
            "negative_doc_col": ["negative_ctxs", "negative_passages", "negative_docs"],
            "image_col": ["image", "img", "picture", "photo"]
        }
        
        for target_col, current_name in detected_mapping.items():
            if current_name not in column_names and target_col in detection_patterns:
                # Try to find a matching column
                for pattern in detection_patterns[target_col]:
                    if pattern in column_names:
                        detected_mapping[target_col] = pattern
                        logger.info(f"Auto-detected column '{pattern}' for {target_col}")
                        break
        
        return detected_mapping
    
    def _get_nested_field(self, item: Dict[str, Any], field_path: str) -> Any:
        """Access nested fields using dot notation.
        
        Args:
            item: Data item from HF dataset
            field_path: Path to field (e.g., "document.content.main_text")
            
        Returns:
            Value at nested path, or None if not found
        """
        if self.config.nested_field_separator not in field_path:
            return item.get(field_path)
        
        try:
            current = item
            for part in field_path.split(self.config.nested_field_separator):
                if isinstance(current, dict):
                    current = current.get(part)
                elif isinstance(current, (list, tuple)) and part.isdigit():
                    current = current[int(part)]
                else:
                    return None
                if current is None:
                    return None
            return current
        except (KeyError, IndexError, TypeError):
            return None
    
    def _save_image(self, image, doc_id: str) -> str:
        """Save PIL image to disk and return relative path.
        
        Args:
            image: PIL Image object
            doc_id: Document ID for filename
            
        Returns:
            Relative path to saved image
        """
        if not PIL_AVAILABLE:
            logger.warning("PIL not available, cannot save images")
            return ""
        
        if not isinstance(image, Image.Image):
            logger.warning(f"Expected PIL Image, got {type(image)}")
            return ""
        
        # Create images directory
        images_dir = self.config.dataset_path / self.config.images_output_subdir
        images_dir.mkdir(parents=True, exist_ok=True)
        
        # Save image with safe filename
        safe_filename = re.sub(r'[^\w\-_.]', '_', doc_id)
        image_path = images_dir / f"{safe_filename}.png"
        
        try:
            image.save(image_path, "PNG")
            # Return relative path from dataset root
            return f"{self.config.images_output_subdir}/{safe_filename}.png"
        except Exception as e:
            logger.error(f"Failed to save image for doc {doc_id}: {e}")
            return ""
    
    def load_corpus(self) -> Dict[str, Dict[str, Any]]:
        """Load corpus (documents) from Hugging Face dataset.
        
        Returns:
            Dictionary mapping doc_id to document data
            
        Raises:
            DatasetLoadingError: If corpus cannot be loaded
        """
        if self.config.cache_enabled and self._corpus is not None:
            return self._corpus
        
        try:
            dataset = self._load_hf_dataset('corpus')
            corpus = {}
            
            # Detect columns if auto-detection is enabled
            corpus_mapping = self._detect_columns(dataset, self.config.corpus_feature_mapping)
            
            logger.info(f"Loading corpus with column mapping: {corpus_mapping}")
            
            count = 0
            for item in dataset:
                # Apply corpus-specific sampling limit 
                corpus_limit = self.config.max_corpus_samples or self.config.max_samples
                if corpus_limit and count >= corpus_limit:
                    break
                
                # Extract document ID with nested field support
                doc_id_field = corpus_mapping.get("doc_id", "id")
                doc_id_value = self._get_nested_field(item, doc_id_field)
                
                if doc_id_value is None:
                    doc_id = f"doc_{count}"  # Generate ID if not available
                    logger.warning(f"Missing doc_id field '{doc_id_field}', using generated ID: {doc_id}")
                else:
                    doc_id = str(doc_id_value)
                
                # Extract text content with nested field support
                text_field = corpus_mapping.get("text", "text")
                text_content = self._get_nested_field(item, text_field)
                if text_content is None:
                    text_content = ""
                    logger.warning(f"Empty text content for document {doc_id}")
                
                # Create document entry
                doc_entry = {
                    "text": str(text_content),
                    "doc_id": doc_id
                }
                
                # Add title if available with nested field support
                title_field = corpus_mapping.get("title", "title")
                title_value = self._get_nested_field(item, title_field)
                if title_value:
                    doc_entry["title"] = str(title_value)
                
                # Handle images if present
                image_field = self.config.image_field_hf or corpus_mapping.get("image", "image")
                if image_field:
                    image_value = self._get_nested_field(item, image_field)
                    if image_value and PIL_AVAILABLE:
                        image_path = self._save_image(image_value, doc_id)
                        if image_path:
                            doc_entry["image_path"] = image_path
                
                # Add OCR text field (initially null as per specification)
                doc_entry["ocr_text"] = None
                
                # Add metadata for additional fields
                metadata = {}
                for key, value in item.items():
                    if key not in [doc_id_field.split(self.config.nested_field_separator)[0], 
                                  text_field.split(self.config.nested_field_separator)[0], 
                                  title_field.split(self.config.nested_field_separator)[0],
                                  image_field.split(self.config.nested_field_separator)[0] if image_field else None]:
                        # Add as metadata, but limit to simple types
                        if isinstance(value, (str, int, float, bool)):
                            metadata[key] = value
                        elif isinstance(value, (list, dict)) and len(str(value)) < 1000:
                            metadata[key] = value
                
                if metadata:
                    doc_entry["metadata"] = metadata
                
                # Apply preprocessing if configured
                if self.config.preprocessing_options:
                    doc_entry["text"] = self._apply_preprocessing(doc_entry["text"])
                
                corpus[doc_id] = doc_entry
                count += 1
            
            logger.info(f"Loaded {len(corpus)} documents from corpus")
            
            if self.config.cache_enabled:
                self._corpus = corpus
            
            return corpus
            
        except Exception as e:
            raise DatasetLoadingError(f"Failed to load corpus from HF dataset: {e}")
            
            if self.config.cache_enabled:
                self._corpus = corpus
            
            return corpus
            
        except Exception as e:
            raise DatasetLoadingError(f"Failed to load corpus from HF dataset: {e}")
    
    def load_queries(self) -> Dict[str, str]:
        """Load queries from Hugging Face dataset.
        
        Returns:
            Dictionary mapping query_id to query text
            
        Raises:
            DatasetLoadingError: If queries cannot be loaded
        """
        if self.config.cache_enabled and self._queries is not None:
            return self._queries
        
        try:
            dataset = self._load_hf_dataset('queries')
            queries = {}
            
            # Detect columns if auto-detection is enabled
            query_mapping = self._detect_columns(dataset, self.config.queries_feature_mapping)
            
            logger.info(f"Loading queries with column mapping: {query_mapping}")
            
            count = 0
            for item in dataset:
                # Apply query-specific sampling limit
                query_limit = self.config.max_query_samples or self.config.max_samples
                if query_limit and count >= query_limit:
                    break
                
                # Extract query ID with nested field support
                query_id_field = query_mapping.get("query_id", "id")
                query_id_value = self._get_nested_field(item, query_id_field)
                
                if query_id_value is None:
                    query_id = f"query_{count}"  # Generate ID if not available
                    logger.warning(f"Missing query_id field '{query_id_field}', using generated ID: {query_id}")
                else:
                    query_id = str(query_id_value)
                
                # Extract query text with nested field support
                text_field = query_mapping.get("text", "question")
                query_text = self._get_nested_field(item, text_field)
                if query_text is None:
                    logger.warning(f"Empty query text for query {query_id}")
                    query_text = ""
                
                # Apply preprocessing if configured
                if self.config.preprocessing_options:
                    query_text = self._apply_preprocessing(str(query_text))
                
                queries[query_id] = str(query_text)
                count += 1
            
            logger.info(f"Loaded {len(queries)} queries")
            
            if self.config.cache_enabled:
                self._queries = queries
            
            return queries
            
        except Exception as e:
            raise DatasetLoadingError(f"Failed to load queries from HF dataset: {e}")
    
    def load_qrels(self) -> Dict[str, Dict[str, int]]:
        """Load relevance judgments from Hugging Face dataset.
        
        This method handles various qrels formats:
        1. Explicit qrels with query_id, doc_id, relevance columns
        2. Positive/negative document lists per query
        3. Direct query-document pairs (assumes positive relevance)
        
        Returns:
            Dictionary mapping query_id to dict of doc_id -> relevance_score
            
        Raises:
            DatasetLoadingError: If qrels cannot be loaded
        """
        if self.config.cache_enabled and self._qrels is not None:
            return self._qrels
        
        try:
            dataset = self._load_hf_dataset('qrels')
            qrels = defaultdict(dict)
            
            # Detect columns if auto-detection is enabled
            qrels_mapping = self._detect_columns(dataset, self.config.qrels_feature_mapping)
            
            logger.info(f"Loading qrels with column mapping: {qrels_mapping}")
            
            count = 0
            for item in dataset:
                # Apply general sampling limit for qrels (uses max_samples as qrels typically don't need separate limits)
                if self.config.max_samples and count >= self.config.max_samples:
                    break
                
                # Extract query ID with nested field support
                query_id_field = qrels_mapping.get("query_id", "query_id")
                query_id_value = self._get_nested_field(item, query_id_field)
                
                if query_id_value is None:
                    # Try to use same ID as in queries/corpus
                    fallback_field = self.config.queries_feature_mapping.get("query_id", "id")
                    query_id_value = self._get_nested_field(item, fallback_field)
                
                if query_id_value is None:
                    query_id = f"query_{count}"
                    logger.warning(f"Missing query_id field, using generated ID: {query_id}")
                else:
                    query_id = str(query_id_value)
                
                # Method 1: Explicit relevance column with nested field support
                relevance_field = qrels_mapping.get("relevance", "relevance")
                doc_id_field = qrels_mapping.get("doc_id", "doc_id")
                
                relevance_value = self._get_nested_field(item, relevance_field)
                doc_id_value = self._get_nested_field(item, doc_id_field)
                
                if relevance_value is not None and doc_id_value is not None:
                    doc_id = str(doc_id_value)
                    relevance = int(relevance_value) if relevance_value is not None else self.config.hf_default_relevance_score
                    qrels[query_id][doc_id] = relevance
                
                # Method 2: Positive documents list with nested field support
                positive_field = qrels_mapping.get("positive_doc_col", "positive_ctxs")
                positive_docs = self._get_nested_field(item, positive_field)
                if positive_docs:
                    if isinstance(positive_docs, list):
                        for doc in positive_docs:
                            if isinstance(doc, dict) and "passage_id" in doc:
                                doc_id = str(doc["passage_id"])
                            elif isinstance(doc, dict) and "id" in doc:
                                doc_id = str(doc["id"])
                            elif isinstance(doc, str):
                                doc_id = doc
                            else:
                                doc_id = str(doc)
                            qrels[query_id][doc_id] = 2  # High relevance for positive
                
                # Method 3: Negative documents list (optional) with nested field support
                negative_field = qrels_mapping.get("negative_doc_col", "negative_ctxs")
                negative_docs = self._get_nested_field(item, negative_field)
                if negative_docs:
                    if isinstance(negative_docs, list):
                        for doc in negative_docs:
                            if isinstance(doc, dict) and "passage_id" in doc:
                                doc_id = str(doc["passage_id"])
                            elif isinstance(doc, dict) and "id" in doc:
                                doc_id = str(doc["id"])
                            elif isinstance(doc, str):
                                doc_id = doc
                            else:
                                doc_id = str(doc)
                            qrels[query_id][doc_id] = 0  # Irrelevant
                
                # Method 4: Direct query-document pairing (fallback)
                if not qrels[query_id]:
                    # Check if there's a direct doc reference in corpus mapping
                    corpus_doc_field = self.config.corpus_feature_mapping.get("doc_id", "id")
                    corpus_doc_value = self._get_nested_field(item, corpus_doc_field)
                    if corpus_doc_value is not None:
                        doc_id = str(corpus_doc_value)
                        qrels[query_id][doc_id] = self.config.hf_default_relevance_score
                
                count += 1
            
            # Convert defaultdict to regular dict
            qrels_dict = {qid: dict(judgments) for qid, judgments in qrels.items() if judgments}
            
            logger.info(f"Loaded qrels for {len(qrels_dict)} queries with {sum(len(j) for j in qrels_dict.values())} total judgments")
            
            if self.config.cache_enabled:
                self._qrels = qrels_dict
            
            return qrels_dict
            
        except Exception as e:
            raise DatasetLoadingError(f"Failed to load qrels from HF dataset: {e}")
