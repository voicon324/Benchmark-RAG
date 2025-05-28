"""
Document image dataset loader for NewAIBench framework.

This module provides specialized dataset loaders for document image datasets
used in visual information retrieval and document analysis tasks.
"""

import json
import csv
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
from collections import defaultdict
import mimetypes
from dataclasses import dataclass

from .base import BaseDatasetLoader, DatasetConfig, DatasetLoadingError, DataValidationError

logger = logging.getLogger(__name__)


@dataclass
class DocumentImageDatasetConfig(DatasetConfig):
    """Extended configuration for document image datasets.
    
    Attributes:
        image_root_path: Root directory containing image files
        supported_image_formats: List of supported image file extensions
        require_ocr_text: Whether OCR text is required for each document
        validate_images: Whether to validate image file existence and format
        image_preprocessing_options: Options for image preprocessing
    """
    image_root_path: Optional[Union[str, Path]] = None
    supported_image_formats: List[str] = None
    require_ocr_text: bool = True
    validate_images: bool = True
    image_preprocessing_options: Dict[str, Any] = None
    
    def __post_init__(self):
        """Extended validation for image-specific configuration."""
        super().__post_init__()
        
        # Set default supported formats
        if self.supported_image_formats is None:
            self.supported_image_formats = ['.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp', '.gif', '.pdf']
        
        # Set default image root path
        if self.image_root_path is None:
            self.image_root_path = self.dataset_path / "images"
        else:
            self.image_root_path = Path(self.image_root_path)
        
        # Set default image preprocessing options
        if self.image_preprocessing_options is None:
            self.image_preprocessing_options = {
                "validate_format": True,
                "check_corruption": False,
                "extract_metadata": True
            }


class DocumentImageDatasetLoader(BaseDatasetLoader):
    """Dataset loader for document image datasets.
    
    This loader handles datasets containing document images with associated
    metadata, OCR text, and relevance judgments. It's designed for tasks like:
    - Visual document retrieval
    - Multimodal IR (text + images)
    - Document image analysis
    - OCR-based text extraction evaluation
    
    Features:
    - Image path validation and verification
    - OCR text extraction and validation
    - Metadata handling (page numbers, document types, etc.)
    - Support for various image formats
    - Efficient handling of large image collections
    - Integration with OCR engines for text extraction
    """
    
    def __init__(self, config: DocumentImageDatasetConfig) -> None:
        """Initialize DocumentImageDatasetLoader.
        
        Args:
            config: DocumentImageDatasetConfig with image-specific settings
            
        Raises:
            ValueError: If config is not DocumentImageDatasetConfig
            FileNotFoundError: If image root path doesn't exist
        """
        if not isinstance(config, DocumentImageDatasetConfig):
            raise ValueError("config must be an instance of DocumentImageDatasetConfig")
        
        super().__init__(config)
        self.config: DocumentImageDatasetConfig = config
        
        # Validate image root path
        if not self.config.image_root_path.exists():
            logger.warning(f"Image root path does not exist: {self.config.image_root_path}")
            if self.config.validate_images:
                raise FileNotFoundError(f"Image root path not found: {self.config.image_root_path}")
        
        logger.info(f"Initialized DocumentImageDatasetLoader with image root: {self.config.image_root_path}")
    
    def load_corpus(self) -> Dict[str, Dict[str, Any]]:
        """Load document image corpus.
        
        Returns:
            Dictionary mapping document IDs to document dictionaries.
            Each document contains:
            - 'text': OCR extracted text or provided text
            - 'image_path': Path to the image file
            - 'metadata': Additional metadata (dimensions, format, etc.)
            - Optional: 'title', 'page_number', 'document_type', etc.
            
        Raises:
            DatasetLoadingError: If corpus cannot be loaded
            DataValidationError: If image validation fails
        """
        if self.config.cache_enabled and self._corpus is not None:
            logger.debug("Returning cached image corpus")
            return self._corpus
        
        corpus_path = self.config.dataset_path / self.config.corpus_file
        if not corpus_path.exists():
            raise DatasetLoadingError(f"Corpus file not found: {corpus_path}")
        
        logger.info(f"Loading image corpus from: {corpus_path}")
        
        try:
            # Load base corpus data
            if self.config.format_type == "jsonl":
                corpus = self._load_jsonl_image_corpus(corpus_path)
            elif self.config.format_type == "json":
                corpus = self._load_json_image_corpus(corpus_path)
            elif self.config.format_type == "tsv":
                corpus = self._load_tsv_image_corpus(corpus_path)
            elif self.config.format_type == "csv":
                corpus = self._load_csv_image_corpus(corpus_path)
            else:
                raise DatasetLoadingError(f"Unsupported format for image corpus: {self.config.format_type}")
            
            # Validate and process images
            if self.config.validate_images:
                corpus = self._validate_and_process_images(corpus)
            
            # Apply text preprocessing
            corpus = self._preprocess_image_corpus(corpus)
            
            # Apply sampling if specified
            if self.config.max_samples:
                corpus_items = list(corpus.items())[:self.config.max_samples]
                corpus = dict(corpus_items)
                logger.info(f"Limited image corpus to {len(corpus)} samples")
            
            # Cache if enabled
            if self.config.cache_enabled:
                self._corpus = corpus
            
            logger.info(f"Successfully loaded {len(corpus)} image documents")
            return corpus
            
        except Exception as e:
            logger.error(f"Failed to load image corpus: {e}")
            raise DatasetLoadingError(f"Cannot load image corpus: {e}") from e
    
    def load_queries(self) -> Dict[str, str]:
        """Load queries for document image retrieval.
        
        Queries can be text-based (searching for images by text description)
        or image-based (searching for similar images).
        
        Returns:
            Dictionary mapping query IDs to query text strings.
        """
        if self.config.cache_enabled and self._queries is not None:
            logger.debug("Returning cached queries")
            return self._queries
        
        queries_path = self.config.dataset_path / self.config.queries_file
        if not queries_path.exists():
            raise DatasetLoadingError(f"Queries file not found: {queries_path}")
        
        logger.info(f"Loading queries from: {queries_path}")
        
        try:
            if self.config.format_type == "jsonl":
                queries = self._load_jsonl_queries(queries_path)
            elif self.config.format_type == "json":
                queries = self._load_json_queries(queries_path)
            elif self.config.format_type == "tsv":
                queries = self._load_tsv_queries(queries_path)
            elif self.config.format_type == "csv":
                queries = self._load_csv_queries(queries_path)
            else:
                raise DatasetLoadingError(f"Unsupported format for queries: {self.config.format_type}")
            
            # Apply preprocessing to query text
            queries = {qid: self._apply_preprocessing(text) for qid, text in queries.items()}
            
            # Cache if enabled
            if self.config.cache_enabled:
                self._queries = queries
            
            logger.info(f"Successfully loaded {len(queries)} queries")
            return queries
            
        except Exception as e:
            logger.error(f"Failed to load queries: {e}")
            raise DatasetLoadingError(f"Cannot load queries: {e}") from e
    
    def load_qrels(self) -> Dict[str, Dict[str, int]]:
        """Load relevance judgments for document images.
        
        Returns:
            Nested dictionary: {query_id: {doc_id: relevance_score}}
        """
        if self.config.cache_enabled and self._qrels is not None:
            logger.debug("Returning cached qrels")
            return self._qrels
        
        qrels_path = self.config.dataset_path / self.config.qrels_file
        if not qrels_path.exists():
            raise DatasetLoadingError(f"Qrels file not found: {qrels_path}")
        
        logger.info(f"Loading qrels from: {qrels_path}")
        
        try:
            # Load qrels (format typically doesn't depend on main format)
            if qrels_path.suffix in ['.tsv', '.txt']:
                qrels = self._load_tsv_qrels(qrels_path)
            elif qrels_path.suffix == '.json':
                qrels = self._load_json_qrels(qrels_path)
            elif qrels_path.suffix == '.jsonl':
                qrels = self._load_jsonl_qrels(qrels_path)
            else:
                qrels = self._load_tsv_qrels(qrels_path)
            
            # Cache if enabled
            if self.config.cache_enabled:
                self._qrels = qrels
            
            total_judgments = sum(len(judgments) for judgments in qrels.values())
            logger.info(f"Successfully loaded {len(qrels)} queries with {total_judgments} judgments")
            return qrels
            
        except Exception as e:
            logger.error(f"Failed to load qrels: {e}")
            raise DatasetLoadingError(f"Cannot load qrels: {e}") from e
    
    def _load_jsonl_image_corpus(self, file_path: Path) -> Dict[str, Dict[str, Any]]:
        """Load image corpus from JSONL format."""
        corpus = {}
        
        with open(file_path, 'r', encoding=self.config.encoding) as f:
            for line_num, line in enumerate(f, 1):
                try:
                    line = line.strip()
                    if not line:
                        continue
                    
                    doc = json.loads(line)
                    
                    # Extract document ID
                    if "_id" in doc:
                        doc_id = str(doc["_id"])
                    elif "id" in doc:
                        doc_id = str(doc["id"])
                    elif "doc_id" in doc:
                        doc_id = str(doc["doc_id"])
                    else:
                        doc_id = str(line_num)
                        logger.warning(f"No ID field found in line {line_num}, using line number")
                    
                    # Process image document
                    processed_doc = self._process_image_document(doc, doc_id)
                    if processed_doc:
                        corpus[doc_id] = processed_doc
                    
                except json.JSONDecodeError as e:
                    logger.warning(f"Invalid JSON on line {line_num}: {e}")
                    continue
                except Exception as e:
                    logger.warning(f"Error processing image document on line {line_num}: {e}")
                    continue
        
        return corpus
    
    def _load_json_image_corpus(self, file_path: Path) -> Dict[str, Dict[str, Any]]:
        """Load image corpus from JSON format."""
        with open(file_path, 'r', encoding=self.config.encoding) as f:
            data = json.load(f)
        
        # Handle different JSON structures
        if isinstance(data, dict):
            if "corpus" in data:
                data = data["corpus"]
            elif "documents" in data:
                data = data["documents"]
        
        corpus = {}
        if isinstance(data, list):
            for i, doc in enumerate(data):
                if isinstance(doc, dict):
                    doc_id = doc.get("_id", doc.get("id", doc.get("doc_id", str(i))))
                    processed_doc = self._process_image_document(doc, str(doc_id))
                    if processed_doc:
                        corpus[str(doc_id)] = processed_doc
        elif isinstance(data, dict):
            for doc_id, doc in data.items():
                if isinstance(doc, dict):
                    processed_doc = self._process_image_document(doc, str(doc_id))
                    if processed_doc:
                        corpus[str(doc_id)] = processed_doc
        
        return corpus
    
    def _load_tsv_image_corpus(self, file_path: Path) -> Dict[str, Dict[str, Any]]:
        """Load image corpus from TSV format."""
        corpus = {}
        
        with open(file_path, 'r', encoding=self.config.encoding) as f:
            reader = csv.DictReader(f, delimiter='\t')
            
            for row_num, row in enumerate(reader, 1):
                try:
                    # Extract document ID
                    if "id" in row:
                        doc_id = str(row["id"])
                    elif "doc_id" in row:
                        doc_id = str(row["doc_id"])
                    elif "_id" in row:
                        doc_id = str(row["_id"])
                    else:
                        doc_id = str(row_num)
                    
                    processed_doc = self._process_image_document(dict(row), doc_id)
                    if processed_doc:
                        corpus[doc_id] = processed_doc
                    
                except Exception as e:
                    logger.warning(f"Error processing TSV row {row_num}: {e}")
                    continue
        
        return corpus
    
    def _load_csv_image_corpus(self, file_path: Path) -> Dict[str, Dict[str, Any]]:
        """Load image corpus from CSV format."""
        corpus = {}
        
        with open(file_path, 'r', encoding=self.config.encoding) as f:
            reader = csv.DictReader(f)
            
            for row_num, row in enumerate(reader, 1):
                try:
                    doc_id = row.get("id", row.get("doc_id", str(row_num)))
                    processed_doc = self._process_image_document(dict(row), str(doc_id))
                    if processed_doc:
                        corpus[str(doc_id)] = processed_doc
                    
                except Exception as e:
                    logger.warning(f"Error processing CSV row {row_num}: {e}")
                    continue
        
        return corpus
    
    def _process_image_document(self, doc: Dict[str, Any], doc_id: str) -> Optional[Dict[str, Any]]:
        """Process a single image document entry.
        
        Args:
            doc: Raw document dictionary
            doc_id: Document ID
            
        Returns:
            Processed document dictionary or None if invalid
        """
        processed_doc = {}
        
        # Extract image path
        image_path = None
        for field in ["image_path", "image", "path", "file_path", "filename"]:
            if field in doc:
                image_path = doc[field]
                break
        
        if not image_path:
            logger.warning(f"No image path found for document {doc_id}")
            return None
        
        # Resolve full image path
        if not Path(image_path).is_absolute():
            full_image_path = self.config.image_root_path / image_path
        else:
            full_image_path = Path(image_path)
        
        processed_doc["image_path"] = str(full_image_path)
        
        # Extract text content (OCR text or provided text)
        text_content = ""
        for field in ["text", "ocr_text", "content", "extracted_text"]:
            if field in doc:
                text_content = str(doc[field])
                break
        
        if not text_content and self.config.require_ocr_text:
            logger.warning(f"No text content found for document {doc_id}")
            return None
        
        processed_doc["text"] = text_content
        
        # Extract title if available
        if "title" in doc:
            processed_doc["title"] = str(doc["title"])
        
        # Extract metadata
        metadata = {}
        for key, value in doc.items():
            if key not in ["id", "_id", "doc_id", "image_path", "image", "path", 
                          "file_path", "filename", "text", "ocr_text", "content", 
                          "extracted_text", "title"]:
                metadata[key] = value
        
        processed_doc["metadata"] = metadata
        
        return processed_doc
    
    def _validate_and_process_images(self, corpus: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Validate image files and extract metadata."""
        validated_corpus = {}
        
        for doc_id, doc in corpus.items():
            image_path = Path(doc["image_path"])
            
            # Check if image file exists
            if not image_path.exists():
                logger.warning(f"Image file not found for document {doc_id}: {image_path}")
                continue
            
            # Check file format
            if self.config.image_preprocessing_options.get("validate_format", True):
                if not self._is_valid_image_format(image_path):
                    logger.warning(f"Unsupported image format for document {doc_id}: {image_path}")
                    continue
            
            # Extract image metadata if enabled
            if self.config.image_preprocessing_options.get("extract_metadata", True):
                image_metadata = self._extract_image_metadata(image_path)
                doc["metadata"].update(image_metadata)
            
            # Check for corruption if enabled
            if self.config.image_preprocessing_options.get("check_corruption", False):
                if self._is_image_corrupted(image_path):
                    logger.warning(f"Corrupted image for document {doc_id}: {image_path}")
                    continue
            
            validated_corpus[doc_id] = doc
        
        removed_count = len(corpus) - len(validated_corpus)
        if removed_count > 0:
            logger.info(f"Removed {removed_count} documents due to image validation issues")
        
        return validated_corpus
    
    def _is_valid_image_format(self, image_path: Path) -> bool:
        """Check if image file has a supported format."""
        file_extension = image_path.suffix.lower()
        return file_extension in self.config.supported_image_formats
    
    def _extract_image_metadata(self, image_path: Path) -> Dict[str, Any]:
        """Extract metadata from image file."""
        metadata = {
            "file_size": image_path.stat().st_size,
            "file_extension": image_path.suffix.lower(),
            "mime_type": mimetypes.guess_type(str(image_path))[0]
        }
        
        # Try to extract image dimensions and other metadata
        try:
            # This would require PIL/Pillow
            # from PIL import Image
            # with Image.open(image_path) as img:
            #     metadata["width"] = img.width
            #     metadata["height"] = img.height
            #     metadata["format"] = img.format
            #     metadata["mode"] = img.mode
            pass
        except Exception as e:
            logger.debug(f"Could not extract image metadata for {image_path}: {e}")
        
        return metadata
    
    def _is_image_corrupted(self, image_path: Path) -> bool:
        """Check if image file is corrupted."""
        try:
            # This would require PIL/Pillow
            # from PIL import Image
            # with Image.open(image_path) as img:
            #     img.verify()
            # return False
            return False
        except Exception:
            return True
    
    def _preprocess_image_corpus(self, corpus: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Apply preprocessing to image corpus."""
        processed_corpus = {}
        options = self.config.preprocessing_options
        
        for doc_id, doc in corpus.items():
            # Apply text preprocessing to OCR text
            if "text" in doc and doc["text"]:
                doc["text"] = self._apply_preprocessing(doc["text"])
                
                # Apply length filtering
                if options.get("min_length", 1):
                    if len(doc["text"].split()) < options["min_length"]:
                        continue
                
                if options.get("max_length"):
                    if len(doc["text"].split()) > options["max_length"]:
                        words = doc["text"].split()[:options["max_length"]]
                        doc["text"] = " ".join(words)
            
            processed_corpus[doc_id] = doc
        
        logger.info(f"Preprocessing retained {len(processed_corpus)} of {len(corpus)} image documents")
        return processed_corpus
    
    def _load_jsonl_queries(self, file_path: Path) -> Dict[str, str]:
        """Load queries from JSONL format."""
        queries = {}
        
        with open(file_path, 'r', encoding=self.config.encoding) as f:
            for line_num, line in enumerate(f, 1):
                try:
                    line = line.strip()
                    if not line:
                        continue
                    
                    query = json.loads(line)
                    
                    # Extract query ID
                    if "_id" in query:
                        query_id = str(query["_id"])
                    elif "id" in query:
                        query_id = str(query["id"])
                    elif "query_id" in query:
                        query_id = str(query["query_id"])
                    else:
                        query_id = str(line_num)
                    
                    # Extract query text
                    if "text" in query:
                        query_text = str(query["text"])
                    elif "query" in query:
                        query_text = str(query["query"])
                    elif "question" in query:
                        query_text = str(query["question"])
                    else:
                        raise DataValidationError(f"No query text found for query {query_id}")
                    
                    queries[query_id] = query_text
                    
                except json.JSONDecodeError as e:
                    logger.warning(f"Invalid JSON on line {line_num}: {e}")
                    continue
                except Exception as e:
                    logger.warning(f"Error processing query line {line_num}: {e}")
                    continue
        
        return queries
    
    def _load_json_queries(self, file_path: Path) -> Dict[str, str]:
        """Load queries from JSON format."""
        with open(file_path, 'r', encoding=self.config.encoding) as f:
            data = json.load(f)
        
        queries = {}
        if isinstance(data, dict):
            if "queries" in data:
                data = data["queries"]
        
        if isinstance(data, list):
            for i, query in enumerate(data):
                if isinstance(query, dict):
                    query_id = query.get("_id", query.get("id", str(i)))
                    query_text = query.get("text", query.get("query", ""))
                    queries[str(query_id)] = query_text
                else:
                    queries[str(i)] = str(query)
        elif isinstance(data, dict):
            for query_id, query_data in data.items():
                if isinstance(query_data, dict):
                    query_text = query_data.get("text", query_data.get("query", ""))
                else:
                    query_text = str(query_data)
                queries[str(query_id)] = query_text
        
        return queries
    
    def _load_tsv_queries(self, file_path: Path) -> Dict[str, str]:
        """Load queries from TSV format."""
        queries = {}
        
        with open(file_path, 'r', encoding=self.config.encoding) as f:
            reader = csv.DictReader(f, delimiter='\t')
            
            for row in reader:
                query_id = row.get("id", row.get("query_id", ""))
                query_text = row.get("text", row.get("query", ""))
                
                if query_id and query_text:
                    queries[str(query_id)] = query_text
        
        return queries
    
    def _load_csv_queries(self, file_path: Path) -> Dict[str, str]:
        """Load queries from CSV format."""
        queries = {}
        
        with open(file_path, 'r', encoding=self.config.encoding) as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                query_id = row.get("id", row.get("query_id", ""))
                query_text = row.get("text", row.get("query", ""))
                
                if query_id and query_text:
                    queries[str(query_id)] = query_text
        
        return queries
    
    def _load_tsv_qrels(self, file_path: Path) -> Dict[str, Dict[str, int]]:
        """Load qrels from TSV format."""
        qrels = defaultdict(dict)
        
        with open(file_path, 'r', encoding=self.config.encoding) as f:
            for line_num, line in enumerate(f, 1):
                try:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    
                    parts = line.split('\t')
                    if len(parts) >= 3:
                        if len(parts) == 3:
                            query_id, doc_id, score = parts
                        elif len(parts) == 4:
                            query_id, _, doc_id, score = parts
                        else:
                            query_id, doc_id, score = parts[0], parts[-2], parts[-1]
                        
                        qrels[str(query_id)][str(doc_id)] = int(score)
                    
                except (ValueError, IndexError) as e:
                    logger.warning(f"Invalid qrels format on line {line_num}: {line}")
                    continue
        
        return dict(qrels)
    
    def _load_json_qrels(self, file_path: Path) -> Dict[str, Dict[str, int]]:
        """Load qrels from JSON format."""
        with open(file_path, 'r', encoding=self.config.encoding) as f:
            data = json.load(f)
        
        qrels = {}
        for query_id, judgments in data.items():
            if isinstance(judgments, dict):
                qrels[str(query_id)] = {str(doc_id): int(score) for doc_id, score in judgments.items()}
        
        return qrels
    
    def _load_jsonl_qrels(self, file_path: Path) -> Dict[str, Dict[str, int]]:
        """Load qrels from JSONL format."""
        qrels = defaultdict(dict)
        
        with open(file_path, 'r', encoding=self.config.encoding) as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    query_id = str(data["query_id"])
                    doc_id = str(data["doc_id"])
                    score = int(data["score"])
                    qrels[query_id][doc_id] = score
                except (json.JSONDecodeError, KeyError, ValueError) as e:
                    logger.warning(f"Invalid qrels JSONL entry: {line}")
                    continue
        
        return dict(qrels)
    
    def get_image_paths(self) -> List[str]:
        """Get list of all image paths in the corpus.
        
        Returns:
            List of image file paths
        """
        if self._corpus is None:
            self.load_corpus()
        
        return [doc["image_path"] for doc in self._corpus.values() if "image_path" in doc]
    
    def get_documents_with_ocr(self) -> Dict[str, Dict[str, Any]]:
        """Get documents that have OCR text.
        
        Returns:
            Dictionary of documents with non-empty OCR text
        """
        if self._corpus is None:
            self.load_corpus()
        
        return {
            doc_id: doc for doc_id, doc in self._corpus.items()
            if doc.get("text", "").strip()
        }
    
    def get_image_statistics(self) -> Dict[str, Any]:
        """Get image-specific statistics.
        
        Returns:
            Dictionary with image corpus statistics
        """
        stats = self.get_statistics()
        
        if self._corpus:
            # Count documents with/without OCR text
            docs_with_ocr = len(self.get_documents_with_ocr())
            stats["documents_with_ocr"] = docs_with_ocr
            stats["documents_without_ocr"] = len(self._corpus) - docs_with_ocr
            
            # Image format distribution
            format_counts = defaultdict(int)
            for doc in self._corpus.values():
                if "image_path" in doc:
                    ext = Path(doc["image_path"]).suffix.lower()
                    format_counts[ext] += 1
            stats["image_format_distribution"] = dict(format_counts)
            
            # File size statistics (if available in metadata)
            file_sizes = []
            for doc in self._corpus.values():
                if "metadata" in doc and "file_size" in doc["metadata"]:
                    file_sizes.append(doc["metadata"]["file_size"])
            
            if file_sizes:
                stats["avg_file_size_bytes"] = sum(file_sizes) / len(file_sizes)
                stats["total_corpus_size_bytes"] = sum(file_sizes)
                stats["max_file_size_bytes"] = max(file_sizes)
                stats["min_file_size_bytes"] = min(file_sizes)
        
        return stats