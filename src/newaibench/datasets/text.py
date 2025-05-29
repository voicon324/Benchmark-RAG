"""
Text dataset loader for NewAIBench framework.

This module provides specialized dataset loaders for text-based information retrieval
datasets, supporting various formats like BEIR, TREC, MS MARCO, etc.
"""

import json
import csv
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Iterator
from collections import defaultdict

from .base import BaseDatasetLoader, DatasetConfig, DatasetLoadingError, DataValidationError

logger = logging.getLogger(__name__)


class TextDatasetLoader(BaseDatasetLoader):
    """Dataset loader for text-based information retrieval datasets.
    
    This loader supports various text dataset formats commonly used in IR research,
    including BEIR, TREC, MS MARCO, and custom formats. It handles JSONL, TSV, 
    JSON, and CSV files with comprehensive preprocessing options.
    
    Supported formats:
    - BEIR format: corpus.jsonl, queries.jsonl, qrels/*.tsv
    - TREC format: *.txt files with specific structures
    - MS MARCO format: *.tsv files
    - Custom JSONL/JSON formats
    
    Features:
    - Automatic format detection
    - Text preprocessing (lowercase, normalization, etc.)
    - Memory-efficient streaming for large datasets
    - Robust error handling and validation
    - Caching for improved performance
    """
    
    def __init__(self, config: DatasetConfig) -> None:
        """Initialize TextDatasetLoader with configuration.
        
        Args:
            config: DatasetConfig object with text-specific settings
        """
        super().__init__(config)
        
        # Text-specific preprocessing defaults
        if not config.preprocessing_options:
            config.preprocessing_options = {
                "lowercase": False,
                "normalize_whitespace": True,
                "remove_special_chars": False,
                "strip_html": False,
                "min_length": 1,
                "max_length": None
            }
        
        logger.info(f"Initialized TextDatasetLoader for format: {config.format_type}")
    
    def load_corpus(self) -> Dict[str, Dict[str, Any]]:
        """Load text corpus from various formats.
        
        Returns:
            Dictionary mapping document IDs to document dictionaries.
            Each document contains at least 'text' field and optionally
            'title', 'metadata', etc.
            
        Raises:
            DatasetLoadingError: If corpus cannot be loaded
            DataValidationError: If loaded data fails validation
        """
        if self.config.cache_enabled and self._corpus is not None:
            logger.debug("Returning cached corpus")
            return self._corpus
        
        corpus_path = self.config.dataset_path / self.config.corpus_file
        if not corpus_path.exists():
            raise DatasetLoadingError(f"Corpus file not found: {corpus_path}")
        
        logger.info(f"Loading corpus from: {corpus_path}")
        
        try:
            if self.config.format_type == "jsonl":
                corpus = self._load_jsonl_corpus(corpus_path)
            elif self.config.format_type == "json":
                corpus = self._load_json_corpus(corpus_path)
            elif self.config.format_type == "tsv":
                corpus = self._load_tsv_corpus(corpus_path)
            elif self.config.format_type == "csv":
                corpus = self._load_csv_corpus(corpus_path)
            else:
                raise DatasetLoadingError(f"Unsupported format for corpus: {self.config.format_type}")
            
            # Apply preprocessing
            corpus = self._preprocess_corpus(corpus)
            
            # Apply sampling if specified
            if self.config.max_samples:
                corpus_items = list(corpus.items())[:self.config.max_samples]
                corpus = dict(corpus_items)
                logger.info(f"Limited corpus to {len(corpus)} samples")
            
            # Cache if enabled
            if self.config.cache_enabled:
                self._corpus = corpus
            
            logger.info(f"Successfully loaded {len(corpus)} documents")
            return corpus
            
        except Exception as e:
            logger.error(f"Failed to load corpus: {e}")
            raise DatasetLoadingError(f"Cannot load corpus: {e}") from e
    
    def load_queries(self) -> Dict[str, str]:
        """Load queries from various formats.
        
        Returns:
            Dictionary mapping query IDs to query text strings.
            
        Raises:
            DatasetLoadingError: If queries cannot be loaded
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
            
            # Apply preprocessing to queries
            queries = {qid: self._apply_preprocessing(text) for qid, text in queries.items()}
            
            # Apply sampling if specified
            if self.config.max_samples:
                query_items = list(queries.items())[:self.config.max_samples]
                queries = dict(query_items)
                logger.info(f"Limited queries to {len(queries)} samples")
            
            # Cache if enabled
            if self.config.cache_enabled:
                self._queries = queries
            
            logger.info(f"Successfully loaded {len(queries)} queries")
            return queries
            
        except Exception as e:
            logger.error(f"Failed to load queries: {e}")
            raise DatasetLoadingError(f"Cannot load queries: {e}") from e
    
    def load_qrels(self) -> Dict[str, Dict[str, int]]:
        """Load relevance judgments from various formats.
        
        Returns:
            Nested dictionary: {query_id: {doc_id: relevance_score}}
            
        Raises:
            DatasetLoadingError: If qrels cannot be loaded
        """
        if self.config.cache_enabled and self._qrels is not None:
            logger.debug("Returning cached qrels")
            return self._qrels
        
        qrels_path = self.config.dataset_path / self.config.qrels_file
        if not qrels_path.exists():
            raise DatasetLoadingError(f"Qrels file not found: {qrels_path}")
        
        logger.info(f"Loading qrels from: {qrels_path}")
        
        try:
            # Qrels are typically in TSV/TXT format regardless of main format
            if qrels_path.suffix in ['.tsv', '.txt']:
                qrels = self._load_tsv_qrels(qrels_path)
            elif qrels_path.suffix == '.json':
                qrels = self._load_json_qrels(qrels_path)
            elif qrels_path.suffix == '.jsonl':
                qrels = self._load_jsonl_qrels(qrels_path)
            else:
                # Try TSV format as fallback
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
    
    def _load_jsonl_corpus(self, file_path: Path) -> Dict[str, Dict[str, Any]]:
        """Load corpus from JSONL format."""
        corpus = {}
        
        with open(file_path, 'r', encoding=self.config.encoding) as f:
            for line_num, line in enumerate(f, 1):
                try:
                    line = line.strip()
                    if not line:
                        continue
                    
                    doc = json.loads(line)
                    
                    # Handle different JSONL schemas
                    if "_id" in doc:
                        doc_id = str(doc["_id"])
                    elif "id" in doc:
                        doc_id = str(doc["id"])
                    elif "doc_id" in doc:
                        doc_id = str(doc["doc_id"])
                    else:
                        doc_id = str(line_num)
                        logger.warning(f"No ID field found in line {line_num}, using line number")
                    
                    # Extract text content
                    if "text" in doc:
                        doc["text"] = str(doc["text"])
                    elif "contents" in doc:
                        doc["text"] = str(doc["contents"])
                        del doc["contents"]
                    elif "content" in doc:
                        doc["text"] = str(doc["content"])
                        del doc["content"]
                    else:
                        raise DataValidationError(f"No text field found in document {doc_id}")
                    
                    corpus[doc_id] = doc
                    
                except json.JSONDecodeError as e:
                    logger.warning(f"Invalid JSON on line {line_num}: {e}")
                    continue
                except Exception as e:
                    logger.warning(f"Error processing line {line_num}: {e}")
                    continue
        
        return corpus
    
    def _load_json_corpus(self, file_path: Path) -> Dict[str, Dict[str, Any]]:
        """Load corpus from JSON format."""
        with open(file_path, 'r', encoding=self.config.encoding) as f:
            data = json.load(f)
        
        # Handle different JSON structures
        if isinstance(data, dict):
            if "corpus" in data:
                data = data["corpus"]
            elif "documents" in data:
                data = data["documents"]
        
        if isinstance(data, list):
            # Convert list to dict
            corpus = {}
            for i, doc in enumerate(data):
                if isinstance(doc, dict):
                    doc_id = doc.get("_id", doc.get("id", doc.get("doc_id", str(i))))
                    corpus[str(doc_id)] = doc
                else:
                    corpus[str(i)] = {"text": str(doc)}
            return corpus
        elif isinstance(data, dict):
            return data
        else:
            raise DatasetLoadingError("Unsupported JSON structure for corpus")
    
    def _load_tsv_corpus(self, file_path: Path) -> Dict[str, Dict[str, Any]]:
        """Load corpus from TSV format."""
        corpus = {}
        
        with open(file_path, 'r', encoding=self.config.encoding) as f:
            reader = csv.DictReader(f, delimiter='\t')
            
            for row_num, row in enumerate(reader, 1):
                try:
                    # Handle different TSV schemas
                    if "id" in row:
                        doc_id = str(row["id"])
                    elif "doc_id" in row:
                        doc_id = str(row["doc_id"])
                    elif "_id" in row:
                        doc_id = str(row["_id"])
                    else:
                        doc_id = str(row_num)
                    
                    # Extract text
                    if "text" in row:
                        text = row["text"]
                    elif "passage" in row:
                        text = row["passage"]
                    elif "content" in row:
                        text = row["content"]
                    else:
                        # Use all non-id fields as text
                        text_fields = [v for k, v in row.items() if "id" not in k.lower()]
                        text = " ".join(text_fields)
                    
                    doc = {"text": text}
                    
                    # Add title if available
                    if "title" in row:
                        doc["title"] = row["title"]
                    
                    # Add other metadata
                    for key, value in row.items():
                        if key not in ["id", "doc_id", "_id", "text", "passage", "content"]:
                            doc[key] = value
                    
                    corpus[doc_id] = doc
                    
                except Exception as e:
                    logger.warning(f"Error processing TSV row {row_num}: {e}")
                    continue
        
        return corpus
    
    def _load_csv_corpus(self, file_path: Path) -> Dict[str, Dict[str, Any]]:
        """Load corpus from CSV format."""
        corpus = {}
        
        with open(file_path, 'r', encoding=self.config.encoding) as f:
            reader = csv.DictReader(f)
            
            for row_num, row in enumerate(reader, 1):
                try:
                    doc_id = row.get("id", row.get("doc_id", str(row_num)))
                    text = row.get("text", row.get("content", ""))
                    
                    doc = {"text": text}
                    
                    # Add other fields as metadata
                    for key, value in row.items():
                        if key not in ["id", "doc_id", "text", "content"]:
                            doc[key] = value
                    
                    corpus[str(doc_id)] = doc
                    
                except Exception as e:
                    logger.warning(f"Error processing CSV row {row_num}: {e}")
                    continue
        
        return corpus
    
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
        
        # Handle different JSON structures
        if isinstance(data, dict):
            if "queries" in data:
                data = data["queries"]
        
        queries = {}
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
                        # Standard TREC format: query_id doc_id relevance_score
                        if len(parts) == 3:
                            query_id, doc_id, score = parts
                        # Extended format: query_id iteration doc_id relevance_score
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
                    score = int(data["relevance"])
                    qrels[query_id][doc_id] = score
                except (json.JSONDecodeError, KeyError, ValueError) as e:
                    logger.warning(f"Invalid qrels JSONL entry: {line}")
                    continue
        
        return dict(qrels)
    
    def _preprocess_corpus(self, corpus: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Apply preprocessing to corpus documents."""
        processed_corpus = {}
        options = self.config.preprocessing_options
        
        for doc_id, doc in corpus.items():
            if "text" not in doc:
                continue
            
            text = doc["text"]
            
            # Apply preprocessing
            text = self._apply_preprocessing(text)
            
            # Apply length filtering
            if options.get("min_length", 1):
                if len(text.split()) < options["min_length"]:
                    continue
            
            if options.get("max_length"):
                if len(text.split()) > options["max_length"]:
                    words = text.split()[:options["max_length"]]
                    text = " ".join(words)
            
            # Strip HTML if enabled
            if options.get("strip_html", False):
                import re
                text = re.sub(r'<[^>]+>', '', text)
            
            doc["text"] = text
            processed_corpus[doc_id] = doc
        
        logger.info(f"Preprocessing reduced corpus from {len(corpus)} to {len(processed_corpus)} documents")
        return processed_corpus