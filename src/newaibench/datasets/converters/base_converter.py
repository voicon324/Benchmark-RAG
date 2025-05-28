"""
Base converter classes for dataset format conversion in NewAIBench.

This module provides the foundational classes and interfaces for converting
various IR dataset formats to NewAIBench's standardized format.
"""

import json
import csv
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Iterator, Tuple
from dataclasses import dataclass
import hashlib
import gzip
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class ConversionConfig:
    """Configuration for dataset conversion process."""
    source_path: Union[str, Path]
    output_path: Union[str, Path]
    dataset_name: str
    format_type: str  # 'beir', 'msmarco', 'trec', 'custom'
    
    # Processing options
    validate_output: bool = True
    compress_output: bool = False
    preserve_metadata: bool = True
    max_workers: int = 1
    
    # Quality filters
    min_doc_length: int = 10
    min_query_length: int = 3
    max_doc_length: Optional[int] = None
    max_query_length: Optional[int] = None
    
    # Field mapping
    doc_id_field: str = "doc_id"
    doc_text_field: str = "text"
    doc_title_field: str = "title"
    query_id_field: str = "query_id"
    query_text_field: str = "text"


class ConversionError(Exception):
    """Custom exception for dataset conversion errors."""
    pass


class ValidationError(Exception):
    """Custom exception for data validation errors."""
    pass


class BaseDatasetConverter(ABC):
    """
    Abstract base class for all dataset converters.
    
    This class provides the common interface and functionality for converting
    different IR dataset formats to NewAIBench's standardized JSONL format.
    
    All converters should inherit from this class and implement the required
    abstract methods for their specific format.
    """
    
    def __init__(self, config: ConversionConfig):
        """Initialize converter with configuration."""
        if not isinstance(config, ConversionConfig):
            raise ValueError("config must be an instance of ConversionConfig")
        
        self.config = config
        self.source_path = Path(config.source_path)
        self.output_path = Path(config.output_path)
        
        # Validation
        if not self.source_path.exists():
            raise FileNotFoundError(f"Source path does not exist: {self.source_path}")
        
        # Create output directory
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Statistics
        self.stats = {
            'docs_processed': 0,
            'docs_valid': 0,
            'docs_skipped': 0,
            'queries_processed': 0,
            'queries_valid': 0,
            'queries_skipped': 0,
            'qrels_processed': 0,
            'conversion_time': 0.0
        }
        
        logger.info(f"Initialized {self.__class__.__name__} for dataset: {config.dataset_name}")
    
    @abstractmethod
    def convert_corpus(self) -> Path:
        """
        Convert corpus to NewAIBench format.
        
        Returns:
            Path to converted corpus file (corpus.jsonl or corpus.jsonl.gz)
        """
        pass
    
    @abstractmethod
    def convert_queries(self) -> Path:
        """
        Convert queries to NewAIBench format.
        
        Returns:
            Path to converted queries file (queries.jsonl or queries.jsonl.gz)
        """
        pass
    
    @abstractmethod
    def convert_qrels(self) -> Path:
        """
        Convert relevance judgments to NewAIBench format.
        
        Returns:
            Path to converted qrels file (qrels.txt)
        """
        pass
    
    def convert_all(self) -> Dict[str, Path]:
        """
        Convert all dataset components.
        
        Returns:
            Dictionary mapping component names to output file paths
        """
        import time
        start_time = time.time()
        
        logger.info(f"Starting conversion of {self.config.dataset_name}")
        
        try:
            # Convert components
            corpus_path = self.convert_corpus()
            queries_path = self.convert_queries()
            qrels_path = self.convert_qrels()
            
            # Create dataset info file
            info_path = self._create_dataset_info()
            
            self.stats['conversion_time'] = time.time() - start_time
            
            # Log statistics
            self._log_conversion_stats()
            
            # Validate if requested
            if self.config.validate_output:
                self._validate_conversion_output()
            
            results = {
                'corpus': corpus_path,
                'queries': queries_path,
                'qrels': qrels_path,
                'info': info_path
            }
            
            logger.info(f"Conversion completed successfully for {self.config.dataset_name}")
            return results
            
        except Exception as e:
            logger.error(f"Conversion failed for {self.config.dataset_name}: {str(e)}")
            raise ConversionError(f"Failed to convert dataset: {str(e)}") from e
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text content."""
        if not isinstance(text, str):
            text = str(text)
        
        # Basic cleaning
        text = text.strip()
        
        # Remove excessive whitespace
        import re
        text = re.sub(r'\s+', ' ', text)
        
        return text
    
    def _validate_document(self, doc: Dict[str, Any]) -> bool:
        """Validate a document according to configuration."""
        # Check required fields
        if 'doc_id' not in doc or 'text' not in doc:
            return False
        
        # Check text length
        text_len = len(doc['text'].strip())
        if text_len < self.config.min_doc_length:
            return False
        
        if self.config.max_doc_length and text_len > self.config.max_doc_length:
            return False
        
        return True
    
    def _validate_query(self, query: Dict[str, Any]) -> bool:
        """Validate a query according to configuration."""
        # Check required fields
        if 'query_id' not in query or 'text' not in query:
            return False
        
        # Check text length
        text_len = len(query['text'].strip())
        if text_len < self.config.min_query_length:
            return False
        
        if self.config.max_query_length and text_len > self.config.max_query_length:
            return False
        
        return True
    
    def _write_jsonl(self, data: Iterator[Dict[str, Any]], output_file: Path) -> None:
        """Write data to JSONL file, optionally compressed."""
        if self.config.compress_output:
            output_file = output_file.with_suffix('.jsonl.gz')
            open_func = gzip.open
            mode = 'wt'
        else:
            output_file = output_file.with_suffix('.jsonl')
            open_func = open
            mode = 'w'
        
        with open_func(output_file, mode, encoding='utf-8') as f:
            for item in data:
                json.dump(item, f, ensure_ascii=False)
                f.write('\n')
    
    def _write_qrels(self, qrels_data: List[Tuple[str, str, int]], output_file: Path) -> None:
        """Write qrels in TREC format."""
        output_file = output_file.with_suffix('.txt')
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for query_id, doc_id, relevance in qrels_data:
                # TREC format: query_id Q0 doc_id relevance
                f.write(f"{query_id}\tQ0\t{doc_id}\t{relevance}\n")
    
    def _create_dataset_info(self) -> Path:
        """Create dataset info file with metadata."""
        info = {
            'dataset_name': self.config.dataset_name,
            'format_type': self.config.format_type,
            'source_path': str(self.source_path),
            'converted_at': self._get_timestamp(),
            'converter_version': '1.0.0',
            'statistics': self.stats,
            'config': {
                'min_doc_length': self.config.min_doc_length,
                'min_query_length': self.config.min_query_length,
                'max_doc_length': self.config.max_doc_length,
                'max_query_length': self.config.max_query_length,
                'compress_output': self.config.compress_output,
            },
            'file_checksums': self._calculate_checksums()
        }
        
        info_path = self.output_path / 'dataset_info.json'
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(info, f, indent=2, ensure_ascii=False)
        
        return info_path
    
    def _calculate_checksums(self) -> Dict[str, str]:
        """Calculate MD5 checksums for output files."""
        checksums = {}
        
        for file_path in self.output_path.glob('*'):
            if file_path.is_file() and file_path.name != 'dataset_info.json':
                with open(file_path, 'rb') as f:
                    content = f.read()
                    checksum = hashlib.md5(content).hexdigest()
                    checksums[file_path.name] = checksum
        
        return checksums
    
    def _get_timestamp(self) -> str:
        """Get current timestamp in ISO format."""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def _log_conversion_stats(self) -> None:
        """Log conversion statistics."""
        logger.info(f"Conversion statistics for {self.config.dataset_name}:")
        logger.info(f"  Documents: {self.stats['docs_valid']}/{self.stats['docs_processed']} valid")
        logger.info(f"  Queries: {self.stats['queries_valid']}/{self.stats['queries_processed']} valid")
        logger.info(f"  QRels: {self.stats['qrels_processed']} processed")
        logger.info(f"  Conversion time: {self.stats['conversion_time']:.2f}s")
        
        if self.stats['docs_skipped'] > 0:
            logger.warning(f"  Skipped {self.stats['docs_skipped']} documents (validation failed)")
        
        if self.stats['queries_skipped'] > 0:
            logger.warning(f"  Skipped {self.stats['queries_skipped']} queries (validation failed)")
    
    def _validate_conversion_output(self) -> None:
        """Validate the conversion output."""
        logger.info("Validating conversion output...")
        
        # Check required files exist
        required_files = ['corpus.jsonl', 'queries.jsonl', 'qrels.txt']
        if self.config.compress_output:
            required_files = ['corpus.jsonl.gz', 'queries.jsonl.gz', 'qrels.txt']
        
        for filename in required_files:
            file_path = self.output_path / filename
            if not file_path.exists():
                raise ValidationError(f"Required output file missing: {filename}")
        
        # Validate JSONL structure
        self._validate_jsonl_file(self.output_path / 'corpus.jsonl.gz' if self.config.compress_output else self.output_path / 'corpus.jsonl')
        self._validate_jsonl_file(self.output_path / 'queries.jsonl.gz' if self.config.compress_output else self.output_path / 'queries.jsonl')
        
        logger.info("Conversion output validation passed")
    
    def _validate_jsonl_file(self, file_path: Path) -> None:
        """Validate JSONL file structure."""
        open_func = gzip.open if file_path.suffix == '.gz' else open
        mode = 'rt' if file_path.suffix == '.gz' else 'r'
        
        try:
            with open_func(file_path, mode, encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if i >= 100:  # Sample validation
                        break
                    
                    try:
                        json.loads(line.strip())
                    except json.JSONDecodeError as e:
                        raise ValidationError(f"Invalid JSON in {file_path} at line {i+1}: {e}")
        
        except Exception as e:
            raise ValidationError(f"Failed to validate {file_path}: {e}")


class DatasetRegistry:
    """Registry for tracking available dataset converters."""
    
    _converters = {}
    
    @classmethod
    def register(cls, format_type: str, converter_class):
        """Register a converter for a format type."""
        cls._converters[format_type] = converter_class
        logger.debug(f"Registered converter for format: {format_type}")
    
    @classmethod
    def get_converter(cls, format_type: str) -> type:
        """Get converter class for a format type."""
        if format_type not in cls._converters:
            raise ValueError(f"No converter registered for format: {format_type}")
        return cls._converters[format_type]
    
    @classmethod
    def list_formats(cls) -> List[str]:
        """List all supported format types."""
        return list(cls._converters.keys())


def convert_dataset(source_path: Union[str, Path], 
                   output_path: Union[str, Path],
                   dataset_name: str,
                   format_type: str,
                   **kwargs) -> Dict[str, Path]:
    """
    Convenience function to convert a dataset to NewAIBench format.
    
    Args:
        source_path: Path to source dataset
        output_path: Path for output files
        dataset_name: Name of the dataset
        format_type: Format type ('beir', 'msmarco', 'trec', etc.)
        **kwargs: Additional configuration options
        
    Returns:
        Dictionary mapping component names to output file paths
    """
    # Create configuration
    config = ConversionConfig(
        source_path=source_path,
        output_path=output_path,
        dataset_name=dataset_name,
        format_type=format_type,
        **kwargs
    )
    
    # Get appropriate converter
    converter_class = DatasetRegistry.get_converter(format_type)
    converter = converter_class(config)
    
    # Perform conversion
    return converter.convert_all()
