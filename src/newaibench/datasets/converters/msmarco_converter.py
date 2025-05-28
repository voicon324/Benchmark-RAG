"""
MS MARCO dataset converter for NewAIBench.

This module converts MS MARCO Passage Ranking dataset to NewAIBench format,
supporting both the original TSV format and Hugging Face datasets.
"""

import json
import csv
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Iterator, Tuple
import pandas as pd

from .base_converter import BaseDatasetConverter, ConversionConfig, DatasetRegistry

logger = logging.getLogger(__name__)


class MSMARCOConverter(BaseDatasetConverter):
    """
    Converter for MS MARCO Passage Ranking dataset.
    
    Supports conversion from:
    - Original TSV files (collection.tsv, queries.*.tsv, qrels.*.tsv)
    - Hugging Face datasets format
    - Preprocessed BEIR format
    
    Output:
    - corpus.jsonl: Documents in NewAIBench format
    - queries.jsonl: Queries in NewAIBench format  
    - qrels.txt: Relevance judgments in TREC format
    """
    
    def __init__(self, config: ConversionConfig):
        """Initialize MS MARCO converter."""
        super().__init__(config)
        
        # MS MARCO specific files
        self.collection_file = None
        self.queries_file = None
        self.qrels_file = None
        
        self._detect_source_format()
    
    def _detect_source_format(self) -> None:
        """Detect MS MARCO source format and locate files."""
        # Look for different MS MARCO file patterns
        source_files = list(self.source_path.glob('*'))
        
        # Original MS MARCO format
        collection_files = [f for f in source_files if 'collection' in f.name.lower()]
        if collection_files:
            self.collection_file = collection_files[0]
            logger.info(f"Found collection file: {self.collection_file}")
        
        # Query files
        query_files = [f for f in source_files if 'queries' in f.name.lower() and f.suffix in ['.tsv', '.txt']]
        if query_files:
            # Prefer dev queries for evaluation
            dev_queries = [f for f in query_files if 'dev' in f.name.lower()]
            self.queries_file = dev_queries[0] if dev_queries else query_files[0]
            logger.info(f"Found queries file: {self.queries_file}")
        
        # QRels files
        qrels_files = [f for f in source_files if 'qrels' in f.name.lower() or 'qrel' in f.name.lower()]
        if qrels_files:
            # Prefer dev qrels for evaluation
            dev_qrels = [f for f in qrels_files if 'dev' in f.name.lower()]
            self.qrels_file = dev_qrels[0] if dev_qrels else qrels_files[0]
            logger.info(f"Found qrels file: {self.qrels_file}")
        
        # BEIR format fallback
        if not self.collection_file:
            beir_corpus = self.source_path / 'corpus.jsonl'
            if beir_corpus.exists():
                self.collection_file = beir_corpus
                logger.info(f"Using BEIR format corpus: {beir_corpus}")
        
        if not self.queries_file:
            beir_queries = self.source_path / 'queries.jsonl'
            if beir_queries.exists():
                self.queries_file = beir_queries
                logger.info(f"Using BEIR format queries: {beir_queries}")
        
        if not self.qrels_file:
            # Look in qrels subdirectory for BEIR format
            qrels_dir = self.source_path / 'qrels'
            if qrels_dir.exists():
                qrels_files = list(qrels_dir.glob('*.tsv'))
                if qrels_files:
                    self.qrels_file = qrels_files[0]
                    logger.info(f"Using BEIR format qrels: {self.qrels_file}")
    
    def convert_corpus(self) -> Path:
        """Convert MS MARCO corpus to NewAIBench format."""
        if not self.collection_file:
            raise FileNotFoundError("MS MARCO collection file not found")
        
        output_file = self.output_path / 'corpus'
        
        logger.info(f"Converting corpus from: {self.collection_file}")
        
        # Determine input format
        if self.collection_file.suffix == '.jsonl':
            docs_iter = self._convert_beir_corpus()
        else:
            docs_iter = self._convert_tsv_corpus()
        
        # Write output
        self._write_jsonl(docs_iter, output_file)
        
        output_path = output_file.with_suffix('.jsonl.gz' if self.config.compress_output else '.jsonl')
        logger.info(f"Corpus conversion completed: {output_path}")
        return output_path
    
    def _convert_tsv_corpus(self) -> Iterator[Dict[str, Any]]:
        """Convert TSV corpus format."""
        logger.info("Processing TSV corpus format")
        
        # MS MARCO collection.tsv format: pid \t passage
        with open(self.collection_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            
            for row_num, row in enumerate(reader):
                if len(row) < 2:
                    continue
                
                self.stats['docs_processed'] += 1
                
                doc_id = str(row[0]).strip()
                text = str(row[1]).strip() if len(row) > 1 else ""
                
                # Create document
                doc = {
                    'doc_id': doc_id,
                    'text': self._normalize_text(text),
                    'metadata': {
                        'source': 'msmarco',
                        'format': 'passage'
                    }
                }
                
                # Add title if available (some versions have 3 columns)
                if len(row) > 2 and row[2].strip():
                    doc['title'] = self._normalize_text(row[2])
                
                # Validate and yield
                if self._validate_document(doc):
                    self.stats['docs_valid'] += 1
                    yield doc
                else:
                    self.stats['docs_skipped'] += 1
                
                if row_num % 100000 == 0:
                    logger.debug(f"Processed {row_num} documents")
    
    def _convert_beir_corpus(self) -> Iterator[Dict[str, Any]]:
        """Convert BEIR format corpus."""
        logger.info("Processing BEIR corpus format")
        
        with open(self.collection_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                
                self.stats['docs_processed'] += 1
                
                try:
                    data = json.loads(line)
                    
                    # BEIR format: _id, title, text
                    doc_id = str(data.get('_id', data.get('id', '')))
                    text = str(data.get('text', data.get('contents', '')))
                    title = str(data.get('title', ''))
                    
                    # Create document
                    doc = {
                        'doc_id': doc_id,
                        'text': self._normalize_text(text),
                        'metadata': {
                            'source': 'msmarco',
                            'format': 'beir'
                        }
                    }
                    
                    if title:
                        doc['title'] = self._normalize_text(title)
                    
                    # Preserve original metadata
                    if self.config.preserve_metadata:
                        for key, value in data.items():
                            if key not in ['_id', 'id', 'text', 'contents', 'title']:
                                doc['metadata'][key] = value
                    
                    # Validate and yield
                    if self._validate_document(doc):
                        self.stats['docs_valid'] += 1
                        yield doc
                    else:
                        self.stats['docs_skipped'] += 1
                
                except json.JSONDecodeError as e:
                    logger.warning(f"Invalid JSON at line {line_num + 1}: {e}")
                    self.stats['docs_skipped'] += 1
                
                if line_num % 100000 == 0:
                    logger.debug(f"Processed {line_num} documents")
    
    def convert_queries(self) -> Path:
        """Convert MS MARCO queries to NewAIBench format."""
        if not self.queries_file:
            raise FileNotFoundError("MS MARCO queries file not found")
        
        output_file = self.output_path / 'queries'
        
        logger.info(f"Converting queries from: {self.queries_file}")
        
        # Determine input format
        if self.queries_file.suffix == '.jsonl':
            queries_iter = self._convert_beir_queries()
        else:
            queries_iter = self._convert_tsv_queries()
        
        # Write output
        self._write_jsonl(queries_iter, output_file)
        
        output_path = output_file.with_suffix('.jsonl.gz' if self.config.compress_output else '.jsonl')
        logger.info(f"Queries conversion completed: {output_path}")
        return output_path
    
    def _convert_tsv_queries(self) -> Iterator[Dict[str, Any]]:
        """Convert TSV queries format."""
        logger.info("Processing TSV queries format")
        
        # MS MARCO queries format: qid \t query
        with open(self.queries_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            
            for row_num, row in enumerate(reader):
                if len(row) < 2:
                    continue
                
                self.stats['queries_processed'] += 1
                
                query_id = str(row[0]).strip()
                text = str(row[1]).strip()
                
                # Create query
                query = {
                    'query_id': query_id,
                    'text': self._normalize_text(text),
                    'metadata': {
                        'source': 'msmarco',
                        'format': 'passage'
                    }
                }
                
                # Validate and yield
                if self._validate_query(query):
                    self.stats['queries_valid'] += 1
                    yield query
                else:
                    self.stats['queries_skipped'] += 1
    
    def _convert_beir_queries(self) -> Iterator[Dict[str, Any]]:
        """Convert BEIR format queries."""
        logger.info("Processing BEIR queries format")
        
        with open(self.queries_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                
                self.stats['queries_processed'] += 1
                
                try:
                    data = json.loads(line)
                    
                    # BEIR format: _id, text
                    query_id = str(data.get('_id', data.get('id', '')))
                    text = str(data.get('text', data.get('query', '')))
                    
                    # Create query
                    query = {
                        'query_id': query_id,
                        'text': self._normalize_text(text),
                        'metadata': {
                            'source': 'msmarco',
                            'format': 'beir'
                        }
                    }
                    
                    # Preserve original metadata
                    if self.config.preserve_metadata:
                        for key, value in data.items():
                            if key not in ['_id', 'id', 'text', 'query']:
                                query['metadata'][key] = value
                    
                    # Validate and yield
                    if self._validate_query(query):
                        self.stats['queries_valid'] += 1
                        yield query
                    else:
                        self.stats['queries_skipped'] += 1
                
                except json.JSONDecodeError as e:
                    logger.warning(f"Invalid JSON at line {line_num + 1}: {e}")
                    self.stats['queries_skipped'] += 1
    
    def convert_qrels(self) -> Path:
        """Convert MS MARCO qrels to NewAIBench format."""
        if not self.qrels_file:
            logger.warning("No qrels file found, creating empty qrels.txt")
            output_file = self.output_path / 'qrels.txt'
            output_file.touch()
            return output_file
        
        output_file = self.output_path / 'qrels.txt'
        
        logger.info(f"Converting qrels from: {self.qrels_file}")
        
        qrels_data = list(self._parse_qrels())
        self._write_qrels(qrels_data, output_file)
        
        logger.info(f"QRels conversion completed: {output_file}")
        return output_file
    
    def _parse_qrels(self) -> Iterator[Tuple[str, str, int]]:
        """Parse qrels from various formats."""
        with open(self.qrels_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            
            for row_num, row in enumerate(reader):
                if len(row) < 3:
                    continue
                
                try:
                    # MS MARCO qrels format: qid \t 0 \t pid \t rel
                    # TREC format: qid \t Q0 \t pid \t rel
                    query_id = str(row[0]).strip()
                    doc_id = str(row[2]).strip()
                    relevance = int(row[3]) if len(row) > 3 else 1
                    
                    self.stats['qrels_processed'] += 1
                    yield (query_id, doc_id, relevance)
                
                except (ValueError, IndexError) as e:
                    logger.warning(f"Invalid qrels line {row_num + 1}: {e}")


# Register the converter
DatasetRegistry.register('msmarco', MSMARCOConverter)
